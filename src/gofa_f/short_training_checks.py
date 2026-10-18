"""Compare optimized training with the unchanged dense, single-query reference."""
import random

import numpy as np
import torch
from torch.nn import functional as F

from gofa_f.allnode_graph import build_graph
from gofa_f.allnode_model import frozen_cache_binding
from gofa_f.generative_data import Query, sha
from gofa_f.generative_model import GenerativePrefixLRU
from gofa_f.short_codec import numeric_loss
from gofa_f.short_training import compact_forward, compact_loss, teacher_prefix


def assert_value_close(actual, expected, tolerance, name):
    """BF16->FP32 is exact: compare values, not CE preparation tensor dtypes."""
    torch.testing.assert_close(actual.float(), expected.float(),
        rtol=tolerance, atol=tolerance, msg=lambda message: name+"\n"+message)


def optimization_checks(model, codec, series, queries, tolerance, check=lambda: None,
                        checkpoint=None, diagnostic=None):
    device = next(model.native.parameters()).device
    original, version = model.trainable_state(), model.parameter_version
    gradients_before = {n: p.grad for n, p in model.adapter.named_parameters()}
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
           torch.cuda.get_rng_state_all() if device.type == "cuda" else None)
    proof = {"passed": False, "tolerance": tolerance, "differences": {},
             "microbatch": 2, "finite_vocabulary": 10000, "generation_path_changed": False,
             "diagnostics": {}}
    cache = GenerativePrefixLRU(frozen_cache_binding(model, "optimization-proof"), max_bytes=2*2**30)
    def compare(a, b, name):
        error = (a.detach().float()-b.detach().float()).abs()
        proof["differences"][name] = float(error.max())
        bad = error > tolerance+tolerance*b.detach().float().abs()
        flat = int(error.flatten().argmax())
        proof["diagnostics"][name] = {"violations": int(bad.sum()), "elements": bad.numel(),
            "actual_dtype": str(a.dtype), "reference_dtype": str(b.dtype),
            "actual_at_max": float(a.detach().flatten()[flat]),
            "reference_at_max": float(b.detach().flatten()[flat])}
        assert_value_close(a, b, tolerance, name)
    try:
        if checkpoint is not None:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model.load_trainable(saved["adapter"])
            model.parameter_version = saved["parameter_version"]
            proof["checkpoint_sha256"] = sha(checkpoint)
            proof["checkpoint_updates"] = saved["state"]["updates"]
            proof["checkpoint_pending"] = saved["state"]["pending"]
        all_queries = [Query(**q) for q in queries if q["source"] == "pems04"]
        largest = [q for q in all_queries if (q.length, q.horizon) == (48, 24)][:2]
        short = next(q for q in all_queries if (q.length, q.horizon) == (10, 1))
        if len(largest) != 2:
            raise ValueError("Two distinct maximum-shape validation queries required")
        for name, pair in [("mixed", [short, largest[0]]), ("maximum", largest)]:
            check()
            records, graphs = [], []
            for q in pair:
                data = series[q.source]
                h = np.where(data.mask[q.origin-q.length:q.origin, q.variable],
                    data.values[q.origin-q.length:q.origin, q.variable], np.nan)
                y = np.where(data.mask[q.origin:q.origin+q.horizon, q.variable],
                    data.values[q.origin:q.origin+q.horizon, q.variable], np.nan)
                records.append(codec.record(h, y))
                graphs.append(build_graph(data, q))
            def execute(kind):
                model.zero_grad(set_to_none=True)
                scores, total, prefixes, hidden_states, native_projections = [], 0., [], [], []
                groups = [(records, graphs)] if kind == "batch2" else \
                    [([r], [g]) for r, g in zip(records, graphs)]
                for own_records, own_graphs in groups:
                    check()
                    tokens = codec.batch(own_records).to(device)
                    contexts = model.encode_origins(own_graphs, codec, cache, check)
                    valid = tokens.labels[:, 1:] != -100
                    captured = []
                    handle = model.native.model.norm.register_forward_hook(
                        lambda module, args, value: captured.append(value.detach()))
                    try:
                        if kind == "dense":
                            prefix = model.cached_prefix(tokens, cache, "stage19_teacher_numeric")
                            logits = model(tokens, contexts=contexts, prefix=prefix).logits
                            loss, _ = numeric_loss(logits, tokens.labels, codec)
                            scores.append(logits[:, :-1][valid].index_select(-1, codec.alphabet(device)).detach())
                        else:
                            prefix = teacher_prefix(model, tokens, cache)
                            result = compact_forward(model, codec, tokens, contexts, prefix)
                            scores.append(result[0].detach())
                            loss, _ = compact_loss(result)
                    finally:
                        handle.remove()
                    prefixes.append(prefix[0][:, :-1][valid].detach())
                    selected = captured[-1][:, :-1][valid]
                    hidden_states.append(selected)
                    with torch.no_grad():
                        # Isolate the projection from all hidden-state differences.
                        parts = []
                        for first in range(0, len(selected), 128):
                            own = selected[first:first+128]
                            full = F.linear(F.pad(own, (0, 0, 0, 128-len(own))),
                                model.native.lm_head.weight, model.native.lm_head.bias)
                            parts.append(full[:len(own)].index_select(-1, codec.alphabet(device)))
                        native_projections.append(torch.cat(parts))
                    total += float(loss.detach())
                    loss.backward()
                gradients = {n: p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                             for n, p in model.adapter.named_parameters()}
                return (torch.cat(scores), torch.tensor(total), gradients,
                        torch.cat(prefixes), torch.cat(hidden_states), torch.cat(native_projections))
            expected, loss, expected_gradients, expected_prefix, expected_hidden, expected_native = execute("dense")
            for kind in ("compact1", "batch2"):
                actual, actual_loss, gradients, prefix, hidden, native = execute(kind)
                compare(prefix, expected_prefix, name+":"+kind+":frozen_prefix")
                compare(hidden, expected_hidden, name+":"+kind+":final_hidden")
                compare(native, expected_native, name+":"+kind+":same_hidden_native_projection")
                compare(actual, expected, name+":"+kind+":numeric_logits")
                compare(actual_loss, loss, name+":"+kind+":loss")
                for key, value in gradients.items():
                    compare(value, expected_gradients[key], name+":"+kind+":"+key)
        proof["maximum_shape"] = {"queries": 2, "nodes_per_query": 9, "history": 48, "horizon": 24,
            "allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
            "reserved_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else None}
        if device.type == "cuda" and proof["maximum_shape"]["reserved_bytes"] > 38*2**30:
            raise AssertionError("Two-query training exceeds Torch38GiB")
        proof["passed"] = True
        return proof
    except Exception as exc:
        proof.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        model.load_trainable(original)
        model.parameter_version = version
        for name, p in model.adapter.named_parameters():
            p.grad = gradients_before[name]
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state_all(rng[3])
        if diagnostic:
            diagnostic(proof)
