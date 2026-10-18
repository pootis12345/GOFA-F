"""Separate differentiable-training proofs from actual no-grad numeric decoding."""
import copy
import random

import numpy as np
import torch

from gofa_f.allnode_graph import build_graph
from gofa_f.generative_data import Query
from gofa_f.generative_model import GenerativePrefixLRU
from gofa_f.allnode_model import frozen_cache_binding
from gofa_f.short_codec import numeric_loss
from gofa_f.short_forecast import forecast_batch
from gofa_f.short_data import TEMPLATES


def mechanisms(model, codec, series, queries, tolerance, diagnostic=None, check=lambda: None):
    device = next(model.native.parameters()).device
    original = model.trainable_state()
    version = model.parameter_version
    gradients_before = {n: p.grad for n, p in model.adapter.named_parameters()}
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
           torch.cuda.get_rng_state_all() if device.type == "cuda" else None)
    frozen_versions = [p._version for p in model.native.parameters()]
    proof = {"passed": False, "dtype": str(model.native.dtype), "tolerance": tolerance,
             "differences": {}, "gradient_checks_retain_graph": True, "production_generation_no_grad": True}
    visited, handles = set(), []
    for module in model.modules():
        params = list(module.parameters(recurse=False))
        if params:
            handles.append(module.register_forward_pre_hook(lambda m, a, ps=params: visited.update(id(p) for p in ps)))
    cache = GenerativePrefixLRU(frozen_cache_binding(model, "real-proof"), max_bytes=2*2**30)
    def compare(a, b, name, exact=False):
        proof["differences"][name] = float((a.detach().float()-b.detach().float()).abs().max())
        torch.testing.assert_close(a, b, rtol=0 if exact else tolerance, atol=0 if exact else tolerance, msg=name)
    def grads():
        return {n: p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                for n, p in model.adapter.named_parameters()}
    try:
        selected = [Query(**next(q for q in queries if q["source"] == "pems04" and
                    (q["length"], q["horizon"]) == template)) for template in TEMPLATES]
        for q in selected:
            check()
            data = series[q.source]
            h = np.where(data.mask[q.origin-q.length:q.origin, q.variable],
                         data.values[q.origin-q.length:q.origin, q.variable], np.nan)
            y = np.where(data.mask[q.origin:q.origin+q.horizon, q.variable],
                         data.values[q.origin:q.origin+q.horizon, q.variable], np.nan)
            tokens = codec.batch([codec.record(h, y)]).to(device)
            graph = build_graph(data, q)
            with torch.no_grad():
                context = model.encode_origins([graph], codec, cache, check)
                compare(model(tokens, contexts=context).logits, model(tokens).logits,
                        f"initial_identity_L{q.length}", exact=True)
            if context[0].nodes != 9 or model.origin_executions[-1]["layers"] != len(model.native.model.layers):
                raise AssertionError("All nine sampled nodes must execute the complete shared LLM")
        q = selected[-1]
        data = series[q.source]
        h = data.values[q.origin-q.length:q.origin, q.variable].astype(float)
        y = data.values[q.origin:q.origin+q.horizon, q.variable].astype(float)
        graph = build_graph(data, q)
        tokens = codec.batch([codec.record(h, y)]).to(device)
        optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=2e-4, weight_decay=1e-4)
        for _ in range(3):
            check()
            optimizer.zero_grad(set_to_none=True)
            context = model.encode_origins([graph], codec, cache, check)
            prefix = model.cached_prefix(tokens, cache, "stage19_teacher_numeric")
            loss, _ = numeric_loss(model(tokens, contexts=context, prefix=prefix).logits, tokens.labels, codec)
            loss.backward()
            optimizer.step()
            model.mark_updated()
        for i, exchange in enumerate(model.adapter.exchanges):
            for name in ("down", "query", "key", "value", "edge_score", "gate", "up"):
                if not any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                           for p in getattr(exchange, name).parameters()):
                    raise AssertionError(f"Unused branch after actual updates: {i}/{name}")
        proof["actual_nonzero_diagnostic_updates"] = 3
        from gofa_f.short_training_checks import optimization_checks
        def optimization_diagnostic(value):
            proof["training_optimizations_fresh"] = value
            if diagnostic is not None:
                diagnostic(proof)
        proof["training_optimizations_fresh"] = optimization_checks(model, codec, series, queries,
            tolerance, check=check, diagnostic=optimization_diagnostic)
        def training_path(kind, current=None):
            check()
            current = tokens if current is None else current
            model.zero_grad(set_to_none=True)
            context = model.encode_origins([graph], codec, cache if kind != "full" else None, check)
            if kind == "differentiable_incremental":
                cut = int(current.prompt_ends[0])+1
                part = current.first(cut)
                output = model(part, contexts=context, use_cache=True)
                pieces, kv = [output.logits], output.past_key_values
                for index in range(cut, current.input_ids.shape[1]):
                    part = part.append(current.input_ids[:, index:index+1])
                    output = model(part, contexts=context, cache=kv, use_cache=True)
                    pieces.append(output.logits)
                    kv = output.past_key_values
                logits = torch.cat(pieces, 1)
            else:
                prefix = model.cached_prefix(current, cache, "stage19_teacher_numeric") if kind == "prefix" else None
                logits = model(current, contexts=context, prefix=prefix).logits
            numeric_loss(logits, current.labels, codec)[0].backward()
            return logits.detach(), grads()
        reference, expected_gradients = training_path("full")
        for kind in ("prefix", "differentiable_incremental"):
            current = tokens
            if kind == "differentiable_incremental" and device.type == "cpu":
                numeric_positions = torch.nonzero(tokens.labels[0] != -100).flatten()
                current = tokens.first(int(numeric_positions[1])+1)
                reference, expected_gradients = training_path("full", current)
            actual, gradients = training_path(kind, current)
            compare(actual, reference, kind+":logits")
            for name, gradient in gradients.items():
                compare(gradient, expected_gradients[name], kind+":"+name)
        proof["nonzero_training_paths"] = True
        proof["differentiable_incremental_numeric_slots"] = 24 if device.type == "cuda" else 2
        with torch.no_grad():
            context = model.encode_origins([graph], codec, cache, check)
            horizon = 24 if device.type == "cuda" else 3
            cached = forecast_batch(model, codec, [h], horizon, context, check, cached=True)
            full = forecast_batch(model, codec, [h], horizon, context, check, cached=False)
            if cached["numeric_ids"] != full["numeric_ids"] or cached["generated_ids"] != full["generated_ids"]:
                raise AssertionError("Actual greedy tokens diverged; logit tolerance is not sufficient")
            np.testing.assert_array_equal(cached["predictions"], full["predictions"])
            if not all(cached["format_complete"]):
                raise AssertionError("Numeric slot completion failed")
            proof["actual_no_grad_generation"] = {"horizon": horizon,
                "tokens": len(cached["generated_ids"][0]), "forward_calls": cached["forward_calls"],
                "min_margin": cached["min_numeric_margin"], "exact_tokens": True}
        saved, moments = model.trainable_state(), copy.deepcopy(optimizer.state_dict())
        materialized = grads()
        optimizer.step()
        expected = model.trainable_state()
        model.load_trainable(saved)
        optimizer.load_state_dict(moments)
        for name, p in model.adapter.named_parameters():
            p.grad = materialized[name].clone()
        optimizer.step()
        for name, value in model.trainable_state().items():
            compare(value, expected[name], "exact_materialized_optimizer_replay:"+name, exact=True)
        proof["exact_saved_state_not_claimed_bitwise_new_cuda_gradients"] = True
        if any(p.requires_grad or p.grad is not None or p._version != old
               for p, old in zip(model.native.parameters(), frozen_versions)):
            raise AssertionError("Frozen native backbone changed")
        model._verified_forward_parameters = sum(p.numel() for p in model.parameters() if id(p) in visited)
        proof["parameters"] = model.parameter_counts()
        if sum(p.numel() for p in model.adapter.parameters() if id(p) in visited) != 1060998:
            raise AssertionError("All added modules must actually execute")
        proof["maximum_shape"] = {"nodes": 9, "history": 48, "horizon": 24,
            "teacher_tokens": tokens.input_ids.shape[1],
            "allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
            "reserved_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else None}
        if device.type == "cuda" and proof["maximum_shape"]["reserved_bytes"] > 38*2**30:
            raise AssertionError("Maximum shape exceeds the Torch cap")
        proof["passed"] = True
        return proof
    except Exception as exc:
        proof.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        for hook in handles:
            hook.remove()
        model.load_trainable(original)
        model.parameter_version = version
        for name, p in model.adapter.named_parameters():
            p.grad = gradients_before[name]
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state_all(rng[3])
        cache.entries.clear()
        if diagnostic is not None:
            diagnostic(proof)
