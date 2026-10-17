"""Real-weight all-node acceptance. CPU FP32 and CUDA BF16 are separate experiments."""
import copy
import gc

import numpy as np
import torch

from gofa_f.allnode_graph import build_graph
from gofa_f.allnode_forecast import forecast_batch
from gofa_f.generative_data import Query
from gofa_f.generative_model import GenerativePrefixLRU, completion_loss


def mechanisms(model, codec, series, queries, tolerance, diagnostic=None, check=lambda: None):
    device = next(model.native.parameters()).device
    initial = model.trainable_state()
    version = model.parameter_version
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    native_versions = [p._version for p in model.native.parameters()]
    original_grads = {n: p.grad for n, p in model.adapter.named_parameters()}
    visited = set()
    def visit(module, *_):
        visited.update(id(p) for p in module.parameters(recurse=False))
    handles = [module.register_forward_hook(visit) for module in model.modules()
               if list(module.parameters(recurse=False))]
    result = {"passed": False, "dtype": str(model.native.dtype), "tolerance": tolerance,
        "graph_layers": list(model.graph_layers), "prefix_layers": model.prefix_layers,
        "target_queries_used": 0, "differences": {}, "state_restore": "exact",
        "cuda_replay": "same_execution_tolerance_not_cross_dtype_bitwise"}
    cache = GenerativePrefixLRU("acceptance_only", max_bytes=512*2**20)
    def compare(actual, expected, label, exact=False):
        error = (actual.detach().float()-expected.detach().float()).abs()
        result["differences"][label] = float(error.max())
        allowed = 0 if exact else tolerance+tolerance*expected.detach().float().abs()
        if (error > allowed).any():
            index = int(error.flatten().argmax())
            position = np.unravel_index(index, error.shape)
            result["failed_comparison"] = {"label": label, "shape": list(error.shape),
                "max_position": [int(i) for i in position], "violations": int((error > allowed).sum()),
                "actual_at_max": float(actual[position]), "expected_at_max": float(expected[position])}
        torch.testing.assert_close(actual, expected, rtol=0 if exact else tolerance,
                                   atol=0 if exact else tolerance, msg=label)
    def gradients():
        return {n: p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                for n, p in model.adapter.named_parameters()}
    try:
        selected = [Query(**next(q for q in queries if q["source"] == "pems04" and q["length"] == length))
                    for length in (10, 12, 14, 48, 192, 288)]
        for q in selected:
            check()
            print(f"all-node real {device} L={q.length}", flush=True)
            source = series[q.source]
            history, response = source.training_record(q)
            graphs = [build_graph(source, q)]
            batch = codec.batch([codec.record(history, response[:1])]).to(device)
            with torch.no_grad():
                contexts = model.encode_origins(graphs, codec, cache, check)
                native = model(batch).logits
                complete = model(batch, contexts=contexts).logits
                compare(complete, native, f"initial_L{q.length}", exact=True)
                prefix = model.cached_prefix(batch, cache, "teacher_forced")
                compare(model(batch, contexts=contexts, prefix=prefix).logits,
                        complete, f"initial_prefix_L{q.length}")
                if contexts[0].nodes != 9 or model.origin_executions[-1]["layers"] != len(model.native.model.layers):
                    raise AssertionError("Maximum nine-node shared-LLM path not exercised")
        result["all_six_shapes_nine_nodes"] = True
        q = selected[0]
        source = series[q.source]
        history, response = source.training_record(q)
        graphs = [build_graph(source, q)]
        batch = codec.batch([codec.record(history, response[:1])]).to(device)
        optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=2e-4, weight_decay=1e-4)
        for _ in range(3):
            check()
            optimizer.zero_grad(set_to_none=True)
            contexts = model.encode_origins(graphs, codec, cache, check)
            prefix = model.cached_prefix(batch, cache, "teacher_forced")
            completion_loss(model(batch, contexts=contexts, prefix=prefix).logits, batch.labels)[0].backward()
            optimizer.step()
            model.mark_updated()
        for index, exchange in enumerate(model.adapter.exchanges):
            for name in ("down", "query", "key", "value", "edge_score", "gate", "up"):
                module = getattr(exchange, name)
                if not any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                           for p in module.parameters()):
                    raise AssertionError(f"Exchange {index+1}/{name} did not participate after diagnostic updates")
        result["nonzero_diagnostic_updates"] = 3
        def run(kind, current):
            model.zero_grad(set_to_none=True)
            contexts = model.encode_origins(graphs, codec, cache if kind != "full" else None, check)
            if kind == "incremental":
                cut = int(current.prompt_ends[0])+1
                part = current.first(cut)
                output = model(part, contexts=contexts, use_cache=True)
                pieces, kv = [output.logits], output.past_key_values
                for index in range(cut, current.input_ids.shape[1]):
                    check()
                    part = part.append(current.input_ids[:, index:index+1])
                    output = model(part, contexts=contexts, use_cache=True, cache=kv)
                    pieces.append(output.logits)
                    kv = output.past_key_values
                logits = torch.cat(pieces, 1)
            else:
                prefix = model.cached_prefix(current, cache, "teacher_forced") if kind == "prefix" else None
                logits = model(current, contexts=contexts, prefix=prefix).logits
            completion_loss(logits, current.labels)[0].backward()
            return logits.detach().clone(), gradients()
        # Three rolling views with deliberate new extremes: original graph input never changes.
        for offset in (0, 16, 32):
            check()
            rolled = np.r_[history, np.full(offset, 1e4 if offset == 16 else -1e4)]
            current = codec.batch([codec.record(rolled, response[:1])]).to(device)
            reference, grads = run("full", current)
            for kind in ("prefix", "incremental"):
                actual, actual_grads = run(kind, current)
                compare(actual, reference, f"nonzero_{kind}_chunk{offset}")
                for name, value in actual_grads.items():
                    compare(value, grads[name], f"{kind}_chunk{offset}_gradient:{name}")
        result["nonzero_three_rolling_views_predictions_gradients"] = True
        # Same-precision actual generation, not just teacher-forced logits.
        if device.type == "cuda":
            with torch.no_grad():
                contexts = model.encode_origins(graphs, codec, cache, check)
                trace_cached, trace_complete = [], []
                cached = forecast_batch(model, codec, [history], 24, contexts, check,
                    cached=True, trace=trace_cached.append)
                complete = forecast_batch(model, codec, [history], 24, contexts, check,
                    cached=False, trace=trace_complete.append)
                np.testing.assert_array_equal(cached[0], complete[0])
                np.testing.assert_array_equal(cached[1], complete[1])
                if trace_cached != trace_complete:
                    raise AssertionError("Actual greedy tokens differ even if fallback metrics coincide")
            result["two_block_actual_generation_identical"] = True
        else:
            result["real_two_block_greedy_cuda_required"] = True
        # Exact state restoration and numerical next-step replay are distinct assertions.
        saved_adapter, saved_optimizer = model.trainable_state(), copy.deepcopy(optimizer.state_dict())
        saved_grads = gradients()
        optimizer.step()
        expected = model.trainable_state()
        model.load_trainable(saved_adapter)
        optimizer.load_state_dict(saved_optimizer)
        for name, value in model.trainable_state().items():
            compare(value, saved_adapter[name], "state_restore:"+name, exact=True)
        for name, p in model.adapter.named_parameters():
            p.grad = saved_grads[name].clone()
        optimizer.step()  # replay of already materialized identical gradients
        for name, value in model.trainable_state().items():
            compare(value, expected[name], "same_gradient_optimizer_replay:"+name, exact=True)
        result["exact_materialized_state_and_optimizer_replay"] = True
        if device.type == "cuda":
            # Longest origin and the final H=144 teacher-forced block, with populated AdamW state.
            check()
            long_q = selected[-1]
            long_source = series[long_q.source]
            long_history, _ = long_source.training_record(long_q)
            extra = long_source.values[long_q.origin:long_q.origin+128, long_q.variable]
            answer = long_source.values[long_q.origin+128:long_q.origin+144, long_q.variable]
            long_tokens = codec.batch([codec.record(np.r_[long_history, extra], answer)]).to(device)
            model.zero_grad(set_to_none=True)
            long_context = model.encode_origins([build_graph(long_source, long_q)], codec, check=check)
            loss = completion_loss(model(long_tokens, contexts=long_context).logits, long_tokens.labels)[0]
            loss.backward()
            optimizer.step()
            model.mark_updated()
            result["maximum_shape"] = {"nodes": 9, "origin_history": 288, "horizon": 144,
                "teacher_chunk": 128, "teacher_tokens": long_tokens.input_ids.shape[1],
                "allocated_bytes": torch.cuda.max_memory_allocated(),
                "reserved_bytes": torch.cuda.max_memory_reserved()}
            if result["maximum_shape"]["reserved_bytes"] > 38*2**30:
                raise AssertionError("Maximum shape exceeds 38GiB Torch cap")
            del long_context, long_tokens, loss
        if any(p.requires_grad or p.grad is not None or p._version != old
               for p, old in zip(model.native.parameters(), native_versions)):
            raise AssertionError("Frozen native changed")
        model._verified_forward_parameters = sum(p.numel() for p in model.parameters() if id(p) in visited)
        result["parameter_counts"] = model.parameter_counts()
        result["forward_added_parameters"] = sum(p.numel() for p in model.adapter.parameters() if id(p) in visited)
        if result["forward_added_parameters"] != sum(p.numel() for p in model.adapter.parameters()):
            raise AssertionError("Unused registered adapter module")
        result["passed"] = True
        return result
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        for handle in handles:
            handle.remove()
        if diagnostic is not None:
            diagnostic(result)
        model.load_trainable(initial)
        model.parameter_version = version
        for name, p in model.adapter.named_parameters():
            p.grad = original_grads[name]
        torch.set_rng_state(rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        cache.entries.clear()
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
