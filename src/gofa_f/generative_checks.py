"""Preregistered real-weight Stage17 checks. Never loads a target query."""
import copy

import torch

from gofa_f.generative_data import Query
from gofa_f.generative_model import QueryTokens, completion_loss


def mechanisms(model, codec, series, queries, tolerance, diagnostic=None, check=lambda: None):
    device = next(model.parameters()).device
    before = model.trainable_state()
    version = model.parameter_version
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    native_versions = [p._version for p in model.native.parameters()]
    original_grads = {n: p.grad for n, p in model.adapter.named_parameters()}
    results = {"tolerance": tolerance, "graph_layers": list(model.graph_layers),
               "prefix_layers": model.prefix_layers, "target_queries_used": 0, "differences": {}}
    def compare(actual, expected, name, exact=False):
        results["differences"][name] = float((actual.detach().float()-expected.detach().float()).abs().max())
        try:
            torch.testing.assert_close(actual, expected, rtol=0 if exact else tolerance,
                                       atol=0 if exact else tolerance)
        except AssertionError as exc:
            raise AssertionError(f"{name}: {exc}") from exc
    try:
        selected = []
        for length in (10, 12, 14, 48, 192, 288):
            q = next(q for q in queries if q["length"] == length and q["source"] == "pems04")
            selected.append(Query(**q))
        for q in selected:
            check()
            print(f"real-weight {device.type} history={q.length} nodes=307", flush=True)
            source = series[q.source]
            history, response = source.training_record(q)
            batch = codec.batch([codec.record(history, response[:1])]).to(device)
            graphs = [source.graph(q).to(device)]
            with torch.no_grad():
                plain = model(batch).logits
                graph = model(batch, graphs).logits
                compare(graph, plain, f"initial_native_L{q.length}", exact=True)
                prefix = model.prefix(batch)
                compare(model(batch, graphs, prefix=prefix).logits, graph, f"prefix_L{q.length}")
            if not torch.isfinite(graph).all():
                raise AssertionError("Nonfinite real-shape logits")
        results["six_real_history_shapes"] = True
        # Do not let zero output projections hide gradient/cache problems.
        q = selected[0]
        source = series[q.source]
        history, response = source.training_record(q)
        batch = codec.batch([codec.record(history, response[:1])]).to(device)
        graphs = [source.graph(q).to(device)]
        optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=2e-4)
        for _ in range(3):
            check()
            optimizer.zero_grad()
            completion_loss(model(batch, graphs).logits, batch.labels)[0].backward()
            optimizer.step()
            model.mark_updated()
        if not all(any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
            for p in layer.parameters()) for layer in
            [model.adapter.encoder, *model.adapter.gnns, *model.adapter.interactions]):
            raise AssertionError("An added branch did not learn")
        optimizer.zero_grad()
        complete = model(batch, graphs).logits
        completion_loss(complete, batch.labels)[0].backward()
        gradients = {n: p.grad.detach().clone() for n, p in model.adapter.named_parameters()}
        model.zero_grad()
        prefix = model.prefix(batch)
        cached = model(batch, graphs, prefix=prefix).logits
        compare(cached, complete, "nonzero_prefix_logits")
        completion_loss(cached, batch.labels)[0].backward()
        for n, p in model.adapter.named_parameters():
            compare(p.grad, gradients[n], "prefix_gradient:"+n)
        results["nonzero_graph_prefix_predictions_gradients"] = True
        model.zero_grad()
        # Prefill prompt then decode the answer, retaining autograd for equivalence only.
        cut = int(batch.prompt_ends[0])+1
        part = QueryTokens(batch.input_ids[:, :cut], batch.attention_mask[:, :cut],
                           batch.labels[:, :cut], batch.prompt_ends)
        output = model(part, graphs, use_cache=True)
        pieces, cache = [output.logits], output.past_key_values
        for index in range(cut, batch.input_ids.shape[1]):
            check()
            part = QueryTokens(batch.input_ids[:, index:index+1], batch.attention_mask[:, :index+1],
                               batch.labels[:, index:index+1], batch.prompt_ends)
            output = model(part, graphs, use_cache=True, cache=cache)
            cache = output.past_key_values
            pieces.append(output.logits)
        incremental = torch.cat(pieces, 1)
        compare(incremental, complete, "incremental_logits")
        completion_loss(incremental, batch.labels)[0].backward()
        for n, p in model.adapter.named_parameters():
            compare(p.grad, gradients[n], "incremental_gradient:"+n)
        results["incremental_logits_gradients"] = True
        model.mark_updated()
        try:
            model(part, graphs, use_cache=True, cache=cache)
        except ValueError:
            results["kv_invalidated_after_update"] = True
        else:
            raise AssertionError("Stale graph KV accepted")
        # Exact checkpoint model/optimizer/gradient restoration and next-step trajectory.
        saved = copy.deepcopy(optimizer.state_dict())
        state = model.trainable_state()
        current_grads = {n: p.grad.detach().clone() for n, p in model.adapter.named_parameters()}
        optimizer.step()
        expected = model.trainable_state()
        model.load_trainable(state)
        optimizer.load_state_dict(saved)
        for n, p in model.adapter.named_parameters():
            p.grad = current_grads[n].clone()
        optimizer.step()
        for n, value in model.trainable_state().items():
            torch.testing.assert_close(value, expected[n], rtol=0, atol=0)
        results["exact_optimizer_restore"] = True
        if any(p.requires_grad or p.grad is not None or p._version != old
               for p, old in zip(model.native.parameters(), native_versions)):
            raise AssertionError("Frozen native parameters changed")
        results["native_frozen"] = True
        results["passed"] = True
        return results
    except Exception as exc:
        results.update(passed=False, error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if diagnostic is not None:
            diagnostic(results)
        model.load_trainable(before)
        model.parameter_version = version
        for n, p in model.adapter.named_parameters():
            p.grad = original_grads[n]
        torch.set_rng_state(rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
