"""Real new-source nonzero training/sample paths; diagnostic updates are rolled back."""
from pathlib import Path

import numpy as np
import torch

from gofa_f.allnode_graph import build_graph
from gofa_f.allnode_stage import cpu_copy, rng_state, restore_rng
from gofa_f.generative_data import Query
from gofa_f.generative_model import GenerativePrefixLRU
from gofa_f.short_migration import assert_equal
from gofa_f.short_sampled_state import atomic_torch
from gofa_f.short_sampling_diagnostic import generate, sample_seed
from gofa_f.short_source_diagnostic import compare_scores
from gofa_f.short_training import compact_forward, compact_loss, teacher_prefix, training_batch
from gofa_f.source_retraining_data import VIEWS


def full_reference_loss(logits, labels, codec):
    """Dense-model reference uses the production observed-slot CE execution.

    The legacy spatial [B,V,T] CE kernel is not the current training loss
    implementation. At3500 it differed from rowwise CE by1.67e-5 in FP32,
    despite identical logits. Do not compare different reduction executions
    when accepting the transformer/full-prefix paths. Vocabulary, label shift,
    masks and query weighting remain independently reconstructed here.
    """
    shifted = labels[:, 1:]
    valid = shifted != -100
    counts = valid.sum(-1)
    if (counts == 0).any():
        raise ValueError('Every query needs observed target slots')
    rows, steps = torch.nonzero(valid, as_tuple=True)
    scores = logits[rows, steps].float().index_select(-1, codec.alphabet(logits.device))
    targets = codec.lookup(logits.shape[-1], logits.device)[shifted[rows, steps]]
    if (targets < 0).any():
        raise ValueError('Invalid supervised numeric token')
    losses = torch.nn.functional.cross_entropy(scores, targets, reduction='none')
    return (scores.new_zeros(len(counts)).scatter_add(0, rows, losses)/counts).sum()


def mechanisms(model, codec, series, queries, tolerance, output, check=lambda: None):
    original, version, rng = model.trainable_state(), model.parameter_version, rng_state()
    original_grads = {n: cpu_copy(p.grad) for n, p in model.adapter.named_parameters()}
    frozen = [p._version for p in model.native.parameters()]
    device = next(model.native.parameters()).device
    cache = GenerativePrefixLRU('stage23-mechanisms', max_bytes=2*2**30)
    result = dict(passed=False, tolerance=tolerance, actual_diagnostic_updates=0, records=[])
    optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=5e-5, weight_decay=1e-4)
    selected = []
    for name in VIEWS:
        own = [Query(**q) for q in queries if q['source'] == name]
        # Choose the maximum sampled-node case using history only, not labels.
        selected.append(max(own, key=lambda q: len(build_graph(series[name], q).variable_ids)))
    def records(qs):
        rows, graphs = [], []
        for q in qs:
            data = series[q.source]
            h = np.where(data.mask[q.origin-q.length:q.origin, q.variable],
                         data.values[q.origin-q.length:q.origin, q.variable], np.nan)
            y = np.where(data.mask[q.origin:q.origin+q.horizon, q.variable],
                         data.values[q.origin:q.origin+q.horizon, q.variable], np.nan)
            rows.append(codec.record(h, y))
            graphs.append(build_graph(data, q))
        return rows, graphs
    def gradients():
        return {n: cpu_copy(p.grad) for n, p in model.adapter.named_parameters()}
    try:
        for step in range(3):
            check()
            q = selected[step % 4]
            rows, graphs = records([q])
            model.zero_grad(set_to_none=True)
            tokens = codec.batch(rows).to(device)
            context = model.encode_origins(graphs, codec, cache, check)
            loss, _ = compact_loss(compact_forward(model, codec, tokens, context,
                                                    teacher_prefix(model, tokens, cache)))
            loss.backward()
            optimizer.step()
            model.mark_updated()
            result['actual_diagnostic_updates'] += 1
        for name, p in model.adapter.named_parameters():
            if p.grad is None or not torch.isfinite(p.grad).all() or not p.grad.abs().sum() > 0:
                raise ValueError('Unused new-source adapter parameter: '+name)
        for q in selected:
            check()
            rows, graphs = records([q])
            graph = graphs[0]
            if len(graph.variable_ids) != 9:
                raise ValueError('Actual new-source maximum must exercise nine nodes')
            data = series[q.source]
            # The entire graph is fixed before t0: mutate future inputs only.
            future, mask = data.values[q.origin:].copy(), data.mask[q.origin:].copy()
            try:
                data.values[q.origin:] += 12345
                data.mask[q.origin:] = ~data.mask[q.origin:]
                altered = build_graph(data, q)
                if altered.identity != graph.identity:
                    raise ValueError('Future data changed origin graph')
            finally:
                data.values[q.origin:], data.mask[q.origin:] = future, mask
            tokens = codec.batch(rows).to(device)
            model.zero_grad(set_to_none=True)
            context = model.encode_origins(graphs, codec, check=check)
            full = model(tokens, contexts=context).logits
            valid = tokens.labels[:, 1:] != -100
            reference = full[:, :-1][valid].index_select(-1, codec.alphabet(device)).detach()
            full_reference_loss(full, tokens.labels, codec).backward()
            expected = gradients()
            del full, context
            model.zero_grad(set_to_none=True)
            context = model.encode_origins(graphs, codec, cache, check)
            actual = compact_forward(model, codec, tokens, context, teacher_prefix(model, tokens, cache))
            loss, _ = compact_loss(actual)
            torch.testing.assert_close(actual[0].float(), reference.float(), rtol=tolerance, atol=tolerance)
            loss.backward()
            for name, p in model.adapter.named_parameters():
                torch.testing.assert_close(p.grad.float().cpu(), expected[name].float(), rtol=tolerance, atol=tolerance,
                                           msg='new_source_gradient:'+name)
            model.zero_grad(set_to_none=True)
            with torch.no_grad():
                context = model.encode_origins(graphs, codec, cache, check)
                a, b = [], []
                h = np.where(data.mask[q.origin-q.length:q.origin, q.variable],
                             data.values[q.origin-q.length:q.origin, q.variable], np.nan)
                kwargs = dict(contexts=context, seed=sample_seed(q.__dict__, 0), check=check)
                sampled_horizon = q.horizon if device.type == 'cuda' else 3
                cached = generate(model, codec, h, sampled_horizon, trace=a, **kwargs)
                full_sample = generate(model, codec, h, sampled_horizon, cached=False, trace=b, **kwargs)
                comparison = compare_scores(a, b, tolerance)
                if comparison['violations'] or cached['generated_ids'] != full_sample['generated_ids']:
                    raise ValueError('New-source full/incremental sampling diverged')
            result['records'].append(dict(query=q.__dict__, nodes=len(graph.variable_ids),
                logits_max_difference=float((actual[0].detach().float()-reference.float()).abs().max()),
                full_prefix_gradients_passed=True, future_causality_passed=True,
                sampling_horizon=sampled_horizon, sampling_tokens_equal=True, comparison=comparison))
            del actual, context
        result['two_query_training'] = []
        for pair in ([selected[0], selected[2]], selected[2:]):
            model.zero_grad(set_to_none=True)
            expected_loss = 0.
            for q in pair:
                rows, graphs = records([q])
                tokens = codec.batch(rows).to(device)
                full = model(tokens, contexts=model.encode_origins(graphs, codec, check=check)).logits
                loss = full_reference_loss(full, tokens.labels, codec)
                expected_loss += float(loss.detach())
                loss.backward()
                del full
            expected = gradients()
            model.zero_grad(set_to_none=True)
            loss, _ = training_batch(model, codec, series, pair, cache, check)
            torch.testing.assert_close(loss.detach().float(), torch.tensor(expected_loss, device=device),
                                       rtol=tolerance, atol=tolerance)
            loss.backward()
            for name, p in model.adapter.named_parameters():
                torch.testing.assert_close(p.grad.float().cpu(), expected[name].float(), rtol=tolerance, atol=tolerance,
                                           msg='two_query_new_source:'+name)
            result['two_query_training'].append(dict(queries=[q.__dict__ for q in pair], passed=True))
        model.zero_grad(set_to_none=True)
        # Exact materialized partial optimizer state, RNG and adapter restoration.
        rows, graphs = records([selected[-1]])
        tokens = codec.batch(rows).to(device)
        loss, _ = compact_loss(compact_forward(model, codec, tokens, model.encode_origins(graphs, codec, check=check)))
        loss.backward()
        payload = dict(adapter=model.trainable_state(), optimizer=cpu_copy(optimizer.state_dict()),
                       gradients=gradients(), rng=rng_state(), parameter_version=model.parameter_version,
                       pending=1, pending_tokens=int((tokens.labels != -100).sum()))
        test_path = Path(output)/'mechanism-partial-state.pt'
        atomic_torch(test_path, payload)
        loaded = torch.load(test_path, map_location='cpu', weights_only=False)
        assert_equal(payload, loaded)
        optimizer.step()
        expected_after = model.trainable_state()
        model.load_trainable(loaded['adapter'])
        optimizer.load_state_dict(loaded['optimizer'])
        for name, p in model.adapter.named_parameters():
            p.grad = loaded['gradients'][name].to(device)
        restore_rng(loaded['rng'])
        optimizer.step()
        assert_equal(expected_after, model.trainable_state())
        test_path.unlink()
        if any(p.requires_grad or p.grad is not None or p._version != old
               for p, old in zip(model.native.parameters(), frozen)):
            raise ValueError('Frozen backbone changed')
        result.update(passed=True, exact_materialized_state_replay=True,
            does_not_claim_bitwise_new_cuda_gradients=True, backbone_unchanged=True,
            trainable_parameters=sum(p.numel() for p in model.adapter.parameters()))
        if result['trainable_parameters'] != 1060998:
            raise ValueError('Adapter parameter count changed')
        return result
    finally:
        model.load_trainable(original)
        model.parameter_version = version
        for n, p in model.adapter.named_parameters():
            p.grad = original_grads[n].to(p.device) if original_grads[n] is not None else None
        restore_rng(rng)
