"""Actual-weight acceptance for mixed generated-input/true-label and KL replay."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from gofa_f.allnode_stage import cpu_copy, restore_rng, rng_state
from gofa_f.generative_data import Query
from gofa_f.robust_loss import parent_adapter
from gofa_f.short_migration import assert_equal
from gofa_f.short_training import compact_forward
from gofa_f.source_retraining_checks import mechanisms as inference_checks
from gofa_f.weather_loss import batch, loss, prefix_key, records


def fixtures(codec, series, queries):
    events, prefixes = [], {}
    for name, kind in [('boston2017', 'weather_true'), ('pems04_flow', 'keep'),
                       ('boston2017', 'weather_generated'), ('penmanshiel2017', 'keep')]:
        q = dict(next(q for q in queries if q['source'] == name))
        data = series[name]
        history = np.where(data.mask[q['origin']-q['length']:q['origin'], q['variable']],
                           data.values[q['origin']-q['length']:q['origin'], q['variable']], np.nan)
        # Deliberately different finite answers, without reading future anchor values.
        row = codec.record(history, np.full(q['horizon'], float(np.nanmean(history))))
        prefixes[prefix_key(q)] = dict(query=q, generated_ids=row['ids'][row['prompt_end']+1:])
        events.append(dict(query=q, kind=kind, weight=2/3 if name == 'boston2017' else 4/3))
    return events, prefixes


def mechanisms(model, codec, series, queries, tolerance, output, check=lambda: None):
    device = next(model.native.parameters()).device
    result = dict(passed=False, tolerance=tolerance)
    if device.type == 'cuda':
        result['unchanged_inference_acceptance'] = inference_checks(model, codec, series, queries, tolerance, output, check)
    original, version, rng = model.trainable_state(), model.parameter_version, rng_state()
    original_grads = {n: cpu_copy(p.grad) for n, p in model.adapter.named_parameters()}
    frozen = [p._version for p in model.native.parameters()]
    teacher = deepcopy(model.adapter).requires_grad_(False).train()
    teacher_state = {n: cpu_copy(p) for n, p in teacher.named_parameters()}
    optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=1e-5, weight_decay=1e-4)
    events, prefixes = fixtures(codec, series, queries)
    def gradients():
        return {n: cpu_copy(p.grad) for n, p in model.adapter.named_parameters()}
    try:
        for _ in range(3):
            check()
            model.zero_grad(set_to_none=True)
            value, _, _ = batch(model, teacher, codec, series, events[2:3], prefixes, check)
            value.backward()
            optimizer.step()
            model.mark_updated()
        for pair in (events[:2], events[2:]):
            rec, graphs = records(codec, series, pair, prefixes)
            tokens = codec.batch(rec).to(device)
            with parent_adapter(model, teacher):
                contexts = model.encode_origins(graphs, codec, check=check)
                parent = compact_forward(model, codec, tokens, contexts)[0].detach()
            model.zero_grad(set_to_none=True)
            contexts = model.encode_origins(graphs, codec, check=check)
            full = model(tokens, contexts=contexts).logits
            labels = tokens.labels[:, 1:]
            valid = labels != -100
            rows, steps = torch.nonzero(valid, as_tuple=True)
            scores = full[rows, steps].float().index_select(-1, codec.alphabet(device))
            target = codec.lookup(full.shape[-1], device)[labels[rows, steps]]
            expected_loss, _ = loss((scores, target, rows, valid.sum(-1)), parent, pair)
            value_reference = expected_loss.detach().clone()
            expected_loss.backward()
            expected = gradients()
            del full, scores, contexts, expected_loss
            model.zero_grad(set_to_none=True)
            value, count, terms = batch(model, teacher, codec, series, pair, prefixes, check)
            torch.testing.assert_close(value.detach(), value_reference, rtol=tolerance, atol=tolerance)
            value.backward()
            for n, p in model.adapter.named_parameters():
                torch.testing.assert_close(p.grad.cpu(), expected[n], rtol=tolerance, atol=tolerance,
                                           msg='stage25_full_cached_gradient:'+n)
            expected = gradients()
            model.zero_grad(set_to_none=True)
            for event in pair:
                value, _, _ = batch(model, teacher, codec, series, [event], prefixes, check)
                value.backward()
            for n, p in model.adapter.named_parameters():
                torch.testing.assert_close(p.grad.cpu(), expected[n], rtol=tolerance, atol=tolerance,
                                           msg='stage25_microbatch_gradient:'+n)
        payload = dict(adapter=model.trainable_state(), optimizer=cpu_copy(optimizer.state_dict()),
            gradients=gradients(), rng=rng_state(), pending=2, pending_tokens=count,
            pending_terms=terms, version=model.parameter_version, schedule_cursor=2)
        path = Path(output)/'weather-partial-state.pt'
        torch.save(payload, path)
        saved = torch.load(path, map_location='cpu', weights_only=False)
        assert_equal(payload, saved)
        optimizer.step()
        after = model.trainable_state()
        model.load_trainable(saved['adapter'])
        optimizer.load_state_dict(saved['optimizer'])
        for n, p in model.adapter.named_parameters():
            p.grad = saved['gradients'][n].to(device)
        restore_rng(saved['rng'])
        optimizer.step()
        assert_equal(after, model.trainable_state())
        path.unlink()
        if any(p.grad is not None or not torch.equal(p.cpu(), teacher_state[n]) for n, p in teacher.named_parameters()) \
                or any(p.requires_grad or p.grad is not None or p._version != old for p, old in zip(model.native.parameters(), frozen)):
            raise ValueError('Frozen parent/backbone changed')
        result.update(passed=True, nonzero_diagnostic_updates=3, mixed_prefix_and_true_labels=True,
            dense_cached_gradients=True, microbatch2_vs1_gradients=True,
            exact_materialized_state_replay=True, fixed_parent_and_backbone_unchanged=True,
            trainable_parameters=1060998, bitwise_new_cuda_arithmetic_not_claimed=True)
        return result
    finally:
        model.load_trainable(original)
        model.parameter_version = version
        for n, p in model.adapter.named_parameters():
            p.grad = original_grads[n].to(p.device) if original_grads[n] is not None else None
        restore_rng(rng)
