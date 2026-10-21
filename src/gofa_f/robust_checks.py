"""Actual-weight Stage24 teacher, combined-loss gradients and partial-state checks."""
from copy import deepcopy
from pathlib import Path

import torch

from gofa_f.allnode_stage import cpu_copy, restore_rng, rng_state
from gofa_f.generative_data import Query
from gofa_f.robust_loss import batch, distribution_loss, parent_adapter, records
from gofa_f.short_migration import assert_equal
from gofa_f.short_training import compact_forward
from gofa_f.source_retraining_checks import mechanisms as original_mechanisms


def mechanisms(model, codec, series, queries, tolerance, output, check=lambda: None):
    device = next(model.native.parameters()).device
    result = dict(passed=False, tolerance=tolerance)
    if device.type == 'cuda':
        result['unchanged_inference_acceptance'] = original_mechanisms(
            model, codec, series, queries, tolerance, output, check)
    original, version, rng = model.trainable_state(), model.parameter_version, rng_state()
    original_grads = {n: cpu_copy(p.grad) for n, p in model.adapter.named_parameters()}
    frozen = [p._version for p in model.native.parameters()]
    teacher = deepcopy(model.adapter).requires_grad_(False).train()
    teacher_state = {n: p.detach().clone() for n, p in teacher.named_parameters()}
    selected = [Query(**next(q for q in queries if q['source'] == n))
                for n in ('pems04_flow', 'penmanshiel2017')]
    optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=2e-5, weight_decay=1e-4)
    def grads():
        return {n: cpu_copy(p.grad) for n, p in model.adapter.named_parameters()}
    try:
        for _ in range(3):
            check()
            model.zero_grad(set_to_none=True)
            loss, _, _ = batch(model, teacher, codec, series, selected[:1], check)
            loss.backward()
            optimizer.step()
            model.mark_updated()
        rec, graphs = records(codec, series, selected)
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
        targets = codec.lookup(full.shape[-1], device)[labels[rows, steps]]
        dense = scores, targets, rows, valid.sum(-1)
        loss, _ = distribution_loss(dense, parent)
        expected_loss = loss.detach().clone()
        loss.backward()
        expected = grads()
        del full, contexts, dense, loss, scores
        model.zero_grad(set_to_none=True)
        cached_loss, count, parts = batch(model, teacher, codec, series, selected, check)
        torch.testing.assert_close(cached_loss.detach(), expected_loss, rtol=tolerance, atol=tolerance)
        cached_loss.backward()
        for name, p in model.adapter.named_parameters():
            torch.testing.assert_close(p.grad.cpu(), expected[name], rtol=tolerance, atol=tolerance,
                                       msg='combined_full_shared_gradient:'+name)
        expected_pair = grads()
        model.zero_grad(set_to_none=True)
        for q in selected:
            loss, _, _ = batch(model, teacher, codec, series, [q], check)
            loss.backward()
        for name, p in model.adapter.named_parameters():
            torch.testing.assert_close(p.grad.cpu(), expected_pair[name], rtol=tolerance, atol=tolerance,
                                       msg='combined_microbatch_gradient:'+name)
        payload = dict(adapter=model.trainable_state(), optimizer=cpu_copy(optimizer.state_dict()),
            gradients=grads(), rng=rng_state(), pending=2, pending_terms=parts, pending_tokens=count,
            version=model.parameter_version, cursor=2, validation_progress={'sealed': False})
        path = Path(output)/'robust-partial-state.pt'
        torch.save(payload, path)
        loaded = torch.load(path, weights_only=False, map_location='cpu')
        assert_equal(payload, loaded)
        optimizer.step()
        after = model.trainable_state()
        model.load_trainable(loaded['adapter'])
        optimizer.load_state_dict(loaded['optimizer'])
        for n, p in model.adapter.named_parameters():
            p.grad = loaded['gradients'][n].to(device)
        restore_rng(loaded['rng'])
        optimizer.step()
        assert_equal(after, model.trainable_state())
        path.unlink()
        if any(p.grad is not None or not torch.equal(p, teacher_state[n]) for n, p in teacher.named_parameters()):
            raise ValueError('Fixed teacher was updated')
        if any(p.requires_grad or p.grad is not None or p._version != old for p, old in zip(model.native.parameters(), frozen)):
            raise ValueError('Frozen backbone changed')
        result.update(passed=True, nonzero_diagnostic_updates=3, dense_shared_gradients=True,
            microbatch2_vs1_gradients=True, exact_saved_state_replay=True,
            bitwise_new_cuda_arithmetic_not_claimed=True, fixed_teacher_unchanged=True,
            trainable_parameters=1060998, observed_tokens=count)
        return result
    finally:
        model.load_trainable(original)
        model.parameter_version = version
        for n, p in model.adapter.named_parameters():
            p.grad = original_grads[n].to(p.device) if original_grads[n] is not None else None
        restore_rng(rng)
