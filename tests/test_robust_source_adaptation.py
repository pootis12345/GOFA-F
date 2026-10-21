from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from gofa_f.generative_data import Query
from gofa_f.robust_loss import batch, distribution_loss, parent_adapter
from gofa_f.robust_progress import Progress
from gofa_f.robust_state import (confirmation_pool, gate100, passed, quick_pool, recorded_windows,
                                select, validate, validation_boundary)
from gofa_f.robust_targets import run as target_run
from gofa_f.source_retraining_data import VIEWS
from gofa_f.source_retraining_policy import pools
from test_allnode import source, tiny
from test_short_training import codec  # noqa: F401
from test_source_retraining import data


def test_cdf_ordinal_order_and_identical_parent_kl():
    logits = torch.zeros(1, 3, requires_grad=True)
    value, parts = distribution_loss((logits, torch.tensor([1]), torch.tensor([0]), torch.tensor([1])),
                                      logits.detach())
    assert float(parts['ce']) == pytest.approx(np.log(3), abs=1e-6)
    assert float(parts['cdf']) == pytest.approx(1/9, abs=1e-6)
    assert float(parts['kl']) == 0
    value.backward()
    assert torch.isfinite(logits.grad).all()
    losses = []
    for index in range(3):
        scores = torch.full((1, 3), -40.)
        scores[0, index] = 0
        _, terms = distribution_loss((scores, torch.tensor([0]), torch.tensor([0]), torch.tensor([1])), scores)
        losses.append(float(terms['cdf']))
    assert losses == pytest.approx([0, .5, 1])


def test_loss_per_query_normalization_not_long_horizon_weighting():
    torch.manual_seed(42)
    logits = torch.randn(5, 7, requires_grad=True)
    teacher = torch.randn(5, 7)
    target, rows, counts = torch.tensor([0, 6, 2, 1, 3]), torch.tensor([0, 1, 1, 1, 1]), torch.tensor([1, 4])
    combined, _ = distribution_loss((logits, target, rows, counts), teacher)
    individual = []
    for i in range(2):
        mask = rows == i
        individual.append(distribution_loss((logits[mask], target[mask], torch.zeros(int(mask.sum()), dtype=torch.long),
                                            torch.tensor([int(mask.sum())])), teacher[mask])[0])
    torch.testing.assert_close(combined, sum(individual))
    with pytest.raises(ValueError, match='detached'):
        distribution_loss((logits, target, rows, counts), teacher.requires_grad_())
    with pytest.raises(ValueError):
        distribution_loss((logits, target, rows, torch.tensor([0, 5])), teacher.detach())


def test_gate_selection_ties_and_strict_final_boundaries():
    reference = dict(W=1.1)
    native = dict(W=1.1*.995, S=.98, domains=dict(traffic=.9, weather=1.02, wind=.99))
    relative = dict(domains=dict(traffic=1., weather=1.02, wind=1.))
    assert gate100(native, relative, reference)
    assert not gate100(dict(native, W=1.1*.995+1e-8), relative, reference)
    assert passed(native, relative)
    assert not passed(dict(native, S=.9900001), relative)
    rows = [dict(update=u, native_score=native, parent_score=relative) for u in [250, 100]]
    assert select(rows)['update'] == 100
    assert select([dict(rows[0], parent_score=dict(domains=dict(wind=1.020001)))]) is None


def test_validation_boundary_after_resume_or_next_microbatch():
    # The saved update number does not change while the next64-query batch is
    # accumulated. Pending2 must not re-enter the update50/100 validation guard.
    assert validation_boundary(50, 0) and validation_boundary(100, 0) and validation_boundary(250, 0)
    for update in (50, 100, 250):
        assert all(not validation_boundary(update, pending) for pending in (1, 2, 62, 64))
    assert not validation_boundary(51, 0)


def test_fixed_config_and_target_guard(tmp_path):
    cfg = yaml.safe_load(Path('configs/stage24_robust_source_adaptation.yaml').read_text())
    validate(cfg)
    for key, change in [('loss', dict(ce=1., cdf=0., parent_kl=.1, temperature=1.)),
                        ('training', dict(cfg['training'], max_updates=500))]:
        with pytest.raises(ValueError):
            validate(dict(cfg, **{key: change}))
    with pytest.raises(ValueError, match='confirmation'):
        target_run(None, None, {}, tmp_path, None, lambda: None)
    (tmp_path/'manifest.json').write_text('{}')
    (tmp_path/'final_selection.json').write_text(json.dumps(dict(checkpoint_sha256='same', binding='wrong',
                                                               source_gate_passed=True)))
    (tmp_path/'source_confirmation.json').write_text(json.dumps(dict(complete=True, passed=True,
        checkpoint_sha256='same', binding='wrong')))
    with pytest.raises(ValueError, match='binding'):
        target_run(None, None, {}, tmp_path, None, lambda: None)


def test_fresh_confirmation_coverage_and_old_window_exclusion(tmp_path):
    series = {n: data(count) for n, (count, _, _) in VIEWS.items()}
    old = pools(series)
    old_output = tmp_path/'old'
    old_output.mkdir()
    windows = [dict(source=n, variable=v, origin=4800, horizon=200) for n in VIEWS
               for v in range(VIEWS[n][0])]
    (old_output/'manifest.json').write_text(json.dumps(dict(pools=dict(confirmation=windows))))
    exclusions, receipts = recorded_windows(tmp_path, series)
    result = confirmation_pool(series, old, exclusions)
    assert len(result) == 192 and len(receipts) == 1
    assert {n: sum(q['source'] == n for q in result) for n in VIEWS} == dict(
        pems04_flow=32, pems04_speed=32, boston2017=64, penmanshiel2017=64)
    assert all(q['origin'] >= 5000 for q in result)
    quick = quick_pool(old['validation'])
    assert len(quick) == 96 and all(q in old['validation'] for q in quick)
    assert {n: len({q['variable'] for q in quick if q['source'] == n}) for n in VIEWS} == dict(
        pems04_flow=16, pems04_speed=16, boston2017=16, penmanshiel2017=14)


def test_teacher_reference_swap_restores_rng_parameters_and_existing_gradients():
    model = tiny()
    teacher = deepcopy(model.adapter).requires_grad_(False)
    student = model.adapter
    parameter = next(student.parameters())
    parameter.grad = torch.ones_like(parameter)
    old = parameter.detach().clone()
    rng = torch.get_rng_state().clone()
    generation = model.cache_generation
    with pytest.raises(RuntimeError):
        with parent_adapter(model, teacher):
            assert model.adapter is teacher
            torch.rand(5)
            raise RuntimeError('interrupted')
    assert model.adapter is student and model.cache_generation > generation
    assert torch.equal(torch.get_rng_state(), rng) and torch.equal(old, parameter)
    assert torch.equal(parameter.grad, torch.ones_like(parameter))


def test_real_codec_combined_loss_shared_prefix_microbatch_and_parent_freeze(codec):  # noqa: F811
    model = tiny()
    with torch.no_grad():
        for layer in model.adapter.exchanges:
            layer.up.weight.normal_(std=.001)
    teacher = deepcopy(model.adapter).requires_grad_(False).train()
    series = dict(pems04=source(9))
    queries = [Query('pems04', 0, 500, 12, 12), Query('pems04', 1, 600, 48, 24)]
    loss, count, parts = batch(model, teacher, codec, series, queries)
    assert count == 36 and parts['kl'] == pytest.approx(0., abs=1e-7)
    loss.backward()
    expected = {n: p.grad.clone() for n, p in model.adapter.named_parameters()}
    model.zero_grad(set_to_none=True)
    for q in queries:
        value, _, _ = batch(model, teacher, codec, series, [q], shared=False)
        value.backward()
    for n, p in model.adapter.named_parameters():
        torch.testing.assert_close(p.grad, expected[n], rtol=1e-6, atol=1e-6)
    assert all(p.grad is None for p in teacher.parameters())
    assert all(p.grad is None for p in model.native.parameters())


def test_progress_update_cap_conditional_targets_and_epoch_counts(tmp_path):
    t = [0.]
    p = Progress(tmp_path, clock=lambda: t[0])
    p.tick('train', force=True)
    t[0] = 20.
    p.state.update(updates=10, measured_updates=10, evaluation_seconds_per_slot=[.001])
    value = p.tick(force=True)
    assert value['maximum_new_updates'] == 250
    assert value['overall_training_fraction'] == 10/250
    assert value['eta_is_conditional_on_all_gates_passing']
    assert sum(v['query_presentations'] for v in value['views'].values()) == 640
    assert value['overall_eta_seconds'] > value['training_eta_seconds']
    assert p.tick('complete', force=True)['overall_eta_seconds'] == 0
