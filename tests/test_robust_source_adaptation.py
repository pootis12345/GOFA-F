from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from gofa_f.generative_data import Query
from gofa_f.robust_loss import batch, distribution_loss, parent_adapter
from gofa_f.robust_diagnostic import observe_gates
from gofa_f.robust_progress import Progress
from gofa_f.robust_state import (confirmation_pool, gate100, passed, quick_pool, recorded_windows,
                                select, validate, validation_boundary, parent_replay_bindings)
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


def test_diagnostic_cdf_matches_training_cdf_with_missing_slot(codec):  # noqa: F811
    from gofa_f.robust_diagnostic import cdf_distance
    record = codec.record(np.arange(48, dtype=float), np.array([12., np.nan, 19.]))
    scores = torch.randn(3, len(codec.finite_ids), dtype=torch.float32)
    labels = torch.tensor([record['labels'][i] for i in record['answer_positions']])
    valid = labels != -100
    targets = codec.lookup(max(codec.finite_ids)+1, torch.device('cpu'))[labels[valid]]
    _, terms = distribution_loss((scores[valid], targets, torch.zeros(int(valid.sum()), dtype=torch.long),
                                 torch.tensor([int(valid.sum())])), scores[valid].detach())
    assert cdf_distance(scores, None, codec, record) == pytest.approx(float(terms['cdf']), abs=1e-6)


def test_all_diagnostic_conditions_common_prefix_and_gradient_smoke(tmp_path, codec):  # noqa: F811
    from dataclasses import asdict
    from types import SimpleNamespace
    from gofa_f.robust_diagnostic import run
    from gofa_f.short_migration import assert_equal
    model, prior, output = tiny(), tmp_path/'prior', tmp_path/'new'
    prior.mkdir()
    output.mkdir()
    with torch.no_grad():
        for layer in model.adapter.exchanges:
            layer.up.weight.normal_(std=.001)
    parent = model.trainable_state()
    torch.save(parent, output/'parent.pt')
    for update in (250, 500):
        torch.save(parent, prior/f'candidate-{update:06d}.pt')
    series = {n: source(9) for n in VIEWS}
    queries = [asdict(Query(n, 0, 500, length, horizon)) for n, (_, length, horizon) in VIEWS.items()]
    manifest = dict(configuration=dict(stage23_output=str(prior)),
                    pools=dict(validation=queries, diagnostic=queries,
                               train={q['source']: [q]*4 for q in queries}))
    (output/'manifest.json').write_text(json.dumps(manifest))
    for q in queries:
        data = series[q['source']]
        record = codec.record(data.values[500-q['length']:500, 0], data.values[500:500+q['horizon'], 0])
        for label in ('validation-native', 'validation-parent', 'validation-update-000250', 'validation-update-000500'):
            folder = prior/'evaluations'/(label+'-'+q['source'])
            folder.mkdir(parents=True)
            (folder/'000000.json').write_text(json.dumps(dict(query=q)))
            if label == 'validation-parent':
                (folder/'trajectories').mkdir()
                (folder/'trajectories/000000-00.json').write_text(json.dumps(
                    dict(generated_ids=record['ids'][record['prompt_end']+1:])))
    rng = torch.get_rng_state().clone()
    result = run(model, codec, series, manifest, output, lambda: None, SimpleNamespace(tick=lambda *a, **k: None))
    assert result['complete'] and result['records'] == 16 and result['optimizer_updates'] == 0
    assert result['new_free_trajectories'] == 0 and len(result['gradient_norms']) == 4
    assert_equal(parent, model.trainable_state())
    assert torch.equal(rng, torch.get_rng_state())
    assert run(model, codec, series, manifest, output, lambda: None, SimpleNamespace()) == result


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


def test_parent_replay_requires_original_source_hashes_not_view_hashes():
    prior = dict(source_configuration={'data': 'bound'}, source_raw_hashes={'nn5': 'hash'},
                 probes=[dict(query=dict(source='nn5'), references=dict(native={}, gofa={}))])
    assert parent_replay_bindings(prior) == dict(source_raw_hashes={'nn5': 'hash'})
    with pytest.raises(ValueError, match='raw hashes'):
        parent_replay_bindings(dict(prior, source_raw_hashes={}))
    with pytest.raises(ValueError, match='incomplete'):
        parent_replay_bindings(dict(prior, source_raw_hashes={'pems04_flow': 'wrong namespace'}))


def test_fixed_config_and_target_guard(tmp_path):
    cfg = yaml.safe_load(Path('configs/stage24_robust_source_adaptation.yaml').read_text())
    validate(cfg)
    for key, change in [('loss', dict(ce=1., cdf=0., parent_kl=.1, temperature=1.)),
                        ('training', dict(cfg['training'], max_updates=500))]:
        with pytest.raises(ValueError):
            validate(dict(cfg, **{key: change}))
    with pytest.raises(ValueError):
        validate(dict(cfg, resources=dict(cfg['resources'], admission_policy='idle', admission_samples=3)))
    with pytest.raises(ValueError, match='confirmation'):
        target_run(None, None, {}, tmp_path, None, lambda: None)
    (tmp_path/'manifest.json').write_text('{}')
    (tmp_path/'final_selection.json').write_text(json.dumps(dict(checkpoint_sha256='same', binding='wrong',
                                                               source_gate_passed=True)))
    (tmp_path/'source_confirmation.json').write_text(json.dumps(dict(complete=True, passed=True,
        checkpoint_sha256='same', binding='wrong')))
    with pytest.raises(ValueError, match='binding'):
        target_run(None, None, {}, tmp_path, None, lambda: None)


def test_resource_handoff_accepts_only_stopped_unstarted_preparation(tmp_path):
    from gofa_f.robust_state import preparation_handoff
    root = tmp_path/'stage24-robust-source-adaptation-v2'
    root.mkdir()
    def write(name, value):
        (root/name).write_text(json.dumps(value))
    write('status.json', dict(state='prepared', new_updates=0, target_evaluations=0))
    write('manifest.json', dict(configuration=dict(resources=dict(admission_policy='idle'))))
    write('activity.json', dict(worker=None))
    write('launcher.json', dict(state='waiting_for_idle'))
    with pytest.raises(ValueError, match='safely exited'):
        preparation_handoff(root)
    write('launcher.json', dict(state='stopped_by_user'))
    assert preparation_handoff(root)['configuration']['resources']['admission_policy'] == 'idle'
    write('status.json', dict(state='prepared', new_updates=1, target_evaluations=0))
    with pytest.raises(ValueError, match='unstarted'):
        preparation_handoff(root)


@pytest.mark.parametrize('own_memory', [20000, 40*1024+1, None])
def test_stage24_shared_supervisor_immediate_admission_audit_and_recovery(tmp_path, monkeypatch, own_memory):
    import gofa_f.short_launcher as launcher
    cfg = yaml.safe_load(Path('configs/stage24_robust_source_adaptation.yaml').read_text())
    config = tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump(cfg))
    clock, children, signals, queries = [0.], [], [], []
    monkeypatch.setattr(launcher.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(launcher.time, 'sleep', lambda n: clock.__setitem__(0, clock[0]+n))
    monkeypatch.setattr(launcher, 'process_identity', lambda _: 'ticks')
    monkeypatch.setattr(launcher, 'own_gpu_pids', lambda _: {800})
    monkeypatch.setattr(launcher, 'IdleGate', lambda: pytest.fail('Shared Stage24 must not use idle gate'))
    def query():
        queries.append(clock[0])
        if len(queries) == 1:
            raise RuntimeError('Temporary GPU query failure')
        return dict(uuid='GPU-test', compute_mode='Default', pids=[999, 800], utilization=100,
                    used_mib=80000, free_mib=1500, process_used_mib={999: 60000, 800: own_memory})
    monkeypatch.setattr(launcher, 'query_gpu', query)
    class Child:
        pid, returncode = 800, None
        def __init__(self, command, cwd, env, start_new_session):
            assert start_new_session and env['CUDA_VISIBLE_DEVICES'] == 'GPU-test'
            assert env['GOFA_STAGE19_ADMITTED'] == '1'
            if not children:
                assert clock[0] == 10  # only the failed identification caused a retry
            else:
                self.returncode = 0
                launcher.atomic_json(tmp_path/'status.json', dict(state='complete'))
            self.polls = 0
            children.append(self)
        def poll(self):
            self.polls += 1
            if self.polls > 25 and self.returncode is None:
                self.returncode = 0
                launcher.atomic_json(tmp_path/'status.json', dict(state='complete'))
            return self.returncode
    monkeypatch.setattr(launcher.subprocess, 'Popen', Child)
    def stop(pid, sig):
        signals.append((pid, sig))
        children[-1].returncode = 75
        launcher.atomic_json(tmp_path/'status.json', dict(state='resource_paused'))
    monkeypatch.setattr(launcher.os, 'killpg', stop)
    launcher.supervise(tmp_path, config, Path('.'), 'robust_source_adaptation')
    binding = launcher.read(tmp_path/'launcher_manifest.json')
    assert binding['admission_policy'] == 'memory_cap_only' and binding['admission_samples'] == 1
    assert binding['memory_cap_gib'] == 40 and binding['torch_cap_gib'] == 38
    assert 'idle_max_used_mib' not in binding and binding['leave_free_gib'] == 0
    ledger = launcher.read(tmp_path/'activity.json')
    assert ledger['worker'] is None and ledger['waiting_seconds'] >= 10
    if own_memory == 20000:
        assert len(children) == 1 and not signals and ledger['resource_pause_count'] == 0
        assert queries[2:4] == [10., 20.]  # audits continue every10s with foreign process present
    else:
        assert len(children) == 2 and signals == [(800, launcher.signal.SIGTERM)]
        assert ledger['resource_pause_count'] == 1
    with pytest.raises(ValueError, match='Legacy'):
        launcher.admission_gate(cfg['resources'], 'source_retraining')


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


def test_all_node_gate_observer_does_not_change_delta_or_rng():
    from gofa_f.allnode_graph import build_graph
    model = tiny()
    with torch.no_grad():
        model.adapter.exchanges[0].up.weight.normal_(std=.001)
    graph = build_graph(source(9), Query('pems04', 0, 500, 12, 12)).graph
    hidden = torch.randn(len(graph.history[0]), 12, model.native.config.hidden_size)
    rng = torch.get_rng_state().clone()
    expected = model.adapter.exchanges[0](hidden, graph)
    records = []
    with observe_gates(model, records):
        actual = model.adapter.exchanges[0](hidden, graph)
    assert torch.equal(expected, actual) and torch.equal(rng, torch.get_rng_state())
    assert len(records) == 1 and len(records[0]['by_node']) == hidden.shape[0]


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
