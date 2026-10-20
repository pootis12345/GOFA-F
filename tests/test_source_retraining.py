from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from gofa_f.generative_data import Query
from gofa_f.source_retraining_data import VIEWS, isd_temperature, distance_km
from gofa_f.source_retraining_policy import (commit_selection, confirmation_pass, initial_selection,
                                             pools, scheduled, score)
from gofa_f.source_retraining_state import validate


def data(nodes, n=6000):
    rng = np.random.default_rng(42)
    values = rng.normal(size=(n, nodes)).astype(np.float32)
    mask = np.ones_like(values, bool)
    return SimpleNamespace(values=values, mask=mask,
        prefix_counts=np.vstack([np.zeros(nodes, np.int64), np.cumsum(mask, axis=0)]),
        segments=np.zeros(n, np.int64))


def test_real_budget_paired_traffic_and_no_disjoint_label_leak():
    series = {n: data(count) for n, (count, _, _) in VIEWS.items()}
    a = pools(series)
    b = pools(series)
    assert a == b
    assert len(a['validation']) == len(a['confirmation']) == 384
    assert all(len(q) == 8192 for q in a['train'].values())
    for split in ('validation', 'confirmation'):
        q = a[split]
        paired = [[(v['variable'], v['origin']) for v in q if v['source'] == name]
                  for name in ('pems04_flow', 'pems04_speed')]
        assert paired[0] == paired[1] and len(paired[0]) == 64
        for name in VIEWS:
            own = [v for v in q if v['source'] == name]
            for variable in set(v['variable'] for v in own):
                starts = sorted(v['origin'] for v in own if v['variable'] == variable)
                assert all(b-a >= VIEWS[name][2] for a, b in zip(starts, starts[1:]))
                low, high = (3600, 4800) if split == 'validation' else (4800, 6000)
                assert all(low <= t and t+VIEWS[name][2] <= high for t in starts)


def test_domain_cycle_equal_over_global_presentations_not_reset_at_updates():
    pool = {name: [dict(source=name, variable=0, origin=i, length=lh[1], horizon=lh[2])
                   for i in range(8192)] for name, lh in VIEWS.items()}
    result = [scheduled(pool, i) for i in range(384)]
    counts = {n: sum(q.source == n for q in result) for n in VIEWS}
    assert counts == dict(pems04_flow=64, pems04_speed=64, boston2017=128, penmanshiel2017=128)
    assert scheduled(pool, 64).source == 'boston2017'
    assert [scheduled(pool, i) for i in range(70)] == result[:70]


def test_confirmation_excludes_already_observed_stage22_labels():
    series = {n: data(count) for n, (count, _, _) in VIEWS.items()}
    forbidden = {n: {str(v): [[4800, 5100]] for v in range(307)}
                 for n in ('pems04_flow', 'pems04_speed')}
    value = pools(series, forbidden)
    assert all(q['origin'] >= 5100 for q in value['confirmation'] if q['source'].startswith('pems04'))


def scores(factors=None):
    factors = factors or {n: .9 for n in VIEWS}
    queries = [dict(source=n, variable=v) for n in VIEWS for v in range(2) for _ in range(v+1)]
    eps = {n: [1e-6, 1e-6] for n in VIEWS}
    native = [2 if q['variable'] else 0 for q in queries]
    errors = [e*factors[q['source']] for e, q in zip(native, queries)]
    return score(errors, native, queries, eps), (queries, eps, native)


def test_hierarchy_near_zero_views_domains_and_forgetting():
    value, (queries, eps, native) = scores(dict(pems04_flow=.64, pems04_speed=1., boston2017=.81, penmanshiel2017=1.))
    assert value['views']['pems04_flow'] == pytest.approx(.8)
    assert value['domains']['traffic'] == pytest.approx(np.sqrt(.8))
    assert value['domains']['weather'] == pytest.approx(.9)
    assert value['W'] == 1
    a = native.copy()
    a[0] = 1
    assert score(a, native, queries, eps)['views']['pems04_flow'] > 1
    with pytest.raises(ValueError, match='Complete'):
        score(a[:-1], native, queries, eps)


def test_separate_best_early_reference_ties_idempotence_and_confirmation():
    parent, _ = scores()
    state = initial_selection(parent, 'parent')
    candidate = deepcopy(parent)
    candidate['W'] *= .9995
    candidate['S'] *= .9995
    relative = dict(domains=dict(traffic=1., weather=1., wind=1.))
    assert commit_selection(state, 250, '250', candidate, relative)
    assert state['best_update'] == 250 and state['bad_validations'] == 1
    reference = state['early_reference_W']
    assert not commit_selection(state, 250, '250', candidate, relative)
    assert state['bad_validations'] == 1 and state['early_reference_W'] == reference
    assert not commit_selection(state, 500, '500', candidate, relative)
    assert state['best_update'] == 250 and state['bad_validations'] == 2
    lower = deepcopy(candidate)
    lower['W'] *= .9
    lower['S'] *= .9
    ineligible = dict(domains=dict(traffic=1., weather=1.020001, wind=1.))
    assert not commit_selection(state, 750, '750', lower, ineligible)
    assert state['bad_validations'] == 3
    assert confirmation_pass(dict(S=.99, domains=dict(a=1.02)), dict(domains=dict(a=1.02)))
    assert not confirmation_pass(dict(S=.99, domains=dict(a=1.02001)), relative)


def test_isd_field5_zero_missing_scale_and_clock():
    series = isd_temperature('2017 1 1 0 0 200 10100 180 30 4 0 0\n'
                             '2017 1 1 1 -9999 200 10100 180 30 4 0 0\n'
                             '2017 1 1 2 -52 200 10100 180 30 4 0 0')
    assert len(series) == 8760 and series.iloc[0] == 0 and np.isnan(series.iloc[1])
    assert series.iloc[2] == -5.2 and distance_km([42, -71], [42, -71]) == 0
    with pytest.raises(ValueError, match='Duplicate'):
        isd_temperature('2017 1 1 0 0 0 0 0 0 0 0 0\n'*2)


def test_pinned_config_and_no_targets():
    cfg = yaml.safe_load(Path('configs/stage23_source_retraining.yaml').read_text())
    validate(cfg)
    for change in ('lr', 'updates', 'target', 'resources'):
        changed = deepcopy(cfg)
        if change == 'lr':
            changed['training']['learning_rate'] = 2e-4
        elif change == 'updates':
            changed['training']['max_updates'] = 4000
        elif change == 'target':
            changed['data']['target_evaluation_enabled'] = True
        else:
            changed['resources']['admission_policy'] = 'memory_cap_only'
        with pytest.raises(ValueError, match='policy'):
            validate(changed)


def test_self_control_same_nodes_and_origin_history_no_cross_messages():
    from test_allnode import source
    from gofa_f.allnode_graph import build_graph
    from gofa_f.short_source_diagnostic import self_only
    graph = build_graph(source(), Query('pems04_flow', 0, 500, 12, 12))
    own = self_only(graph)
    assert own.variable_ids == graph.variable_ids and own.identity != graph.identity
    assert torch.equal(own.graph.history, graph.graph.history)
    assert torch.count_nonzero(own.graph.base-torch.diag(own.graph.base.diag())) == 0


def test_transactional_update_lr_warmup_and_rollback(monkeypatch):
    from test_allnode import tiny
    from gofa_f.short_migration import assert_equal
    from gofa_f.source_retraining_run import apply_update
    model = tiny()
    optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=5e-5, weight_decay=1e-4)
    state = dict(updates=0, pending=64, pending_loss=128., pending_tokens=700)
    progress = SimpleNamespace(state=dict(measured_updates=0))
    progress.tick = lambda **fields: progress.state.update(fields)
    for p in model.adapter.parameters():
        p.grad = torch.ones_like(p)
    apply_update(model, optimizer, state, progress)
    assert state['updates'] == 1 and state['pending'] == 0
    assert progress.state['learning_rate'] == pytest.approx(1e-6) and progress.state['loss'] == 2
    state.update(pending=64, pending_loss=128., pending_tokens=700)
    for p in model.adapter.parameters():
        p.grad = torch.ones_like(p)
    before, moments = model.trainable_state(), deepcopy(optimizer.state_dict())
    gradients = {n: p.grad.clone() for n, p in model.adapter.named_parameters()}
    def fail():
        next(model.adapter.parameters()).data.add_(100)
        raise torch.cuda.OutOfMemoryError('simulated atomic rollback')
    monkeypatch.setattr(optimizer, 'step', fail)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        apply_update(model, optimizer, state, progress)
    assert_equal(model.trainable_state(), before)
    assert_equal(optimizer.state_dict(), moments)
    assert_equal({n: p.grad for n, p in model.adapter.named_parameters()}, gradients)
    assert state['updates'] == 1 and state['pending'] == 64


def test_dense_reference_matches_current_slot_loss_shift_mask_and_query_weights():
    from gofa_f.source_retraining_checks import full_reference_loss
    from gofa_f.short_training import compact_loss
    codec = SimpleNamespace(alphabet=lambda device: torch.tensor([2, 4, 6], device=device),
        lookup=lambda n, device: torch.tensor([-1, -1, 0, -1, 1, -1, 2, -1], device=device))
    torch.manual_seed(42)
    logits = torch.randn(2, 6, 8, requires_grad=True)
    labels = torch.tensor([[-100, -100, 2, -100, 4, 6], [-100, 4, -100, -100, -100, -100]])
    expected = full_reference_loss(logits, labels, codec)
    expected.backward()
    gradient = logits.grad.clone()
    logits.grad = None
    rows, steps = torch.nonzero(labels[:, 1:] != -100, as_tuple=True)
    scores = logits[rows, steps][:, [2, 4, 6]]
    targets = codec.lookup(8, 'cpu')[labels[:, 1:][rows, steps]]
    actual, count = compact_loss((scores, targets, rows, torch.tensor([3, 1])))
    actual.backward()
    assert count == 4
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(logits.grad, gradient, rtol=0, atol=0)
    assert torch.count_nonzero(logits.grad[:, -1]) == 0


def script(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, Path('scripts')/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('barrier', ['none', 'supervisor', 'worker', 'complete', 'bad_proof'])
def test_bootstrap_optional_records_bound_proof_and_duplicate_barriers(tmp_path, monkeypatch, barrier):
    import sys
    from gofa_f import idle_launcher, source_retraining, source_retraining_state
    from gofa_f.generative_data import sha
    from gofa_f.graph_benchmark import atomic, read
    boot = script('bootstrap_source_retraining')
    output = tmp_path/'run'
    output.mkdir()
    atomic(output/'manifest.json', {'bound': True})
    atomic(output/'cpu_preflight.json', dict(passed=barrier != 'bad_proof',
        manifest_sha256=sha(output/'manifest.json'), real_weights=dict(passed=True), cuda_disabled=True))
    if barrier == 'supervisor':
        atomic(output.with_suffix('.launch.json'), dict(pid=123, start_ticks='1234'))
    elif barrier == 'worker':
        atomic(output/'activity.json', dict(worker=dict(pid=123, start_ticks='1234')))
    elif barrier == 'complete':
        atomic(output/'status.json', dict(state='complete'))
    monkeypatch.setattr(idle_launcher, 'process_identity', lambda pid: str(pid)+'4')
    monkeypatch.setattr(source_retraining, 'configuration', lambda path: {'tested': True})
    monkeypatch.setattr(source_retraining_state, 'verify', lambda cfg, path: None)
    launched = []
    def launch(command, **kwargs):
        launched.append((command, kwargs))
        return SimpleNamespace(pid=567)
    monkeypatch.setattr(boot.subprocess, 'Popen', launch)
    monkeypatch.setattr(boot.subprocess, 'run', lambda *a, **k: pytest.fail('Bound proof must not rerun weights'))
    monkeypatch.setattr(sys, 'argv', ['bootstrap', '--config', 'bound.yaml', '--output', str(output)])
    if barrier != 'none':
        with pytest.raises((ValueError, RuntimeError)):
            boot.main()
        assert not launched
        return
    assert boot.main() == 0
    assert len(launched) == 1
    command, fields = launched[0]
    assert command[-2:] == ['--worker-mode', 'source_retraining']
    assert fields['env']['CUDA_VISIBLE_DEVICES'] == '' and fields['start_new_session']
    assert fields['env']['OMP_NUM_THREADS'] == fields['env']['MKL_NUM_THREADS'] == '2'
    assert read(output.with_suffix('.launch.json'))['start_ticks'] == '5674'
    assert read(output/'bootstrap.json')['state'] == 'cpu_passed_idle_supervisor_deployed'


def test_cpu_report_optional_records_before_gpu_start(tmp_path, monkeypatch, capsys):
    import sys
    from gofa_f import source_retraining
    cfg = yaml.safe_load(Path('configs/stage23_source_retraining.yaml').read_text())
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(source_retraining, 'configuration', lambda path: cfg)
    monkeypatch.setattr(source_retraining, 'report', lambda path: {'new_updates': 0})
    monkeypatch.setattr(sys, 'argv', ['source_retraining', '--config', 'bound.yaml',
                                    '--output', str(tmp_path), '--report-only'])
    source_retraining.main()
    assert '"new_updates": 0' in capsys.readouterr().out


def test_partial_run_report_preserves_absent_selection_and_results(tmp_path):
    from gofa_f.graph_benchmark import atomic
    from gofa_f.source_retraining import report
    atomic(tmp_path/'manifest.json', dict(protocol='stage23-source-retraining-v1',
        configuration=dict(adapter_sha256='parent')))
    value = report(tmp_path)
    assert value['selection'] is None and value['source_results'] == {}
    assert value['activity'] is None and value['target_predictions'] == 0


def test_reuse_numerical_proof_rejects_changed_science_and_tensor_code(tmp_path):
    from gofa_f.generative_data import sha
    from gofa_f.graph_benchmark import atomic
    checker = script('check_source_retraining_cpu')
    previous = tmp_path/'old'
    previous.mkdir()
    manifest = dict(configuration=dict(adapter_sha256='parent'), pools=dict(train=[1]),
                    code={'source_retraining.py': 'old', 'allnode_model.py': 'same'})
    atomic(previous/'manifest.json', manifest)
    (previous/'cpu_tests.xml').write_text('<testsuites><testsuite><testcase/></testsuite></testsuites>')
    atomic(previous/'cpu_preflight.json', dict(passed=True, cuda_disabled=True,
        real_weight_dtype='float32', real_weights=dict(passed=True), cases=1,
        manifest_sha256=sha(previous/'manifest.json')))
    changed = deepcopy(manifest)
    changed['pools']['train'] = [2]
    with pytest.raises(ValueError, match='Scientific'):
        checker.reuse_real_weights(previous, tmp_path, changed, tmp_path)
    changed = deepcopy(manifest)
    changed['code'].update({'source_retraining.py': 'new', 'allnode_model.py': 'different'})
    with pytest.raises(ValueError, match='Numerical'):
        checker.reuse_real_weights(previous, tmp_path, changed, tmp_path)


def test_reuse_numerical_proof_allows_only_exact_cli_import_fix(tmp_path):
    from gofa_f.generative_data import sha
    from gofa_f.graph_benchmark import atomic
    checker = script('check_source_retraining_cpu')
    old, snapshot = tmp_path/'old', tmp_path/'snapshot'
    old.mkdir()
    source = snapshot/'src/gofa_f'
    source.mkdir(parents=True)
    current = Path('src/gofa_f/source_retraining.py').read_text()
    original = current.replace(
        '    from gofa_f.allnode_stage import read\n    from gofa_f.graph_benchmark import atomic, sha',
        '    from gofa_f.graph_benchmark import atomic, read, sha').replace(
        '        from gofa_f.allnode_stage import read', '        from gofa_f.graph_benchmark import read')
    (source/'source_retraining.py').write_text(original)
    (source/'tensor.py').write_text('numerical unchanged')
    (old/'parent.pt').write_bytes(b'fixed parent')
    manifest = dict(configuration=dict(adapter_sha256=sha(old/'parent.pt')), pools=dict(train=[1]),
        code={p.name: sha(p) for p in source.glob('*.py')})
    atomic(old/'manifest.json', manifest)
    (old/'cpu_tests.xml').write_text('<testsuites><testsuite><testcase/></testsuite></testsuites>')
    atomic(old/'cpu_preflight.json', dict(passed=True, cuda_disabled=True,
        real_weight_dtype='float32', real_weights=dict(passed=True, exact_tokens=True), cases=1,
        alphabet={'tokens': 10000}, manifest_sha256=sha(old/'manifest.json')))
    new = deepcopy(manifest)
    new['code']['source_retraining.py'] = sha(Path('src/gofa_f/source_retraining.py'))
    proof = checker.reuse_real_weights(old, snapshot, new, tmp_path)
    assert proof['real_weights']['exact_tokens']
    assert proof['reuse_provenance']['all_numerical_files_identical']
    atomic(old/'activity.json', dict(worker=None))
    with pytest.raises(ValueError, match='never-launched'):
        checker.reuse_real_weights(old, snapshot, new, tmp_path)
