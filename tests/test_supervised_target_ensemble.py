import copy
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor

from gofa_f.allnode_stage import read
from gofa_f.generative_data import sha
from gofa_f.prefix_cache import atomic_json
from gofa_f.supervised_ensemble_data import (CADENCE_SECONDS, FitLog, disjoint_chain, feature_names,
    features, quotas, training_arrays, training_queries)
import gofa_f.supervised_ensemble_math as math
import gofa_f.supervised_ensemble_cpu as cpu
import gofa_f.supervised_ensemble_models as models
from gofa_f.supervised_ensemble_state import DATA_FLAGS, validate_config
from test_history_target_calibration import row


def series(size=600, nodes=3, source='sdwpf'):
    values = np.stack([np.arange(size)*(.1+i)-10 for i in range(nodes)], 1)
    mask = np.ones_like(values, bool)
    return SimpleNamespace(values=values, mask=mask,
        times=np.arange(size, dtype=np.int64)*CADENCE_SECONDS[source]*10**9,
        segments=np.zeros(size, np.int64), ids=[str(i) for i in range(nodes)],
        prefix_counts=np.concatenate((np.zeros((1, nodes), np.int64), mask.cumsum(0)), 0))


def test_features_causal_calendar_clock_and_no_future_missing_marker():
    d = series()
    q = row()['query']
    x, last = features(d, q)
    assert x.shape == (24, 121) and len(feature_names(48)) == 121
    assert last == pytest.approx(-.1)
    d.values[100:] = 1e30
    d.mask[100:] = False
    d.times[100:] = -1
    other, again = features(d, q)
    assert np.array_equal(x, other, equal_nan=True) and again == last
    d.values[:100, 1:] = 1e30
    assert np.array_equal(features(d, q)[0], x, equal_nan=True)
    assert feature_names(12)[-5:] == ['horizon', 'day_sin', 'day_cos', 'week_sin', 'week_cos']


def test_missing_constant_negative_age_and_temporal_breaks():
    d = series()
    q = row()['query']
    d.values[:100, 0] = -3.
    d.mask[88:100, 0] = False
    x, last = features(d, q)
    names = feature_names(48)
    assert last == -3. and x[0, names.index('last_age')] == 12.
    assert np.isnan(x[0, names.index('mean_12')])
    assert x[0, names.index('slope_12')] == 0. and x[0, names.index('coverage_12')] == 0.
    assert x[0, names.index('std_48')] == 0.
    d.times[99] += 1
    with pytest.raises(ValueError, match='cadence'):
        features(d, q)
    d.times[99] -= 1
    d.mask[:100, 0] = False
    with pytest.raises(ValueError, match='historical observation'):
        features(d, q)


def test_capacity_rotation_retains_saturated_variable_and_deterministic_train_pools():
    allocation = quotas({0: 1, 1: 10, 2: 10}, 12, np.random.default_rng(42))
    assert allocation[0] == 1 and sum(allocation.values()) == 12 and set(allocation) == {0, 1, 2}
    with pytest.raises(ValueError, match='Insufficient'):
        quotas({0: 1}, 12, np.random.default_rng(42))
    assert disjoint_chain([2, 3, 4, 6, 10], 4).tolist() == [2, 6, 10]
    d = series()
    a, coverage = training_queries(d, 'sdwpf', 48, 24, 12)
    assert (a, coverage) == training_queries(d, 'sdwpf', 48, 24, 12)
    assert len(a) == 12 and coverage['eligible_variables'] == 3
    assert all(q['origin']+24 <= int(.6*len(d.values)) for q in a)
    for v in range(3):
        times = sorted(q['origin'] for q in a if q['variable'] == v)
        assert all(b-a >= 24 for a, b in zip(times, times[1:]))


def test_training_slots_missing_targets_variable_equal_mean1_and_boundary_rejection():
    d = series()
    qs = [dict(row(origin=100+i*24)['query'], variable=0 if i < 2 else 1) for i in range(3)]
    d.mask[100:124, 0] = False
    d.mask[100, 0] = True
    x, y, weights, stats = training_arrays(d, qs)
    assert len(y) == 49 and weights.mean() == pytest.approx(1.) and x.shape == (49, 121)
    assert weights[:25].sum() == pytest.approx(weights[25:].sum())
    assert stats['weight_normalization'].startswith('mean1')
    with pytest.raises(ValueError, match='boundary'):
        training_arrays(d, [dict(qs[0], origin=350)])
    d.segments[120:] = 1
    with pytest.raises(ValueError, match='segment'):
        training_arrays(d, qs[:1])


def test_tree_serialization_nan_support_and_iteration_observer(tmp_path):
    x = np.random.default_rng(42).normal(size=(160, 5))
    x[:8, 0] = np.nan
    y = np.nan_to_num(x[:, 0])-x[:, 1]
    model = HistGradientBoostingRegressor(loss='absolute_error', max_iter=3, max_leaf_nodes=5,
        min_samples_leaf=10, early_stopping=False, random_state=42).fit(x, y, sample_weight=np.ones(len(y)))
    path = tmp_path/'model.joblib'
    joblib.dump(model, path)
    assert np.allclose(model.predict(x), joblib.load(path).predict(x), rtol=1e-6, atol=1e-6)
    import io
    seen = []
    log = FitLog(io.StringIO(), lambda i, total, now: seen.append((i, total)))
    log.write('[1/150]')
    log.write(' tree\n')
    log.write('[150/150] done\n')
    assert seen[0] == (1, 150) and seen[-1] == (150, 150)


def paired_data(count=4):
    n = [row(origin=100+i*24, truth=2.) for i in range(count)]
    g = [row(origin=100+i*24, truth=2., prediction=2.) for i in range(count)]
    c = [np.full(24, 2.) for _ in n]
    return n, g, c


def test_convex_endpoints_graph_skip_lp_ties_and_control_independence():
    n, g, c = paired_data()
    p = math.fit(n, g, c)
    assert p['weights'] == pytest.approx([0., 0., 1.])
    assert np.array_equal(math.prediction([1.], None, [2.], [1., 0., 0.]), [1.])
    assert np.array_equal(math.prediction([1.], [3.], [2.], [0., 1., 0.]), [3.])
    assert np.array_equal(math.prediction([1.], None, [2.], [0., 0., 1.]), [2.])
    with pytest.raises(ValueError, match='graph weight'):
        math.prediction([1.], None, [2.], [0., 1., 0.])
    with pytest.raises(ValueError, match='convex'):
        math.prediction([1.], [2.], [3.], [-.1, 0., 1.1])
    control = math.fit(n, g, c, True)
    changed = copy.deepcopy(g)
    for r in changed:
        r['prediction'] = [1000.]*24
    assert math.fit(n, changed, c, True) == control
    for r in n:
        r['prediction'] = [2.]*24
    assert math.fit(n, g, c)['weights'] == pytest.approx([1., 0., 0.])


def test_macro_weighting_missing_slots_and_five_system_reporting():
    n, g, c = paired_data(2)
    n[0]['query']['variable'] = g[0]['query']['variable'] = 1
    n[0]['observed'] = g[0]['observed'] = [True]+[False]*23
    n[0]['target'][1:] = g[0]['target'][1:] = [float('nan')]*23
    p = math.fit(n, g, c)
    rows, metric = math.derived(n, g, c, p['weights'])
    assert p['fit_mae'] == pytest.approx(metric['macro']['mae'])
    result = math.comparison(n, None, c, dict(full=p, native_cpu=math.fit(n, g, c, True)), bootstrap=True)
    assert result['metrics']['gofa'] is None and not result['graph_confirmation_executed']
    assert result['strictly_improved'] and not result['extra_graph_gain_demonstrated']
    assert result['metrics']['ensemble']['derived_prediction']['generation_coverage'] is None
    assert result['paired_variable_cluster_bootstrap']['draws'] == 2000
    for a in n:
        a['representable_range'] = [-.1, .1]
    _, metric = math.derived(n, None, c, [0., 0., 1.])
    assert metric['derived_prediction']['outside_original_range_fraction'] == 1.


def setup_cpu(tmp_path, monkeypatch, failed=False, identity=False):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    output = tmp_path/'output'
    output.mkdir()
    atomic_json(output/'manifest.json', dict(fixture=True))
    atomic_json(output/'cpu_preflight.json', dict(passed=True, cuda_disabled=True, manifest_sha256=sha(output/'manifest.json')))
    marker = dict(models={n: n for n in cpu.TARGETS}, complete=True)
    atomic_json(output/'models_complete.json', marker)
    pairs, cpus, queries = {}, {}, {}
    for name, (_, _, L, H) in cpu.TARGETS.items():
        count = 128 if name == 'sdwpf' else 64
        n = [row(name, origin=1000+i*H, horizon=H, length=L, truth=2.) for i in range(count)]
        g = copy.deepcopy(n)
        c = [np.full(H, 2.) for _ in n]
        if failed and name == 'engrad':
            c[32:48] = [np.full(H, 5.) for _ in range(16)]
        pairs[name], cpus[name] = dict(native=n, gofa=g), c
        queries[name] = [dict(row(name, origin=1000+i*H, horizon=H, length=L)['query']) for i in range(128)]
    m = dict(development_inputs_sha256='original', queries=queries, test_queries=queries,
        development_queries={name: qs[:len(cpus[name])] for name, qs in queries.items()},
        initial_confirmation_estimate_seconds=3600., prior_inference_seconds_per_numeric_slot={'native': .1, 'gofa': .2},
        weights_sha256='native', alphabet=dict(alphabet_sha256='alphabet'))
    monkeypatch.setattr(cpu, 'own_manifest', lambda *_: m)
    monkeypatch.setattr(cpu, 'require_models', lambda *_: marker)
    monkeypatch.setattr(cpu, 'original_development', lambda *_: (pairs, 'original'))
    def cpu_predictions(out, manifest, name, qs, phase):
        path = out/'cpu_predictions'/f'{phase}-{name}.json'
        path.parent.mkdir(exist_ok=True)
        if not path.exists():
            atomic_json(path, dict(fixture=True))
        return cpus[name][:len(qs)]
    monkeypatch.setattr(cpu, 'cpu_predictions', cpu_predictions)
    calls = []
    def fit(n, g, c, native_only=False):
        calls.append((n[0]['query']['source'], len(n), native_only))
        weights = [1., 0., 0.] if identity and n[0]['query']['source'] == 'sdwpf' and len(n) == 128 else [0., 0., 1.]
        return dict(weights=weights)
    monkeypatch.setattr(cpu, 'fit', fit)
    return {}, output, m, pairs, cpus, calls


def test_development_gate_failure_stops_before_any_confirmation_or_test(tmp_path, monkeypatch, capsys):
    cfg, output, m, pairs, cpus, calls = setup_cpu(tmp_path, monkeypatch, failed=True)
    monkeypatch.setattr(cpu, 'read_shard', lambda *_: pytest.fail('Confirmation read before sealing'))
    monkeypatch.setattr(cpu, 'test_records', lambda *_: pytest.fail('Test read after CPU failure'))
    value = cpu.develop(cfg, output)
    assert not value['all_six_passed'] and not value['datasets']['engrad']['both_folds_nondegraded']
    assert len(calls) == 24
    with pytest.raises(ValueError, match='All six development'):
        cpu.seal(cfg, output)
    assert not (output/'coefficients.json').exists()
    from gofa_f.supervised_ensemble_progress import show
    show(output)
    assert 'ETA=not applicable' in capsys.readouterr().out


def test_final_identity_and_input_binding_block_gpu_admission(tmp_path, monkeypatch):
    cfg, output, m, pairs, cpus, calls = setup_cpu(tmp_path, monkeypatch, identity=True)
    assert cpu.develop(cfg, output)['all_six_passed']
    cpu.seal(cfg, output)
    assert read(output/'postfit_gate.json')['identity_datasets'] == ['sdwpf']
    assert len(calls) == 36
    with pytest.raises(ValueError, match='nonidentity'):
        cpu.load_coefficients(cfg, output)
    monkeypatch.setattr(cpu, 'original_development', lambda *_: (pairs, 'changed'))
    with pytest.raises(ValueError, match='evidence changed'):
        cpu.load_development(cfg, output)


def setup_confirm(tmp_path, monkeypatch):
    cfg, output, m, pairs, cpus, calls = setup_cpu(tmp_path, monkeypatch)
    cpu.develop(cfg, output)
    cpu.seal(cfg, output)
    for name in cpu.TARGETS:
        path = output/'evaluations'/f'confirm-{name}-native'
        path.mkdir(parents=True)
        binding = cpu.inference_binding(m, sha(output/'manifest.json'), name, 'native')
        atomic_json(path/'complete.json', dict(complete=True, binding=binding))
    def read_shard(root, i, q, binding, hashes):
        assert root.name.endswith('-native'), 'A zero-weight graph path was read'
        return row(q['source'], origin=q['origin'], horizon=q['horizon'], length=q['length'], truth=2.)
    monkeypatch.setattr(cpu, 'read_shard', read_shard)
    return cfg, output, m, pairs, cpus, calls


def test_zero_weight_skips_graph_confirmation_and_all_six_gate_enables_one_cpu_test(tmp_path, monkeypatch):
    cfg, output, m, pairs, cpus, calls = setup_confirm(tmp_path, monkeypatch)
    before = sha(output/'coefficients.json')
    monkeypatch.setattr(cpu, 'test_records', lambda cfg, m, name, mode: pairs[name][mode])
    with pytest.raises(ValueError, match='All six confirmations'):
        cpu.replay_test(cfg, output)
    assert cpu.complete_confirmation(cfg, output)['all_six_strictly_improved']
    assert read(output/'confirmation/pems03.json')['comparison']['metrics']['gofa'] is None
    result = cpu.replay_test(cfg, output)
    assert result['all_six_strictly_improved'] and result['test_gpu_predictions_generated'] == 0
    monkeypatch.setattr(cpu, 'test_records', lambda *_: pytest.fail('A cached final test was read again'))
    assert cpu.replay_test(cfg, output) == result
    assert len(calls) == 36 and sha(output/'coefficients.json') == before


def test_first_confirmation_failure_stops_remaining_datasets_and_tests(tmp_path, monkeypatch):
    cfg, output, m, pairs, cpus, calls = setup_confirm(tmp_path, monkeypatch)
    cpus['pems03'][:] = [np.full(12, 5.) for _ in range(64)]
    assert not cpu.confirm_dataset(cfg, output, 'pems03')['comparison']['strictly_improved']
    with pytest.raises(ValueError, match='Earlier fixed'):
        cpu.confirm_dataset(cfg, output, 'pems_bay')
    monkeypatch.setattr(cpu, 'test_records', lambda *_: pytest.fail('Test after failed confirmation'))
    with pytest.raises(ValueError, match='All six confirmations'):
        cpu.replay_test(cfg, output)
    assert read(output/'status.json')['stop_reason'] == 'failed_confirm_pems03'
    assert read(output/'progress.json')['test_replay'] == 'skipped'


def test_explicit_supervision_flag_scope_and_cpu_finalization_no_gpu(tmp_path, monkeypatch):
    import gofa_f.short_launcher as launcher
    cfg = yaml.safe_load(Path('configs/stage29_supervised_target_ensemble.yaml').read_text())
    validate_config(cfg)
    assert cfg['data'] == DATA_FLAGS and cfg['data']['target_training_enabled']
    changed = copy.deepcopy(cfg)
    changed['data']['target_training_enabled'] = False
    with pytest.raises(ValueError, match='protocol'):
        validate_config(changed)
    assert type(launcher.admission_gate(cfg['resources'], 'supervised_target_ensemble')).__name__ == 'CapOnlyGate'
    config = tmp_path/'cfg.yaml'
    config.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(launcher.signal, 'signal', lambda *_: None)
    monkeypatch.setattr(launcher, 'query_gpu', lambda: pytest.fail('CPU finalization queried GPU'))
    atomic_json(tmp_path/'status.json', dict(state='confirm_complete'))
    def run(command, **kwargs):
        assert 'gofa_f.supervised_target_ensemble' in command and command[-1] == '--advance-cpu'
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == '' and read(tmp_path/'activity.json')['worker'] is None
        atomic_json(tmp_path/'status.json', dict(state='complete'))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(launcher.subprocess, 'run', run)
    launcher.supervise(tmp_path, config, Path('.'), 'supervised_target_ensemble')
    assert read(tmp_path/'launcher.json')['state'] == 'complete'
    with pytest.raises(ValueError, match='disabled target training'):
        launcher.supervise(tmp_path, config, Path('.'), 'state_residual_calibration')


def test_model_checksum_guard_and_cost_resume_do_not_reset(tmp_path):
    manifest = dict(code={}, configuration={}, training_queries={'sdwpf': []}, raw_hashes={'sdwpf': 'raw'},
        metadata_hashes={'sdwpf': 'meta'}, feature_names={'sdwpf': ['a']})
    (tmp_path/'models').mkdir()
    atomic_json(tmp_path/'models/sdwpf.json', dict(binding=models.model_binding(manifest, 'sdwpf'),
        model_sha256='incorrect', sklearn_version='1.5.2'))
    (tmp_path/'models/sdwpf.joblib').write_bytes(b'not a model')
    with pytest.raises(ValueError, match='binding changed'):
        models.load_model(tmp_path, manifest, 'sdwpf')
    atomic_json(tmp_path/'cpu_activity.json', dict(spent_seconds=10.))
    charge = models.CPUCharge(tmp_path)
    charge.close()
    assert read(tmp_path/'cpu_activity.json')['spent_seconds'] >= 10.
    assert read(tmp_path/'cpu_activity.json')['active'] is None
