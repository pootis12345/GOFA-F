import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from gofa_f.allnode_stage import read
from gofa_f.generative_data import sha
from gofa_f.prefix_cache import atomic_json
import gofa_f.state_residual_math as math
import gofa_f.state_residual_cpu as cpu
import gofa_f.state_residual_state as state
from test_history_target_calibration import row


def fixture(monkeypatch):
    values = np.stack((np.arange(200)*2., np.arange(200)*20.+500., np.ones(200)*4.), 1)
    data = SimpleNamespace(values=values, mask=np.ones_like(values, bool), ids=['a', 'b', 'c'],
        external=np.array([[0., 1., 1.], [1., 0., 0.], [1., 0., 0.]]))
    monkeypatch.setattr(math, 'build_graph', lambda *_: SimpleNamespace(variable_ids=('a', 'b', 'c'), identity='frozen'))
    n, g = row(), row(prediction=5.)
    n['scaling'] = g['scaling'] = [1./94., -52./47.]
    return data, n, g


def test_eight_features_node_independence_future_causality_and_control(monkeypatch):
    data, n, g = fixture(monkeypatch)
    f = math.frame(n, g, data)
    assert f['matrix'].shape == (24, 8) and f['features'] == list(math.FEATURES)
    assert f['history']['scale'] == 94. and f['history']['neighbors'] == 2
    assert f['matrix'][0, 1] == 0. and f['matrix'][-1, 1] == 1.
    before = f['matrix'].copy()
    data.values[100:] = 1e25
    data.mask[100:] = False
    assert np.array_equal(math.frame(n, g, data)['matrix'], before)
    # Changing a neighbor's absolute offset cannot contaminate target or neighbor MinMax state.
    data.values[:100, 1] += 100000.
    assert np.allclose(math.frame(n, g, data)['matrix'], before)
    cs = math.frame(n, g, data, True)
    assert cs['matrix'].shape == (24, 7) and 'graph_difference' not in cs['features']
    changed = copy.deepcopy(g)
    changed['prediction'] = [999.]*24
    assert np.array_equal(math.frame(n, changed, data, True)['matrix'], cs['matrix'])


def test_missing_neighbor_constant_short_history_and_scaling_refusal(monkeypatch):
    data, n, g = fixture(monkeypatch)
    data.mask[:100, 1:] = False
    f = math.frame(n, g, data)
    assert f['history']['neighbors'] == 0 and np.all(f['matrix'][:, 6:] == 0.)
    data.values[:100, 0] = 2.
    n['scaling'] = g['scaling'] = [1., -2.]
    f = math.frame(n, g, data)
    assert f['scale'] == 1. and f['history']['trend'] == 0.
    data.mask[88:99, 0] = False
    assert math.frame(n, g, data)['history']['trend'] == 0.
    data.mask[:100, 0] = False
    with pytest.raises(ValueError, match='observed target'):
        math.frame(n, g, data)


def toy(count=8, truth=-.1):
    n = [row(origin=100+i*24, truth=truth) for i in range(count)]
    g = copy.deepcopy(n)
    fs = [dict(features=list(math.FEATURES), matrix=np.column_stack((np.ones(24), np.zeros((24, 7)))),
        native=np.zeros(24), scale=1.) for _ in n]
    return n, g, fs


@pytest.mark.parametrize('tau', math.TAUS)
def test_negative_beta_l1_bound_objective_exact_endpoint_and_ties(tau):
    n, g, fs = toy()
    p = math.fit(n, g, fs, tau)
    assert p['beta'][0] == pytest.approx(-.1) and sum(abs(x) for x in p['beta']) <= .25+1e-10
    assert math.fit(n, g, fs, tau) == p
    assert p['objective'] == pytest.approx(p['fit_mae']/p['native_fit_mae']+tau*p['penalty'])
    assert np.max(np.abs(math.predict(fs[0], p)-fs[0]['native'])) <= .25
    identity = dict(p, beta=[0.]*8)
    assert np.array_equal(math.predict(fs[0], identity), fs[0]['native'])
    with pytest.raises(ValueError, match='constraint'):
        math.predict(fs[0], dict(p, beta=[1.]*8))


def test_zero_error_mask_and_variable_equal_reduction():
    n, g, fs = toy(2, truth=0.)
    p = math.fit(n, g, fs, .1)
    assert p['beta'] == [0.]*8 and p['fit_mae'] == 0.
    n[0]['query']['variable'] = g[0]['query']['variable'] = 1
    n[0]['observed'] = g[0]['observed'] = [True]+[False]*23
    n[0]['target'][0] = g[0]['target'][0] = 1.
    x, y, base, weights = math.arrays(n, g, fs)
    assert weights[0] == pytest.approx(.5) and weights[1:].sum() == pytest.approx(.5)
    assert weights @ np.abs(base-y) == pytest.approx(.5)
    with pytest.raises(ValueError, match='preregistered'):
        math.fit(n, g, fs, .2)


def test_derived_outputs_not_clipped_and_control_interpretation():
    n, g, fs = toy(2)
    for r in n+g:
        r['representable_range'] = [-.01, .01]
    p = math.fit(n, g, fs, .1)
    rows, metric = math.derived(n, g, fs, p)
    assert metric['derived_prediction']['outside_original_range_fraction'] == 1.
    assert metric['derived_prediction']['generation_coverage'] is None
    assert rows[0]['prediction'][0] < -.01
    result = math.compare(n, g, rows, metric, rows, metric, graph_used=True, bootstrap=True)
    assert result['strictly_improved'] and not result['extra_graph_gain_demonstrated']
    assert result['paired_variable_cluster_bootstrap']['draws'] == 2000


def test_additional_queries_opposite_block_fixed64_and_disjoint(monkeypatch):
    original = []
    for block, count in enumerate((16, 16, 16, 16, 64)):
        for j in range(count):
            original.append(dict(source='sdwpf', variable=len(original), origin=1000+block*1000+j,
                length=48, horizon=24, chunk=0))
    data = SimpleNamespace(ids=[str(i) for i in range(134)])
    monkeypatch.setattr(state, 'origins', lambda *_args, **_kwargs: np.arange(1000, 5900))
    a, order = state.additional_queries(data, original, dict(boundaries=[1000, 2000, 3000, 4000, 5000, 6000]))
    assert len(a) == 64 and len(order) == 128
    assert (a, order) == state.additional_queries(data, original, dict(boundaries=[1000, 2000, 3000, 4000, 5000, 6000]))
    for i, q in enumerate(a):
        assert 1000+((i//16+2) % 4)*1000 <= q['origin'] < 2000+((i//16+2) % 4)*1000-23
        assert q['variable'] == original[i]['variable'] and abs(q['origin']-original[i]['origin']) >= 24
    assert max(r['query']['origin'] for r in order[:64]) < min(r['query']['origin'] for r in order[64:96])


def test_source_protocol_scope_and_supervisor_cpu_gate(tmp_path, monkeypatch):
    import gofa_f.short_launcher as launcher
    cfg = yaml.safe_load(Path('configs/stage28_state_residual_calibration.yaml').read_text())
    state.validate_config(cfg)
    assert type(launcher.admission_gate(cfg['resources'], 'state_residual_calibration')).__name__ == 'CapOnlyGate'
    config = tmp_path/'cfg.yaml'
    config.write_text(yaml.safe_dump(cfg))
    atomic_json(tmp_path/'status.json', dict(state='development_complete'))
    monkeypatch.setattr(launcher.signal, 'signal', lambda *_: None)
    monkeypatch.setattr(launcher, 'query_gpu', lambda: pytest.fail('CPU gate queried GPU'))
    def run(command, **kwargs):
        assert 'gofa_f.state_residual_calibration' in command and command[-1] == '--advance-cpu'
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == '' and read(tmp_path/'activity.json')['worker'] is None
        atomic_json(tmp_path/'status.json', dict(state='complete', stop_reason='failed_sdwpf_development_gate'))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(launcher.subprocess, 'run', run)
    launcher.supervise(tmp_path, config, Path('.'), 'state_residual_calibration')
    assert read(tmp_path/'launcher.json')['state'] == 'complete'


def setup_cpu(tmp_path, monkeypatch, poor=False, final_identity=False):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    output = tmp_path/'output'
    output.mkdir()
    atomic_json(output/'manifest.json', dict(fixture=True))
    atomic_json(output/'cpu_preflight.json', dict(passed=True, cuda_disabled=True, manifest_sha256=sha(output/'manifest.json')))
    n, g, fs = toy(128, truth=1.)
    frozen = {name: dict(full=dict(stored_parameters=7), native_history={})
              for name in ('pems03', 'pems_bay', 'meteonet_nw2018', 'engrad', 'kelmarsh2019')}
    from gofa_f.target_calibration_state import digest
    m = dict(data_directory=str(tmp_path), frozen_coefficients=frozen, frozen_coefficients_sha256=digest(frozen))
    monkeypatch.setattr(cpu, 'own_manifest', lambda *_: m)
    monkeypatch.setattr(cpu, 'development_pair', lambda *_args, **_kwargs: (dict(native=n, gofa=g), 'input-sha'))
    monkeypatch.setattr(cpu, 'Series', lambda *_: None)
    monkeypatch.setattr(cpu, 'previous', lambda *_: None)
    monkeypatch.setattr(cpu, 'frame', lambda a, b, data, native_only=False: dict(fs[0],
        matrix=fs[0]['matrix'] if not native_only else fs[0]['matrix'][:, :2],
        features=list(math.FEATURES) if not native_only else list(math.FEATURES[:2])))
    calls = []
    def fit(a, b, frames, tau):
        calls.append((len(a), tau, frames[0]['features']))
        beta = [0.]*len(frames[0]['features'])
        if not (poor or final_identity and len(a) == 128):
            beta[0] = .1
        return dict(features=frames[0]['features'], beta=beta, stored_parameters=len(beta))
    monkeypatch.setattr(cpu, 'fit', fit)
    # Frames have repeated intercept and the toy native is0; correction .1 improves truth1.
    return dict(folds=[[64, 96], [96, 128]], development_minimum_improvement=.005), output, m, calls


def test_development_failure_never_refits_or_reads_confirmation(tmp_path, monkeypatch, capsys):
    cfg, output, m, calls = setup_cpu(tmp_path, monkeypatch, poor=True)
    monkeypatch.setattr(cpu, 'prediction_records', lambda *_: pytest.fail('Unauthorized confirmation read'))
    monkeypatch.setattr(cpu, 'test_records', lambda *_: pytest.fail('Unauthorized test read'))
    assert not cpu.develop(cfg, output)['passed']
    assert len(calls) == 12 and all(c[0] < 128 for c in calls)
    with pytest.raises(ValueError, match='gate must pass'):
        cpu.seal(cfg, output)
    from gofa_f.state_residual_progress import show
    show(output)
    assert 'ETA=not applicable' in capsys.readouterr().out
    assert not (output/'coefficients.json').exists()


def test_final_identity_blocks_confirmation_after_development_pass(tmp_path, monkeypatch):
    cfg, output, m, calls = setup_cpu(tmp_path, monkeypatch, final_identity=True)
    assert cpu.develop(cfg, output)['passed']
    p = cpu.seal(cfg, output)
    assert p['coefficients']['sdwpf']['full']['beta'] == [0.]*8 and len(calls) == 14
    assert read(output/'status.json')['state'] == 'complete'
    with pytest.raises(ValueError, match='nonidentity'):
        cpu.load_coefficients(cfg, output)


def test_frozen_five_binding_and_one_way_test_admission(tmp_path, monkeypatch):
    cfg, output, m, calls = setup_cpu(tmp_path, monkeypatch)
    assert cpu.develop(cfg, output)['passed']
    p = cpu.seal(cfg, output)
    assert p['coefficients']['pems03'] == m['frozen_coefficients']['pems03']
    assert len(calls) == 14 and cpu.seal(cfg, output) == p
    monkeypatch.setattr(cpu, 'test_records', lambda *_: pytest.fail('Test read before confirmation'))
    with pytest.raises(ValueError, match='All six independent'):
        cpu.replay_test(cfg, output)
    p['coefficients']['pems03']['full']['stored_parameters'] = 8
    atomic_json(output/'coefficients.json', p)
    lock = read(output/'coefficient_lock.json')
    lock['coefficients_sha256'] = sha(output/'coefficients.json')
    atomic_json(output/'coefficient_lock.json', lock)
    post = read(output/'postfit_gate.json')
    post['coefficients_sha256'] = lock['coefficients_sha256']
    atomic_json(output/'postfit_gate.json', post)
    with pytest.raises(ValueError, match='Frozen five'):
        cpu.load_coefficients(cfg, output)


def test_confirmation_failure_skips_test_and_cannot_refit(tmp_path, monkeypatch):
    cfg, output, m, calls = setup_cpu(tmp_path, monkeypatch)
    cpu.develop(cfg, output)
    cpu.seal(cfg, output)
    before = sha(output/'coefficients.json')
    monkeypatch.setattr(cpu, 'prediction_records', lambda *_: ({n: {} for n in cpu.TARGETS}, 'confirm-input'))
    monkeypatch.setattr(cpu, 'comparison', lambda name, *_args, **_kwargs: dict(strictly_improved=name != 'sdwpf'))
    monkeypatch.setattr(cpu, 'test_records', lambda *_: pytest.fail('Test read after failed confirmation'))
    assert not cpu.confirmation(cfg, output)['all_six_strictly_improved']
    with pytest.raises(ValueError, match='All six independent'):
        cpu.replay_test(cfg, output)
    assert len(calls) == 14 and sha(output/'coefficients.json') == before
    assert read(output/'status.json')['stop_reason'] == 'failed_independent_confirmation'
    assert read(output/'progress.json')['test_replay'] == 'skipped'


def test_complete_confirmation_and_cpu_test_are_replayable_without_fitting(tmp_path, monkeypatch):
    cfg, output, m, calls = setup_cpu(tmp_path, monkeypatch)
    cpu.develop(cfg, output)
    cpu.seal(cfg, output)
    before = sha(output/'coefficients.json')
    monkeypatch.setattr(cpu, 'prediction_records', lambda *_: ({n: {} for n in cpu.TARGETS}, 'confirm-input'))
    monkeypatch.setattr(cpu, 'comparison', lambda *_args, **_kwargs: dict(strictly_improved=True))
    monkeypatch.setattr(cpu, 'test_records', lambda *_: [])
    assert cpu.confirmation(cfg, output)['all_six_strictly_improved']
    assert cpu.replay_test(cfg, output)['test_gpu_predictions_generated'] == 0
    assert cpu.replay_test(cfg, output)['all_six_strictly_improved']
    assert len(calls) == 14 and sha(output/'coefficients.json') == before
    assert read(output/'status.json')['goal_achieved']


def test_dev_input_tampering_and_preseal_import_rejected(tmp_path, monkeypatch):
    cfg, output, m, calls = setup_cpu(tmp_path, monkeypatch)
    cpu.develop(cfg, output)
    with pytest.raises(ValueError, match='Sealed nonidentity'):
        state.import_confirmation(cfg, output, m)
    monkeypatch.setattr(cpu, 'development_pair', lambda *_args, **_kwargs: ({}, 'changed-input'))
    with pytest.raises(ValueError, match='inputs changed'):
        cpu.load_development(cfg, output)
