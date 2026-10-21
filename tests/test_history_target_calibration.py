import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from gofa_f.allnode_stage import read
from gofa_f.generative_data import sha
from gofa_f.graph_benchmark_data import TARGETS
from gofa_f.history_calibration_math import TAUS, comparison, derived, fit, frame, predict
from gofa_f.history_calibration_state import validate_config
from gofa_f.prefix_cache import atomic_json
import gofa_f.history_calibration_cpu as cpu
from test_target_calibration import entry


def series(size=12000, nodes=1, hourly=False):
    return SimpleNamespace(values=np.full((size, nodes), 2.), mask=np.ones((size, nodes), bool),
        times=np.arange(size, dtype=np.int64)*(3600 if hourly else 600)*10**9)


def row(name='sdwpf', origin=100, horizon=24, prediction=0., truth=2., length=48):
    result = entry(name=name, origin=origin, horizon=horizon, native=prediction, truth=truth)
    result['query']['length'] = length
    return result


def test_causal_experts_ignore_every_future_value_and_mask():
    data = series(hourly=True)
    data.values[:100, 0] = np.arange(100)*.1
    n, g = row('engrad'), row('engrad', prediction=1.)
    f = frame(n, g, data)
    assert f['names'] == ['native', 'gofa', 'persistence', 'mean', 'trend', 'daily']
    assert f['experts'][0, 2] == pytest.approx(9.9)
    assert f['experts'][0, 4] == pytest.approx(10.)
    assert f['experts'][-1, 4] == pytest.approx(12.3)
    assert np.allclose(f['experts'][:, 5], data.values[76:100, 0])
    data.values[100:] = 1e20
    data.mask[100:] = False
    other = frame(n, g, data)
    assert np.array_equal(f['experts'], other['experts']) and f['scale'] == other['scale']


def test_missing_constant_daily_fallback_and_elapsed_trend_alignment():
    data = series(hourly=True)
    n, g = row('engrad'), row('engrad', prediction=1.)
    data.mask[88:100] = False
    f = frame(n, g, data)
    assert np.all(f['experts'][:, 2:] == 2.)
    data.mask[:] = True
    data.values[:100, 0] = np.arange(100)
    data.mask[98:100] = False
    f = frame(n, g, data)
    assert f['experts'][0, 2] == 97. and f['experts'][0, 4] == pytest.approx(100.)
    data.mask[76] = False
    assert frame(n, g, data)['experts'][0, 5] == 97.
    data.mask[:100] = False
    with pytest.raises(ValueError, match='observed history'):
        frame(n, g, data)


def test_wind_has_no_daily_expert_and_invalid_unit_or_scale_rejected():
    data = series()
    n, g = row(), row(prediction=1.)
    f = frame(n, g, data)
    assert 'daily' not in f['names'] and f['experts'].shape == (24, 5)
    n['scaling'] = g['scaling'] = [0., 0.]
    with pytest.raises(ValueError, match='scale'):
        frame(n, g, data)
    n, g = row('engrad'), row('engrad', prediction=1.)
    with pytest.raises(ValueError, match='hourly'):
        frame(n, g, data)


def toy_frames(native, graph, perfect=2., native_only=False):
    names = ['native']+([] if native_only else ['gofa'])+['persistence', 'mean', 'trend']
    result = []
    for n, g in zip(native, graph):
        h = n['query']['horizon']
        columns = [n['prediction']]+([] if native_only else [g['prediction']])+[[perfect]*h]*3
        result.append(dict(names=names, experts=np.asarray(columns, np.float64).T,
            scale=1., u=np.arange(h)/(h-1)))
    return result


@pytest.mark.parametrize('tau', TAUS)
def test_regularized_lp_objective_simplex_bias_bounds_and_primary_ties(tau):
    n = [row(horizon=2, truth=2., origin=100+i*2) for i in range(4)]
    g = [row(horizon=2, prediction=4., truth=2., origin=100+i*2) for i in range(4)]
    fs = toy_frames(n, g)
    value = fit(n, g, fs, tau)
    assert sum(value['weights'].values()) == pytest.approx(1.)
    assert min(value['weights'].values()) >= 0
    assert max(abs(value['bias'][0]), abs(sum(value['bias']))) <= .25+1e-10
    assert value['objective'] == pytest.approx(value['fit_mae']/value['native_fit_mae']+tau*value['penalty'])
    assert np.isfinite(predict(fs[0], value)).all()
    if tau < 1:
        assert value['fit_mae'] < 1e-8
        # Half graph + half Native has smaller shrinkage cost than all persistence.
        assert value['weights']['gofa'] == pytest.approx(.5)
    assert fit(n, g, fs, tau) == value


def test_native_endpoint_zero_baseline_missing_labels_and_macro_weighting():
    n = [row(horizon=2, prediction=2.)]
    g = [row(horizon=2, prediction=4.)]
    fs = toy_frames(n, g)
    p = fit(n, g, fs, .1)
    assert p['weights']['native'] == 1. and p['bias'] == [0., 0.]
    assert np.array_equal(predict(fs[0], p), n[0]['prediction'])
    a = row(horizon=2, truth=1.)
    a['observed'] = [True, False]
    a['target'][1] = np.nan
    b = copy.deepcopy(a)
    b['prediction'] = [3., 3.]
    value = fit([a], [b], toy_frames([a], [b], perfect=1.), .01)
    assert value['fit_mae'] < 1e-8
    with pytest.raises(ValueError, match='preregistered'):
        fit(n, g, fs, .2)


def test_derived_predictions_are_unclipped_and_graph_gain_requires_control_win():
    n, g = [row(horizon=2)], [row(horizon=2, prediction=4.)]
    fs, cs = toy_frames(n, g), toy_frames(n, g, native_only=True)
    p, c = fit(n, g, fs, .01), fit(n, g, cs, .01)
    result = comparison(n, g, fs, cs, p, c, bootstrap=True)
    assert result['strictly_improved'] and not result['extra_graph_gain_demonstrated']
    assert result['paired_variable_cluster_bootstrap']['draws'] == 2000
    assert result['metrics']['calibrated']['derived_prediction']['generation_coverage'] is None
    n[0]['representable_range'] = g[0]['representable_range'] = [-.1, .1]
    rows, metrics = derived(n, g, fs, p)
    assert rows[0]['prediction'][0] > .1
    assert metrics['derived_prediction']['outside_original_range_fraction'] == 1.


def setup_cpu(tmp_path, monkeypatch, fail=False):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    output = tmp_path/'output'
    output.mkdir()
    cfg = dict(folds=[[32, 48], [48, 64]])
    manifest = dict(data_directory=str(tmp_path), development_inputs_sha256='frozen-input', queries={})
    datasets, data = {}, {}
    for name, (_, _, length, horizon) in TARGETS.items():
        data[name] = series(hourly=name in ('engrad', 'meteonet_nw2018'))
        truth = 0. if fail and name == 'sdwpf' else 2.
        n = [row(name, origin=1000+i*horizon, horizon=horizon, length=length, truth=truth) for i in range(64)]
        g = copy.deepcopy(n)
        for r in g:
            r['prediction'] = [4.]*horizon
        datasets[name] = dict(native=n, gofa=g)
        manifest['queries'][name] = [r['query'] for r in n]
    atomic_json(output/'manifest.json', dict(fixture=True))
    atomic_json(output/'cpu_preflight.json', dict(passed=True, cuda_disabled=True,
        manifest_sha256=sha(output/'manifest.json')))
    monkeypatch.setattr(cpu, 'own_manifest', lambda *_: manifest)
    monkeypatch.setattr(cpu, 'verify_data', lambda *_: None)
    monkeypatch.setattr(cpu, 'development_records', lambda *_: (datasets, 'frozen-input'))
    monkeypatch.setattr(cpu, 'Series', lambda p: data[p.stem])
    events = []
    def simple_fit(n, g, fs, tau):
        events.append((len(n), [r['query']['origin'] for r in n], tau, fs[0]['names']))
        target = 'native' if n[0]['target'][0] == 0 else 'persistence'
        return dict(experts=fs[0]['names'], weights={k: float(k == target) for k in fs[0]['names']},
            bias=[0., 0.], tau=tau, stored_parameters=len(fs[0]['names'])+2)
    monkeypatch.setattr(cpu, 'fit', simple_fit)
    return cfg, output, manifest, datasets, events


def test_forward_folds_control_separation_tau_ties_and_sealed_resume(tmp_path, monkeypatch):
    cfg, output, manifest, data, events = setup_cpu(tmp_path, monkeypatch)
    value = cpu.develop(cfg, output)
    assert value['all_six_strictly_improved'] and len(events) == 72
    assert {x[0] for x in events} == {32, 48}
    for name in TARGETS:
        assert value['datasets'][name]['full']['selected_tau'] == 1.
        assert value['datasets'][name]['native_history']['selected_tau'] == 1.
        for mode in ('full', 'native_history'):
            for candidate in value['datasets'][name][mode]['candidates']:
                for fold in candidate['folds']:
                    assert fold['train_queries'] == fold['validation_begin']
                    if mode == 'native_history':
                        assert 'gofa' not in fold['parameters']['experts']
    assert cpu.develop(cfg, output) == value and len(events) == 72
    seal = cpu.seal(cfg, output)
    assert len(events) == 84 and seal['stored_deployment_parameters'] == 44
    assert cpu.seal(cfg, output) == seal and len(events) == 84
    assert not seal['independent_confirmation_or_test_read']
    monkeypatch.setattr(cpu, 'development_records', lambda *_: (data, 'changed'))
    with pytest.raises(ValueError, match='changed'):
        cpu.load_coefficients(cfg, output)


def test_cpu_failure_never_fits_all64_or_opens_confirmation_test(tmp_path, monkeypatch, capsys):
    cfg, output, manifest, data, events = setup_cpu(tmp_path, monkeypatch, fail=True)
    monkeypatch.setattr(cpu, 'records', lambda *_: pytest.fail('Unauthorized confirmation read'))
    monkeypatch.setattr(cpu, 'test_records', lambda *_: pytest.fail('Unauthorized test read'))
    value = cpu.develop(cfg, output)
    assert not value['all_six_strictly_improved'] and len(events) == 72
    assert read(output/'activity.json')['spent_seconds'] == 0.
    assert read(output/'activity.json')['worker'] is None
    with pytest.raises(ValueError, match='CPU checks'):
        cpu.seal(cfg, output)
    with pytest.raises(ValueError, match='CPU improvements'):
        cpu.confirmation(cfg, output)
    assert not (output/'coefficients.json').exists() and not (output/'test_replay.json').exists()
    from gofa_f.history_target_progress import show
    show(output)
    text = capsys.readouterr().out
    assert 'ETA=not applicable' in text and 'skipped' in text
    assert read(output/'progress.json')['overall_eta_seconds'] is None


def test_confirm_and_test_require_both_seals_and_do_not_refit(tmp_path, monkeypatch):
    cfg, output, manifest, datasets, events = setup_cpu(tmp_path, monkeypatch)
    monkeypatch.setattr(cpu, 'test_records', lambda *_: pytest.fail('Test opened before confirmation'))
    with pytest.raises(ValueError):
        cpu.replay_test(cfg, output)
    cpu.develop(cfg, output)
    cpu.seal(cfg, output)
    before = sha(output/'coefficients.json')
    with pytest.raises(ValueError, match='independent confirmations'):
        cpu.replay_test(cfg, output)
    monkeypatch.setattr(cpu, 'records', lambda *_: (datasets, 'confirmation-input'))
    assert cpu.confirmation(cfg, output)['all_six_strictly_improved']
    monkeypatch.setattr(cpu, 'test_records', lambda cfg, m, name, mode: datasets[name][mode])
    value = cpu.replay_test(cfg, output)
    assert value['all_six_strictly_improved'] and value['test_gpu_predictions_generated'] == 0
    assert len(events) == 84 and sha(output/'coefficients.json') == before
    assert cpu.replay_test(cfg, output) == value


def test_cap_only_scope_and_finalizer_has_no_gpu_admission(tmp_path, monkeypatch):
    import gofa_f.short_launcher as launcher
    cfg = yaml.safe_load(Path('configs/stage27_history_target_calibration.yaml').read_text())
    validate_config(cfg)
    assert type(launcher.admission_gate(cfg['resources'], 'history_target_calibration')).__name__ == 'CapOnlyGate'
    config = tmp_path/'cfg.yaml'
    config.write_text(yaml.safe_dump(cfg))
    atomic_json(tmp_path/'status.json', dict(state='confirm_complete'))
    monkeypatch.setattr(launcher.signal, 'signal', lambda *_: None)
    monkeypatch.setattr(launcher, 'query_gpu', lambda: pytest.fail('CPU completion queried GPU'))
    def run(command, **kwargs):
        assert 'gofa_f.history_target_calibration' in command and command[-1] == '--advance-cpu'
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == '' and read(tmp_path/'activity.json')['worker'] is None
        atomic_json(tmp_path/'status.json', dict(state='complete', stop_reason='failed_independent_confirmation'))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(launcher.subprocess, 'run', run)
    launcher.supervise(tmp_path, config, Path('.'), 'history_target_calibration')
    assert read(tmp_path/'launcher.json')['state'] == 'complete'


def test_all64_refit_identity_stops_before_confirmation_even_after_oos_pass(tmp_path, monkeypatch):
    cfg, output, manifest, data, events = setup_cpu(tmp_path, monkeypatch)
    assert cpu.develop(cfg, output)['all_six_strictly_improved']
    original = cpu.fit
    def refit(n, g, fs, tau):
        result = original(n, g, fs, tau)
        if n[0]['query']['source'] == 'sdwpf' and 'gofa' in fs[0]['names']:
            result['weights'] = {k: float(k == 'native') for k in fs[0]['names']}
            result['bias'] = [-0., 0.]
        return result
    monkeypatch.setattr(cpu, 'fit', refit)
    before = len(events)
    value = cpu.seal(cfg, output)
    assert len(events) == before+12
    assert cpu.exact_native_identity(value['coefficients']['sdwpf']['full'])
    gate = read(output/'postfit_gate.json')
    assert gate['identity_datasets'] == ['sdwpf'] and not gate['all_datasets_nonidentity']
    assert read(output/'status.json')['state'] == 'complete'
    assert read(output/'activity.json')['spent_seconds'] == 0.
    with pytest.raises(ValueError, match='Postfit feasibility'):
        cpu.load_coefficients(cfg, output)
    assert not (output/'confirm_complete.json').exists()
