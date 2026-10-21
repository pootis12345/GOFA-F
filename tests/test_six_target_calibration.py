import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from gofa_f.allnode_stage import read
from gofa_f.generative_data import sha
from gofa_f.graph_benchmark_data import TARGETS
from gofa_f.prefix_cache import atomic_json
from gofa_f.short_sampling_diagnostic import sample_seed
from gofa_f.six_calibration_math import comparison, derived, fit, prediction
from gofa_f.six_calibration_state import label, query_pool, result_binding, validate_config
import gofa_f.six_calibration_cpu as cpu
from test_target_calibration import entry


def rows(truths, graph, variables=None):
    variables = variables or list(range(len(truths)))
    n = [entry(variable=v, origin=500+i, truth=y) for i, (v, y) in enumerate(zip(variables, truths))]
    g = copy.deepcopy(n)
    for r, p in zip(g, graph):
        r['prediction'] = [p]
    return n, g


@pytest.mark.parametrize('shift', [-3., 0., 2.])
def test_lp_bias_sign_identical_predictions_and_native_control(shift):
    n, g = rows([shift]*3, [0.]*3)
    for native_only in (False, True):
        value = fit(n, g, native_only=native_only)
        assert value['coefficient'] == 0.
        assert value['bias'] == pytest.approx(shift, abs=1e-12)
        assert value['fit_mae'] < 1e-12


@pytest.mark.parametrize('coefficient', [0., .3, 1.])
def test_lp_unique_blend_solution_and_lexicographic_priority(coefficient):
    n, g = rows([2.-coefficient, 2.+coefficient], [-1., 1.])
    result = fit(n, g)
    assert result['coefficient'] == pytest.approx(coefficient, abs=1e-10)
    assert result['bias'] == pytest.approx(2.)
    assert result['fit_mae'] < 1e-9
    assert len(result['solver_runs']) == 4
    assert np.array_equal(prediction([1e20], [.123], 0., 0.), [1e20])
    assert np.array_equal(prediction([1e20], [.123], 1., 0.), [.123])


def test_tie_chooses_smallest_lambda_then_smallest_absolute_bias():
    # Constant graph correction is indistinguishable from a shared bias: lambda=0 wins.
    n, g = rows([2., 2.], [4., 4.])
    result = fit(n, g)
    assert result['coefficient'] == 0. and result['bias'] == pytest.approx(2.)
    # Every bias in [-2, 3] minimizes MAE. The absolute-bias criterion chooses zero.
    n, g = rows([-2., 3.], [0., 0.])
    result = fit(n, g)
    assert result['coefficient'] == 0. and result['bias'] == 0.


def test_weighted_variables_masks_and_no_clipping():
    n = [entry(variable=0, truth=-1., horizon=3), entry(variable=1, truth=1.)]
    g = copy.deepcopy(n)
    result = fit(n, g)
    assert result['bias'] == 0. and result['fit_mae'] == 1.
    n[0]['observed'] = g[0]['observed'] = [True, False, False]
    n[0]['target'][1:] = g[0]['target'][1:] = [np.nan, -1e20]
    assert fit(n, g)['fit_mae'] == 1.
    _, metrics = derived(n, g, 0., 200.)
    assert metrics['derived_prediction']['outside_original_range_fraction'] == 1.
    assert metrics['derived_prediction']['generation_coverage'] is None
    with pytest.raises(ValueError, match='Finite'):
        prediction([0.], [0.], 0., np.nan)
    g[0]['scaling'] = [2., 0.]
    with pytest.raises(ValueError, match='scaling'):
        fit(n, g)


def test_bootstrap_uses_variable_clusters_and_all_four_systems():
    n, g = rows([2.]*6, [4.]*6)
    fitted = dict(blend_bias=fit(n, g), native_bias=fit(n, g, native_only=True))
    result = comparison(n, g, fitted, bootstrap=True)
    assert result['strictly_improved'] and not result['extra_graph_gain_demonstrated']
    assert set(result['metrics']) == {'native', 'gofa', 'native_bias', 'blend_bias'}
    interval = result['paired_variable_cluster_bootstrap']
    assert interval['clusters'] == 6 and interval['draws'] == 2000 and interval['seed'] == 4226


def test_query_lists_are_fixed_768_balanced_disjoint_and_five_time_partitions():
    series = {}
    for name, (_, nodes, _, _) in TARGETS.items():
        n = nodes or 216
        series[name] = SimpleNamespace(values=np.broadcast_to(np.zeros((20000, 1)), (20000, n)),
            prefix_counts=np.broadcast_to(np.arange(20001)[:, None], (20001, n)),
            segments=np.zeros(20000), ids=list(map(str, range(n))))
    queries, coverage = query_pool(series)
    assert query_pool(series) == (queries, coverage)
    assert sum(map(len, queries.values())) == 768
    assert 16*sum(q['horizon'] for rows in queries.values() for q in rows) == 245760
    for name, qs in queries.items():
        assert len({q['variable'] for q in qs}) == (6 if name == 'kelmarsh2019' else 128)
        cuts = coverage[name]['boundaries']
        for k, (a, b) in enumerate(((0, 16), (16, 32), (32, 48), (48, 64), (64, 128))):
            for q in qs[a:b]:
                assert cuts[k] <= q['origin'] and q['origin']+q['horizon'] <= cuts[k+1]
        for v in {q['variable'] for q in qs}:
            times = sorted(q['origin'] for q in qs if q['variable'] == v)
            assert all(b-a >= qs[0]['horizon'] for a, b in zip(times, times[1:]))


def archive(tmp_path):
    output = tmp_path/'new'
    output.mkdir()
    cfg = {'adapter_sha256': 'adapter', 'stage21_output': str(tmp_path/'forbidden-test')}
    manifest = dict(configuration=cfg, weights_sha256='backbone', alphabet={'alphabet_sha256': 'alphabet'},
                    queries={})
    atomic_json(output/'manifest.json', {'fixture': True})
    binding = sha(output/'manifest.json')
    for name in TARGETS:
        qs = [entry(name=name, origin=500+i)['query'] for i in range(128)]
        manifest['queries'][name] = qs
        for phase, selected in (('pilot', qs[:64]), ('confirm', qs[64:])):
            for mode, p in (('native', 0.), ('gofa', 4.)):
                root = output/'evaluations'/label(phase, name, mode)
                (root/'trajectories').mkdir(parents=True)
                expected = result_binding(manifest, binding, phase, name, mode)
                for i, q in enumerate(selected):
                    row = entry(name=name, origin=q['origin'], native=p)
                    row['binding'] = expected
                    atomic_json(root/f'{i:06d}.json', row)
                    for sample in range(8):
                        atomic_json(root/'trajectories'/f'{i:06d}-{sample:02d}.json', dict(query=q,
                            binding=expected, prediction=[p], sample=sample, seed=sample_seed(q, sample)))
        for phase in ('pilot', 'confirm'):
            atomic_json(output/f'{phase}_complete.json', dict(binding=binding, complete=True))
    return cfg, output, manifest


def test_fit_cannot_open_confirmation_or_test_and_seals_before_pilot(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    cfg, output, manifest = archive(tmp_path)
    monkeypatch.setattr(cpu, 'own_manifest', lambda *_: manifest)
    original = cpu.read_shard
    reads = []
    def guarded(root, index, *args):
        assert root.parent == output/'evaluations' and root.name.startswith('pilot-') and index < 48
        reads.append((root, index))
        return original(root, index, *args)
    monkeypatch.setattr(cpu, 'read_shard', guarded)
    coefficients = cpu.fit_coefficients(cfg, output)
    assert len(reads) == 6*2*48
    assert coefficients['deployment_scalar_count'] == 12
    assert not coefficients['confirmation_or_test_records_read']
    assert cpu.fit_coefficients(cfg, output) == coefficients
    monkeypatch.setattr(cpu, 'read_shard', original)
    assert cpu.confirm(cfg, output, 'pilot')['all_six_strictly_improved']
    assert cpu.confirm(cfg, output, 'confirm')['all_six_strictly_improved']
    row_path = output/'evaluations/pilot-pems03-native/trajectories/000000-00.json'
    value = read(row_path)
    value['prediction'] = [1.]  # Median stays unchanged but the seal must reject its hash.
    atomic_json(row_path, value)
    with pytest.raises(ValueError, match='changed after sealing'):
        cpu.load_coefficients(cfg, output)


def test_no_test_access_without_both_gates_and_no_fallback_on_failed_pilot(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    cfg, output, manifest = archive(tmp_path)
    monkeypatch.setattr(cpu, 'own_manifest', lambda *_: manifest)
    monkeypatch.setattr(cpu, 'test_records', lambda *_: pytest.fail('Unauthorized test read'))
    with pytest.raises(ValueError, match='sealed'):
        cpu.replay_test(cfg, output)
    cpu.fit_coefficients(cfg, output)
    for mode in ('native', 'gofa'):
        root = output/'evaluations'/label('pilot', 'pems03', mode)
        for i in range(48, 64):
            p = root/f'{i:06d}.json'
            row = read(p)
            row['target'], row['mae'] = [-2.], abs(row['prediction'][0]+2.)
            atomic_json(p, row)
    gate = cpu.confirm(cfg, output, 'pilot')
    assert not gate['all_six_strictly_improved']
    assert read(output/'status.json')['stop_reason'] == 'failed_pilot_gate'
    before = sha(output/'coefficients.json')
    with pytest.raises(ValueError, match='must pass'):
        cpu.replay_test(cfg, output)
    assert sha(output/'coefficients.json') == before


def test_wrong_seed_and_live_worker_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    cfg, output, manifest = archive(tmp_path)
    monkeypatch.setattr(cpu, 'own_manifest', lambda *_: manifest)
    atomic_json(output/'activity.json', {'worker': {'pid': 1}})
    with pytest.raises(ValueError, match='exited'):
        cpu.fit_coefficients(cfg, output)
    atomic_json(output/'activity.json', {'worker': None})
    p = output/'evaluations/pilot-pems03-native/trajectories/000000-00.json'
    value = read(p)
    value['seed'] += 1
    atomic_json(p, value)
    with pytest.raises(ValueError, match='seed differs'):
        cpu.fit_coefficients(cfg, output)


def test_cap_only_is_scoped_and_cpu_gate_never_queries_gpu(tmp_path, monkeypatch):
    import gofa_f.short_launcher as launcher
    cfg = yaml.safe_load(Path('configs/stage26_six_target_calibration.yaml').read_text())
    validate_config(cfg)
    assert type(launcher.admission_gate(cfg['resources'], 'six_target_calibration')).__name__ == 'CapOnlyGate'
    with pytest.raises(ValueError, match='Legacy'):
        launcher.admission_gate(cfg['resources'], 'target_calibration')
    config = tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump(cfg))
    atomic_json(tmp_path/'status.json', {'state': 'pilot_complete'})
    monkeypatch.setattr(launcher.signal, 'signal', lambda *_: None)
    monkeypatch.setattr(launcher, 'query_gpu', lambda: pytest.fail('CPU gate queried GPU'))
    def run(command, **kwargs):
        assert command[-1] == '--advance-cpu' and kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
        assert read(tmp_path/'activity.json')['worker'] is None
        atomic_json(tmp_path/'status.json', dict(state='complete', stop_reason='failed_pilot_gate'))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(launcher.subprocess, 'run', run)
    launcher.supervise(tmp_path, config, Path('.'), 'six_target_calibration')
    assert read(tmp_path/'launcher.json')['state'] == 'complete'


def test_complete_two_gate_flow_replays_test_without_refitting(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    cfg, output, manifest = archive(tmp_path)
    monkeypatch.setattr(cpu, 'own_manifest', lambda *_: manifest)
    cpu.fit_coefficients(cfg, output)
    cpu.confirm(cfg, output, 'pilot')
    with pytest.raises(ValueError, match='must pass'):
        cpu.replay_test(cfg, output)
    cpu.confirm(cfg, output, 'confirm')
    before = sha(output/'coefficients.json')
    calls = []
    def frozen_records(cfg, manifest, name, mode):
        calls.append((name, mode))
        return [entry(name=name, origin=900+i, variable=i, native=0. if mode == 'native' else 4.)
                for i in range(6)]
    monkeypatch.setattr(cpu, 'test_records', frozen_records)
    value = cpu.replay_test(cfg, output)
    assert len(calls) == 12 and value['all_six_strictly_improved']
    assert value['test_gpu_predictions_generated'] == 0 and not value['strict_zero_shot']
    assert sha(output/'coefficients.json') == before
    assert read(output/'status.json')['state'] == 'complete'
    assert cpu.replay_test(cfg, output) == value


def test_gpu_worker_is_no_grad_preserves_flags_and_exits_before_cpu_fit(tmp_path, monkeypatch):
    import torch
    from test_allnode import tiny
    import gofa_f.six_calibration_run as run
    model = tiny()
    torch.save(model.trainable_state(), tmp_path/'adapter.pt')
    cfg = {'adapter_sha256': sha(tmp_path/'adapter.pt')}
    manifest = dict(source_configuration={}, alphabet={'alphabet_sha256': 'alphabet'},
        weights_sha256='backbone', total_numeric_work=245760, queries={name:
        [entry(name=name, origin=500+i)['query'] for i in range(128)] for name in TARGETS},
        data_directory=str(tmp_path))
    atomic_json(tmp_path/'manifest.json', manifest)
    atomic_json(tmp_path/'cpu_preflight.json', dict(passed=True, cuda_disabled=True,
        manifest_sha256=sha(tmp_path/'manifest.json')))
    for key in ('GOFA_STAGE19_ADMITTED', 'GOFA_STAGE19_SUPERVISOR_PID', 'GOFA_STAGE19_SUPERVISOR_TICKS'):
        monkeypatch.setenv(key, '1')
    monkeypatch.setattr(run, 'verify_inference', lambda *_: manifest)
    monkeypatch.setattr(run.signal, 'signal', lambda *_: None)
    monkeypatch.setattr(run, 'supervisor_alive', lambda *_: True)
    monkeypatch.setattr(run, 'load_model', lambda *_: (model, SimpleNamespace(alphabet_sha='alphabet')))
    monkeypatch.setattr(run, 'Series', lambda *_: None)
    monkeypatch.setattr(torch.cuda, 'set_per_process_memory_fraction', lambda *_: None)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda *_: SimpleNamespace(total_memory=80*2**30))
    for key in ('max_memory_allocated', 'max_memory_reserved', 'reset_peak_memory_stats'):
        monkeypatch.setattr(torch.cuda, key, lambda: 0)
    events = []
    def source_cases(*args, **kwargs):
        events.append('four_source_cases')
        return dict(passed=True, gradients_created=False)
    monkeypatch.setattr(run, 'mechanisms', source_cases)
    monkeypatch.setattr(run, 'target_path_acceptance', lambda *_: events.append('new_path_acceptance'))
    def evaluate(*args):
        assert events[0] == 'four_source_cases' and not torch.is_grad_enabled()
        assert model.training and model.adapter.training and not model.native.training
        assert all(p.requires_grad for p in model.adapter.parameters())
        events.append(args[5])
        return dict(complete=True, queries=64, numeric_slots_in_eight_paths=512, active_seconds=1.)
    monkeypatch.setattr(run, 'evaluate', evaluate)
    run.predict(cfg, tmp_path)
    assert read(tmp_path/'status.json')['state'] == 'pilot_complete'
    assert len(events) == 25 and not (tmp_path/'coefficients.json').exists()
    assert not any(p.grad is not None for p in model.parameters())
