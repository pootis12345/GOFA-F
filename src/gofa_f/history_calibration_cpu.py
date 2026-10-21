"""CPU forward-time development gate, immutable fit, confirmation and test."""
from pathlib import Path
import time

from gofa_f.allnode_stage import read
from gofa_f.generative_data import Series, sha
from gofa_f.graph_benchmark_data import TARGETS
from gofa_f.history_calibration_math import TAUS, comparison, derived, fit, frame
from gofa_f.history_calibration_state import development_records, own_manifest, verify_data
from gofa_f.prefix_cache import atomic_json
from gofa_f.short_forecast import summarize
from gofa_f.six_calibration_cpu import no_worker, records, status, test_records
from gofa_f.six_calibration_state import immutable
from gofa_f.target_calibration import cpu_only
from gofa_f.target_calibration_math import improves


def frames_for(native, graph, data, native_only=False):
    return [frame(n, g, data, native_only) for n, g in zip(native, graph)]


def proof_required(output):
    proof = read(output/'cpu_preflight.json', {})
    if not proof.get('passed') or not proof.get('cuda_disabled') or proof['manifest_sha256'] != sha(output/'manifest.json'):
        raise ValueError('Complete bound CPU acceptance required before development')


def develop(cfg, output):
    cpu_only()
    no_worker(output)
    manifest = own_manifest(cfg, output)
    proof_required(output)
    verify_data(manifest)
    datasets, inputs = development_records(cfg)
    if inputs != manifest['development_inputs_sha256']:
        raise ValueError('Development inputs changed')
    if (output/'cpu_gate.json').exists():
        return load_development(cfg, output)[1]
    started = time.monotonic()
    progress, total_fits = [0], 72
    cost = read(output/'cpu_activity.json', dict(spent_seconds=0.))
    def cached_fit(name, mode, tau, part, n, g, fs):
        path = output/'fits'/f'{name}-{mode}-{tau:g}-{part}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        binding = f"{sha(output/'manifest.json')}:{name}:{mode}:{tau:g}:{part}"
        value = read(path)
        if value is None:
            value = dict(binding=binding, parameters=fit(n, g, fs, tau))
            immutable(path, value)
        if value['binding'] != binding:
            raise ValueError('Fit cache binding changed')
        progress[0] += 1
        seconds = time.monotonic()-started
        atomic_json(output/'progress.json', dict(state='cpu_development', dataset=name, model=mode,
            tau=tau, fold=part, fit_tasks_done=progress[0], fit_tasks_total=total_fits,
            overall_eta_seconds=seconds/progress[0]*max(0, total_fits-progress[0]),
            eta_basis='CPU fits only; confirmation GPU cost is conditional', updated_at=time.time()))
        print(f'Stage27 CPU {name}/{mode} tau={tau:g} fold={part} fits={progress[0]}/{total_fits}', flush=True)
        return value['parameters']
    try:
        results = {}
        for name, pair in datasets.items():
            n, g = pair['native'], pair['gofa']
            data = Series(Path(manifest['data_directory'])/'targets'/(name+'.npz'))
            result = {}
            for mode in ('full', 'native_history'):
                fs = frames_for(n, g, data, mode == 'native_history')
                candidates = []
                for tau in TAUS:
                    oos, folds = [], []
                    for a, b in cfg['folds']:
                        parameters = cached_fit(name, mode, tau, f'fold{a}-{b}', n[:a], g[:a], fs[:a])
                        rows, metric = derived(n[a:b], g[a:b], fs[a:b], parameters)
                        oos.extend(rows)
                        folds.append(dict(train_queries=a, validation_begin=a, validation_end=b,
                            parameters=parameters, metrics=metric))
                    metric = summarize(oos)
                    metric.pop('formatting')
                    mae = metric['macro']['mae']
                    candidates.append(dict(tau=tau, oos_mae=mae, metrics=metric, folds=folds,
                        predictions=[dict(query=r['query'], prediction=r['prediction']) for r in oos]))
                selected = min(candidates, key=lambda x: (x['oos_mae'], -x['tau']))
                result[mode] = dict(selected_tau=selected['tau'], oos_mae=selected['oos_mae'],
                    selected_metrics=selected['metrics'], candidates=candidates)
            baseline = summarize(n[32:64])
            result['native_metrics'], result['gofa_metrics'] = baseline, summarize(g[32:64])
            result['strictly_improved'] = improves(result['full']['oos_mae'], baseline['macro']['mae'])
            results[name] = result
        value = dict(binding=sha(output/'manifest.json'), development_inputs_sha256=inputs,
            datasets=results, all_six_strictly_improved=all(x['strictly_improved'] for x in results.values()),
            independent_confirmation=False, old_pilot_reclassified_as_development=True,
            confirmation_records_read=False, test_records_read=False, optimizer_updates=0)
        immutable(output/'development.json', value)
        immutable(output/'cpu_gate.json', dict(binding=value['binding'],
            development_sha256=sha(output/'development.json'), all_six_strictly_improved=value['all_six_strictly_improved'],
            development_inputs_sha256=inputs, confirmation_and_test_read=False))
        if not value['all_six_strictly_improved']:
            status(output, 'complete', 'cpu_development', stop_reason='failed_cpu_development_gate', goal_achieved=False)
            atomic_json(output/'activity.json', dict(spent_seconds=0., worker=None, waiting_seconds=0.,
                resource_pause_count=0, peak_group_mib=0))
            atomic_json(output/'launcher.json', dict(state='complete', gpu_worker_started=False,
                reason='failed_cpu_development_gate'))
            terminal_progress(output, 'failed_cpu_development_gate')
        else:
            status(output, 'cpu_passed', 'cpu_development', goal_achieved=None)
        return value
    finally:
        cost['spent_seconds'] += time.monotonic()-started
        atomic_json(output/'cpu_activity.json', cost)
        report(output)


def load_development(cfg, output):
    manifest = own_manifest(cfg, output)
    gate, value = read(output/'cpu_gate.json'), read(output/'development.json')
    if not gate or not value or gate['binding'] != sha(output/'manifest.json') \
            or value['binding'] != gate['binding'] or sha(output/'development.json') != gate['development_sha256'] \
            or value['all_six_strictly_improved'] != gate['all_six_strictly_improved'] \
            or value['confirmation_records_read'] or value['test_records_read']:
        raise ValueError('Development gate missing/rebound')
    _, inputs = development_records(cfg)
    if inputs != manifest['development_inputs_sha256'] or inputs != gate['development_inputs_sha256']:
        raise ValueError('Development data changed after gate')
    return manifest, value, gate


def seal(cfg, output):
    cpu_only()
    no_worker(output)
    if (output/'coefficient_lock.json').exists():
        return load_coefficients(cfg, output)[1]
    manifest, development, gate = load_development(cfg, output)
    if not gate['all_six_strictly_improved']:
        raise ValueError('All six CPU checks must pass before final fitting')
    datasets, inputs = development_records(cfg)
    parameters = {}
    for name, pair in datasets.items():
        data = Series(Path(manifest['data_directory'])/'targets'/(name+'.npz'))
        parameters[name] = {}
        for mode in ('full', 'native_history'):
            tau = development['datasets'][name][mode]['selected_tau']
            path = output/'fits'/f'{name}-{mode}-{tau:g}-all64.json'
            binding = f"{sha(output/'manifest.json')}:{name}:{mode}:{tau:g}:all64"
            cached = read(path)
            if cached is None:
                fs = frames_for(pair['native'], pair['gofa'], data, mode == 'native_history')
                cached = dict(binding=binding, parameters=fit(pair['native'], pair['gofa'], fs, tau))
                immutable(path, cached)
            if cached['binding'] != binding:
                raise ValueError('Final fit binding changed')
            parameters[name][mode] = cached['parameters']
    value = dict(binding=sha(output/'manifest.json'), development_inputs_sha256=inputs,
        cpu_gate_sha256=sha(output/'cpu_gate.json'), coefficients=parameters,
        independent_confirmation_or_test_read=False, optimizer_updates=0, strict_zero_shot=False,
        stored_deployment_parameters=sum(v['full']['stored_parameters'] for v in parameters.values()))
    immutable(output/'coefficients.json', value)
    immutable(output/'coefficient_lock.json', dict(binding=value['binding'],
        coefficients_sha256=sha(output/'coefficients.json'), cpu_gate_sha256=value['cpu_gate_sha256']))
    status(output, 'coefficients_sealed', 'cpu_fit', coefficient_sha256=sha(output/'coefficients.json'))
    return value


def load_coefficients(cfg, output):
    manifest, development, gate = load_development(cfg, output)
    value, lock = read(output/'coefficients.json'), read(output/'coefficient_lock.json')
    if not gate['all_six_strictly_improved'] or value is None or lock is None \
            or value['binding'] != sha(output/'manifest.json') or lock['binding'] != value['binding'] \
            or lock['coefficients_sha256'] != sha(output/'coefficients.json') \
            or lock['cpu_gate_sha256'] != sha(output/'cpu_gate.json') \
            or value['independent_confirmation_or_test_read'] or set(value['coefficients']) != set(TARGETS):
        raise ValueError('Sealed parameters and all six CPU improvements required')
    return manifest, value, lock


def confirmation(cfg, output):
    cpu_only()
    no_worker(output)
    manifest, value, lock = load_coefficients(cfg, output)
    pairs, inputs = records(output, manifest, 'confirm')
    results = {}
    for name, pair in pairs.items():
        data = Series(Path(manifest['data_directory'])/'targets'/(name+'.npz'))
        n, g = pair['native'], pair['gofa']
        fs, control = frames_for(n, g, data), frames_for(n, g, data, True)
        p = value['coefficients'][name]
        results[name] = comparison(n, g, fs, control, p['full'], p['native_history'])
    passed = all(x['strictly_improved'] for x in results.values())
    gate = dict(binding=value['binding'], coefficient_sha256=lock['coefficients_sha256'],
        inputs_sha256=inputs, datasets=results, all_six_strictly_improved=passed, refitted=False)
    immutable(output/'confirm_gate.json', gate)
    if not passed:
        status(output, 'complete', 'independent_confirmation', stop_reason='failed_independent_confirmation', goal_achieved=False)
        terminal_progress(output, 'failed_independent_confirmation')
        report(output)
    else:
        status(output, 'confirmed', 'independent_confirmation')
    return gate


def require_confirmation(cfg, output):
    manifest, value, lock = load_coefficients(cfg, output)
    gate = read(output/'confirm_gate.json')
    if not gate or not gate['all_six_strictly_improved'] or gate['binding'] != value['binding'] \
            or gate['coefficient_sha256'] != lock['coefficients_sha256'] \
            or set(gate['datasets']) != set(TARGETS) or not all(x['strictly_improved'] for x in gate['datasets'].values()):
        raise ValueError('All six independent confirmations required before test access')
    _, inputs = records(output, manifest, 'confirm')
    if inputs != gate['inputs_sha256']:
        raise ValueError('Confirmation changed after gate')
    return manifest, value, lock


def replay_test(cfg, output):
    cpu_only()
    no_worker(output)
    manifest, parameters, lock = require_confirmation(cfg, output)
    verify_data(manifest)
    immutable(output/'test_admission.json', dict(binding=parameters['binding'],
        coefficients_sha256=lock['coefficients_sha256'], confirm_gate_sha256=sha(output/'confirm_gate.json'),
        test_gpu_predictions_allowed=False))
    results = {}
    for name in TARGETS:
        n, g = (test_records(cfg, manifest, name, mode) for mode in ('native', 'gofa'))
        data = Series(Path(manifest['data_directory'])/'targets'/(name+'.npz'))
        fs, control = frames_for(n, g, data), frames_for(n, g, data, True)
        p = parameters['coefficients'][name]
        value = comparison(n, g, fs, control, p['full'], p['native_history'], bootstrap=True)
        immutable(output/f'test-{name}.json', value)
        results[name] = value
    value = dict(binding=parameters['binding'], coefficient_sha256=lock['coefficients_sha256'],
        datasets=results, all_six_strictly_improved=all(x['strictly_improved'] for x in results.values()),
        complete=True, test_gpu_predictions_generated=0, strict_zero_shot=False)
    immutable(output/'test_replay.json', value)
    status(output, 'complete', 'test_replay', stop_reason='completed_test_replay', goal_achieved=value['all_six_strictly_improved'])
    terminal_progress(output, 'completed_test_replay')
    report(output)
    return value


def terminal_progress(output, reason):
    p = read(output/'progress.json', {})
    p.update(state='complete', stop_reason=reason, overall_eta_seconds=None,
        evaluation_eta_seconds=None, remaining_work='not scheduled; skipped work is not counted complete',
        confirmation='complete' if (output/'confirm_gate.json').exists() else 'skipped',
        test_replay='complete' if (output/'test_replay.json').exists() else 'skipped')
    atomic_json(output/'progress.json', p)


def advance(cfg, output):
    cpu_only()
    state = read(output/'status.json', {}).get('state')
    if state == 'confirm_complete':
        if not confirmation(cfg, output)['all_six_strictly_improved']:
            return report(output)
    elif state != 'confirmed':
        raise ValueError('CPU finalization requires completed confirmation predictions')
    return replay_test(cfg, output)


def report(output):
    value = dict(status=read(output/'status.json'), development=read(output/'development.json'),
        coefficients=read(output/'coefficients.json'), independent_confirmation=read(output/'confirm_gate.json'),
        test=read(output/'test_replay.json'), activity=read(output/'activity.json'),
        cpu_activity=read(output/'cpu_activity.json'), memory=read(output/'cuda_memory.json'),
        strict_zero_shot=False, optimizer_updates=0,
        limitations=['Target-validation-calibrated development reevaluation; prior test results already seen',
            'Former Stage26 pilot is now development data, not independent confirmation',
            'Historical experts may explain improvement; native-history control is reported separately',
            'Graph weight zero or failure to beat native-history does not demonstrate extra graph gain',
            'Fixed small samples; Kelmarsh only six variable clusters; no population advantage guarantee',
            'Derived predictions have no language-generation coverage claim'])
    atomic_json(output/'report.json', value)
    return value
