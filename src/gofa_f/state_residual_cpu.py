"""CPU-only Stage28 forward folds, frozen five, seal and one-shot test gate."""
from pathlib import Path
import time

import numpy as np

from gofa_f.allnode_stage import read
from gofa_f.generative_data import Series, sha
from gofa_f.graph_benchmark_data import TARGETS
from gofa_f.history_calibration_cpu import frames_for
from gofa_f.history_calibration_math import comparison as history_comparison
from gofa_f.prefix_cache import atomic_json
from gofa_f.short_forecast import summarize
from gofa_f.six_calibration_cpu import no_worker, status, test_records
from gofa_f.six_calibration_state import immutable
from gofa_f.state_residual_math import TAUS, compare, derived, fit, frame
from gofa_f.state_residual_state import (development_pair, import_confirmation, own_manifest,
    prediction_records, previous)
from gofa_f.target_calibration import cpu_only
from gofa_f.target_calibration_state import digest


def proof_required(output):
    p = read(output/'cpu_preflight.json', {})
    if not p.get('passed') or not p.get('cuda_disabled') or p['manifest_sha256'] != sha(output/'manifest.json'):
        raise ValueError('Bound complete CPU proof required')


def terminal(output, reason, phase, goal=False):
    status(output, 'complete', phase, stop_reason=reason, goal_achieved=goal)
    p = read(output/'progress.json', {})
    p.update(state='complete', overall_eta_seconds=None, evaluation_eta_seconds=None,
        confirmation='complete' if (output/'confirm_gate.json').exists() else 'skipped',
        test_replay='complete' if (output/'test_replay.json').exists() else 'skipped', stop_reason=reason)
    atomic_json(output/'progress.json', p)
    report(output)


def develop(cfg, output):
    cpu_only()
    no_worker(output)
    m = own_manifest(cfg, output)
    proof_required(output)
    pair, inputs = development_pair(cfg, output, m)
    if (output/'cpu_gate.json').exists():
        return load_development(cfg, output)[1]
    start = time.monotonic()
    n, g = pair['native'], pair['gofa']
    data = Series(Path(m['data_directory'])/'targets/sdwpf.npz')
    results, done = {}, 0
    native = summarize(n[64:128])
    for mode in ('full', 'native_history'):
        fs = [frame(a, b, data, mode == 'native_history') for a, b in zip(n, g)]
        candidates = []
        for tau in TAUS:
            out, folds = [], []
            for a, b in cfg['folds']:
                path = output/'fits'/f'sdwpf-{mode}-{tau:g}-fold{a}-{b}.json'
                path.parent.mkdir(parents=True, exist_ok=True)
                binding = f"{sha(output/'manifest.json')}:{inputs}:{mode}:{tau:g}:{a}:{b}"
                cached = read(path)
                if cached is None:
                    cached = dict(binding=binding, parameters=fit(n[:a], g[:a], fs[:a], tau))
                    immutable(path, cached)
                if cached['binding'] != binding:
                    raise ValueError('Cached residual fit rebound')
                rows, metric = derived(n[a:b], g[a:b], fs[a:b], cached['parameters'])
                out.extend(rows)
                baseline = summarize(n[a:b])['macro']['mae']
                folds.append(dict(train_queries=a, validation_begin=a, validation_end=b,
                    parameters=cached['parameters'], metrics=metric, native_mae=baseline,
                    nondegraded=metric['macro']['mae'] <= baseline))
                done += 1
                atomic_json(output/'progress.json', dict(state='cpu_development', dataset='sdwpf', model=mode,
                    fit_tasks_done=done, fit_tasks_total=14, overall_eta_seconds=(time.monotonic()-start)/done*(14-done),
                    updated_at=time.time()))
                print(f'Stage28 CPU {mode} tau={tau:g} fold={a}:{b} fits={done}/14', flush=True)
            metric = summarize(out)
            metric.pop('formatting')
            candidates.append(dict(tau=tau, oos_mae=metric['macro']['mae'], metrics=metric, folds=folds,
                predictions=[dict(query=r['query'], prediction=r['prediction']) for r in out]))
        selected = min(candidates, key=lambda x: (x['oos_mae'], -x['tau']))
        results[mode] = dict(selected_tau=selected['tau'], oos_mae=selected['oos_mae'],
            selected_folds=selected['folds'], metrics=selected['metrics'], candidates=candidates)
    fold_pass = all(f['nondegraded'] for f in results['full']['selected_folds'])
    improvement_pass = results['full']['oos_mae'] <= (1.-cfg['development_minimum_improvement'])*native['macro']['mae']
    value = dict(binding=sha(output/'manifest.json'), inputs_sha256=inputs, results=results,
        native_metrics=native, both_folds_nondegraded=fold_pass, minimum_improvement_passed=improvement_pass,
        passed=fold_pass and improvement_pass, confirmation_records_read=False, test_records_read=False,
        frozen_five_sha256=m['frozen_coefficients_sha256'])
    immutable(output/'development.json', value)
    immutable(output/'cpu_gate.json', dict(binding=value['binding'], inputs_sha256=inputs,
        development_sha256=sha(output/'development.json'), passed=value['passed']))
    cost = read(output/'cpu_activity.json', dict(spent_seconds=0.))
    cost['spent_seconds'] += time.monotonic()-start
    atomic_json(output/'cpu_activity.json', cost)
    if not value['passed']:
        terminal(output, 'failed_sdwpf_development_gate', 'cpu_development')
    else:
        status(output, 'cpu_passed', 'cpu_development')
    return value


def load_development(cfg, output, inference=False):
    m = own_manifest(cfg, output)
    gate, value = read(output/'cpu_gate.json'), read(output/'development.json')
    if not gate or not value or gate['binding'] != sha(output/'manifest.json') \
            or value['binding'] != gate['binding'] or gate['development_sha256'] != sha(output/'development.json') \
            or value['passed'] != gate['passed'] or value['confirmation_records_read'] or value['test_records_read']:
        raise ValueError('Residual development gate missing/rebound')
    _, inputs = development_pair(cfg, output, m, inference=inference)
    if inputs != value['inputs_sha256'] or inputs != gate['inputs_sha256']:
        raise ValueError('Residual development inputs changed')
    return m, value


def seal(cfg, output):
    cpu_only()
    no_worker(output)
    if (output/'coefficient_lock.json').exists():
        return load_coefficients(cfg, output)[1]
    m, development = load_development(cfg, output)
    if not development['passed']:
        raise ValueError('SDWPF development gate must pass before all128 fit')
    pair, inputs = development_pair(cfg, output, m)
    n, g = pair['native'], pair['gofa']
    data = Series(Path(m['data_directory'])/'targets/sdwpf.npz')
    parameters = dict(m['frozen_coefficients'])
    parameters['sdwpf'] = {}
    for mode in ('full', 'native_history'):
        tau = development['results'][mode]['selected_tau']
        path = output/'fits'/f'sdwpf-{mode}-{tau:g}-all128.json'
        binding = f"{sha(output/'manifest.json')}:{inputs}:{mode}:{tau:g}:all128"
        cached = read(path)
        if cached is None:
            fs = [frame(a, b, data, mode == 'native_history') for a, b in zip(n, g)]
            cached = dict(binding=binding, parameters=fit(n, g, fs, tau))
            immutable(path, cached)
        if cached['binding'] != binding:
            raise ValueError('Final residual fit rebound')
        parameters['sdwpf'][mode] = cached['parameters']
    value = dict(binding=sha(output/'manifest.json'), inputs_sha256=inputs,
        cpu_gate_sha256=sha(output/'cpu_gate.json'), coefficients=parameters,
        frozen_five_sha256=m['frozen_coefficients_sha256'], confirmation_or_test_read=False,
        stored_deployment_parameters=sum(v['full']['stored_parameters'] for v in parameters.values()),
        optimizer_updates=0, strict_zero_shot=False)
    immutable(output/'coefficients.json', value)
    immutable(output/'coefficient_lock.json', dict(binding=value['binding'],
        coefficients_sha256=sha(output/'coefficients.json'), cpu_gate_sha256=value['cpu_gate_sha256']))
    identity = not np.asarray(parameters['sdwpf']['full']['beta']).any()
    immutable(output/'postfit_gate.json', dict(binding=value['binding'],
        coefficients_sha256=sha(output/'coefficients.json'), sdwpf_native_identity=identity, passed=not identity))
    if identity:
        terminal(output, 'final_refit_native_identity_precludes_strict_improvement', 'postfit_feasibility')
    else:
        status(output, 'coefficients_sealed', 'cpu_fit')
    return value


def load_coefficients(cfg, output, inference=False):
    m, development = load_development(cfg, output, inference=inference)
    p, lock, postfit = (read(output/n) for n in ('coefficients.json', 'coefficient_lock.json', 'postfit_gate.json'))
    if not development['passed'] or not p or not lock or not postfit or not postfit['passed'] \
            or p['binding'] != sha(output/'manifest.json') or lock['binding'] != p['binding'] \
            or lock['coefficients_sha256'] != sha(output/'coefficients.json') \
            or postfit['coefficients_sha256'] != lock['coefficients_sha256'] \
            or lock['cpu_gate_sha256'] != sha(output/'cpu_gate.json') or p['confirmation_or_test_read'] \
            or set(p['coefficients']) != set(TARGETS) or not np.asarray(p['coefficients']['sdwpf']['full']['beta']).any():
        raise ValueError('Sealed nonidentity residual coefficients and development gate required')
    frozen = {n: p['coefficients'][n] for n in TARGETS if n != 'sdwpf'}
    if digest(frozen) != m['frozen_coefficients_sha256'] or frozen != m['frozen_coefficients']:
        raise ValueError('Frozen five calibrators changed')
    previous(cfg)
    return m, p, lock


def comparison(name, pair, data, parameters, bootstrap=False):
    n, g = pair['native'], pair['gofa']
    if name != 'sdwpf':
        return history_comparison(n, g, frames_for(n, g, data), frames_for(n, g, data, True),
            parameters['full'], parameters['native_history'], bootstrap=bootstrap)
    fs = [frame(a, b, data) for a, b in zip(n, g)]
    cs = [frame(a, b, data, True) for a, b in zip(n, g)]
    new, metric = derived(n, g, fs, parameters['full'])
    control, other = derived(n, g, cs, parameters['native_history'])
    graph_used = parameters['full']['beta'][2] != 0.
    return compare(n, g, new, metric, control, other, graph_used, bootstrap=bootstrap)


def confirmation(cfg, output):
    cpu_only()
    no_worker(output)
    m, p, lock = load_coefficients(cfg, output)
    pairs, inputs = prediction_records(output, m, 'confirm')
    results = {}
    for name in TARGETS:
        data = Series(Path(m['data_directory'])/'targets'/(name+'.npz'))
        results[name] = comparison(name, pairs[name], data, p['coefficients'][name])
    gate = dict(binding=p['binding'], coefficients_sha256=lock['coefficients_sha256'], inputs_sha256=inputs,
        datasets=results, all_six_strictly_improved=all(x['strictly_improved'] for x in results.values()), refitted=False)
    immutable(output/'confirm_gate.json', gate)
    if not gate['all_six_strictly_improved']:
        terminal(output, 'failed_independent_confirmation', 'independent_confirmation')
    else:
        status(output, 'confirmed', 'independent_confirmation')
    return gate


def replay_test(cfg, output):
    cpu_only()
    no_worker(output)
    m, p, lock = load_coefficients(cfg, output)
    gate = read(output/'confirm_gate.json')
    if not gate or not gate['all_six_strictly_improved'] or gate['binding'] != p['binding'] \
            or gate['coefficients_sha256'] != lock['coefficients_sha256'] \
            or set(gate['datasets']) != set(TARGETS) or not all(x['strictly_improved'] for x in gate['datasets'].values()):
        raise ValueError('All six independent confirmations required before test access')
    _, inputs = prediction_records(output, m, 'confirm')
    if inputs != gate['inputs_sha256']:
        raise ValueError('Confirmation inputs changed')
    immutable(output/'test_admission.json', dict(binding=p['binding'],
        coefficients_sha256=lock['coefficients_sha256'], confirm_gate_sha256=sha(output/'confirm_gate.json'),
        test_gpu_predictions_allowed=False))
    results = {}
    for name in TARGETS:
        pair = {mode: test_records(cfg, m, name, mode) for mode in ('native', 'gofa')}
        data = Series(Path(m['data_directory'])/'targets'/(name+'.npz'))
        results[name] = comparison(name, pair, data, p['coefficients'][name], bootstrap=True)
        immutable(output/f'test-{name}.json', results[name])
    value = dict(binding=p['binding'], coefficients_sha256=lock['coefficients_sha256'], datasets=results,
        all_six_strictly_improved=all(x['strictly_improved'] for x in results.values()),
        complete=True, test_gpu_predictions_generated=0)
    immutable(output/'test_replay.json', value)
    terminal(output, 'completed_frozen_test_replay', 'test_replay', value['all_six_strictly_improved'])
    return value


def advance(cfg, output):
    cpu_only()
    no_worker(output)
    current = read(output/'status.json', {}).get('state')
    if current == 'development_complete':
        if not develop(cfg, output)['passed']:
            return report(output)
        seal(cfg, output)
        if read(output/'status.json')['state'] != 'complete':
            import_confirmation(cfg, output, own_manifest(cfg, output))
        return report(output)
    if current == 'confirm_complete':
        if not confirmation(cfg, output)['all_six_strictly_improved']:
            return report(output)
    elif current != 'confirmed':
        raise ValueError('CPU advance requires an exited completed prediction phase')
    return replay_test(cfg, output)


def report(output):
    value = {k: read(output/file) for k, file in dict(status='status.json', development='development.json',
        coefficients='coefficients.json', independent_confirmation='confirm_gate.json', test='test_replay.json',
        activity='activity.json', cpu_activity='cpu_activity.json', memory='cuda_memory.json',
        postfit_feasibility='postfit_gate.json', imported_confirmation='confirmation_imports.json').items()}
    value.update(strict_zero_shot=False, optimizer_updates=0,
        limitations=['Target-validation-calibrated development reevaluation; previously viewed test results',
            'Stage27 partial PeMS03 native confirmation provenance retained; never used for fitting',
            'Only SDWPF calibrator changed; other five refits still need independent confirmation',
            'Native plus same history and neighbor control required for additional graph-LLM gain claims',
            'Fixed small sample; Kelmarsh only six clusters; no population-wide advantage guarantee',
            'Derived predictions are not language generation; official pretraining exposure cannot be verified'])
    atomic_json(output/'report.json', value)
    return value
