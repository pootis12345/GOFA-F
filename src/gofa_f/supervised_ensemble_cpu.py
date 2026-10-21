"""Stage29 one-way CPU development, fixed-weight seals and dataset confirmation gates."""
from pathlib import Path
import time

from gofa_f.allnode_stage import read
from gofa_f.generative_data import sha
from gofa_f.graph_benchmark_data import TARGETS
from gofa_f.prefix_cache import atomic_json
from gofa_f.six_calibration_cpu import no_worker, read_shard, test_records
from gofa_f.six_calibration_state import immutable
from gofa_f.supervised_ensemble_math import comparison, derived, fit, prediction
from gofa_f.supervised_ensemble_models import CPUCharge, cpu_predictions, proof_required, require_models
from gofa_f.supervised_ensemble_state import inference_binding, original_development, own_manifest, status
from gofa_f.target_calibration import cpu_only
from gofa_f.target_calibration_math import improves
from gofa_f.target_calibration_state import digest


def terminal(output, reason, phase, goal=False):
    status(output, 'complete', phase, stop_reason=reason, goal_achieved=goal)
    p = read(output/'progress.json', {})
    p.update(state='complete', overall_eta_seconds=None, evaluation_eta_seconds=None,
        confirmation='complete' if (output/'confirm_gate.json').exists() else 'partial_stopped'
            if list((output/'confirmation').glob('*.json')) else 'skipped',
        test_replay='complete' if (output/'test_replay.json').exists() else 'skipped', stop_reason=reason)
    atomic_json(output/'progress.json', p)
    report(output)


def development_inputs(cfg, output, manifest):
    pairs, inputs = original_development(cfg)
    if inputs != manifest['development_inputs_sha256']:
        raise ValueError('Inherited development evidence changed')
    cpus, hashes = {}, {}
    for name in TARGETS:
        cpus[name] = cpu_predictions(output, manifest, name, manifest['development_queries'][name], 'development')
        hashes[name] = sha(output/'cpu_predictions'/f'development-{name}.json')
    return pairs, cpus, digest(dict(llm=inputs, cpu=hashes))


def develop(cfg, output):
    cpu_only()
    no_worker(output)
    m = own_manifest(cfg, output)
    proof_required(output)
    require_models(output, m)
    if (output/'cpu_gate.json').exists():
        return load_development(cfg, output)[1]
    if (output/'coefficient_lock.json').exists():
        raise ValueError('Cannot return to development after sealing')
    cost = CPUCharge(output)
    try:
        pairs, cpus, inputs = development_inputs(cfg, output, m)
        results = {}
        for index, name in enumerate(TARGETS):
            n, g, c = pairs[name]['native'], pairs[name]['gofa'], cpus[name]
            folds = ((64, 96), (96, 128)) if name == 'sdwpf' else ((32, 48), (48, 64))
            modes = {}
            for mode in ('full', 'native_cpu'):
                rows, evidence = [], []
                for a, b in folds:
                    path = output/'fits'/f'{name}-{mode}-{a}-{b}.json'
                    path.parent.mkdir(exist_ok=True)
                    binding = digest(dict(manifest=sha(output/'manifest.json'), inputs=inputs,
                        source=name, mode=mode, prefix=a, end=b))
                    params = read(path)
                    if params is None:
                        params = dict(binding=binding, parameters=fit(n[:a], g[:a], c[:a], mode == 'native_cpu'))
                        immutable(path, params)
                    if params['binding'] != binding:
                        raise ValueError('Development fusion fit changed')
                    own, metric = derived(n[a:b], g[a:b], c[a:b], params['parameters']['weights'])
                    rows.extend(own)
                    from gofa_f.short_forecast import summarize
                    baseline = summarize(n[a:b])['macro']['mae']
                    evidence.append(dict(train_queries=a, validation_begin=a, validation_end=b,
                        parameters=params['parameters'], metrics=metric, native_mae=baseline,
                        nondegraded=metric['macro']['mae'] <= baseline))
                from gofa_f.short_forecast import summarize
                metric = summarize(rows)
                metric.pop('formatting')
                modes[mode] = dict(metrics=metric, oos_mae=metric['macro']['mae'], folds=evidence)
            start = folds[0][0]
            _, cpu_metric = derived(n[start:], g[start:], c[start:], [0., 0., 1.])
            # Only pooled fold-specific predictions define the deployment gate.
            baseline = summarize(n[start:])['macro']['mae']
            score = modes['full']['oos_mae']
            modes.update(native_mae=baseline, cpu_oos_metrics=cpu_metric,
                both_folds_nondegraded=all(f['nondegraded'] for f in modes['full']['folds']),
                minimum_improvement_passed=score <= .995*baseline and improves(score, baseline))
            modes['passed'] = modes['both_folds_nondegraded'] and modes['minimum_improvement_passed']
            results[name] = modes
            cost.tick(True)
            atomic_json(output/'progress.json', dict(state='cpu_development', dataset=name,
                development_datasets_done=index+1, development_datasets_total=6, overall_eta_seconds=None,
                conditional_confirmation_estimate_seconds=m['initial_confirmation_estimate_seconds'], updated_at=time.time()))
            print(f'Stage29 CPU {name} Native={baseline:.8g} OOS={score:.8g} passed={modes["passed"]}', flush=True)
        value = dict(binding=sha(output/'manifest.json'), inputs_sha256=inputs, datasets=results,
            all_six_passed=all(x['passed'] for x in results.values()), confirmation_or_test_read=False,
            models_complete_sha256=sha(output/'models_complete.json'), target_supervised=True)
        immutable(output/'development.json', value)
        immutable(output/'cpu_gate.json', dict(binding=value['binding'], inputs_sha256=inputs,
            development_sha256=sha(output/'development.json'), passed=value['all_six_passed']))
        if not value['all_six_passed']:
            terminal(output, 'failed_cpu_development_gate', 'cpu_development')
        else:
            status(output, 'cpu_passed', 'cpu_development')
        return value
    finally:
        cost.close()
        report(output)


def load_development(cfg, output):
    m = own_manifest(cfg, output)
    require_models(output, m)
    gate, value = read(output/'cpu_gate.json'), read(output/'development.json')
    if not gate or not value or value['binding'] != sha(output/'manifest.json') \
            or gate['binding'] != value['binding'] or gate['development_sha256'] != sha(output/'development.json') \
            or gate['passed'] != value['all_six_passed'] or value['confirmation_or_test_read'] \
            or value['models_complete_sha256'] != sha(output/'models_complete.json'):
        raise ValueError('Complete immutable CPU development gate required')
    _, _, inputs = development_inputs(cfg, output, m)
    if inputs != value['inputs_sha256'] or inputs != gate['inputs_sha256']:
        raise ValueError('Development inputs changed after gate')
    return m, value


def seal(cfg, output):
    cpu_only()
    no_worker(output)
    if (output/'coefficient_lock.json').exists():
        return load_coefficients(cfg, output)[1]
    m, development = load_development(cfg, output)
    if not development['all_six_passed']:
        raise ValueError('All six development gates required before final fitting')
    pairs, cpus, inputs = development_inputs(cfg, output, m)
    params = {}
    for name in TARGETS:
        pair, own = pairs[name], cpus[name]
        params[name] = {}
        for mode in ('full', 'native_cpu'):
            path = output/'fits'/f'{name}-{mode}-all.json'
            binding = digest(dict(manifest=sha(output/'manifest.json'), inputs=inputs, dataset=name, mode=mode, all=True))
            cached = read(path)
            if cached is None:
                cached = dict(binding=binding, parameters=fit(pair['native'], pair['gofa'], own, mode == 'native_cpu'))
                immutable(path, cached)
            if cached['binding'] != binding:
                raise ValueError('Final ensemble fit binding changed')
            params[name][mode] = cached['parameters']
    models = require_models(output, m)
    work = 8*sum(sum(q['horizon'] for q in m['queries'][n][64:])*(1+int(p['full']['weights'][1] != 0.))
        for n, p in params.items())
    estimate = 8*sum(sum(q['horizon'] for q in m['queries'][n][64:])*(
        m['prior_inference_seconds_per_numeric_slot']['native']
        +int(p['full']['weights'][1] != 0.)*m['prior_inference_seconds_per_numeric_slot']['gofa']) for n, p in params.items())
    value = dict(binding=sha(output/'manifest.json'), coefficients=params, inputs_sha256=inputs,
        models_complete_sha256=sha(output/'models_complete.json'), cpu_models_sha256=models['models'],
        cpu_gate_sha256=sha(output/'cpu_gate.json'), stored_deployment_weights=18,
        target_supervised=True, confirmation_or_test_read=False, required_numeric_work=work,
        estimated_confirmation_seconds=estimate, skip_zero_graph_weight=True)
    immutable(output/'coefficients.json', value)
    immutable(output/'coefficient_lock.json', dict(binding=value['binding'],
        coefficients_sha256=sha(output/'coefficients.json'), cpu_gate_sha256=sha(output/'cpu_gate.json')))
    identities = [name for name, p in params.items() if p['full']['weights'] == [1., 0., 0.]]
    immutable(output/'postfit_gate.json', dict(binding=value['binding'], coefficients_sha256=sha(output/'coefficients.json'),
        identity_datasets=identities, passed=not identities))
    if identities:
        terminal(output, 'native_identity_after_refit', 'postfit_feasibility')
    else:
        status(output, 'coefficients_sealed', 'cpu_fit')
    return value


def load_coefficients(cfg, output):
    m, dev = load_development(cfg, output)
    p, lock, post = (read(output/n) for n in ('coefficients.json', 'coefficient_lock.json', 'postfit_gate.json'))
    if not dev['all_six_passed'] or not p or not lock or not post or not post['passed'] \
            or p['binding'] != sha(output/'manifest.json') or lock['binding'] != p['binding'] \
            or lock['coefficients_sha256'] != sha(output/'coefficients.json') \
            or post['coefficients_sha256'] != lock['coefficients_sha256'] \
            or lock['cpu_gate_sha256'] != sha(output/'cpu_gate.json') \
            or p['models_complete_sha256'] != sha(output/'models_complete.json') \
            or p['cpu_models_sha256'] != require_models(output, m)['models'] \
            or p['confirmation_or_test_read'] or set(p['coefficients']) != set(TARGETS):
        raise ValueError('Sealed nonidentity CPU models/weights and six CPU gates required')
    for params in p['coefficients'].values():
        for mode in ('full', 'native_cpu'):
            prediction([0.], [1.], [2.], params[mode]['weights'])
        if params['full']['weights'] == [1., 0., 0.] or params['native_cpu']['weights'][1] != 0:
            raise ValueError('Identity deployment or graph-enabled control rejected')
    return m, p, lock


def confirm_dataset(cfg, output, name, inference=False):
    if not inference:
        cpu_only()
        no_worker(output)
    m, p, lock = load_coefficients(cfg, output)
    before = list(TARGETS)[:list(TARGETS).index(name)]
    for earlier in before:
        accepted = read(output/'confirmation'/(earlier+'.json'))
        if not accepted or not accepted['comparison']['strictly_improved']:
            raise ValueError('Earlier fixed confirmation dataset not passed')
    pair, hashes = {}, {}
    requested = ('native', 'gofa') if p['coefficients'][name]['full']['weights'][1] != 0. else ('native',)
    for mode in requested:
        label = f'confirm-{name}-{mode}'
        marker = read(output/'evaluations'/label/'complete.json')
        binding = inference_binding(m, sha(output/'manifest.json'), name, mode)
        if not marker or not marker['complete'] or marker['binding'] != binding:
            raise ValueError('Required confirmation predictions incomplete')
        pair[mode] = [read_shard(output/'evaluations'/label, i, q, binding, hashes)
            for i, q in enumerate(m['queries'][name][64:])]
    cpu = cpu_predictions(output, m, name, m['queries'][name][64:], 'confirm')
    result = comparison(pair['native'], pair.get('gofa'), cpu, p['coefficients'][name])
    value = dict(binding=p['binding'], coefficients_sha256=lock['coefficients_sha256'],
        models_complete_sha256=p['models_complete_sha256'], comparison=result,
        inputs_sha256=digest({str(Path(f).relative_to(output)): h for f, h in hashes.items()}),
        cpu_predictions_sha256=sha(output/'cpu_predictions'/f'confirm-{name}.json'),
        graph_executed='gofa' in requested, refitted=False)
    (output/'confirmation').mkdir(exist_ok=True)
    immutable(output/'confirmation'/(name+'.json'), value)
    if not result['strictly_improved']:
        terminal(output, 'failed_confirm_'+name, 'independent_confirmation')
    return value


def complete_confirmation(cfg, output):
    cpu_only()
    no_worker(output)
    m, p, lock = load_coefficients(cfg, output)
    datasets = {}
    for name in TARGETS:
        value = confirm_dataset(cfg, output, name)
        if not value['comparison']['strictly_improved']:
            return None
        datasets[name] = value
    gate = dict(binding=p['binding'], coefficients_sha256=lock['coefficients_sha256'],
        datasets=datasets, all_six_strictly_improved=True, refitted=False)
    immutable(output/'confirm_gate.json', gate)
    status(output, 'confirmed', 'independent_confirmation')
    return gate


def replay_test(cfg, output):
    cpu_only()
    no_worker(output)
    m, p, lock = load_coefficients(cfg, output)
    gate = read(output/'confirm_gate.json')
    if not gate or not gate['all_six_strictly_improved'] or gate['binding'] != p['binding'] \
            or gate['coefficients_sha256'] != lock['coefficients_sha256'] or set(gate['datasets']) != set(TARGETS):
        raise ValueError('All six confirmations required BEFORE any test access')
    cached = read(output/'test_replay.json')
    if cached is not None:
        if cached['binding'] != p['binding'] or cached['coefficients_sha256'] != lock['coefficients_sha256'] or not cached['complete']:
            raise ValueError('Existing test replay binding changed')
        return cached
    if complete_confirmation(cfg, output) != gate:
        raise ValueError('Confirmation inputs changed')
    immutable(output/'test_admission.json', dict(binding=p['binding'],
        coefficients_sha256=lock['coefficients_sha256'], confirm_gate_sha256=sha(output/'confirm_gate.json'),
        test_gpu_predictions_allowed=False))
    results = {}
    for name in TARGETS:
        n, g = (test_records(cfg, m, name, mode) for mode in ('native', 'gofa'))
        c = cpu_predictions(output, m, name, m['test_queries'][name], 'test')
        result = comparison(n, g, c, p['coefficients'][name], bootstrap=True)
        immutable(output/f'test-{name}.json', result)
        results[name] = result
    value = dict(binding=p['binding'], coefficients_sha256=lock['coefficients_sha256'], complete=True,
        datasets=results, all_six_strictly_improved=all(x['strictly_improved'] for x in results.values()),
        test_gpu_predictions_generated=0, target_supervised=True)
    immutable(output/'test_replay.json', value)
    terminal(output, 'completed_test_replay', 'test_replay', value['all_six_strictly_improved'])
    return value


def advance(cfg, output):
    if complete_confirmation(cfg, output):
        return replay_test(cfg, output)
    return report(output)


def report(output):
    value = {k: read(output/file) for k, file in dict(status='status.json', cpu_models='models_complete.json',
        development='development.json', coefficients='coefficients.json', independent_confirmation='confirm_gate.json',
        test='test_replay.json', activity='activity.json', cpu_activity='cpu_activity.json', memory='cuda_memory.json',
        postfit_feasibility='postfit_gate.json').items()}
    value.update(target_supervised=True, neural_optimizer_updates=0, strict_zero_shot=False,
        per_dataset_confirmation={p.stem: read(p) for p in (output/'confirmation').glob('*.json')},
        limitations=['Target first60 percent labels train six CPU models; this is NOT zero-shot GOFA-F',
            'The parent3500 adapter is frozen; CPU gains are not new adapter training results',
            'Old target tests and development results were viewed; not independent blind evaluation',
            'Zero graph weight or no win over Native+same-CPU does not prove additional graph-LLM gain',
            'Derived numeric forecasts have no language-generation coverage; skipped graph metrics remain null',
            'Fixed small target samples; Kelmarsh has only six clusters; no population-wide guarantee'])
    atomic_json(output/'report.json', value)
    return value
