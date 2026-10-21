"""Stage26 CPU-only fitting, confirmation gates and sealed test replay."""
from pathlib import Path
import time

import numpy as np

from gofa_f.allnode_stage import read
from gofa_f.generative_data import sha
from gofa_f.graph_benchmark_data import TARGETS
from gofa_f.prefix_cache import atomic_json
from gofa_f.short_sampled_forecast import checked
from gofa_f.short_sampling_diagnostic import SETTINGS, sample_seed
from gofa_f.six_calibration_math import comparison, fit
from gofa_f.six_calibration_state import SLICES, immutable, label, own_manifest, result_binding
from gofa_f.target_calibration import cpu_only
from gofa_f.target_calibration_math import paired
from gofa_f.target_calibration_state import digest


def status(output, state, phase, **fields):
    atomic_json(output/'status.json', dict(state=state, phase=phase, optimizer_updates=0,
        strict_zero_shot=False, updated_at=time.time(), **fields))


def no_worker(output):
    if read(output/'activity.json', {}).get('worker') is not None:
        raise ValueError('CPU calibration requires the GPU worker to have exited')


def read_shard(root, index, query, binding, hashes):
    path = root/f'{index:06d}.json'
    row = checked(path, binding, query)
    if row is None or row['sampling'] != SETTINGS \
            or row['seeds'] != [sample_seed(query, s) for s in range(8)]:
        raise ValueError('Missing/rebound sampled median')
    trajectories = []
    for sample in range(8):
        p = root/'trajectories'/f'{index:06d}-{sample:02d}.json'
        item = checked(p, binding, query, seed=sample_seed(query, sample), sample=sample)
        if item is None or np.asarray(item['prediction']).shape != (query['horizon'],) \
                or not np.isfinite(item['prediction']).all():
            raise ValueError('Median lacks eight finite bound sampled trajectories')
        hashes[str(p)] = sha(p)
        trajectories.append(item['prediction'])
    if not np.array_equal(np.median(np.asarray(trajectories, np.float64), axis=0), row['prediction']):
        raise ValueError('Median differs from sampled trajectories')
    mask, y, p = np.asarray(row['observed'], bool), np.asarray(row['target']), np.asarray(row['prediction'])
    if mask.shape != (query['horizon'],) or not mask.any() or not np.isfinite(y[mask]).all() \
            or not np.isfinite(p).all() or float(np.abs(p-y)[mask].mean()) != row['mae']:
        raise ValueError('Invalid target/mask/MAE binding')
    hashes[str(path)] = sha(path)
    return row


def records(output, manifest, subset, inference=False):
    """Fit reads only 0:48 shards; it never opens pilot/confirm/test records."""
    if not inference:
        no_worker(output)
    if subset not in SLICES:
        raise ValueError('Only explicitly partitioned validation records may be read')
    phase = 'confirm' if subset == 'confirm' else 'pilot'
    marker = read(output/f'{phase}_complete.json')
    binding = sha(output/'manifest.json')
    if marker is None or marker['binding'] != binding or not marker['complete']:
        raise ValueError('Prediction phase is incomplete')
    datasets, hashes = {}, {}
    begin, end = SLICES[subset]
    for name in TARGETS:
        datasets[name] = {}
        for mode in ('native', 'gofa'):
            own = label(phase, name, mode)
            expected = result_binding(manifest, binding, phase, name, mode)
            root = output/'evaluations'/own
            entries = []
            for global_index in range(begin, end):
                local = global_index-64 if phase == 'confirm' else global_index
                entries.append(read_shard(root, local, manifest['queries'][name][global_index], expected, hashes))
            datasets[name][mode] = entries
        paired(datasets[name]['native'], datasets[name]['gofa'])
    relative = {str(Path(p).relative_to(output)): value for p, value in hashes.items()}
    return datasets, digest(relative)


def fit_coefficients(cfg, output):
    cpu_only()
    manifest = own_manifest(cfg, output)
    datasets, inputs = records(output, manifest, 'fit')
    value = dict(binding=sha(output/'manifest.json'), fit_inputs_sha256=inputs,
        coefficients={name: dict(blend_bias=fit(pair['native'], pair['gofa']),
                                native_bias=fit(pair['native'], pair['gofa'], native_only=True))
                      for name, pair in datasets.items()},
        deployment_scalar_count=12, native_control_scalar_count=6, graph_alpha=1.,
        fit_queries_per_dataset=48, confirmation_or_test_records_read=False,
        refit_allowed=False, optimizer_updates=0, strict_zero_shot=False)
    immutable(output/'coefficients.json', value)
    immutable(output/'coefficient_lock.json', dict(binding=value['binding'],
        coefficients_sha256=sha(output/'coefficients.json'), fit_inputs_sha256=inputs))
    return value


def load_coefficients(cfg, output, inference=False):
    manifest = own_manifest(cfg, output)
    value, lock = read(output/'coefficients.json'), read(output/'coefficient_lock.json')
    if value is None or lock is None or value['binding'] != sha(output/'manifest.json') \
            or lock['binding'] != value['binding'] or sha(output/'coefficients.json') != lock['coefficients_sha256'] \
            or set(value['coefficients']) != set(TARGETS) or value['deployment_scalar_count'] != 12 \
            or value['confirmation_or_test_records_read'] or value['strict_zero_shot']:
        raise ValueError('Twelve validation-only deployment scalars must be sealed')
    _, inputs = records(output, manifest, 'fit', inference=inference)
    if inputs != lock['fit_inputs_sha256']:
        raise ValueError('Fit inputs changed after sealing')
    return manifest, value, lock


def load_gate(cfg, output, phase, inference=False):
    manifest, coefficients, lock = load_coefficients(cfg, output, inference=inference)
    gate = read(output/f'{phase}_gate.json')
    if gate is None or not gate['all_six_strictly_improved'] \
            or gate['coefficient_seal_sha256'] != lock['coefficients_sha256'] \
            or gate['binding'] != coefficients['binding'] or set(gate['datasets']) != set(TARGETS) \
            or not all(v['strictly_improved'] for v in gate['datasets'].values()):
        raise ValueError('All six '+phase+' confirmations must pass before access')
    _, inputs = records(output, manifest, phase, inference=inference)
    if inputs != gate['inputs_sha256']:
        raise ValueError('Confirmation inputs changed after gate sealing')
    return manifest, coefficients, lock, gate


def confirm(cfg, output, phase):
    cpu_only()
    if phase not in ('pilot', 'confirm'):
        raise ValueError('Unknown confirmation phase')
    if phase == 'confirm':
        load_gate(cfg, output, 'pilot')
    manifest, coefficients, lock = load_coefficients(cfg, output)
    datasets, inputs = records(output, manifest, phase)
    results = {name: comparison(pair['native'], pair['gofa'], coefficients['coefficients'][name])
               for name, pair in datasets.items()}
    passed = all(v['strictly_improved'] for v in results.values())
    value = dict(binding=coefficients['binding'], coefficient_seal_sha256=lock['coefficients_sha256'],
        inputs_sha256=inputs, phase=phase, all_six_strictly_improved=passed, datasets=results,
        strict_zero_shot=False, parent_preservation_gate=False, refitted=False)
    immutable(output/f'{phase}_gate.json', value)
    if not passed:
        status(output, 'complete', phase, stop_reason=f'failed_{phase}_gate', goal_achieved=False)
        report(output)
    else:
        status(output, 'pilot_passed' if phase == 'pilot' else 'confirmed', phase,
               goal_achieved=None, coefficient_seal_sha256=lock['coefficients_sha256'])
    return value


def test_records(cfg, manifest, name, mode):
    parent = Path(cfg['stage21_output'])
    own = f'target-{name}-{mode}'
    weight = manifest['weights_sha256'] if mode == 'native' else cfg['adapter_sha256']
    binding = f"{cfg['stage21_manifest_sha256']}:sample8:{own}:{weight}:"+manifest['alphabet']['alphabet_sha256']
    hashes = {}
    entries = [read_shard(parent/'evaluations'/own, i, q, binding, hashes)
               for i, q in enumerate(manifest['test_queries'][name])]
    for p, value in hashes.items():
        if manifest['test_hashes'].get(str(Path(p).relative_to(parent))) != value:
            raise ValueError('Frozen test shard changed')
    result = parent/(own+'-result.json')
    if sha(result) != manifest['test_hashes'][str(result.relative_to(parent))] \
            or read(result)['binding'] != binding or not read(result)['complete']:
        raise ValueError('Frozen test result changed')
    return entries


def replay_test(cfg, output):
    cpu_only()
    no_worker(output)
    # Both barriers are checked before any test prediction/label is opened.
    load_gate(cfg, output, 'pilot')
    manifest, coefficients, lock, gate = load_gate(cfg, output, 'confirm')
    immutable(output/'test_admission.json', dict(binding=coefficients['binding'],
        coefficients_sha256=lock['coefficients_sha256'],
        pilot_gate_sha256=sha(output/'pilot_gate.json'), confirm_gate_sha256=sha(output/'confirm_gate.json'),
        all_six_confirmed=True, test_gpu_predictions_allowed=False))
    results = {}
    for name in TARGETS:
        n, g = (test_records(cfg, manifest, name, mode) for mode in ('native', 'gofa'))
        value = comparison(n, g, coefficients['coefficients'][name], bootstrap=True)
        immutable(output/f'test-{name}.json', value)
        results[name] = value
    value = dict(binding=coefficients['binding'], coefficient_seal_sha256=lock['coefficients_sha256'],
        complete=True, datasets=results, all_six_strictly_improved=all(v['strictly_improved'] for v in results.values()),
        test_queries=768, test_gpu_predictions_generated=0, strict_zero_shot=False)
    immutable(output/'test_replay.json', value)
    status(output, 'complete', 'test_replay', stop_reason='completed_frozen_test_replay',
        goal_achieved=value['all_six_strictly_improved'])
    report(output)
    return value


def advance(cfg, output):
    cpu_only()
    no_worker(output)
    current = read(output/'status.json', {}).get('state')
    if current == 'pilot_complete':
        fit_coefficients(cfg, output)
        return confirm(cfg, output, 'pilot')
    if current in ('confirm_complete', 'confirmed'):
        if current == 'confirm_complete' and not confirm(cfg, output, 'confirm')['all_six_strictly_improved']:
            return report(output)
        return replay_test(cfg, output)
    raise ValueError('CPU advance requires an exited completed prediction phase')


def report(output):
    value = dict(status=read(output/'status.json'), coefficients=read(output/'coefficients.json'),
        pilot=read(output/'pilot_gate.json'), independent_confirmation=read(output/'confirm_gate.json'),
        test=read(output/'test_replay.json'), activity=read(output/'activity.json'),
        memory=read(output/'cuda_memory.json'), optimizer_updates=0, strict_zero_shot=False,
        limitations=['Target validation labels used; this is a target-validation-calibrated system',
            'Previously viewed fixed test results: development reevaluation, not an independent blind test',
            'Small disjoint confirmation sets and selected variables; no guarantee of population advantage',
            'Native bias control required to interpret any extra graph gain',
            'Derived numeric fusion has no language generation coverage claim',
            'Eight-sample BF16 constrained protocol differs from official FP16 execution'])
    atomic_json(output/'report.json', value)
    return value
