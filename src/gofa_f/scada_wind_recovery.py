"""Exact inference-state handoff after the Stage33 CPU finalization failure only."""
import argparse
import copy
from pathlib import Path
import shutil
import time

import numpy as np

from gofa_f.allnode_stage import read
from gofa_f.generative_data import sha
from gofa_f.graph_benchmark_data import TARGETS
from gofa_f.idle_launcher import process_identity
from gofa_f.prefix_cache import atomic_json
from gofa_f.scada_wind_iteration import (advance, compare, confirmation_fingerprint,
    inference_binding, load_coefficients, manifest, status)
from gofa_f.short_sampled_forecast import checked
from gofa_f.short_sampling_diagnostic import sample_seed
from gofa_f.six_calibration_cpu import read_shard
from gofa_f.six_calibration_state import immutable
from gofa_f.source_retraining import configuration
from gofa_f.target_calibration import cpu_only
from gofa_f.target_calibration_math import improves
from gofa_f.target_calibration_state import code_hashes, digest

SOURCE_MANIFEST = '37d33deba228d0a8cef6e098bbf6709835ac9aacfc60bd9f2774272fb3a80398'
SOURCE_COEFFICIENTS = '72aed90e4ff81e7621bed714324eb2389bee8b1bde837bc8c337416497e04968'


def numeric_replay(left, right):
    """State/gates exact; CPU FP32 diagnostic tolerance is still1e-6."""
    differences = []
    def visit(a, b):
        if type(a) is not type(b):
            raise ValueError('Replay structure/type changed')
        if isinstance(a, dict):
            if a.keys() != b.keys():
                raise ValueError('Replay fields changed')
            for k in a:
                visit(a[k], b[k])
        elif isinstance(a, list):
            if len(a) != len(b):
                raise ValueError('Replay list length changed')
            for x, y in zip(a, b):
                visit(x, y)
        elif isinstance(a, float):
            if not np.isfinite([a, b]).all() or not np.isclose(a, b, rtol=1e-6, atol=1e-6):
                raise ValueError('CPU numeric replay exceeded original tolerance')
            differences.append(abs(a-b))
        elif a != b:
            raise ValueError('Discrete replay state/gate changed')
    visit(left, right)
    return max(differences, default=0.)


def source_exited(source):
    for suffix in ('.launch.json', '.bootstrap.launch.json'):
        identity = read(source.with_suffix(suffix), {})
        if identity.get('pid') and process_identity(identity['pid']) == str(identity['start_ticks']):
            raise ValueError('Source controller still alive')
    if read(source/'activity.json', {}).get('worker') is not None \
            or read(source/'launcher.json', {}).get('state') != 'failed' \
            or read(source/'launcher.json', {}).get('cpu_returncode') != 1 \
            or read(source/'status.json', {}).get('state') != 'confirm_complete' \
            or (source/'confirm_gate.json').exists() or (source/'test_replay.json').exists():
        raise ValueError('Only exited pre-test CPU-finalization failure is repairable here')
    tail = source.with_suffix('.log').read_text()[-6000:]
    if 'ValueError: Sealed artifact changed:' not in tail or '/confirmation/pems03.json' not in tail:
        raise ValueError('Different source failure; do not use this handoff')


def migrate(cfg, source, output, cpu_proof):
    cpu_only()
    source_exited(source)
    if output.resolve() == source.resolve() or output.exists() \
            or Path(cfg['output_directory']).resolve() != output.resolve():
        raise ValueError('New independent repair output required')
    if sha(source/'manifest.json') != SOURCE_MANIFEST \
            or sha(source/'coefficients.json') != SOURCE_COEFFICIENTS:
        raise ValueError('Pinned source manifest/coefficients changed')
    old = read(source/'manifest.json')
    oldp = read(source/'coefficients.json')
    if old['configuration']['sampling'] != cfg['sampling'] \
            or old['configuration']['deadline_unix'] != cfg['deadline_unix']:
        raise ValueError('Sampling or deadline changed')
    code = code_hashes()
    if not cpu_proof.get('passed') or cpu_proof.get('cuda_initialized') is not False \
            or cpu_proof.get('code') != code or cpu_proof.get('cases', 0) < 719 \
            or any(code.get(k) != v for k, v in old['unchanged_tensor_files'].items()):
        raise ValueError('Full current CPU proof and unchanged numerical core required')
    source_confirmations = {name: sha(source/'confirmation'/(name+'.json')) for name in TARGETS}
    if not all(read(source/'confirmation'/(name+'.json'))['comparison']['strictly_improved']
               for name in TARGETS):
        raise ValueError('All six original sealed confirmation gates must have passed')
    output.mkdir(parents=True)
    for name in ['adapter.pt', 'scada.npz', 'scada.json', 'preparation_queries.json',
                 'gpu_identity.json', 'cpu_activity.json', 'cuda_memory.json']:
        shutil.copyfile(source/name, output/name)
    m = copy.deepcopy(old)
    m.update(configuration=cfg, code=code, cpu_finalization_repair=dict(
        source_manifest_sha256=SOURCE_MANIFEST, source_coefficients_sha256=SOURCE_COEFFICIENTS,
        source_confirmation_sha256=source_confirmations, neural_updates=0,
        model_refits=0, new_gpu_predictions=0))
    immutable(output/'manifest.json', m)
    binding = sha(output/'manifest.json')
    development = copy.deepcopy(read(source/'development.json'))
    development['binding'] = binding
    immutable(output/'development.json', development)
    parameters = copy.deepcopy(oldp)
    parameters.update(binding=binding, development_sha256=sha(output/'development.json'))
    immutable(output/'coefficients.json', parameters)
    lock = dict(binding=binding, coefficients_sha256=sha(output/'coefficients.json'))
    immutable(output/'coefficient_lock.json', lock)
    mechanism = copy.deepcopy(read(source/'mechanisms.json'))
    if not mechanism['passed'] or mechanism['binding'] != SOURCE_MANIFEST \
            or mechanism['gradients_created']:
        raise ValueError('Original CUDA acceptance proof changed')
    mechanism.update(binding=binding, cpu_repair_reuse=dict(
        source_sha256=sha(source/'mechanisms.json'), original_binding=SOURCE_MANIFEST,
        unchanged_numerical_core=True, new_gpu_execution=False))
    immutable(output/'mechanisms.json', mechanism)
    for directory in ('path_checks', 'acceptance'):
        if (source/directory).exists():
            shutil.copytree(source/directory, output/('inherited_'+directory))
    if parameters['coefficients'] != oldp['coefficients']:
        raise ValueError('Any scientific parameter change is forbidden in repair')
    for path in (source/'candidates/sdwpf').glob('*.joblib'):
        target = output/'candidates/sdwpf'/path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        meta = copy.deepcopy(read(path.with_suffix('.json')))
        if sha(target) != meta['model_sha256']:
            raise ValueError('CPU model bytes changed')
        meta['binding'] = digest(dict(manifest=binding, recipe=path.stem))
        immutable(target.with_suffix('.json'), meta)
    activity = copy.deepcopy(read(source/'activity.json'))
    activity['inherited_before_cpu_repair_seconds'] = activity['spent_seconds']
    immutable(output/'activity.json', activity)
    proof = dict(cpu_proof, manifest_sha256=binding, cuda_disabled=True,
                 inherited_frozen_core_proof_sha256=m['inherited_cpu_proof_sha256'])
    immutable(output/'cpu_preflight.json', proof)
    # Verify new header bindings and all unchanged models before importing any decision.
    manifest(cfg, output)
    load_coefficients(cfg, output)
    receipts, deltas = {}, {}
    for name in TARGETS:
        hashes, pair = {}, {}
        modes = ['native']+(['gofa'] if parameters['coefficients'][name]['full']['weights'].get('gofa', 0) != 0 else [])
        for mode in modes:
            label = f'confirm-{name}-{mode}'
            folder, dest = source/'evaluations'/label, output/'evaluations'/label
            original = inference_binding(old, SOURCE_MANIFEST, name, mode)
            rebound = inference_binding(m, binding, name, mode)
            complete = read(folder/'complete.json')
            if not complete['complete'] or complete['binding'] != original:
                raise ValueError('Source sampling incomplete or rebound')
            for i, q in enumerate(m['queries'][name][64:]):
                read_shard(folder, i, q, original, {})
                for sample in range(8):
                    rel = Path('trajectories')/f'{i:06d}-{sample:02d}.json'
                    p = folder/rel
                    row = checked(p, original, q, seed=sample_seed(q, sample), sample=sample)
                    receipt = dict(source_sha256=sha(p), source_binding=original)
                    (dest/rel).parent.mkdir(parents=True, exist_ok=True)
                    immutable(dest/rel, dict(row, binding=rebound, cpu_repair_reuse=receipt))
                    receipts[str((dest/rel).relative_to(output))] = receipt
                p = folder/f'{i:06d}.json'
                row = read(p)
                receipt = dict(source_sha256=sha(p), source_binding=original)
                immutable(dest/p.name, dict(row, binding=rebound, cpu_repair_reuse=receipt))
                receipts[str((dest/p.name).relative_to(output))] = receipt
            immutable(dest/'complete.json', dict(complete, binding=rebound,
                cpu_repair_reuse=dict(source_sha256=sha(folder/'complete.json'), new_gpu_seconds=0)))
            hashes[str(dest/'complete.json')] = sha(dest/'complete.json')
            pair[mode] = [read_shard(dest, i, q, rebound, hashes)
                          for i, q in enumerate(m['queries'][name][64:])]
        old_result = read(source/'confirmation'/(name+'.json'))
        current = compare(cfg, output, name, pair['native'], pair.get('gofa'))
        deltas[name] = numeric_replay(old_result['comparison'], current)
        result = copy.deepcopy(old_result)
        if result['comparison']['strictly_improved'] != improves(
                current['metrics']['calibrated']['macro']['mae'], current['metrics']['native']['macro']['mae']):
            raise ValueError('Original strict confirmation decision changed')
        result.update(binding=binding, coefficients_sha256=lock['coefficients_sha256'],
            inputs_sha256=confirmation_fingerprint(output, m, parameters, lock, name, hashes),
            cpu_repair_reuse=dict(source_confirmation_sha256=source_confirmations[name],
                                  original_comparison_preserved_exactly=True))
        (output/'confirmation').mkdir(exist_ok=True)
        immutable(output/'confirmation'/(name+'.json'), result)
    immutable(output/'cpu_recovery.json', dict(source_manifest_sha256=SOURCE_MANIFEST,
        source_coefficients_sha256=SOURCE_COEFFICIENTS, source_confirmations=source_confirmations,
        parameters_unchanged=True, model_bytes_unchanged=True, sampling_tokens_unchanged=True,
        gpu_activity_added=0, source_gpu_seconds=activity['spent_seconds'], cpu_numeric_rtol=1e-6,
        cpu_numeric_atol=1e-6, numeric_max_abs_by_dataset=deltas, trajectories=receipts,
        frozen_test_read=False, recovered_at=time.time()))
    status(output, 'confirm_complete', 'cpu_repair', cpu_gate_pending=True, new_gpu_predictions=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-proof', type=Path, required=True)
    args = parser.parse_args()
    cfg = configuration(args.config)
    started = time.monotonic()
    migrate(cfg, args.source, args.output, read(args.cpu_proof))
    result = advance(cfg, args.output)
    elapsed = time.monotonic()-started
    cost = read(args.output/'cpu_activity.json')
    cost['inherited_before_cpu_repair_seconds'] = cost['spent_seconds']
    cost['cpu_repair_seconds'] = elapsed
    cost['spent_seconds'] += elapsed
    atomic_json(args.output/'cpu_activity.json', cost)
    from gofa_f.scada_wind_iteration import report
    report(args.output)
    return result


if __name__ == '__main__':
    main()
