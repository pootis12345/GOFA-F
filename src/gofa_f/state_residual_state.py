"""Stage28 immutable provenance, additional development identities and safe imports."""
import copy
from dataclasses import asdict
import os
from pathlib import Path
import shutil

import numpy as np

from gofa_f.allnode_stage import read
from gofa_f.generative_data import Query, Series, sha
from gofa_f.graph_benchmark_data import TARGETS
from gofa_f.history_calibration_state import development_records, verify_data
from gofa_f.short_data import origins, stratified
from gofa_f.short_sampled_forecast import checked
from gofa_f.short_sampled_state import exited
from gofa_f.short_sampling_diagnostic import SETTINGS, sample_seed
from gofa_f.six_calibration_cpu import no_worker, read_shard
from gofa_f.six_calibration_state import ADAPTER, immutable, result_binding
from gofa_f.target_calibration_state import code_hashes, digest


def validate_config(cfg):
    resources = dict(admission_policy='memory_cap_only', admission_samples=1, resource_check_seconds=10,
        memory_cap_gib=40, torch_cap_gib=38, activity_limit_seconds=None, wall_deadline=None,
        yield_on_foreign=False, leave_free_gib=0)
    if cfg['protocol'] != 'stage28-state-residual-calibration-v1' or cfg['seed'] != 42 \
            or cfg['adapter_sha256'] != ADAPTER or cfg['targets'] != list(TARGETS) \
            or cfg['sampling'] != SETTINGS or cfg['resources'] != resources \
            or cfg['training'] != {'max_updates': 0} or cfg['taus'] != [.01, .1, 1.] \
            or cfg['l1_limit'] != .25 or cfg['folds'] != [[64, 96], [96, 128]] \
            or cfg['development_minimum_improvement'] != .005 or cfg['additional_queries'] != 64 \
            or cfg['data'] != dict(target_training_enabled=False, target_evaluation_enabled=True,
                target_label_calibration_enabled=True, strict_zero_shot=False):
        raise ValueError('Fixed Stage28 protocol changed')


def previous(cfg):
    root = Path(cfg['stage27_output'])
    for file, key in (('manifest.json', 'stage27_manifest_sha256'), ('coefficients.json', 'stage27_coefficients_sha256')):
        if sha(root/file) != cfg[key]:
            raise ValueError('Stage27 immutable archive changed')
    m, p = read(root/'manifest.json'), read(root/'coefficients.json')
    lock = read(root/'coefficient_lock.json')
    if lock['coefficients_sha256'] != cfg['stage27_coefficients_sha256'] or p['binding'] != cfg['stage27_manifest_sha256'] \
            or p['independent_confirmation_or_test_read'] or not read(root/'cpu_gate.json')['all_six_strictly_improved']:
        raise ValueError('Stage27 frozen five parameter provenance invalid')
    for key in ('stage26_output', 'stage26_manifest_sha256', 'stage26_report_sha256',
                'stage26_pilot_gate_sha256', 'stage21_output', 'stage21_manifest_sha256', 'stage21_report_sha256'):
        if cfg[key] != m['configuration'][key]:
            raise ValueError('Inherited archive differs: '+key)
    return m, p


def additional_queries(data, original, coverage):
    old = original[:64]
    if len(old) != 64 or len({q['variable'] for q in old}) != 64:
        raise ValueError('Exactly64 distinct original SDWPF development variables required')
    cuts = coverage['boundaries']
    added = []
    for index, q in enumerate(old):
        block = (index//16+2) % 4
        points = origins(data, q['variable'], 48, 24, 'validation', require_history=True)
        points = points[(points >= cuts[block]) & (points+24 <= cuts[block+1])]
        for other in original:
            if other['variable'] == q['variable']:
                points = points[(points+24 <= other['origin']) | (points >= other['origin']+other['horizon'])]
        rng = np.random.default_rng(np.random.SeedSequence([42, 28, q['variable'], block]))
        origin = stratified(points, 1, 24, rng, disjoint=True)[0]
        added.append(asdict(Query('sdwpf', q['variable'], int(origin), 48, 24)))
    order = [dict(kind='prior', index=i, query=q) for i, q in enumerate(old)] \
        +[dict(kind='added', index=i, query=q) for i, q in enumerate(added)]
    order.sort(key=lambda r: (r['query']['origin'], str(data.ids[r['query']['variable']])))
    if len({Query(**r['query']).key for r in order}) != 128:
        raise ValueError('Duplicate development query')
    for block in range(4):
        if sum(cuts[block] <= r['query']['origin'] and r['query']['origin']+24 <= cuts[block+1] for r in order) != 32:
            raise ValueError('Each forward-time development block must contain32 queries')
    return added, order


def prepare(cfg, output):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU preparation requires CUDA disabled')
    validate_config(cfg)
    for key in ('stage21_output', 'stage26_output', 'stage27_output'):
        if output.resolve() == Path(cfg[key]).resolve():
            raise ValueError('Independent Stage28 output required')
        exited(Path(cfg[key]))
    prior, parameters = previous(cfg)
    if read(Path(cfg['stage27_output'])/'confirm_gate.json') or read(Path(cfg['stage27_output'])/'test_replay.json'):
        raise ValueError('No completed Stage27 confirmation/test may be reused for selection')
    code = code_hashes()
    if any(code.get(n) != h for n, h in prior['unchanged_tensor_files'].items()):
        raise ValueError('Accepted inference numerical core changed')
    datasets, inputs = development_records(cfg)
    manifest = {k: copy.deepcopy(prior[k]) for k in ('source_configuration', 'weights_sha256', 'weight_stats',
        'alphabet', 'source_selection', 'source_raw_hashes', 'probes', 'unchanged_tensor_files',
        'data_directory', 'raw_hashes', 'metadata_hashes', 'queries', 'coverage', 'test_queries', 'test_hashes')}
    verify_data(manifest)
    data = Series(Path(manifest['data_directory'])/'targets/sdwpf.npz')
    added, order = additional_queries(data, manifest['queries']['sdwpf'], manifest['coverage']['sdwpf'])
    from gofa_f.state_residual_math import frame, history_state
    # Verify real-data per-node scaling and all added graph contexts before GPU work.
    for n, g in zip(datasets['sdwpf']['native'], datasets['sdwpf']['gofa']):
        frame(n, g, data)
    added_states = [history_state(data, q) for q in added]
    frozen = {n: parameters['coefficients'][n] for n in TARGETS if n != 'sdwpf'}
    proof = read(Path(cfg['stage27_output'])/'cpu_preflight.json')
    if not proof['passed'] or proof['manifest_sha256'] != cfg['stage27_manifest_sha256']:
        raise ValueError('Inherited bound CPU numerical proof missing')
    manifest.update(configuration=cfg, code=code, prior_development_inputs_sha256=inputs,
        added_queries=added, development_order=order, frozen_coefficients=frozen,
        cpu_feature_audit=dict(prior_paired_frames_checked=64, added_histories_checked=64,
            added_neighbor_counts=[s['neighbors'] for s in added_states], future_labels_read=False),
        frozen_coefficients_sha256=digest(frozen), inherited_cpu_proof_sha256=sha(Path(cfg['stage27_output'])/'cpu_preflight.json'),
        confirmation_previously_partially_generated=True, neural_training_allowed=False, optimizer_updates=0,
        graph_alpha=1., strict_zero_shot=False,
        total_numeric_work=16*(64*24+sum(q['horizon'] for rows in manifest['queries'].values() for q in rows[64:])))
    output.mkdir(parents=True, exist_ok=True)
    immutable(output/'manifest.json', manifest)
    if not (output/'adapter.pt').exists():
        shutil.copyfile(Path(cfg['stage27_output'])/'adapter.pt', output/'adapter.pt')
    if sha(output/'adapter.pt') != ADAPTER:
        raise ValueError('Frozen adapter changed')
    immutable(output/'gpu_identity.json', read(Path(cfg['stage27_output'])/'gpu_identity.json'))
    if not (output/'status.json').exists():
        immutable(output/'status.json', dict(state='prepared', phase='prepared', optimizer_updates=0))
    return manifest


def own_manifest(cfg, output):
    validate_config(cfg)
    m = read(output/'manifest.json')
    if not m or m['configuration'] != cfg or m['code'] != code_hashes():
        raise ValueError('Stage28 code/config/manifest binding changed')
    return m


def verify_inference(cfg, output):
    m = own_manifest(cfg, output)
    previous(cfg)
    verify_data(m)
    root = Path(m['source_configuration']['weights']['directory'])
    if sha(output/'adapter.pt') != ADAPTER or sha(root/'snapshot.json') != m['weights_sha256'] \
            or any((root/n).stat().st_size != v['size'] or (root/n).stat().st_mtime_ns != v['mtime_ns']
                   for n, v in m['weight_stats'].items()):
        raise ValueError('Frozen inference weights changed')
    return m


def prediction_binding(m, binding, phase, name, mode):
    weight = m['weights_sha256'] if mode == 'native' else ADAPTER
    return f'{binding}:sample8:{phase}-{name}-{mode}:{weight}:'+m['alphabet']['alphabet_sha256']


def prediction_records(output, manifest, phase, inference=False):
    if not inference:
        no_worker(output)
    if phase not in ('development', 'confirm'):
        raise ValueError('Explicit inference subset required')
    binding = sha(output/'manifest.json')
    complete = read(output/f'{phase}_complete.json')
    if not complete or complete['binding'] != binding or not complete['complete']:
        raise ValueError('Prediction phase incomplete')
    pairs, hashes = {}, {}
    names = ('sdwpf',) if phase == 'development' else tuple(TARGETS)
    for name in names:
        pairs[name] = {}
        queries = manifest['added_queries'] if phase == 'development' else manifest['queries'][name][64:]
        for mode in ('native', 'gofa'):
            root = output/'evaluations'/f'{phase}-{name}-{mode}'
            expected = prediction_binding(manifest, binding, phase, name, mode)
            pairs[name][mode] = [read_shard(root, i, q, expected, hashes) for i, q in enumerate(queries)]
    return pairs, digest({str(Path(p).relative_to(output)): h for p, h in hashes.items()})


def development_pair(cfg, output, manifest, inference=False):
    old, original = development_records(cfg)
    if original != manifest['prior_development_inputs_sha256']:
        raise ValueError('Original development inputs changed')
    added, extra = prediction_records(output, manifest, 'development', inference=inference)
    pair = {mode: [(old['sdwpf'] if item['kind'] == 'prior' else added['sdwpf'])[mode][item['index']]
        for item in manifest['development_order']] for mode in ('native', 'gofa')}
    return pair, digest(dict(prior=original, added=extra, order=manifest['development_order']))


def import_confirmation(cfg, output, manifest, inference=False):
    """Called only AFTER the parameter seal; old evidence remains untouched."""
    from gofa_f.state_residual_cpu import load_coefficients
    load_coefficients(cfg, output, inference=inference)
    root = Path(cfg['stage27_output'])
    old_manifest = read(root/'manifest.json')
    for key in ('queries', 'raw_hashes', 'metadata_hashes', 'unchanged_tensor_files', 'alphabet', 'weights_sha256'):
        if old_manifest[key] != manifest[key]:
            raise ValueError('Partial confirmation numerical provenance differs')
    receipts = {}
    for name in TARGETS:
        for mode in ('native', 'gofa'):
            label = f'confirm-{name}-{mode}'
            source = root/'evaluations'/label
            destination = output/'evaluations'/label
            old_binding = result_binding(old_manifest, cfg['stage27_manifest_sha256'], 'confirm', name, mode)
            new_binding = prediction_binding(manifest, sha(output/'manifest.json'), 'confirm', name, mode)
            for i, q in enumerate(manifest['queries'][name][64:]):
                for sample in range(8):
                    relative = Path('trajectories')/f'{i:06d}-{sample:02d}.json'
                    path = source/relative
                    if not path.exists():
                        continue
                    value = checked(path, old_binding, q, seed=sample_seed(q, sample), sample=sample)
                    if np.asarray(value['prediction']).shape != (q['horizon'],) or not np.isfinite(value['prediction']).all():
                        raise ValueError('Invalid imported trajectory')
                    target = destination/relative
                    receipt = dict(original_path=str(path), original_sha256=sha(path), original_binding=old_binding)
                    imported = dict(value, binding=new_binding, imported_evidence=receipt)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():
                        immutable(target, imported)
                    elif read(target) != imported:
                        raise ValueError('Imported trajectory rebound')
                    receipts[str(target.relative_to(output))] = receipt
                median = source/f'{i:06d}.json'
                if median.exists():
                    hashes = {}
                    value = read_shard(source, i, q, old_binding, hashes)
                    target = destination/median.name
                    receipt = dict(original_path=str(median), original_sha256=sha(median), original_binding=old_binding)
                    imported = dict(value, binding=new_binding, imported_evidence=receipt)
                    if not target.exists():
                        immutable(target, imported)
                    elif read(target) != imported:
                        raise ValueError('Imported median rebound')
                    receipts[str(target.relative_to(output))] = receipt
    return immutable(output/'confirmation_imports.json', dict(binding=sha(output/'manifest.json'),
        records=receipts, used_for_fitting=False, old_evidence_modified=False,
        inherited_activity_seconds=read(root/'activity.json')['spent_seconds']))
