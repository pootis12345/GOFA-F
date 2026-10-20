"""Stage23 immutable data/model protocol and source-only lineage barriers."""
import os
from pathlib import Path
import shutil

import numpy as np

from gofa_f.generative_data import Series, sha
from gofa_f.graph_benchmark import ADAPTER_SHA, atomic, read
from gofa_f.short_sampled_state import exited
from gofa_f.source_gate_state import TARGETS
from gofa_f.source_retraining_data import VIEWS
from gofa_f.source_retraining_policy import pools


def validate(cfg):
    expected = dict(admission_policy='idle', admission_samples=3, resource_check_seconds=10,
        memory_cap_gib=40, torch_cap_gib=38, activity_limit_seconds=None, wall_deadline=None,
        yield_on_foreign=False, leave_free_gib=0)
    training = dict(max_updates=1000, effective_batch=64, microbatch=2, learning_rate=5e-5,
        warmup_updates=50, weight_decay=1e-4, gradient_clip=1, validation_every=250,
        early_min_updates=500, early_patience=2, significant_improvement=.001)
    if cfg['protocol'] != 'stage23-source-retraining-v1' or cfg['seed'] != 42 \
            or cfg['adapter_sha256'] != ADAPTER_SHA or cfg['targets'] != list(TARGETS) \
            or cfg['resources'] != expected or cfg['training'] != training \
            or cfg['sampling'] != dict(samples=8, top_k=100, temperature=1., top_p=1.) \
            or cfg['data']['target_training_enabled'] or cfg['data']['target_evaluation_enabled']:
        raise ValueError('Authorized Stage23 policy changed')


def prior(cfg):
    validate(cfg)
    for root in (cfg['stage22_output'], cfg['stage21_output'], cfg['source_output']):
        exited(Path(root))
    source, target = Path(cfg['source_output']), Path(cfg['stage21_output'])
    if sha(source/'sampled_best.pt') != ADAPTER_SHA \
            or sha(source/'manifest.json') != cfg['source_manifest_sha256'] \
            or sha(target/'manifest.json') != cfg['stage21_manifest_sha256']:
        raise ValueError('Parent3500/provenance changed')
    manifest = read(target/'manifest.json')
    if manifest['source_selection']['updates'] != 3500:
        raise ValueError('Unique3500 source selection required')
    for name, digest in manifest['unchanged_tensor_files'].items():
        if sha(Path(__file__).parent/name) != digest:
            raise ValueError('Accepted numerical tensor core changed: '+name)
    for name in ('short_training.py', 'short_sampled_forecast.py', 'short_sampling_diagnostic.py'):
        original = read(source/'manifest.json')
        if sha(Path(__file__).parent/name) != original['code'][name]:
            raise ValueError('Accepted teacher/sample execution changed: '+name)
    return manifest


def isolation(series, prior_manifest):
    """Metadata/observation barriers; never reads target values or target scores."""
    target_root = Path(prior_manifest['data_directory'])/'targets'
    targets = {}
    for n in TARGETS:
        path = target_root/(n+'.json')
        if sha(path) != prior_manifest['metadata_hashes'][n] \
                or sha(target_root/(n+'.npz')) != prior_manifest['raw_hashes'][n]:
            raise ValueError('Frozen target isolation evidence changed: '+n)
        targets[n] = read(path)
    if set(series) != set(VIEWS):
        raise ValueError('Four fixed views required')
    records = []
    for name, data in series.items():
        start, end = int(data.times[0]), int(data.times[-1])
        coords = data.meta['provenance'].get('coordinates')
        for target, meta in targets.items():
            t0, t1 = np.datetime64(meta['start']).astype('datetime64[ns]').astype(np.int64), \
                np.datetime64(meta['end']).astype('datetime64[ns]').astype(np.int64)
            same_observation = not (end < t0 or t1 < start)
            geographically_distinct = False
            other = meta['provenance'].get('coordinates')
            if coords is not None and other is not None and target != 'sdwpf':
                from gofa_f.source_retraining_data import distance_km
                geographically_distinct = min(distance_km(a, b) for a in coords for b in other) > 1
            source_domain = 'traffic' if name.startswith('pems04') else 'weather' if name == 'boston2017' else 'wind'
            target_domain = 'traffic' if target in ('pems03', 'pems_bay') else \
                'weather' if target in ('meteonet_nw2018', 'engrad') else 'wind'
            different_quantity = source_domain != target_domain
            permitted = not same_observation or geographically_distinct or different_quantity
            if not permitted:
                raise ValueError('Cannot exclude source/target observation overlap: '+name+'/'+target)
            records.append(dict(source=name, target=target, disjoint_times=not same_observation,
                coordinates_distinct=geographically_distinct, different_physical_quantity=different_quantity, passed=True))
    return dict(passed=True, comparisons=records, target_values_read=False,
        traffic_same_entity_different_times_possible=True, pretraining_exposure_independently_unverifiable=True,
        identifiers='BostonUS2017; PenmanshielScotland2017; SDWPFChina2020-21; KelmarshEngland2019')


def prepare(cfg, output):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU preparation requires CUDA_VISIBLE_DEVICES empty')
    original = prior(cfg)
    root, output = Path(cfg['data']['directory']), Path(output)
    series = {n: Series(root/'sources'/(n+'.npz')) for n in VIEWS}
    for name, data in series.items():
        if data.meta['sha256'] != sha(data.path) or data.values.shape[1] != VIEWS[name][0]:
            raise ValueError('Source conversion binding changed: '+name)
    required = [root/'boston_station_audit.json', root/'wind_field_audit.json']
    if not all(read(p)['passed'] for p in required):
        raise ValueError('Station or wind-field audit missing')
    # Stage22 read a few old PEMS04 holdout windows. Do not call overlapping
    # labels independently new confirmation data; exclude by IDs/windows only.
    stage22_manifest = Path(cfg['stage22_output'])/'manifest.json'
    exclusions = {n: {} for n in VIEWS}
    for row in read(stage22_manifest)['pools']['confirmation']:
        q = row['query']
        if q['source'] == 'pems04':
            for name in ('pems04_flow', 'pems04_speed'):
                exclusions[name].setdefault(str(q['variable']), []).append([q['origin'], q['origin']+q['horizon']])
    query_pools = pools(series, exclusions)
    audit = isolation(series, original)
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(protocol=cfg['protocol'], configuration=cfg,
        source_configuration=original['source_configuration'], source_raw_hashes=original['source_raw_hashes'],
        weights_sha256=original['weights_sha256'], weight_stats=original['weight_stats'],
        alphabet=original['alphabet'], probes=original['probes'],
        unchanged_tensor_files=original['unchanged_tensor_files'],
        source_hashes={n: sha(d.path) for n, d in series.items()},
        metadata_hashes={n: sha(d.path.with_suffix('.json')) for n, d in series.items()},
        audit_hashes={str(p): sha(p) for p in required}, isolation=audit,
        earlier_source_confirmation_manifest_sha256=sha(stage22_manifest),
        confirmation_excluded_prior_label_windows=exclusions,
        pools=query_pools, parent_update=3500, new_optimizer=True, exact_stage19_resume=False,
        trainable_parameters=1060998, target_predictions_allowed=False,
        code={p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
    if (output/'manifest.json').exists() and read(output/'manifest.json') != manifest:
        raise ValueError('Immutable Stage23 manifest changed')
    atomic(output/'manifest.json', manifest)
    if not (output/'parent.pt').exists():
        shutil.copyfile(Path(cfg['source_output'])/'sampled_best.pt', output/'parent.pt')
    if sha(output/'parent.pt') != ADAPTER_SHA:
        raise ValueError('Parent adapter copy changed')
    if not (output/'gpu_identity.json').exists():
        atomic(output/'gpu_identity.json', read(Path(cfg['stage21_output'])/'gpu_identity.json'))
    if not (output/'status.json').exists():
        atomic(output/'status.json', dict(state='prepared', phase='cpu_preparation', parent_update=3500,
            new_updates=0, target_evaluations=0))
    return manifest


def verify(cfg, output):
    manifest = read(Path(output)/'manifest.json')
    validate(cfg)
    if manifest['configuration'] != cfg or manifest['code'] != {
            p.name: sha(p) for p in Path(__file__).parent.glob('*.py')} \
            or sha(Path(output)/'parent.pt') != ADAPTER_SHA:
        raise ValueError('Stage23 execution binding changed')
    root = Path(cfg['data']['directory'])/'sources'
    for n in VIEWS:
        if sha(root/(n+'.npz')) != manifest['source_hashes'][n] \
                or sha(root/(n+'.json')) != manifest['metadata_hashes'][n]:
            raise ValueError('Source observation/metadata changed')
    for p, digest in manifest['audit_hashes'].items():
        if sha(p) != digest:
            raise ValueError('Prepared station/field evidence changed')
    weights = Path(manifest['source_configuration']['weights']['directory'])
    if sha(weights/'snapshot.json') != manifest['weights_sha256'] or any(
            (weights/n).stat().st_size != v['size'] or (weights/n).stat().st_mtime_ns != v['mtime_ns']
            for n, v in manifest['weight_stats'].items()):
        raise ValueError('Official backbone files changed')
    return manifest
