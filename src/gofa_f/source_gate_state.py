"""Immutable Stage22 source pools, cached forecast provenance and label barriers."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil

import numpy as np

from gofa_f.graph_benchmark import ADAPTER_SHA, atomic, read, sha
from gofa_f.source_gate import AMPLITUDES, SOURCES, TEMPLATES


TARGETS = ('pems03', 'pems_bay', 'meteonet_nw2018', 'engrad', 'sdwpf', 'kelmarsh2019')


def validate(cfg):
    expected_resources = dict(admission_policy='idle', admission_samples=3, resource_check_seconds=10,
        memory_cap_gib=40, torch_cap_gib=38, activity_limit_seconds=None, wall_deadline=None,
        yield_on_foreign=False, leave_free_gib=0)
    if cfg['protocol'] != 'stage22-source-reliability-gate-v1' or cfg['seed'] != 42 \
            or cfg['adapter_sha256'] != ADAPTER_SHA or cfg['amplitudes'] != list(AMPLITUDES) \
            or cfg['source_variables'] != 8 or cfg['source_templates'] != [list(t) for t in TEMPLATES] \
            or cfg['minimum_predicted_gain'] != .01 or cfg['maximum_source_score'] != .99 \
            or cfg['maximum_single_source_ratio'] != 1.02 or cfg['targets'] != list(TARGETS) \
            or cfg['neural_training_enabled'] or cfg['target_calibration_enabled'] \
            or cfg['resources'] != expected_resources or cfg['data']['target_training_enabled']:
        raise ValueError('Authorized Stage22 fixed pilot policy changed')


def parent(cfg):
    from gofa_f.short_sampled_state import exited
    stage21, source = Path(cfg['stage21_output']), Path(cfg['source_output'])
    for root in (stage21, source):
        exited(root)
        if read(root/'status.json')['state'] != 'complete':
            raise ValueError('Prior work must be complete and exited')
    if sha(stage21/'manifest.json') != cfg['stage21_manifest_sha256'] \
            or sha(source/'manifest.json') != cfg['source_manifest_sha256'] \
            or sha(source/'pools/metric_epsilons.json') != cfg['epsilon_sha256'] \
            or sha(source/'sampled_best.pt') != ADAPTER_SHA or sha(stage21/'adapter.pt') != ADAPTER_SHA:
        raise ValueError('Locked source selection/parent inputs changed')
    manifest = read(stage21/'manifest.json')
    if manifest['source_selection']['updates'] != 3500 \
            or manifest['source_selection']['checkpoint_sha256'] != ADAPTER_SHA:
        raise ValueError('Not the unique 3500 source-selected checkpoint')
    for name, digest in manifest['unchanged_tensor_files'].items():
        if sha(Path(__file__).parent/name) != digest:
            raise ValueError('Previously accepted numerical core changed: '+name)
    for name in ('short_generative_stage.py', 'short_sampling_diagnostic.py', 'short_sampled_forecast.py',
                 'short_source_diagnostic.py', 'prefix_cache.py'):
        if sha(Path(__file__).parent/name) != manifest['code'][name]:
            raise ValueError('Reused forecast execution changed: '+name)
    return manifest


def make_pools(series, pool):
    from gofa_f.generative_data import Query
    from gofa_f.short_data import origins
    fit, confirmation, selected = [], [], {}
    index = {(q['source'], q['variable'], q['length'], q['horizon']): (i, q)
             for i, q in enumerate(pool['validation'])}
    if len(index) != 2264 or len(pool['validation']) != 2264:
        raise ValueError('Original full source pool coverage changed')
    for s, name in enumerate(SOURCES):
        data = series[name]
        variables = np.random.default_rng(np.random.SeedSequence([42, 22, s, 0])).permutation(
            data.values.shape[1])[:8].tolist()
        selected[name] = variables
        for variable in variables:
            used = []
            for length, horizon in TEMPLATES:
                i, q = index[name, variable, length, horizon]
                fit.append(dict(index=len(fit), query=q, source_index=i))
                points = origins(data, variable, length, horizon, 'test', require_history=True)
                for a, b in used:
                    points = points[(points+horizon <= a) | (points >= b)]
                if not len(points):
                    raise ValueError('Cannot meet disjoint source confirmation coverage')
                rng = np.random.default_rng(np.random.SeedSequence([42, 22, s, variable, length, horizon, 1]))
                origin = int(rng.choice(points))
                used.append((origin, origin+horizon))
                confirmation.append(dict(index=len(confirmation),
                    query=asdict(Query(name, variable, origin, length, horizon))))
    return dict(fit=fit, confirmation=confirmation, selected_variables=selected)


def holdout_audit(output_parent, series):
    """Audit prior source train/validation pools and diagnostic source manifests.

    Evidence is restricted to recorded project lineage, not an assertion about
    unlogged computations or the official model's pretraining exposure.
    """
    files, violations = {}, []
    def queries(value):
        if isinstance(value, dict):
            if {'source', 'origin', 'horizon'} <= set(value):
                yield value
            else:
                for child in value.values():
                    yield from queries(child)
        elif isinstance(value, list):
            for child in value:
                yield from queries(child)
    for root in sorted(Path(output_parent).iterdir()):
        if not root.is_dir() or not root.name.startswith(('stage17-', 'stage18-', 'stage19-')):
            continue
        candidates = [root/'pools/queries.json']+sorted(root.glob('*manifest.json'))
        for path in candidates:
            if not path.exists():
                continue
            value = read(path)
            sections = {k: value[k] for k in ('train', 'validation') if k in value}
            if path.parent.name != 'pools':
                rows = value.get('queries', {})
                sections['diagnostics'] = ({k: v for k, v in rows.items() if k != 'targets'}
                                          if isinstance(rows, dict) else rows)
                sections['probes'] = value.get('probes', [])
            found = [q for q in queries(sections) if q['source'] in series]
            if found:
                files[str(path)] = dict(sha256=sha(path), queries=len(found))
            for q in found:
                if q['origin']+q['horizon'] > int(.8*len(series[q['source']].values)):
                    violations.append(dict(path=str(path), query=q))
    if not files or violations:
        raise ValueError('Source holdout isolation failed: '+json.dumps(violations[:5]))
    return dict(passed=True, audited_source_lineage_files=files, violations=[],
        scope='Recorded Stage17/18/19 train/validation pools and diagnostic manifests; not pretraining audit')


def cached_reference(root, label, index, query, manifest_digest, weight_digest, alphabet_digest):
    """Validate the eight original paths and derived median, including missing masks."""
    from gofa_f.short_sampling_diagnostic import SETTINGS, sample_seed
    directory = Path(root)/'evaluations'/label
    path = directory/f'{index:06d}.json'
    item = read(path)
    expected = f'{manifest_digest}:sample8:{label}:{weight_digest}:{alphabet_digest}'
    if item['binding'] != expected or item['query'] != query or item['sampling'] != SETTINGS \
            or item['seeds'] != [sample_seed(query, j) for j in range(8)]:
        raise ValueError('Cached source/target record binding differs')
    paths, predictions = [], []
    for sample in range(8):
        p = directory/'trajectories'/f'{index:06d}-{sample:02d}.json'
        value = read(p)
        if value['binding'] != expected or value['query'] != query or value['sample'] != sample \
                or value['seed'] != sample_seed(query, sample):
            raise ValueError('Cached trajectory binding differs')
        paths.append(dict(path=str(p), sha256=sha(p)))
        predictions.append(value['prediction'])
    if not np.array_equal(np.median(np.asarray(predictions, np.float64), axis=0), item['prediction']):
        raise ValueError('Cached median differs from eight trajectories')
    return dict(path=str(path), sha256=sha(path), trajectories=paths), item


def prepare(cfg, output):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU preparation must explicitly disable CUDA')
    validate(cfg)
    prior = parent(cfg)
    from gofa_f.generative_data import Series
    root = Path(prior['source_configuration']['data']['directory'])
    series = {s: Series(root/'sources'/(s+'.npz')) for s in SOURCES}
    for s in SOURCES:
        if sha(series[s].path) != prior['source_raw_hashes'][s]:
            raise ValueError('Original source data changed')
    source = Path(cfg['source_output'])
    if sha(source/'pools/queries.json') != read(source/'manifest.json')['source_pool_sha256']:
        raise ValueError('Full original source query pool differs')
    pools = make_pools(series, read(source/'pools/queries.json'))
    isolation = holdout_audit(source.parent, series)
    references = {}
    for row in pools['fit']:
        key = str(row['index'])
        references[key] = {}
        for label, alpha in (('sampled-native', 0.), ('sampled-update-003500', 1.)):
            ref, item = cached_reference(source, label, row['source_index'], row['query'],
                cfg['source_manifest_sha256'], prior['weights_sha256'] if not alpha else ADAPTER_SHA,
                prior['alphabet']['alphabet_sha256'])
            data, q = series[row['query']['source']], row['query']
            mask = data.mask[q['origin']:q['origin']+q['horizon'], q['variable']]
            y = data.values[q['origin']:q['origin']+q['horizon'], q['variable']]
            if not np.array_equal(mask, item['observed']) or not np.array_equal(y, item['target']):
                raise ValueError('Source cached labels/masks changed')
            references[key][str(alpha)] = ref
    manifest = dict(protocol=cfg['protocol'], configuration=cfg,
        source_configuration=prior['source_configuration'], source_raw_hashes=prior['source_raw_hashes'],
        source_metadata_hashes={s: sha(series[s].path.with_suffix('.json')) for s in SOURCES},
        weights_sha256=prior['weights_sha256'], weight_stats=prior['weight_stats'],
        unchanged_tensor_files=prior['unchanged_tensor_files'], alphabet=prior['alphabet'], probes=prior['probes'],
        pools=pools, source_references=references, source_holdout_audit=isolation,
        target_queries=prior['queries']['targets'], target_data_directory=prior['data_directory'],
        target_raw_hashes=prior['raw_hashes'], target_metadata_hashes=prior['metadata_hashes'],
        code={p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        maximum_new_numeric_work=167936, neural_optimizer_updates=0,
        target_label_fitting_allowed=False, prior_activity_is_reference_only=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'manifest.json').exists() and read(output/'manifest.json') != manifest:
        raise ValueError('Immutable Stage22 preparation binding changed')
    atomic(output/'manifest.json', manifest)
    if not (output/'adapter.pt').exists():
        shutil.copyfile(source/'sampled_best.pt', output/'adapter.pt')
    atomic(output/'gpu_identity.json', read(Path(cfg['stage21_output'])/'gpu_identity.json'))
    atomic(output/'status.json', dict(state='prepared', phase='cpu_preparation', neural_optimizer_updates=0))
    return manifest


def verify(cfg, output):
    validate(cfg)
    manifest = read(Path(output)/'manifest.json')
    if manifest['configuration'] != cfg or manifest['code'] != {
            p.name: sha(p) for p in Path(__file__).parent.glob('*.py')} \
            or sha(Path(output)/'adapter.pt') != ADAPTER_SHA:
        raise ValueError('Stage22 configuration/code/checkpoint changed')
    prior = parent(cfg)
    for path, record in manifest['source_holdout_audit']['audited_source_lineage_files'].items():
        if sha(path) != record['sha256']:
            raise ValueError('Audited source isolation evidence changed')
    root = Path(manifest['source_configuration']['data']['directory'])
    for s in SOURCES:
        if sha(root/'sources'/(s+'.npz')) != manifest['source_raw_hashes'][s] \
                or sha(root/'sources'/(s+'.json')) != manifest['source_metadata_hashes'][s]:
            raise ValueError('Source data/metadata changed')
    if prior['queries']['targets'] != manifest['target_queries']:
        raise ValueError('Frozen six-target query list changed')
    for name in TARGETS:
        p = Path(manifest['target_data_directory'])/'targets'/(name+'.npz')
        if sha(p) != manifest['target_raw_hashes'][name] or sha(p.with_suffix('.json')) != manifest['target_metadata_hashes'][name]:
            raise ValueError('Frozen target data/metadata changed')
    weights = Path(manifest['source_configuration']['weights']['directory'])
    if sha(weights/'snapshot.json') != manifest['weights_sha256'] or any(
            (weights/n).stat().st_size != v['size'] or (weights/n).stat().st_mtime_ns != v['mtime_ns']
            for n, v in manifest['weight_stats'].items()):
        raise ValueError('Official backbone changed')
    return manifest
