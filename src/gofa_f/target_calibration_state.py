"""Immutable Stage20 validation identities and inherited Stage19 test archive."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from gofa_f.allnode_stage import read
from gofa_f.generative_data import Query, Series, sha
from gofa_f.prefix_cache import atomic_json
from gofa_f.short_data import TARGETS, origins, stratified
from gofa_f.short_sampled_state import exited
from gofa_f.short_sampling_diagnostic import SETTINGS


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def code_hashes():
    return {p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def validation_queries(series, seed=42):
    queries, coverage = {}, {}
    for dataset, (name, (length, horizon, nodes)) in enumerate(TARGETS.items()):
        data = series[name]
        variables = [104] if name == 'nasdaq100' else list(range(nodes))
        rng = np.random.default_rng(np.random.SeedSequence([seed, dataset, 20, 128]))
        selected = rng.permutation(variables).tolist()[:128]
        allocation = [selected[i % len(selected)] for i in range(128)]
        a, b = int(.6*len(data.values)), int(.8*len(data.values))
        cuts = [a + i*(b-a)//4 for i in range(5)]
        rows = []
        for quarter in range(4):
            own = allocation[32*quarter:32*(quarter+1)]
            for variable in dict.fromkeys(own):
                points = origins(data, variable, length, horizon, 'validation',
                    require_history=False, calendar_breaks=name != 'exchange')
                points = points[(points >= cuts[quarter]) & (points+horizon <= cuts[quarter+1])]
                own_rng = np.random.default_rng(np.random.SeedSequence([seed, dataset, 20, 128, quarter, variable]))
                for origin in stratified(points, own.count(variable), horizon, own_rng, disjoint=True):
                    rows.append(asdict(Query(name, variable, int(origin), length, horizon)))
            if len(rows) != (quarter+1)*32:
                raise ValueError('Each temporal quarter must have exactly 32 queries')
        if len({Query(**q).key for q in rows}) != 128:
            raise ValueError('Validation queries must be distinct')
        queries[name] = rows
        coverage[name] = dict(queries=128,fit_queries=96,confirmation_queries=32,
            selected_variables=selected,available_target_variables=len(variables),
            selected_variable_count=len(selected),quarter_boundaries=cuts,
            history=length,horizon=horizon,fit_quarters=[0,1,2],confirmation_quarters=[3])
    return queries, coverage


def check_config(cfg, prior):
    if cfg['protocol'] != 'stage20-target-calibration-v1' or cfg['seed'] != 42 \
            or cfg['validation_queries_per_dataset'] != 128 or cfg['fit_queries'] != 96 \
            or cfg['confirmation_queries'] != 32 or cfg['sampling'] != SETTINGS \
            or cfg['training'] != {'max_updates': 4000} \
            or cfg['data'] != dict(target_training_enabled=False,target_evaluation_enabled=True,
                target_label_calibration_enabled=True,strict_zero_shot=False) \
            or cfg['resources'] != prior['configuration']['resources']:
        raise ValueError('Fixed Stage20 calibration protocol or resource policy differs')


def prepare(cfg, output):
    parent = Path(cfg['stage19_output'])
    exited(parent)
    if output.resolve() == parent.resolve() or sha(parent/'manifest.json') != cfg['stage19_manifest_sha256'] \
            or sha(parent/'report.json') != cfg['stage19_report_sha256']:
        raise ValueError('Independent output and pinned complete Stage19 archive required')
    prior = read(parent/'manifest.json')
    check_config(cfg, prior)
    if not read(parent/'report.json')['complete'] or not read(parent/'mechanisms.json')['passed'] \
            or sha(parent/'adapter.pt') != cfg['adapter_sha256']:
        raise ValueError('Complete accepted Stage19 checkpoint required')
    source = Path(prior['configuration']['source_output'])
    exited(source)
    if sha(source/'source_final_selection.json') != prior['configuration']['source_selection_sha256'] \
            or read(source/'source_final_selection.json')['updates'] != 3500:
        raise ValueError('Source-only selected checkpoint changed')
    code = code_hashes()
    inherited = prior['unchanged_tensor_files'] | {
        n: prior['code'][n] for n in ('short_sampled_targets.py','short_target_progress.py')}
    if any(code.get(n) != value for n, value in inherited.items()):
        raise ValueError('Accepted numerical execution or sampling changed')
    cpu = read(parent/'cpu_preflight.json')
    if not cpu['passed'] or cpu['manifest_sha256'] != cfg['stage19_manifest_sha256']:
        raise ValueError('Bound prior CPU proof missing')
    root = Path(prior['source_configuration']['data']['directory'])
    data = {name: Series(root/'targets'/(name+'.npz')) for name in TARGETS}
    for name, item in data.items():
        if sha(item.path) != prior['raw_hashes'][name] \
                or sha(item.path.with_suffix('.json')) != prior['metadata_hashes'][name] \
                or item.values.shape[1] != TARGETS[name][2]:
            raise ValueError('Audited raw data or variable order changed')
    queries, coverage = validation_queries(data, cfg['seed'])
    # Hash old test files without using their predictions/errors to choose
    # validation identities or coefficients. Fitting never opens these files.
    archive, results = {}, {}
    for name, rows in prior['queries']['targets'].items():
        archive[name], results[name] = {}, {}
        for mode in ('native','gofa'):
            label = f'target-{name}-{mode}'
            archive[name][mode] = [sha(parent/'evaluations'/label/f'{i:06d}.json') for i in range(len(rows))]
            results[name][mode] = sha(parent/(label+'-result.json'))
    manifest = dict(configuration=cfg,code=code,source_configuration=prior['source_configuration'],
        weights_sha256=prior['weights_sha256'],weight_stats=prior['weight_stats'],
        alphabet=prior['alphabet'],source_raw_hashes=prior['source_raw_hashes'],probes=prior['probes'],
        source_selection=prior['source_selection'],source_output=str(source),
        source_selection_sha256=prior['configuration']['source_selection_sha256'],
        raw_hashes=prior['raw_hashes'],metadata_hashes=prior['metadata_hashes'],
        inherited_numerical_files=inherited,prior_cpu_proof_sha256=sha(parent/'cpu_preflight.json'),
        queries=queries,coverage=coverage,test_queries=prior['queries']['targets'],
        test_query_hashes=archive,test_result_hashes=results,
        total_numeric_work=16*sum(q['horizon'] for rows in queries.values() for q in rows),
        validation_queries=1152,validation_model_queries=2304,validation_trajectories=18432,
        optimizer_updates=0,graph_alpha=1.,strict_zero_shot=False,
        target_test_results_already_seen=True,fit_reads_test_records=False,
        objective='original-unit observed-point MAE within variable, then equal variables',
        prediction_rule='convex fusion after independent sample8 pointwise medians')
    if read(output/'manifest.json', manifest) != manifest:
        raise ValueError('Immutable Stage20 manifest changed')
    atomic_json(output/'manifest.json', manifest)
    if not (output/'adapter.pt').exists():
        shutil.copyfile(parent/'adapter.pt', output/'adapter.pt')
    if sha(output/'adapter.pt') != cfg['adapter_sha256']:
        raise ValueError('Never overwrite a different destination adapter')
    atomic_json(output/'gpu_identity.json', read(parent/'gpu_identity.json'))
    return manifest


def own_manifest(cfg, output):
    """Local validation fitting needs no test directory or raw test labels."""
    value = read(output/'manifest.json')
    if value is None or value['configuration'] != cfg or value['code'] != code_hashes():
        raise ValueError('Stage20 code/config/manifest binding differs')
    return value


def verify_inference(cfg, output):
    manifest = own_manifest(cfg, output)
    parent = Path(cfg['stage19_output'])
    exited(parent)
    source = Path(manifest['source_output'])
    exited(source)
    if sha(parent/'manifest.json') != cfg['stage19_manifest_sha256'] \
            or sha(parent/'report.json') != cfg['stage19_report_sha256'] \
            or sha(output/'adapter.pt') != cfg['adapter_sha256'] \
            or sha(parent/'adapter.pt') != cfg['adapter_sha256'] \
            or sha(source/'source_final_selection.json') != manifest['source_selection_sha256']:
        raise ValueError('Inherited source checkpoint or Stage19 archive changed')
    root = Path(manifest['source_configuration']['data']['directory'])
    for name in TARGETS:
        if sha(root/'targets'/(name+'.npz')) != manifest['raw_hashes'][name] \
                or sha(root/'targets'/(name+'.json')) != manifest['metadata_hashes'][name]:
            raise ValueError('Target observations or variable metadata changed')
    weights = Path(manifest['source_configuration']['weights']['directory'])
    if sha(weights/'snapshot.json') != manifest['weights_sha256'] or any(
            (weights/n).stat().st_size != s['size'] or (weights/n).stat().st_mtime_ns != s['mtime_ns']
            for n,s in manifest['weight_stats'].items()):
        raise ValueError('Official frozen backbone changed')
    return manifest


def result_binding(manifest, binding, name, mode):
    weight = manifest['weights_sha256'] if mode == 'native' else manifest['configuration']['adapter_sha256']
    return f'{binding}:sample8:validation-{name}-{mode}:{weight}:'+manifest['alphabet']['alphabet_sha256']
