"""Stage26 immutable six-target query and inference provenance bindings."""
from dataclasses import asdict
import os
from pathlib import Path
import shutil

import numpy as np

from gofa_f.allnode_stage import read
from gofa_f.generative_data import Query, Series, sha
from gofa_f.graph_benchmark_data import TARGETS
from gofa_f.prefix_cache import atomic_json
from gofa_f.short_data import origins, stratified
from gofa_f.short_sampled_state import exited
from gofa_f.short_sampling_diagnostic import SETTINGS
from gofa_f.target_calibration_state import code_hashes

ADAPTER = '816d38b577a6664f30dd6e6b6d50986a77b098ce9c4c0b0ffb63d70078b26f15'
SLICES = {'fit': [0, 48], 'pilot': [48, 64], 'confirm': [64, 128]}


def immutable(path, value):
    if path.exists() and read(path) != value:
        raise ValueError('Sealed artifact changed: '+str(path))
    atomic_json(path, value)
    return value


def validate_config(cfg):
    resources = dict(admission_policy='memory_cap_only', admission_samples=1,
        resource_check_seconds=10, memory_cap_gib=40, torch_cap_gib=38,
        activity_limit_seconds=None, wall_deadline=None, yield_on_foreign=False, leave_free_gib=0)
    if cfg['protocol'] != 'stage26-six-target-calibration-v1' or cfg['seed'] != 42 \
            or cfg['adapter_sha256'] != ADAPTER or cfg['targets'] != list(TARGETS) \
            or cfg['sampling'] != SETTINGS or cfg['training'] != {'max_updates': 0} \
            or cfg['resources'] != resources or cfg['query_counts'] != {'fit': 48, 'pilot': 16, 'confirm': 64} \
            or cfg['data'] != dict(target_training_enabled=False, target_evaluation_enabled=True,
                target_label_calibration_enabled=True, strict_zero_shot=False):
        raise ValueError('Fixed Stage26 protocol changed')


def query_pool(series, seed=42):
    queries, coverage = {}, {}
    for index, (name, (_, nodes, length, horizon)) in enumerate(TARGETS.items()):
        data = series[name]
        n = data.values.shape[1]
        if nodes is not None and nodes != n:
            raise ValueError('Original node count changed: '+name)
        begin, end = int(.6*len(data.values)), int(.8*len(data.values))
        cuts = [begin+(end-begin)*i//5 for i in range(6)]
        pools = {}
        for v in range(n):
            points = origins(data, v, length, horizon, 'validation', require_history=True)
            blocks = [points[(points >= cuts[k]) & (points+horizon <= cuts[k+1])] for k in range(5)]
            if all(len(p) >= (11 if n == 6 and k == 4 else 3 if n == 6 else 1)
                   for k, p in enumerate(blocks)):
                pools[v] = blocks
        if len(pools) < min(128, n):
            raise ValueError('Insufficient legal variables across five blocks: '+name)
        rng = np.random.default_rng(np.random.SeedSequence([seed, 26, index]))
        selected = rng.permutation(sorted(pools)).tolist()[:min(128, n)]
        allocation = [selected[i % len(selected)] for i in range(128)]
        rows, offset = [], 0
        for block, count in enumerate((16, 16, 16, 16, 64)):
            own = allocation[offset:offset+count]
            # Keep the preassigned variable order; each variable's starts are stratified.
            assigned = {}
            for v in dict.fromkeys(own):
                vrng = np.random.default_rng(np.random.SeedSequence([seed, 26, index, block, v]))
                assigned[v] = iter(stratified(pools[v][block], own.count(v), horizon, vrng, disjoint=True))
            rows.extend(asdict(Query(name, v, int(next(assigned[v])), length, horizon)) for v in own)
            offset += count
        if len({Query(**q).key for q in rows}) != 128:
            raise ValueError('Duplicate calibration identities')
        for v in selected:
            times = sorted(q['origin'] for q in rows if q['variable'] == v)
            if len(times) > 1 and np.any(np.diff(times) < horizon):
                raise ValueError('Overlapping same-variable label windows')
        queries[name] = rows
        coverage[name] = dict(selected_variables=selected, variable_ids=[str(data.ids[v]) for v in selected],
            all_nodes=n, eligible_variables=len(pools), boundaries=cuts, queries=128,
            fit_queries=48, pilot_queries=16, independent_confirmation_queries=64,
            missing_eligibility='at least one observed history and target; contiguous segment',
            label_windows_disjoint=True, selection='seed42 IDs/time strata only; never predictions or errors')
    return queries, coverage


def prepare(cfg, output):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Preparation requires CUDA disabled')
    validate_config(cfg)
    parent = Path(cfg['stage21_output'])
    exited(parent)
    if output.resolve() == parent.resolve() or sha(parent/'manifest.json') != cfg['stage21_manifest_sha256'] \
            or sha(parent/'report.json') != cfg['stage21_report_sha256'] \
            or sha(parent/'adapter.pt') != ADAPTER or not read(parent/'report.json')['complete']:
        raise ValueError('Pinned complete Stage21 archive required')
    prior = read(parent/'manifest.json')
    proof = read(parent/'cpu_preflight.json')
    if not proof['passed'] or proof['manifest_sha256'] != cfg['stage21_manifest_sha256']:
        raise ValueError('Prior bound CPU acceptance missing')
    code = code_hashes()
    if any(code.get(n) != value for n, value in prior['unchanged_tensor_files'].items()):
        raise ValueError('Accepted numerical core changed')
    root = Path(prior['data_directory'])
    series = {name: Series(root/'targets'/(name+'.npz')) for name in TARGETS}
    for name, data in series.items():
        if sha(data.path) != prior['raw_hashes'][name] \
                or sha(data.path.with_suffix('.json')) != prior['metadata_hashes'][name]:
            raise ValueError('Audited target observations/metadata changed')
    queries, coverage = query_pool(series)
    test_hashes = {}
    # Provenance hashes only; no test numerical records are loaded here.
    for name in TARGETS:
        for mode in ('native', 'gofa'):
            label = f'target-{name}-{mode}'
            paths = [parent/(label+'-result.json'), parent/'evaluations'/label/'complete.json']
            paths += sorted((parent/'evaluations'/label).glob('[0-9]*.json'))
            paths += sorted((parent/'evaluations'/label/'trajectories').glob('*.json'))
            if len(paths) != 2+128*9:
                raise ValueError('Frozen test archive incomplete: '+label)
            test_hashes.update({str(p.relative_to(parent)): sha(p) for p in paths})
    manifest = dict(configuration=cfg, code=code, data_directory=str(root), queries=queries,
        coverage=coverage, slices=SLICES, test_queries=prior['queries']['targets'],
        test_hashes=test_hashes, weights_sha256=prior['weights_sha256'], weight_stats=prior['weight_stats'],
        source_configuration=prior['source_configuration'], alphabet=prior['alphabet'],
        source_selection=prior['source_selection'], source_raw_hashes=prior['source_raw_hashes'],
        probes=prior['probes'], unchanged_tensor_files=prior['unchanged_tensor_files'],
        raw_hashes=prior['raw_hashes'], metadata_hashes=prior['metadata_hashes'],
        inherited_cpu_proof_sha256=sha(parent/'cpu_preflight.json'),
        total_numeric_work=16*sum(q['horizon'] for rows in queries.values() for q in rows),
        optimizer_updates=0, strict_zero_shot=False, graph_alpha=1.,
        test_results_previously_seen=True, neural_training_allowed=False)
    immutable(output/'manifest.json', manifest)
    if not (output/'adapter.pt').exists():
        shutil.copyfile(parent/'adapter.pt', output/'adapter.pt')
    if sha(output/'adapter.pt') != ADAPTER:
        raise ValueError('Destination adapter changed')
    immutable(output/'gpu_identity.json', read(parent/'gpu_identity.json'))
    if not (output/'status.json').exists():
        atomic_json(output/'status.json', dict(state='prepared', optimizer_updates=0))
    return manifest


def own_manifest(cfg, output):
    validate_config(cfg)
    value = read(output/'manifest.json')
    if value is None or value['configuration'] != cfg or value['code'] != code_hashes():
        raise ValueError('Code/config/manifest binding differs')
    return value


def verify_inference(cfg, output):
    manifest = own_manifest(cfg, output)
    parent = Path(cfg['stage21_output'])
    if sha(parent/'manifest.json') != cfg['stage21_manifest_sha256'] \
            or sha(parent/'report.json') != cfg['stage21_report_sha256'] \
            or sha(output/'adapter.pt') != ADAPTER or sha(parent/'adapter.pt') != ADAPTER:
        raise ValueError('Frozen parent archive changed')
    root = Path(manifest['data_directory'])
    for name in TARGETS:
        if sha(root/'targets'/(name+'.npz')) != manifest['raw_hashes'][name] \
                or sha(root/'targets'/(name+'.json')) != manifest['metadata_hashes'][name]:
            raise ValueError('Target observations or metadata changed')
    weights = Path(manifest['source_configuration']['weights']['directory'])
    if sha(weights/'snapshot.json') != manifest['weights_sha256'] or any(
            (weights/n).stat().st_size != s['size'] or (weights/n).stat().st_mtime_ns != s['mtime_ns']
            for n, s in manifest['weight_stats'].items()):
        raise ValueError('Frozen backbone changed')
    return manifest


def label(stage, name, mode):
    if stage not in ('pilot', 'confirm') or name not in TARGETS or mode not in ('native', 'gofa'):
        raise ValueError('Invalid validation identity')
    return f'{stage}-{name}-{mode}'


def result_binding(manifest, binding, phase, name, mode):
    weights = manifest['weights_sha256'] if mode == 'native' else ADAPTER
    return f'{binding}:sample8:{label(phase, name, mode)}:{weights}:'+manifest['alphabet']['alphabet_sha256']
