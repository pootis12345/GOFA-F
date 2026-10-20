"""Immutable Stage21 v2 data, source checkpoint and label-blind query bindings."""
from dataclasses import asdict
import os
from pathlib import Path
import shutil

import numpy as np

from gofa_f.graph_benchmark import ADAPTER_SHA, atomic, read, sha, stage20_stopped
from gofa_f.graph_benchmark_data import TARGETS, convert_all


def validate_config(cfg):
    expected = {n: dict(domain=d, nodes=c, history=length, horizon=h)
                for n, (d, c, length, h) in TARGETS.items()}
    resources = dict(admission_policy='idle', admission_samples=3, resource_check_seconds=10,
        memory_cap_gib=40, torch_cap_gib=38, activity_limit_seconds=None, wall_deadline=None,
        yield_on_foreign=False, leave_free_gib=0)
    sampling = dict(samples=8, top_k=100, temperature=1., top_p=1., seed=42,
        aggregate='pointwise_median', execution='serial_single_query_independent_kv')
    if cfg['protocol'] != 'stage21-graph-benchmark-v2' or cfg['seed'] != 42 \
            or cfg['datasets'] != expected or cfg['queries_per_dataset'] != 128 \
            or cfg['adapter_sha256'] != ADAPTER_SHA or cfg['conditions'] != ['native', 'gofa'] \
            or cfg['training_enabled'] or cfg['calibration_enabled'] \
            or cfg['data'] != dict(target_training_enabled=False, target_evaluation_enabled=True) \
            or cfg['training'] != dict(max_updates=4000) or cfg['sampling'] != sampling:
        raise ValueError('Authorized MeteoNet replacement/six-target protocol changed')
    if cfg['resources'] != resources:
        raise ValueError('Fully-idle admission/resource policy changed')


def query_pool(series, seed=42, budget=128):
    """IDs + temporal strata + observed-label eligibility only; never read errors."""
    from gofa_f.generative_data import Query
    from gofa_f.short_data import origins, stratified
    queries, coverage = {}, {}
    for index, (name, (_, nodes, length, horizon)) in enumerate(TARGETS.items()):
        data = series[name]
        n = data.values.shape[1]
        if nodes is not None and n != nodes:
            raise ValueError('Original sensor count changed: '+name)
        rng = np.random.default_rng(np.random.SeedSequence([seed, index, 21, 2]))
        eligible_by_variable = {v: origins(data, v, length, horizon, 'test', require_history=False)
                                for v in range(n)}
        eligible_variables = [v for v in range(n) if len(eligible_by_variable[v]) >= (budget+n-1)//n]
        if len(eligible_variables) < min(n, budget):
            raise ValueError('Insufficient target variables with legal observed labels: '+name)
        selected = rng.permutation(eligible_variables).tolist()[:min(n, budget)]
        own = []
        for rank, variable in enumerate(selected):
            count = budget//len(selected)+int(rank < budget % len(selected))
            eligible = eligible_by_variable[variable]
            for origin in stratified(eligible, count, horizon, rng, disjoint=True):
                own.append(asdict(Query(name, variable, int(origin), length, horizon)))
        if len(own) != budget or len({tuple(q.values()) for q in own}) != budget:
            raise ValueError('Incomplete/duplicate fixed target queries')
        queries[name] = own
        coverage[name] = dict(queries=budget, all_nodes=n, selected_target_variables=selected,
            variable_ids=[data.ids[v] for v in selected], variables=len(selected),
            min_origins_per_variable=budget//len(selected),
            max_origins_per_variable=(budget+len(selected)-1)//len(selected),
            eligible_variables=len(eligible_variables),
            ineligible_variable_ids=[data.ids[v] for v in range(n) if v not in eligible_variables],
            history=length, horizon=horizon, auxiliary_nodes='all original nodes eligible',
            test_begin_index=int(.8*len(data.values)), label_windows_disjoint_within_variable=True)
    return dict(targets=queries, coverage=coverage,
        selection='seed42 ID permutation and temporal strata; no forecast/error selection')


def source_parent(cfg):
    stage20_stopped(cfg['stage20_output'])
    parent = Path(cfg['stage19_output'])
    if sha(parent/'manifest.json') != cfg['stage19_manifest_sha256'] \
            or sha(parent/'report.json') != cfg['stage19_report_sha256'] \
            or sha(parent/'adapter.pt') != ADAPTER_SHA:
        raise ValueError('Locked Stage19 checkpoint/parent provenance changed')
    from gofa_f.short_sampled_state import exited, TENSOR_FILES
    exited(parent)
    prior = read(parent/'manifest.json')
    source = Path(prior['configuration']['source_output'])
    exited(source)
    if sha(source/'source_final_selection.json') != prior['configuration']['source_selection_sha256'] \
            or sha(source/'sampled_best.pt') != ADAPTER_SHA \
            or prior['source_selection']['updates'] != 3500:
        raise ValueError('Unique source-only selection no longer matches')
    unchanged = TENSOR_FILES+('short_sampled_forecast.py', 'short_sampling_diagnostic.py')
    for name in unchanged:
        if sha(Path(__file__).parent/name) != prior['unchanged_tensor_files'][name]:
            raise ValueError('Accepted numerical execution changed: '+name)
    for probe in prior['probes']:
        for ref in probe['references'].values():
            if sha(ref['path']) != ref['sha256']:
                raise ValueError('Source replay reference changed')
    return prior


def isolation_audit(series, prior):
    """Entity/geography/time provenance, not an error-based numerical screen."""
    from gofa_f.generative_data import Series
    root = Path(prior['source_configuration']['data']['directory'])
    sources = {}
    for name, digest in prior['source_raw_hashes'].items():
        path = root/'sources'/(name+'.npz')
        if sha(path) != digest:
            raise ValueError('Audited source observations changed')
        data = Series(path)
        sources[name] = dict(start=data.meta['start'], train_end=str(
            np.datetime64(int(data.times[int(.6*len(data.values))-1]), 'ns')),
            end=data.meta['end'], sha256=digest, provenance=data.meta['provenance'])
    # Distinct traffic periods prove no identical source-training observations;
    # they do not prove that physical sensor identities never recur.
    pems_start = np.datetime64(sources['pems04']['start'])
    pems_end = np.datetime64(sources['pems04']['end'])
    for name in ('pems03', 'pems_bay'):
        data = series[name]
        if not (np.datetime64(data.meta['end']) < pems_start or np.datetime64(data.meta['start']) > pems_end):
            raise ValueError('Traffic source/target observation periods overlap')
    if series['meteonet_nw2018'].meta['provenance']['unit'] != 'Celsius':
        raise ValueError('MeteoNet original Kelvin conversion missing')
    return dict(passed=True, sources=sources,
        targets={n: dict(start=d.meta['start'], end=d.meta['end'], provenance=d.meta['provenance'])
                 for n, d in series.items()},
        evidence=['PeMS04 Jan-Feb2018 source and PeMS03 Sep-Nov2018/BAY2017 target periods disjoint',
            'MeteoNet France2018 and EngRAD England2018-2020 are not Beijing Shunyi2013-2017 observations',
            'Chinese wind2020-2021 and UK wind2019 are not US Solar2006 or NN5 bank withdrawals',
            'AQI36 rejected and excluded; no Shunyi aliases retained'],
        limitations=['Traffic sensors may represent recurring entities at other times',
            'ChatTime pretraining exposure cannot be independently verified',
            'Original releases may already contain preprocessing; exposed in each provenance'])


def prepare(cfg, output, convert=True):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU preparation must explicitly disable CUDA')
    validate_config(cfg)
    prior = source_parent(cfg)
    output, root = Path(output), Path(cfg['data_directory'])
    if convert:
        convert_all(root)
    from gofa_f.generative_data import Series
    series = {n: Series(root/'targets'/(n+'.npz')) for n in TARGETS}
    audit = isolation_audit(series, prior)
    pools = query_pool(series)
    code = {p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    manifest = dict(configuration=cfg, source_configuration=prior['source_configuration'],
        weights_sha256=prior['weights_sha256'], weight_stats=prior['weight_stats'],
        source_selection=prior['source_selection'], source_raw_hashes=prior['source_raw_hashes'],
        probes=prior['probes'], alphabet=prior['alphabet'],
        unchanged_tensor_files=prior['unchanged_tensor_files'], code=code,
        data_directory=str(root), raw_hashes={n: sha(d.path) for n,d in series.items()},
        metadata_hashes={n: sha(d.path.with_suffix('.json')) for n,d in series.items()},
        queries=pools, isolation_audit=audit, optimizer_updates=0, training_allowed=False,
        calibration_allowed=False, total_paired_queries=768, total_model_queries=1536,
        total_trajectories=12288, total_numeric_work=16*sum(
            q['horizon'] for rows in pools['targets'].values() for q in rows),
        interpretation='Frozen source-selected checkpoint; six targets cannot select model, protocol or graph rules')
    if (output/'manifest.json').exists() and read(output/'manifest.json') != manifest:
        raise ValueError('Immutable Stage21 manifest changed; never overwrite a prepared experiment')
    if (output/'adapter.pt').exists() and sha(output/'adapter.pt') != ADAPTER_SHA:
        raise ValueError('Destination adapter changed')
    atomic(output/'manifest.json', manifest)
    if not (output/'adapter.pt').exists():
        shutil.copyfile(Path(cfg['stage19_output'])/'adapter.pt', output/'adapter.pt')
    atomic(output/'gpu_identity.json', read(Path(cfg['stage19_output'])/'gpu_identity.json'))
    atomic(output/'audit.json', dict(passed=True, protocol=cfg['protocol'],
        source_target_isolation=audit, adapter_sha256=ADAPTER_SHA, optimizer_updates=0,
        target_predictions=0, replacement='AQI36 -> MeteoNet NW2018; explicitly authorized'))
    atomic(output/'status.json', dict(state='prepared', optimizer_updates=0, target_predictions=0))
    return manifest


def verify(cfg, output):
    validate_config(cfg)
    manifest = read(Path(output)/'manifest.json')
    source_parent(cfg)
    code = {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')}
    if manifest['configuration'] != cfg or manifest['code'] != code \
            or sha(Path(output)/'adapter.pt') != ADAPTER_SHA:
        raise ValueError('Stage21 code/config/adapter binding changed')
    root = Path(cfg['data_directory'])
    for name in TARGETS:
        if sha(root/'targets'/(name+'.npz')) != manifest['raw_hashes'][name] \
                or sha(root/'targets'/(name+'.json')) != manifest['metadata_hashes'][name]:
            raise ValueError('Stage21 target data/metadata binding changed')
    weights = Path(manifest['source_configuration']['weights']['directory'])
    if sha(weights/'snapshot.json') != manifest['weights_sha256'] or any(
            (weights/n).stat().st_size != v['size'] or (weights/n).stat().st_mtime_ns != v['mtime_ns']
            for n,v in manifest['weight_stats'].items()):
        raise ValueError('Official backbone changed')
    return manifest
