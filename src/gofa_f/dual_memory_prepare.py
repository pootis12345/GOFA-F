"""Seal both-shape source pools, metadata and graph sampling before model fitting."""
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from gofa_f.dual_memory_data import GROUPS, SHAPES, SAMPLER, memory_input, task_text
from gofa_f.generative_data import Query, Series, sha
from gofa_f.graph_memory_data import flatten_queries, fresh_queries
from gofa_f.prefix_cache import atomic_json
from gofa_f.source_graph_probe import digest, immutable, read


def source_text(data, name, length, horizon):
    info = data.meta.get('provenance', data.meta)
    # Explicit audited field meaning; no dataset-name embeddings or target tuning.
    quantities = {'pems04_flow': ('traffic flow', 'publisher flow unit (not independently verified)'),
        'pems04_speed': ('traffic speed', 'unknown (publisher speed unit not independently verified)'),
        'boston2017': ('air temperature', 'degrees Celsius'),
        'penmanshiel2017': ('active electrical power', 'kilowatts')}
    quantity, unit = quantities[name]
    times = np.asarray(data.times, np.int64)
    steps = np.diff(times)
    same_segment = data.segments[1:] == data.segments[:-1]
    values = np.unique(steps[same_segment])
    if len(values) != 1 or values[0] <= 0:
        raise ValueError('A verified regular within-segment sampling interval is required')
    interval = float(values[0])/1e9
    return task_text(quantity, unit, interval, length, horizon)


def training_pool(data, name, variables, length, horizon, count):
    variables = sorted(variables, key=lambda v: str(data.ids[v]))
    rng = np.random.default_rng(np.random.SeedSequence([42,59, list(GROUPS).index(name), length]))
    order = rng.permutation(variables).tolist()
    result = []; selected = {int(v): set() for v in variables}
    for index in range(count):
        variable = int(order[index % len(order)])
        candidates = data.candidates(variable, length, horizon, 'train')
        # Series' older token-chunk eligibility is tightened to full-H here.
        candidates = candidates[(data.prefix_counts[candidates+horizon, variable]-data.prefix_counts[candidates, variable]) > 0]
        candidates = np.array([x for x in candidates if int(x) not in selected[variable]])
        if not len(candidates):
            raise ValueError('Fixed source training pool lacks lawful unique queries')
        number_of_strata = max(1, (count+len(order)-1)//len(order))
        stratum = min(index//len(order), number_of_strata-1)
        # Fixed time stratum with an explicit empty-bin failure, no error access.
        block = np.array_split(candidates, min(number_of_strata, len(candidates)))[min(stratum, min(number_of_strata,len(candidates))-1)]
        origin = int(rng.choice(block)); selected[variable].add(origin)
        result.append(asdict(Query(name, variable, origin, length, horizon)))
    return result


def prepare(cfg, output):
    if torch.cuda.is_initialized() or __import__('os').environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Preparation must remain CPU-only')
    started = time.monotonic(); parent = Path(cfg['source_output'])
    if sha(parent/'manifest.json') != cfg['source_manifest_sha256']:
        raise ValueError('Audited source manifest changed')
    old = read(parent/'manifest.json'); partitions = read(Path(cfg['partition_output'])/'manifest.json')['partition']
    directory = Path(old['configuration']['data']['directory'])/'sources'
    exclusions = []
    for key in ('validation','confirmation','diagnostic'):
        exclusions.extend(flatten_queries(old['pools'][key]))
    reserved = read(Path(cfg['reservation_output'])/'manifest.json')
    if reserved is None:
        raise ValueError('Prior unopened source reservation manifest required')
    for split in reserved['queries'].values():
        exclusions.extend(split['development']); exclusions.extend(split['reserved'])
    queries = {}; source_hashes = {}; metadata_hashes = {}; texts = {}; rosters = {}; losses = {}
    for name in GROUPS:
        data = Series(directory/(name+'.npz'))
        if sha(data.path) != old['source_hashes'][name] or sha(data.path.with_suffix('.json')) != old['metadata_hashes'][name]:
            raise ValueError('Audited raw source or metadata changed')
        if name.startswith('pems04'):
            data.external = np.maximum(data.external, data.external.T)
            own = partitions
        else:
            own = {k:list(range(len(data.ids))) for k in ('train','validation','confirmation')}
        source_hashes[name] = sha(data.path); metadata_hashes[name] = sha(data.path.with_suffix('.json'))
        queries[name] = {}; texts[name] = {}; rosters[name] = []
        # Training-only unit normalization; no source/target holdout statistics.
        end = int(.6*len(data.values)); delta = np.abs(np.diff(data.values[:end].astype(np.float64),axis=0))
        valid = data.mask[1:end]&data.mask[:end-1]&np.isfinite(delta)
        mean = np.divide(np.where(valid,delta,0).sum(0),valid.sum(0),out=np.zeros(delta.shape[1]),where=valid.sum(0)>0)
        losses[name] = float(max(mean[own['train']].mean(),1e-6))
        for length,horizon in SHAPES:
            shape = f'{length}-{horizon}'
            train = training_pool(data,name,own['train'],length,horizon,cfg['queries']['training_per_view']//2)
            development = fresh_queries(data,name,length,horizon,'validation',cfg['queries']['validation_per_shape_view'],own['validation'],exclusions,59)
            confirmation = fresh_queries(data,name,length,horizon,'test',cfg['queries']['confirmation_per_shape_view'],own['confirmation'],exclusions+development,59)
            exclusions.extend(development); exclusions.extend(confirmation)
            queries[name][shape] = dict(train=train,development=development,confirmation=confirmation)
            texts[name][shape] = source_text(data,name,length,horizon)
            # Profile each task shape, preserving exact real31 cap and node masks.
            for row in train[:8]:
                item = memory_input(data,Query(**row),own['train'])
                rosters[name].append(dict(query=row,input_sha256=item.identity,**item.audit))
        print(json.dumps(dict(state='preparing',source=name,training_queries=cfg['queries']['training_per_view'])),flush=True)
    files = ('dual_memory_data.py','dual_memory_model.py','dual_memory_prepare.py','dual_memory_run.py','ufg_memory_roster.py')
    manifest = dict(protocol=cfg['protocol'],configuration=cfg,source_directory=str(directory),
        source_hashes=source_hashes,metadata_hashes=metadata_hashes,source_configuration=old['source_configuration'],
        weights_sha256=old['weights_sha256'],groups=GROUPS,queries=queries,texts=texts,
        partitions={n: partitions if n.startswith('pems04') else {k:list(range(len(Series(directory/(n+'.npz')).ids))) for k in ('train','validation','confirmation')} for n in GROUPS},
        sampler=SAMPLER,sampling_profiles=rosters,training_unit_normalizers=losses,
        source_confirmation_labels_opened=False,target_labels_opened=False,target_scores_used=False,
        old_source_development_exposure=True,old_target_development_exposure=True,
        prior_reservations_sha256=sha(Path(cfg['reservation_output'])/'manifest.json'),
        code={f:sha(Path(__file__).parent/f) for f in files},
        scientific_initialization='fresh additions; pretrained frozen ChatTime only; no old graph adapter',
        parameter_selection='source-only; PEMS04 channels held out together',
        graph_disclosure='publisher bidirectional traffic; coordinate4NN symmetric proxies for weather/wind')
    output.mkdir(parents=True,exist_ok=True); immutable(output/'manifest.json',manifest)
    atomic_json(output/'preparation.json',dict(manifest_sha256=sha(output/'manifest.json'),
        cpu_seconds=time.monotonic()-started,formal_updates=0,target_predictions=0,passed=True))
    atomic_json(output/'status.json',dict(state='prepared',phase='dual_memory_sources',source_confirmation_opened=False,
        target_predictions=0,training_updates=0,goal_achieved=False))
    return manifest
