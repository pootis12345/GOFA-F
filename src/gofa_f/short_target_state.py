"""Immutable source-selected checkpoint and label-independent target subsampling."""
from pathlib import Path
import shutil

import numpy as np

from gofa_f.allnode_stage import read
from gofa_f.generative_data import Series,sha
from gofa_f.idle_launcher import process_identity
from gofa_f.prefix_cache import atomic_json
from gofa_f.short_data import TARGETS,origins,stratified
from gofa_f.short_sampled_state import TENSOR_FILES,exited
from gofa_f.short_sampling_diagnostic import SETTINGS


def subset(original,seed=42,minimum=128):
    """Use only query identities, never histories, labels or prediction errors."""
    selected,indices,coverage={},{},{}
    if set(original)!=set(TARGETS):
        raise ValueError('Exactly nine original target lists required')
    for dataset,(name,(length,horizon,n)) in enumerate(TARGETS.items()):
        variables=[104] if name=='nasdaq100' else list(range(n))
        rows=original[name]
        if len(rows)!=2048 or len({tuple(sorted(q.items())) for q in rows})!=2048 \
                or any(q['source']!=name or (q['length'],q['horizon'])!=(length,horizon) for q in rows) \
                or set(q['variable'] for q in rows)!=set(variables):
            raise ValueError('Original target coverage or protocol differs: '+name)
        rng=np.random.default_rng(np.random.SeedSequence([seed,dataset,190,minimum]))
        order=rng.permutation(variables).tolist()
        budget=max(minimum,len(variables))
        own=[]
        for rank,v in enumerate(order):
            candidates=sorted((i for i,q in enumerate(rows) if q['variable']==v),key=lambda i:rows[i]['origin'])
            count=budget//len(variables)+int(rank<budget%len(variables))
            positions=stratified(np.arange(len(candidates)),count,horizon,rng)
            own.extend(candidates[p] for p in positions)
        indices[name]=own
        selected[name]=[rows[i] for i in own]
        coverage[name]=dict(queries=budget,variables=len(variables),
            min_origins_per_variable=budget//len(variables),max_origins_per_variable=(budget+len(variables)-1)//len(variables),
            history=length,horizon=horizon)
    return dict(targets=selected,original_indices=indices,coverage=coverage,
        selection='fixed-seed variable-balanced temporal strata within original frozen identities; no values or scores used')


def prepare(cfg,output):
    source,pool_root=Path(cfg['source_output']),Path(cfg['target_pool_output'])
    exited(source)
    base,prior=read(source/'manifest.json'),read(pool_root/'manifest.json')
    seal=read(source/'source_final_selection.json')
    if cfg['protocol']!='stage19-nine-targets-sample8-balanced-v1' or cfg['seed']!=42 \
            or cfg['minimum_queries_per_dataset']!=128 or cfg['expected_queries']!=2121 \
            or cfg['sampling']!=SETTINGS or cfg['data']!={'target_training_enabled':False,'target_evaluation_enabled':True} \
            or cfg['resources']!=base['configuration']['resources'] or cfg['training']!={'max_updates':4000}:
        raise ValueError('Authorized target-only balanced budget/protocol changed')
    if sha(source/'manifest.json')!=cfg['source_manifest_sha256'] \
            or sha(source/'source_final_selection.json')!=cfg['source_selection_sha256'] \
            or sha(source/'sampled_best.pt')!=cfg['adapter_sha256'] \
            or seal['adapter_sha256']!=cfg['adapter_sha256'] or seal['updates']!=3500 \
            or sha(pool_root/'manifest.json')!=cfg['target_pool_manifest_sha256'] \
            or sha(pool_root/'pools/queries.json')!=cfg['target_pool_sha256']:
        raise ValueError('Locked source selection or original target pool changed')
    code={p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    unchanged=TENSOR_FILES+('short_sampled_forecast.py','short_sampling_diagnostic.py')
    if any(code[n]!=base['code'][n] for n in unchanged):
        raise ValueError('Accepted model, training or numerical sampler changed')
    inherited=read(source/'cpu_preflight.json')
    if not inherited['passed'] or inherited['manifest_sha256']!=cfg['source_manifest_sha256'] \
            or not read(source/'restore_audit.json')['passed']:
        raise ValueError('Source acceptance/actual state restoration proof missing')
    queries=subset(read(pool_root/'pools/queries.json')['targets'])
    if sum(len(q) for q in queries['targets'].values())!=2121:
        raise ValueError('Balanced query count differs')
    root=Path(base['source_configuration']['data']['directory'])
    raw={name:sha(root/'targets'/(name+'.npz')) for name in TARGETS}
    metadata={name:sha(root/'targets'/(name+'.json')) for name in TARGETS}
    if any(raw[name]!=prior['raw_hashes']['target:'+name] for name in TARGETS):
        raise ValueError('Original audited target observations differ')
    # Check eligibility without publishing target value/range statistics.
    for name,rows in queries['targets'].items():
        data=Series(root/'targets'/(name+'.npz'))
        if data.values.shape[1]!=TARGETS[name][2]:
            raise ValueError('Target sensor ordering/dimension differs')
        for v in set(q['variable'] for q in rows):
            legal=set(origins(data,v,*TARGETS[name][:2],'test',require_history=False,calendar_breaks=name!='exchange').tolist())
            if any(q['origin'] not in legal for q in rows if q['variable']==v):
                raise ValueError('Target origin crosses split, segment or lacks observed targets')
    probes=[]
    validation=read(source/'pools/queries.json')['validation']
    for name,v in (('nn5',92),('pems04',102)):
        i,q=next((i,q) for i,q in enumerate(validation) if q['source']==name and q['variable']==v and q['length']==48 and q['horizon']==24)
        refs={}
        for mode,label in (('native','sampled-native'),('gofa','sampled-update-003500')):
            path=source/'evaluations'/label/'trajectories'/f'{i:06d}-00.json'
            refs[mode]=dict(path=str(path),sha256=sha(path))
        probes.append(dict(query=q,index=i,references=refs))
    manifest=dict(configuration=cfg,source_configuration=base['source_configuration'],code=code,
        weights_sha256=base['weights_sha256'],weight_stats=base['weight_stats'],
        source_selection=seal,source_proof_sha256=sha(source/'cpu_preflight.json'),
        inherited_target_audit_sha256=prior['source_target_audit_sha256'],
        raw_hashes=raw,metadata_hashes=metadata,source_raw_hashes=base['raw_hashes'],probes=probes,
        unchanged_tensor_files={n:code[n] for n in unchanged},alphabet=inherited['alphabet'],
        queries=queries,optimizer_updates=0,training_allowed=False,
        total_paired_queries=4242,total_trajectories=33936,
        total_numeric_work=16*sum(q['horizon'] for rows in queries['targets'].values() for q in rows),
        interpretation='fixed source-only checkpoint; target results cannot change model or protocol')
    if cfg.get('repair_source_output'):
        failed_preflight(cfg,output,manifest)
    if read(output/'manifest.json',manifest)!=manifest:
        raise ValueError('Immutable target evaluation binding changed')
    atomic_json(output/'manifest.json',manifest)
    if (output/'adapter.pt').exists() and sha(output/'adapter.pt')!=cfg['adapter_sha256']:
        raise ValueError('Destination adapter differs; never overwrite it')
    if not (output/'adapter.pt').exists():
        shutil.copyfile(source/'sampled_best.pt',output/'adapter.pt')
    atomic_json(output/'gpu_identity.json',read(source/'gpu_identity.json'))
    return manifest


def failed_preflight(cfg,output,manifest):
    """Carry costs only from an exited pre-target failure, never erase predictions."""
    previous=Path(cfg['repair_source_output'])
    if previous.resolve()==output.resolve() or sha(previous/'manifest.json')!=cfg['repair_source_manifest_sha256']:
        raise ValueError('Repair must use a separate bound failed output')
    prior=read(previous/'manifest.json')
    activity=read(previous/'activity.json')
    launch=read(Path(str(previous)+'.launch.json'))
    failure=read(previous/'failures.json',{})
    if process_identity(launch['pid'])==str(launch['start_ticks']) or activity['worker'] is not None \
            or read(previous/'launcher.json')['state']!='failed' \
            or failure.get('phase')!='source_replay_mechanisms' \
            or list((previous/'evaluations').rglob('*.json')) \
            or sha(previous/'adapter.pt')!=cfg['adapter_sha256'] \
            or prior['queries']!=manifest['queries'] or prior['source_selection']!=manifest['source_selection']:
        raise ValueError('Only exited pre-target failures with unchanged adapter/subset may migrate')
    migration=dict(source=str(previous),manifest_sha256=cfg['repair_source_manifest_sha256'],
        failures_sha256=sha(previous/'failures.json'),activity_sha256=sha(previous/'activity.json'),
        inherited_activity=activity,adapter_sha256=cfg['adapter_sha256'],target_predictions=0)
    manifest['repair']=migration
    if read(output/'migration.json',migration)!=migration:
        raise ValueError('Repair evidence changed')
    atomic_json(output/'migration.json',migration)
    if not (output/'activity.json').exists():
        atomic_json(output/'activity.json',activity)


def verify(cfg,output):
    manifest=read(output/'manifest.json')
    source,pool=Path(cfg['source_output']),Path(cfg['target_pool_output'])
    exited(source)
    if manifest['configuration']!=cfg or manifest['code']!={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')} \
            or sha(output/'adapter.pt')!=cfg['adapter_sha256'] or sha(source/'sampled_best.pt')!=cfg['adapter_sha256'] \
            or sha(source/'source_final_selection.json')!=cfg['source_selection_sha256'] \
            or sha(source/'manifest.json')!=cfg['source_manifest_sha256'] \
            or sha(pool/'manifest.json')!=cfg['target_pool_manifest_sha256'] \
            or sha(pool/'pools/queries.json')!=cfg['target_pool_sha256']:
        raise ValueError('Source/adapter/pool/code changed after preparation')
    root=Path(manifest['source_configuration']['data']['directory'])
    for name in TARGETS:
        if sha(root/'targets'/(name+'.npz'))!=manifest['raw_hashes'][name] \
                or sha(root/'targets'/(name+'.json'))!=manifest['metadata_hashes'][name]:
            raise ValueError('Target data or sensor metadata changed')
    weights=Path(manifest['source_configuration']['weights']['directory'])
    if sha(weights/'snapshot.json')!=manifest['weights_sha256'] or any(
            (weights/n).stat().st_size!=v['size'] or (weights/n).stat().st_mtime_ns!=v['mtime_ns']
            for n,v in manifest['weight_stats'].items()):
        raise ValueError('Official backbone changed')
    return manifest
