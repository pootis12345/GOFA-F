"""Sealed deployment, history-only prediction inputs and guarded truth access."""
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from gofa_f.dual_memory_data import GROUPS,memory_input,task_text
from gofa_f.dual_memory_policy import digest
from gofa_f.dual_memory_store import atomic_torch
from gofa_f.generative_data import Query,Series,sha
from gofa_f.prefix_cache import atomic_json

TARGETS=('pems03','pems_bay','meteonet_nw2018','engrad','sdwpf','kelmarsh2019')
TARGET_FIELDS={
    'pems03':('traffic flow','vehicles per 5-minute bin','flow'),
    'pems_bay':('traffic speed','mph','speed'),
    'meteonet_nw2018':('air temperature','Celsius','t'),
    'engrad':('air temperature','Celsius','temperature_2m'),
    'sdwpf':('active electrical power','kW','Patv'),
    'kelmarsh2019':('active electrical power','kW','Power (kW)')}
CANONICAL_UNITS={'Celsius':'degrees Celsius','kW':'kilowatts','mph':'miles per hour',
                 'vehicles per 5-minute bin':'vehicles per 5-minute bin'}
MODELS=('native','enhanced','fusion','ufg')
CORE=('dual_memory_model.py','dual_memory_data.py','allnode_codec.py','short_codec.py',
      'short_sampling_diagnostic.py','allnode_kernel.py','generative_kernel.py',
      'ufg_memory_roster.py','ufg.py','ufg_subgraph.py')


def read(path):return json.loads(Path(path).read_text())


def seal(path,value):
    path=Path(path)
    if path.exists():
        if read(path)!=value:raise ValueError('An immutable evaluation binding changed: '+path.name)
    else:atomic_json(path,value)


def source_ready(source):
    """No target file is inspected while waiting for both source-training jobs."""
    source=Path(source)
    needed=('manifest.json','source_model_lock.json','shared_gate.json','status.json','ufg_status.json',
        'activity.json','ufg_activity.json','ufg-12-12/selection.json','ufg-48-24/selection.json')
    if any(not (source/name).exists() for name in needed):return False
    status=read(source/'status.json');ufg=read(source/'ufg_status.json')
    if status['state']!='complete' or ufg['state']!='complete':return False
    if read(source/'activity.json').get('worker') is not None or read(source/'ufg_activity.json').get('worker') is not None:return False
    lock=read(source/'source_model_lock.json')
    if lock['binding']!=sha(source/'manifest.json') or lock['target_labels_used'] or lock['source_confirmation_labels_used']:
        raise ValueError('Source deployment is not an untouched source-only lock')
    if sha(source/'final-source-model/best.pt')!=lock['expert_checkpoint_sha256'] or sha(source/'shared_gate.json')!=lock['shared_gate_sha256']:
        raise ValueError('A locked expert/gate changed')
    return True


def deployment(source,accepted_snapshot):
    source=Path(source);accepted_snapshot=Path(accepted_snapshot)
    if not source_ready(source):raise ValueError('Source fits and UFG locks are not ready')
    lock=read(source/'source_model_lock.json');m=read(source/'manifest.json');gate=read(source/'shared_gate.json')
    if gate['depth']!=lock['depth'] or gate['source_expert_sha256']!=lock['expert_model_sha256'] or gate['target_labels_used']:
        raise ValueError('Shared fusion and expert were not locked together')
    baselines={}
    for shape in ('12-12','48-24'):
        selection=read(source/f'ufg-{shape}/selection.json');path=source/f'ufg-{shape}/best.pt'
        if sha(path)!=selection['checkpoint_sha256'] or selection['target_labels_used']:
            raise ValueError('UFG source best changed or used target labels')
        baselines[shape]=dict(path=str(path),checkpoint_sha256=sha(path),model_sha256=selection['model_sha256'],
            best_update=selection['best_update'],stopped_update=selection['stopped_update'],stop_reason=selection['stop_reason'])
    core={name:sha(Path(__file__).parent/name) for name in CORE}
    if core!={name:sha(accepted_snapshot/'src/gofa_f'/name) for name in CORE}:
        raise ValueError('Accepted prediction tensor/input core differs')
    # Fusion was added with formal source fitting, after the mechanism snapshot.
    # Bind it to its actual tested/trained source runtime rather than inventing
    # a dependency in the earlier architecture acceptance.
    fusion_sha=sha(Path(__file__).parent/'dual_memory_fusion.py')
    if fusion_sha!=m['formal_runtime_code']['dual_memory_fusion.py']:
        raise ValueError('The fitted shared fusion execution code changed')
    return dict(source_manifest_sha256=sha(source/'manifest.json'),source_model_lock_sha256=sha(source/'source_model_lock.json'),
        source_directory=str(source),depth=lock['depth'],weights_sha256=m['weights_sha256'],
        expert=dict(path=str(source/'final-source-model/best.pt'),checkpoint_sha256=lock['expert_checkpoint_sha256'],
                    model_sha256=lock['expert_model_sha256']),
        gate=dict(path=str(source/'shared_gate.json'),sha256=lock['shared_gate_sha256']),ufg=baselines,
        core_sha256=core,shared_fusion_code_sha256=fusion_sha,
        source_confirmation_labels_used_for_fitting=False,target_labels_used=False)


def verify_deployment(value):
    source=Path(value['source_directory'])
    if sha(source/'manifest.json')!=value['source_manifest_sha256'] or sha(source/'source_model_lock.json')!=value['source_model_lock_sha256']:
        raise ValueError('Source deployment lineage changed')
    for entry in [value['expert'],value['gate'],*value['ufg'].values()]:
        expected=entry.get('checkpoint_sha256',entry.get('sha256'))
        if sha(Path(entry['path']))!=expected:raise ValueError('A sealed deployment file changed')
    if value['core_sha256']!={n:sha(Path(__file__).parent/n) for n in CORE}:
        raise ValueError('Prediction core changed after sealing')
    if value['shared_fusion_code_sha256']!=sha(Path(__file__).parent/'dual_memory_fusion.py'):
        raise ValueError('Shared source-fitted fusion code changed after sealing')


def target_text(data,name,q):
    quantity,unit,channel=TARGET_FIELDS[name];metadata=data.meta['provenance']
    if metadata.get('unit')!=unit or metadata.get('channel')!=channel:
        raise ValueError('Target quantity/unit no longer matches the audited registry')
    # SDWPF contains two original frequencies. Use ONLY the actual past window.
    clock=np.asarray(data.times[q.origin-q.length:q.origin],np.int64)
    steps=np.unique(np.diff(clock))
    if len(steps)!=1 or steps[0]<=0:raise ValueError('Historical input is not a regular within-segment window')
    return task_text(quantity,CANONICAL_UNITS[unit],float(steps[0])/1e9,q.length,q.horizon)


def fresh_target_queries(data,name,old_queries):
    """Same variables/quotas as the old128, new nonoverlapping time windows."""
    if len(old_queries)!=128 or {q['source'] for q in old_queries}!={name}:
        raise ValueError('The fixed original128-query identity budget is required')
    quota=Counter(q['variable'] for q in old_queries);shapes={(q['length'],q['horizon']) for q in old_queries}
    if len(shapes)!=1:raise ValueError('A target has more than one forecasting shape')
    length,horizon=next(iter(shapes));result=[]
    variables=sorted(quota,key=lambda v:str(data.ids[v]))
    chosen={v:[] for v in variables}
    # Preserve identities and quotas; select only from legal, error-independent
    # candidates, with the same variable's old/new label intervals excluded.
    for turn in range(max(quota.values())):
        for variable in variables:
            if turn>=quota[variable]:continue
            candidates=data.candidates(variable,length,horizon,'test').copy()
            exclusions=[q for q in old_queries+result if q['variable']==variable]
            for q in exclusions:
                candidates=candidates[(candidates+horizon<=q['origin'])|(candidates>=q['origin']+q['horizon'])]
            if not len(candidates):raise ValueError('Fresh lawful target capacity exhausted; no variable/dataset substitution')
            strata=np.array_split(candidates,min(quota[variable],len(candidates)))
            block=strata[min(turn,len(strata)-1)]
            rng=np.random.default_rng(np.random.SeedSequence([42,59,variable,turn]))
            origin=int(rng.choice(block));row=asdict(Query(name,int(variable),origin,length,horizon))
            result.append(row);chosen[variable].append(origin)
    if Counter(q['variable'] for q in result)!=quota:raise AssertionError('Target variable coverage changed')
    return result


class InputStore:
    """Prediction workers see serialized history only, no future-value accessor."""
    def __init__(self,root):
        self.root=Path(root);self.manifest=read(self.root/'manifest.json');self.index=read(self.root/'inputs/index.json')
        if self.index['binding']!=sha(self.root/'manifest.json'):raise ValueError('Input index belongs to another evaluation')
        self.loaded={}

    def get(self,row):
        key=Query(**row).key
        if key not in self.index['records']:raise ValueError('Query is outside the sealed evaluation')
        filename,offset=self.index['records'][key]
        if filename not in self.loaded:
            path=self.root/'inputs'/filename
            if sha(path)!=self.index['shard_sha256'][filename]:raise ValueError('A sealed history input shard changed')
            value=torch.load(path,map_location='cpu',weights_only=False)
            if value['binding']!=self.index['binding']:raise ValueError('Input shard binding changed')
            self.loaded[filename]=value['records']
        record=self.loaded[filename][offset]
        if record['query']!=row or any(k in record for k in ('target','labels','future','future_observed')):
            raise ValueError('Prediction input contains unexpected labels or wrong query')
        return record['item'],record['text']


def write_inputs(root,manifest,datasets,partition=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);seal(root/'manifest.json',manifest)
    directory=root/'inputs';directory.mkdir(exist_ok=True);binding=sha(root/'manifest.json')
    index={};hashes={};rows=manifest['queries'];scope=manifest['scope']
    for shard,start in enumerate(range(0,len(rows),64)):
        path=directory/f'history-{shard:04d}.pt';own=rows[start:start+64]
        if path.exists():
            old=read(path.with_suffix('.sha.json'))
            if old['binding']!=binding or old['sha256']!=sha(path):raise ValueError('Partial history shard changed')
            value=torch.load(path,map_location='cpu',weights_only=False);records=value['records']
            if value['binding']!=binding or [r['query'] for r in records]!=own:raise ValueError('History shard query binding changed')
        else:
            records=[]
            for row in own:
                q=Query(**row);data=datasets[q.source]
                allowed=None if partition is None else partition[q.source]['confirmation']
                item=memory_input(data,q,allowed)
                text=manifest['source_texts'][q.source][f'{q.length}-{q.horizon}'] if scope=='source_confirmation' else target_text(data,q.source,q)
                records.append(dict(query=row,item=item,text=text,text_sha256=digest(text)))
            atomic_torch(path,dict(binding=binding,records=records))
            atomic_json(path.with_suffix('.sha.json'),dict(binding=binding,sha256=sha(path)))
        for offset,record in enumerate(records):index[Query(**record['query']).key]=[path.name,offset]
        hashes[path.name]=sha(path)
    seal(directory/'index.json',dict(binding=binding,records=index,shard_sha256=hashes,
        contains_future_values=False,contains_future_missingness=False,complete=True))


def truth_records(root,completion_seals):
    """CPU scoring only after all four prediction groups are complete and bound."""
    root=Path(root);m=read(root/'manifest.json');binding=sha(root/'manifest.json')
    if set(completion_seals)!=set(MODELS):raise ValueError('Truth remains locked until all model groups are complete')
    for name,seal_path in completion_seals.items():
        value=read(seal_path)
        if value['binding']!=binding or not value['complete'] or value['queries']!=len(m['queries']) or \
                value.get('model')!=name or value.get('input_index_sha256')!=sha(root/'inputs/index.json'):
            raise ValueError('A model completion seal is incomplete or bound elsewhere')
        for filename,expected in value['shards'].items():
            if sha(root/filename)!=expected:raise ValueError('A sealed prediction shard changed before scoring')
    result=[];data={}
    for row in m['queries']:
        name=row['source'];q=Query(**row)
        if name not in data:
            info=m['datasets'][name];path=Path(info['path'])
            if sha(path)!=info['sha256'] or sha(path.with_suffix('.json'))!=info['metadata_sha256']:raise ValueError('Evaluation observations changed')
            data[name]=Series(path)
        value=data[name];y=np.asarray(value.values[q.origin:q.origin+q.horizon,q.variable],np.float64)
        mask=np.asarray(value.mask[q.origin:q.origin+q.horizon,q.variable],bool)&np.isfinite(y)
        if not mask.any():raise ValueError('Sealed query no longer has a valid evaluation label')
        result.append(dict(query=row,target=np.where(mask,y,0.).tolist(),observed=mask.tolist(),
            origin_time_ns=int(value.times[q.origin])))
    return result
