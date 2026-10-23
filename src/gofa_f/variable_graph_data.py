"""Sealed variable-length source pools and historical-only physical inputs."""
from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict,replace
import json
from pathlib import Path

import numpy as np
import torch

from gofa_f.dual_memory_data import GROUPS,SAMPLER,MemoryInput,sample_subgraph,query_seed,all_node_labels
from gofa_f.dual_memory_policy import SourceSchedule,VIEW_CYCLE,digest,priority_rows,summarize
from gofa_f.dual_memory_prepare import training_pool
from gofa_f.dual_memory_store import SourceStore
from gofa_f.generative_data import Query,Series,sha
from gofa_f.graph_memory_data import fresh_queries
from gofa_f.prefix_cache import atomic_json

TRAIN_SHAPES=((10,1),(12,12),(48,24),(96,48),(192,96),(288,144))
HELD_SHAPES=((14,7),(24,7),(96,24),(192,48))


def interval_seconds(data,q):
    steps=np.diff(np.asarray(data.times[q.origin-q.length:q.origin],np.int64))
    if len(steps)==0 or not (steps>0).all() or not (steps==steps[0]).all():
        raise ValueError('Actual regular historical clock required; no invented resampling')
    return float(steps[0])/1e9


def variable_input(data,query,allowed=None):
    if not 10<=query.length<=288 or not 1<=query.horizon<=144 or data.external is None:
        raise ValueError('Declared length range and physical graph required')
    # Horizon-independent roster. Match the original seed at both primary tasks.
    anchor=replace(query,horizon={12:12,48:24}.get(query.length,1),chunk=0)
    nodes,topology,audit=sample_subgraph(data.external,query.variable,query_seed(anchor),allowed)
    start,end=query.origin-query.length,query.origin
    if start<0 or end>len(data.values) or data.segments[start]!=data.segments[end-1]:
        raise ValueError('Historical window crosses data/segment boundary')
    raw=np.asarray(data.values[start:end,nodes],np.float64).T.copy()
    mask=np.asarray(data.mask[start:end,nodes],bool).T.copy()&np.isfinite(raw)
    present=mask.any(1)
    if not present[0]:raise ValueError('Target requires observed historical data')
    low=np.where(present,np.where(mask,raw,np.inf).min(1),0.)
    high=np.where(present,np.where(mask,raw,-np.inf).max(1),0.)
    scale=np.where(high-low<10*np.finfo(np.float64).eps,1.,high-low)
    normalized=np.where(mask,(raw-low[:,None])/scale[:,None]-.5,0.)
    last_index=np.where(mask,np.arange(query.length)[None],-1).max(1)
    last=np.where(present,raw[np.arange(len(nodes)),last_index.clip(0)],0.)
    adjacency=(topology.numpy().copy()|np.eye(len(nodes),dtype=bool))&present[:,None]&present[None,:]
    hop=np.array(audit['hop'],np.float64)
    edges=np.stack(np.broadcast_arrays(np.log(scale[None,:]/scale[:,None]),
        np.arcsinh((low[None,:]-low[:,None])/scale[:,None]),hop[:,None]/3,hop[None,:]/3),-1).astype(np.float32)
    fields=dict(history=torch.from_numpy(np.where(mask,raw,0.).astype(np.float32)),
        observed=torch.from_numpy(mask),normalized=torch.from_numpy(normalized.astype(np.float32)),
        minimum=torch.from_numpy(low),extent=torch.from_numpy(scale),adjacency=torch.from_numpy(adjacency),
        edges=torch.from_numpy(edges),present=torch.from_numpy(present),last=torch.from_numpy(last))
    import hashlib
    value=hashlib.sha256()
    for x in fields.values():value.update(x.contiguous().numpy().tobytes())
    ids=tuple(str(data.ids[n]) for n in nodes)
    value.update(json.dumps([query.key,ids,audit],sort_keys=True).encode())
    audit.update(global_indices=nodes,node_ids=ids,length=query.length,horizon=query.horizon)
    return MemoryInput(**fields,node_ids=ids,identity=value.hexdigest(),audit=audit,horizon=query.horizon)


class VariableStore(SourceStore):
    def get(self,row,split,role='fit'):
        q=self._verify(row,split,role)
        if q.key in self.index:return super().get(row,split,role)
        data=self.source(q.source)
        partition={'train':'train','development':'validation','confirmation':'confirmation'}[split]
        item=variable_input(data,q,self.manifest['partitions'][q.source][partition])
        y,mask=all_node_labels(data,q,item)
        return item,y,mask&item.present[:,None]

    def interval(self,row):return interval_seconds(self.source(row['source']),Query(**row))


class VariableSchedule(SourceSchedule):
    def __init__(self,queries,held_group=None,shape=None,initial=8192,expand=True):
        if held_group is not None or shape is not None:raise ValueError('One all-source variable-length fit')
        self.held_group=None;self.shape=None;self.initial=initial;self.expand=False
        self.cycle=list(VIEW_CYCLE);self.shapes=[f'{l}-{h}' for l,h in TRAIN_SHAPES]
        self.rows={(n,s):priority_rows(v['train']) for n,d in queries.items() for s,v in d.items() if s in self.shapes}
        self.state=dict(total=0,view_visits={},pools={})

    def peek(self):
        state=deepcopy(self.state);view=self.cycle[state['total']%len(self.cycle)]
        visits=state['view_visits'].get(view,0);shape=self.shapes[visits%len(self.shapes)]
        key=view+'/'+shape;rows=self.rows[(view,shape)]
        pool=state['pools'].setdefault(key,dict(size=len(rows),offset=0,epoch=0))
        if pool['offset']==pool['size']:pool.update(offset=0,epoch=pool['epoch']+1)
        import hashlib
        seed=int.from_bytes(hashlib.sha256(f"42:variable-order:{key}:{pool['epoch']}".encode()).digest()[:8],'big')
        index=np.random.default_rng(seed).permutation(pool['size'])[pool['offset']]
        row=rows[int(index)];pool['offset']+=1;state['total']+=1;state['view_visits'][view]=visits+1
        return row,state


def length_balanced_summary(records,normalizers):
    by_shape={}
    for l,h in sorted({(r['query']['length'],r['query']['horizon']) for r in records}):
        own=[r for r in records if (r['query']['length'],r['query']['horizon'])==(l,h)]
        by_shape[f'{l}-{h}']=summarize(own,normalizers)
    if not by_shape:raise ValueError('Complete source validation required')
    score=float(np.exp(np.mean([np.log(v['selection_score']) for v in by_shape.values()])))
    return dict(selection_score=score,per_shape=by_shape,length_equal_weight=True)


def prepare_variable(cfg,root,runtime):
    parent=Path(cfg['source_output']);old=json.loads((parent/'manifest.json').read_text())
    if sha(parent/'manifest.json')!=cfg['source_manifest_sha256']:raise ValueError('Audited source binding changed')
    # This function never opens target observations, predictions, or scores.
    m={k:deepcopy(old[k]) for k in ('source_directory','source_hashes','metadata_hashes','partitions',
        'sampler','training_unit_normalizers','groups','weights_sha256','source_configuration')}
    if m['sampler']!=SAMPLER:raise ValueError('Physical sampler changed')
    m.update(protocol=cfg['protocol'],configuration=dict(training=cfg['training'],queries=dict(initial_training_per_view=8192)),
        variable_configuration=cfg,variable_runtime=runtime,queries={},interval_seconds={},
        target_labels_used=False,source_confirmation_labels_opened=False,
        old_source_development_exposure=True,old_target_development_exposure=True,
        training_shapes=[list(x) for x in TRAIN_SHAPES],held_length_pairs=[list(x) for x in HELD_SHAPES],
        graph_disclosure='Publisher bidirectional roads; audited coordinate4NN proxies for weather/wind',
        horizon_invariant_sampler='Original seed with H12 for L12/H24 for L48/H1 otherwise')
    prior=[q for d in old['queries'].values() for v in d.values() for split in ('development','confirmation') for q in v[split]]
    for name in GROUPS:
        path=Path(m['source_directory'])/(name+'.npz');data=Series(path)
        if sha(path)!=m['source_hashes'][name] or sha(path.with_suffix('.json'))!=m['metadata_hashes'][name]:
            raise ValueError('Audited observations changed')
        queries={};excluded=list(prior)
        counts=[8192//len(TRAIN_SHAPES)+int(i<8192%len(TRAIN_SHAPES)) for i in range(len(TRAIN_SHAPES))]
        for (length,horizon),count in zip(TRAIN_SHAPES,counts):
            train=training_pool(data,name,m['partitions'][name]['train'],length,horizon,count)
            dev=fresh_queries(data,name,length,horizon,'validation',16,m['partitions'][name]['validation'],excluded,66)
            excluded.extend(dev)
            queries[f'{length}-{horizon}']=dict(train=train,development=dev,confirmation=[])
        # Confirmation includes seen and withheld length pairs, but no values enter source selection.
        for length,horizon in (*TRAIN_SHAPES,*HELD_SHAPES):
            confirm=fresh_queries(data,name,length,horizon,'test',8,m['partitions'][name]['confirmation'],excluded,66)
            excluded.extend(confirm)
            queries.setdefault(f'{length}-{horizon}',dict(train=[],development=[],confirmation=[]))['confirmation']=confirm
        m['queries'][name]=queries
        m['interval_seconds'][name]=interval_seconds(data,Query(**next(iter(queries.values()))['train'][0]))
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    path=root/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=m:raise ValueError('Immutable variable source preparation changed')
    atomic_json(path,m)
    return m
