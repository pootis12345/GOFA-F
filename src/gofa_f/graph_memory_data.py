"""Physical-only matched rosters, scaling metadata and source query reservations."""
import hashlib
from dataclasses import asdict

import numpy as np
import torch

from gofa_f.allnode_graph import AllNodeGraph
from gofa_f.generative_graph import OriginGraph,row_normalize
from gofa_f.graph_probe_data import node_roster
from gofa_f.source_sparse_normalized_probe import road_adjacency

CONTROLS=('physical','self','rewired')


def memory_graph(data,query,allowed=None,control='physical'):
    if control not in CONTROLS:raise ValueError('Registered graph-memory control required')
    if data.external is None:raise ValueError('Physical topology required; no correlation fallback')
    allowed=list(range(len(data.ids))) if allowed is None else list(allowed)
    nodes=node_roster(data.external,query.variable,allowed,data.ids)
    raw=np.asarray(data.values[query.origin-query.length:query.origin,nodes],np.float64)
    observed=np.asarray(data.mask[query.origin-query.length:query.origin,nodes],bool)&np.isfinite(raw)
    if len(raw)!=query.length or not observed[:,0].any():raise ValueError('Lawful nonempty target history required')
    present=observed.any(0)
    low=np.where(present,np.where(observed,raw,np.inf).min(0),0.)
    high=np.where(present,np.where(observed,raw,-np.inf).max(0),0.)
    scale=np.where(high-low<10*np.finfo(np.float64).eps,1.,high-low)
    edge=np.stack((np.log(scale[None]/scale[:,None]),np.arcsinh((low[None]-low[:,None])/scale[:,None])),-1).astype(np.float32)
    support=road_adjacency(data.external,np.asarray(nodes),control=='rewired').astype(float)
    if control=='self':support[:]=0
    support+=np.eye(len(nodes))
    base=row_normalize(support,present)
    values=np.where(observed,raw,0).astype(np.float32)
    identity=hashlib.sha256()
    for a in (values,observed,base,edge):identity.update(np.ascontiguousarray(a).tobytes())
    ids=tuple(str(data.ids[n]) for n in nodes)
    identity.update(str((query.key,ids,control,'memory_physical_self_inclusive_v1')).encode())
    graph=OriginGraph(torch.from_numpy(values),torch.from_numpy(observed),torch.from_numpy(base.copy()),
        torch.from_numpy(base),torch.from_numpy(edge),0,identity.hexdigest())
    return AllNodeGraph(graph,ids,query.key)


def flatten_queries(value):
    if isinstance(value,dict):
        if {'source','variable','origin','length','horizon'}<=set(value):yield value
        else:
            for item in value.values():yield from flatten_queries(item)
    elif isinstance(value,list):
        for item in value:yield from flatten_queries(item)


def fresh_queries(data,name,length,horizon,split,count,variables,exclusions,stage_seed=56):
    """Fixed ID/time strata, no error access; labels are not read beyond eligibility."""
    variables=sorted(variables,key=lambda v:str(data.ids[v]));rows=[]
    order=np.random.default_rng(np.random.SeedSequence([42,stage_seed,int(split=='test')])).permutation(variables).tolist()
    for index in range(count):
        variable=order[index%len(order)]
        candidates=data.candidates(variable,length,horizon,split)
        for old in exclusions+rows:
            if old['source']==name and old['variable']==variable:
                candidates=candidates[(candidates+horizon<=old['origin'])|(candidates>=old['origin']+old['horizon'])]
        if not len(candidates):raise ValueError('Fresh source query capacity exhausted; do not replace source')
        # ID/time-only balanced partition. Selection is independent of values/errors.
        blocks=np.array_split(candidates,max(1,(count+len(order)-1)//len(order)))
        block=blocks[min(index//len(order),len(blocks)-1)]
        rng=np.random.default_rng(np.random.SeedSequence([42,stage_seed,variable,index,int(split=='test')]))
        origin=int(rng.choice(block))
        rows.append(dict(source=name,variable=int(variable),origin=origin,length=length,horizon=horizon,chunk=0))
    return rows
