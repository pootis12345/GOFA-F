"""Source-only schedules, metrics, checkpoint selection and exact partial state."""
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from gofa_f.dual_memory_data import GROUPS

DOMAINS = ('traffic','weather','wind')
VIEW_CYCLE = ('pems04_flow','boston2017','penmanshiel2017','pems04_speed','boston2017','penmanshiel2017')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def cpu_tree(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return deepcopy(value)


def rng_state(cuda):
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if cuda else None)


def restore_rng(value):
    random.setstate(value['python']);np.random.set_state(value['numpy']);torch.set_rng_state(value['torch'])
    if value['cuda'] is not None:torch.cuda.set_rng_state_all(value['cuda'])


def capture(model,optimizer,state,binding,cuda=False):
    return dict(binding=binding,model=cpu_tree(model.state_dict()),optimizer=cpu_tree(optimizer.state_dict()),
        gradients={k:None if p.grad is None else cpu_tree(p.grad) for k,p in model.named_parameters()},
        state=deepcopy(state),rng=rng_state(cuda))


def restore(model,optimizer,value,binding):
    if value['binding']!=binding:raise ValueError('Partial optimizer/cursor/selection binding changed')
    model.load_state_dict(value['model'],strict=True);optimizer.load_state_dict(value['optimizer'])
    parameters=dict(model.named_parameters())
    if set(parameters)!=set(value['gradients']):raise ValueError('Trainable parameter layout changed')
    for name,p in parameters.items():
        p.grad=None if value['gradients'][name] is None else value['gradients'][name].to(p.device,dtype=p.dtype).clone()
    restore_rng(value['rng'])
    return deepcopy(value['state'])


def accumulate(loss,parameters):
    """Commit new microgradients only after autograd finishes successfully.

    An OOM during autograd.grad cannot partially corrupt existing accumulation.
    The caller advances its cursor only after this function returns.
    """
    active=[p for p in parameters if p.requires_grad]
    gradients=torch.autograd.grad(loss,active,allow_unused=True)
    finite=torch.stack([torch.isfinite(loss),*[torch.isfinite(g).all() for g in gradients if g is not None]]).all()
    if not bool(finite):
        raise FloatingPointError('Nonfinite loss or microgradient')
    with torch.no_grad():
        for p,g in zip(active,gradients):
            if g is not None:
                if p.grad is None:p.grad=g.detach()
                else:p.grad.add_(g)


def query_loss(prediction,target,observed,present,normalizer):
    """All real nodes: valid slots/node, then valid nodes/query, in raw units."""
    if prediction.shape!=target.shape or observed.shape!=target.shape:
        raise ValueError('Explicit N,H predictions, targets and masks required')
    mask=observed.bool()&present[:,None].bool()&torch.isfinite(target)
    counts=mask.sum(1);valid=counts>0
    if not valid.any() or normalizer<=0:raise ValueError('No lawful observed-origin training label')
    error=(prediction-target).abs().masked_fill(~mask,0)
    return (error.sum(1)/counts.clamp_min(1))[valid].mean()/normalizer


def priority_rows(rows):
    """Variable-balanced nested pools, time-stratified without reading labels."""
    by_variable=defaultdict(list)
    for i,q in enumerate(rows):by_variable[q['variable']].append(i)
    variables=sorted(by_variable)
    queues={}
    for variable,indices in by_variable.items():
        indices=sorted(indices,key=lambda i:rows[i]['origin'])
        # Bit reversal visits distant time strata before filling gaps.
        bits=max(1,(len(indices)-1).bit_length())
        queues[variable]=[indices[j] for j in sorted(range(len(indices)),key=lambda j:int(f'{j:0{bits}b}'[::-1],2))]
    result=[]
    for turn in range(max(map(len,queues.values()))):
        for v in variables:
            if turn<len(queues[v]):result.append(queues[v][turn])
    return [rows[i] for i in result]


class SourceSchedule:
    def __init__(self,queries,held_group=None,shape=None,initial=1024,expand=True):
        if held_group not in (None,*DOMAINS):raise ValueError('Whole-source holdout required')
        if shape not in (None,'12-12','48-24'):raise ValueError('Unknown shape')
        self.held_group=held_group;self.shape=shape;self.initial=initial;self.expand=expand
        self.cycle=[v for v in VIEW_CYCLE if GROUPS[v]!=held_group]
        self.rows={(v,s):priority_rows(q['train']) for v,d in queries.items() if GROUPS[v]!=held_group
            for s,q in d.items() if shape is None or shape==s}
        self.state=dict(total=0,view_visits={},pools={})

    def peek(self):
        state=deepcopy(self.state);view=self.cycle[state['total']%len(self.cycle)]
        visits=state['view_visits'].get(view,0)
        shape=self.shape or ('12-12' if visits%2==0 else '48-24')
        key=view+'/'+shape;rows=self.rows[(view,shape)]
        initial=min(len(rows),max(1,self.initial//(1 if self.shape else 2)))
        pool=state['pools'].setdefault(key,dict(size=initial,offset=0,epoch=0))
        if pool['offset']==pool['size']:
            pool.update(size=min(len(rows),pool['size']*2) if self.expand else pool['size'],offset=0,epoch=pool['epoch']+1)
        seed=int.from_bytes(hashlib.sha256(f"42:source-order:{key}:{pool['epoch']}:{pool['size']}".encode()).digest()[:8],'big')
        order=np.random.default_rng(seed).permutation(pool['size'])
        row=rows[int(order[pool['offset']])]
        pool['offset']+=1;state['view_visits'][view]=visits+1;state['total']+=1
        return row,state

    def commit(self,state):self.state=deepcopy(state)


def summarize(records,normalizers):
    per_view=defaultdict(lambda:defaultdict(lambda:[0.,0]))
    for record in records:
        q=record['query'];p=np.asarray(record['prediction'],np.float64)
        y=np.asarray(record['target'],np.float64);mask=np.asarray(record['observed'],bool)
        if p.shape!=y.shape or mask.shape!=y.shape or not np.isfinite(p).all() or not mask.any():
            raise ValueError('Complete finite target-node predictions and lawful masks required')
        slot=per_view[q['source']][q['variable']]
        slot[0]+=float(np.abs(p-y)[mask].sum());slot[1]+=int(mask.sum())
    views={v:dict(macro_mae=float(np.mean([error/count for error,count in variables.values()])),
        variables=len(variables),points=sum(c for _,c in variables.values()),
        per_variable={str(k):e/c for k,(e,c) in variables.items()}) for v,variables in per_view.items()}
    domains={}
    for domain in DOMAINS:
        names=[v for v in views if GROUPS[v]==domain]
        if names:domains[domain]=float(np.exp(np.mean([np.log(max(views[v]['macro_mae']/normalizers[v],1e-12)) for v in names])))
    return dict(views=views,domains=domains,selection_score=float(np.exp(np.mean(np.log(list(domains.values()))))))


def update_selection(selection,score,update,minimum,patience,significant=.001):
    if selection.get('last_validated',-1)==update:return deepcopy(selection)
    value=deepcopy(selection);value['last_validated']=update
    if score<value.get('best_score',float('inf')):value.update(best_score=score,best_update=update)
    reference=value.get('reference_score')
    if reference is None or score<=reference*(1-significant):
        value.update(reference_score=score,stale=0)
    else:value['stale']=value.get('stale',0)+1
    value['stop']=update>=minimum and value['stale']>=patience
    return value


def fit_groups(fit_id):
    if fit_id=='all':return list(GROUPS)
    if not fit_id.startswith('hold-') or fit_id[5:] not in DOMAINS:raise ValueError('Unknown cross-fit')
    return [name for name,group in GROUPS.items() if group!=fit_id[5:]]
