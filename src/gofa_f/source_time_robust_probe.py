"""Stage41: train-only exponentiated temporal-group weights, with exact saved state."""
from contextlib import contextmanager
from pathlib import Path

import torch

from gofa_f import source_graph_probe as p
from gofa_f import source_neighborhood_probe as n
from gofa_f import source_normalized_probe as bounded
from gofa_f.generative_data import sha

PROTOCOL='stage41-source-time-robust-probe-v1'
PARENT_REPORT='b9508f08f6e65c9e5b6f71c06fccdfe4ebeb7e3cec533fc12f92adef96a04602'
TRAIN_END=10195  # Audited16992 source timestamps, floor(60%).
ETA=.01


def temporal_group(record):
    q=record['query']
    if q['source'] not in p.VIEWS or not 12<=q['origin']<=TRAIN_END-12:
        raise ValueError('Temporal weights may only use the fixed source training segment')
    return p.VIEWS.index(q['source'])*4+min(3,q['origin']*4//TRAIN_END)


class TimeWeights(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('logits',torch.zeros(2,4,dtype=torch.float64))
        self.register_buffer('updates',torch.zeros((),dtype=torch.int64))

    def objective(self,per_query,groups):
        if not per_query.isfinite().all() or groups.shape!=per_query.shape \
                or not ((groups>=0)&(groups<8)).all():
            raise ValueError('Finite training loss and registered temporal groups required')
        losses=[];observed=[];view_losses=[]
        for view in range(2):
            present=torch.stack([(groups==view*4+k).any() for k in range(4)])
            if not present.any():raise ValueError('Both views must remain in each effective batch')
            own=torch.stack([per_query[groups==view*4+k].mean() if present[k] else per_query.new_zeros(())
                for k in range(4)])
            used=self.logits[view].detach().masked_fill(~present,-torch.inf).softmax(0).to(per_query.dtype)
            view_losses.append((own*used).sum())
            losses.append(own);observed.append(present)
        # Update the adversary for the NEXT update using detached training errors.
        with torch.no_grad():
            for view in range(2):
                self.logits[view]+=ETA*losses[view].detach().double()*observed[view]
                self.logits[view]-=self.logits[view].logsumexp(0)
            self.updates+=1
        return torch.stack(view_losses).mean()


def robust_loss(residual,b,state):
    if not b['observed'].any(1).all():raise ValueError('Empty source training label window')
    error=(residual*b['scale'][:,None]-b['delta']).abs()*b['observed']
    return state.objective(error.sum(1)*b['weight'],b['time_group'])


def reused_caches(output,m,charge):
    parent=Path(m['configuration']['parent_output']);old=p.read(parent/'manifest.json')
    if sha(parent/'manifest.json')!=m['parent_manifest_sha256'] or old['queries']!=m['queries']:
        raise ValueError('Parent temporal source lineage changed')
    root=Path(old['configuration']['parent_output'])
    evidence=p.read(parent/'cache_reuse.json')
    if sha(root/'manifest.json')!=evidence['parent_manifest_sha256']:
        raise ValueError('Parent exact source-cache ancestry changed')
    proxy=dict(m,parent_manifest_sha256=sha(root/'manifest.json'),
        configuration=dict(m['configuration'],parent_output=str(root)))
    packed=bounded.reuse_parent_caches(output,proxy,charge)
    for view in p.VIEWS:
        rows=m['queries']['train'][view]
        for r in rows:temporal_group(r)
    return packed


@contextmanager
def robust_runtime():
    with bounded.normalized_runtime():
        old=(p.model_for,p.tensors,p.loss_value,n.PROTOCOL,n.PARENT_REPORT,n.code_hashes,n.caches)
        factory,tensors,_,_,_,hashes,_=old;active=[None]
        def model_for(cfg,seed):
            model=factory(cfg,seed);model.add_module('time_weights',TimeWeights());active[0]=model
            return model
        def robust_tensors(a,indices,records,condition,weight=None,norm=1.):
            result=tensors(a,indices,records,condition,weight,norm)
            if weight is not None:
                result['time_group']=torch.tensor([temporal_group(records[int(i)]) for i in indices],dtype=torch.int64)
            return result
        def loss_value(residual,b):
            if active[0] is None:raise ValueError('Bound source model required')
            return robust_loss(residual,b,active[0].time_weights)
        p.model_for=model_for;p.tensors=robust_tensors;p.loss_value=loss_value
        n.PROTOCOL=PROTOCOL;n.PARENT_REPORT=PARENT_REPORT
        n.code_hashes=lambda:dict(hashes(),source_time_robust_probe=sha(Path(__file__)))
        n.caches=reused_caches
        try:yield
        finally:p.model_for,p.tensors,p.loss_value,n.PROTOCOL,n.PARENT_REPORT,n.code_hashes,n.caches=old


if __name__=='__main__':
    with robust_runtime():n.main()
