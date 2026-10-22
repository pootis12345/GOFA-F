"""Stage43: learn soft shape-state compatibility, retaining every allowed neighbor."""
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.nn import functional as F

from gofa_f import source_graph_probe as p
from gofa_f import source_neighborhood_probe as n
from gofa_f import source_normalized_probe as bounded
from gofa_f.generative_data import sha

PROTOCOL='stage43-source-state-attention-probe-v1'
PARENT_REPORT='b9508f08f6e65c9e5b6f71c06fccdfe4ebeb7e3cec533fc12f92adef96a04602'


def state_features(x):
    """Only origin-frozen normalized history; no targets, IDs or future masks."""
    observed=x[...,1].bool();present=observed.any(-1)
    index=torch.arange(x.shape[2],device=x.device).expand(*x.shape[:3])
    last=index.masked_fill(~observed,-1).amax(-1).clamp_min(0)
    level=x[...,0].gather(-1,last[...,None]).squeeze(-1)*present
    length=min(6,x.shape[2]);mask=observed[...,-length:]
    t=torch.arange(length,dtype=x.dtype,device=x.device)
    count=mask.sum(-1);denominator=count.clamp_min(1)
    mean_t=(mask*t).sum(-1)/denominator
    values=x[...,-length:,0]
    mean_y=(values*mask).sum(-1)/denominator
    dt=t-mean_t[...,None]
    variance=(dt.square()*mask).sum(-1)
    slope=(dt*(values-mean_y[...,None])*mask).sum(-1)/variance.clamp_min(1e-12)
    slope=torch.where(count>=2,slope*length,torch.zeros_like(slope))
    state=torch.stack((level,slope),-1)
    return (state[:,:,None]-state[:,None,:]).square()


class StateScore(torch.nn.Module):
    def __init__(self,original):
        super().__init__()
        # Same original coefficients; two new zero coefficients, no RNG consumed.
        weight=torch.cat((original.weight.detach().clone(),original.weight.new_zeros(1,2)),dim=1)
        self.weight=torch.nn.Parameter(weight)

    def forward(self,edge):
        if edge.shape[-1]!=4:raise ValueError('Scale/offset plus two shape-state differences required')
        return F.linear(edge,self.weight)


def augment_input(module,inputs):
    if len(inputs)<3 or inputs[2].shape[-1]!=2:
        raise ValueError('Unaugmented origin-bound edge inputs required')
    x,adjacency,edge=inputs[:3]
    return (x,adjacency,torch.cat((edge,state_features(x)),dim=-1),*inputs[3:])


def base_value_input(module,inputs):
    if inputs[0].shape[-1]!=4:raise ValueError('Four attention features required')
    return (inputs[0][...,:2],)


def add_state_attention(model):
    if getattr(model,'_stage43_state_attention',False):raise ValueError('State attention already installed')
    for block in model.exchanges:
        block.edge_score=StateScore(block.edge_score)
        block.edge_value.register_forward_pre_hook(base_value_input)
    model.register_forward_pre_hook(augment_input)
    model._stage43_state_attention=True
    return model


def reused_caches(output,m,charge):
    parent=Path(m['configuration']['parent_output']);old=p.read(parent/'manifest.json')
    if sha(parent/'manifest.json')!=m['parent_manifest_sha256'] or old['queries']!=m['queries']:
        raise ValueError('Source parent lineage changed')
    root=Path(old['configuration']['parent_output'])
    if sha(root/'manifest.json')!=p.read(parent/'cache_reuse.json')['parent_manifest_sha256']:
        raise ValueError('Exact cache ancestor changed')
    proxy=dict(m,parent_manifest_sha256=sha(root/'manifest.json'),
        configuration=dict(m['configuration'],parent_output=str(root)))
    return bounded.reuse_parent_caches(output,proxy,charge)


@contextmanager
def state_attention_runtime():
    with bounded.normalized_runtime():
        old=(p.model_for,n.PROTOCOL,n.PARENT_REPORT,n.code_hashes,n.caches)
        factory,_,_,hashes,_=old
        p.model_for=lambda cfg,seed:add_state_attention(factory(cfg,seed))
        n.PROTOCOL=PROTOCOL;n.PARENT_REPORT=PARENT_REPORT
        n.code_hashes=lambda:dict(hashes(),source_state_attention_probe=sha(Path(__file__)))
        n.caches=reused_caches
        try:yield
        finally:p.model_for,n.PROTOCOL,n.PARENT_REPORT,n.code_hashes,n.caches=old


if __name__=='__main__':
    with state_attention_runtime():n.main()
