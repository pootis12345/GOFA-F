"""One fixed source experiment: origin-frozen edge/lag correlation attention."""
from contextlib import contextmanager
import math
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from gofa_f import source_graph_probe as p
from gofa_f import source_neighborhood_probe as n
from gofa_f import source_allnode_probe as allnode
from gofa_f import source_origin_history_probe as history
from gofa_f import source_normalized_probe as bounded
from gofa_f.generative_data import sha

PROTOCOL='stage55-source-edge-lag-v1'
STAGE54_REPORT='a7d5c8e38cd170e2d5e29dcdad2e10b3d0be9b00468bba29d1794452ab2e9b86'


def lag_correlations(x):
    """Receiver increment(t),sender increment(t-lag); only original12past slots."""
    if x.shape[2:]!=(12,2):raise ValueError('Fixed12-point original history required')
    observed=x[...,1].bool();delta=x[...,1:,0]-x[...,:-1,0]
    valid=observed[...,1:] & observed[...,:-1];features=[]
    for lag in (0,1,2,3):
        left=delta[:,:,lag:][:,:,None,:];right=delta[:,:,:11-lag][:,None,:,:]
        mask=valid[:,:,lag:][:,:,None,:] & valid[:,:,:11-lag][:,None,:,:]
        count=mask.sum(-1);denominator=count.clamp_min(1)
        lm=(left*mask).sum(-1)/denominator;rm=(right*mask).sum(-1)/denominator
        a=(left-lm[...,None])*mask;b=(right-rm[...,None])*mask
        variance=a.square().sum(-1)*b.square().sum(-1)
        corr=(a*b).sum(-1)/variance.clamp_min(1e-20).sqrt()
        features.append(torch.where((count>=3)&(variance>1e-20),corr.clamp(-1,1),torch.zeros_like(corr)))
    return torch.stack(features,-1)


class EdgeLagExchange(nn.Module):
    def __init__(self,old):
        super().__init__();self.rank=old.rank;self.lags=old.lags
        for name in ('norm','query','key','value','edge_score','edge_value','gate','up','lag_bias'):
            setattr(self,name,getattr(old,name))
        self.correlation_weight=nn.Parameter(torch.zeros(len(self.lags)))

    def forward(self,hidden,observed,adjacency,edge_features):
        batch,nodes,length,_=hidden.shape
        if edge_features.shape[-1]!=6:raise ValueError('Original2plus4fixed history correlations required')
        z=self.norm(hidden);histories=[];masks=[]
        for lag in self.lags:
            if lag:histories.append(F.pad(z[:,:,:-lag],(0,0,lag,0)));masks.append(F.pad(observed[:,:,:-lag],(lag,0),value=False))
            else:histories.append(z);masks.append(observed)
        histories=torch.stack(histories,-2);valid_history=torch.stack(masks,-1).permute(0,2,1,3)
        q=self.query(z).permute(0,2,1,3);keys=self.key(histories).permute(0,2,1,3,4)
        values=self.value(histories).permute(0,2,1,3,4)
        cross=adjacency.bool() & ~torch.eye(nodes,device=hidden.device,dtype=torch.bool)[None]
        present=observed.any(-1);cross=cross & present[:,:,None] & present[:,None,:]
        support=cross[:,None,:,:,None] & valid_history[:,:,None]
        score=torch.einsum('btir,btjkr->btijk',q,keys)/math.sqrt(self.rank)
        original=edge_features[...,:2]
        score=score+self.edge_score(original).squeeze(-1)[:,None,:,:,None]+self.lag_bias[None,None,None,None]
        score=score+(edge_features[...,2:]*self.correlation_weight)[...,None,:,:,:]
        empty=~support.flatten(-2).any(-1,keepdim=True)
        flat=score.masked_fill(~support,-torch.inf).flatten(-2);flat=torch.where(empty,torch.zeros_like(flat),flat)
        attention=(flat.softmax(-1)*support.flatten(-2)).reshape(batch,length,nodes,nodes,len(self.lags))
        context=torch.einsum('btijk,btjkd->btid',attention,values)
        context=context+torch.einsum('btij,bijd->btid',attention.sum(-1),self.edge_value(original))
        delta=self.up(F.gelu(context))*self.gate(z.permute(0,2,1,3)).sigmoid()
        delta=delta.masked_fill(empty,0.).permute(0,2,1,3)
        return (hidden+delta)*present[:,:,None,None]


def augment(module,inputs):
    x,adj,edge=inputs[:3]
    if edge.shape[-1]!=2:raise ValueError('Do not augment twice')
    return (x,adj,torch.cat((edge,lag_correlations(x)),dim=-1),*inputs[3:])


def add_edge_lag(model):
    if getattr(model,'_stage55_edge_lag',False):raise ValueError('Edge lag already installed')
    blocks=[]
    for old in model.exchanges:
        block=EdgeLagExchange(old);block.register_forward_hook(bounded.bounded_exchange);blocks.append(block)
    model.exchanges=nn.ModuleList(blocks);model.register_forward_pre_hook(augment)
    model._stage55_edge_lag=True;return model


@contextmanager
def edge_lag_runtime():
    with allnode.allnode_runtime():
        old=(p.model_for,p.fit,n.PROTOCOL,n.stage_number,n.code_hashes,n.SEMANTICS)
        factory,_,_,number,hashes,semantics=old
        p.model_for=lambda cfg,seed:add_edge_lag(factory(cfg,seed));p.fit=history.fit_origin_allnode
        n.PROTOCOL=PROTOCOL;n.stage_number=lambda cfg:55 if cfg['protocol']==PROTOCOL else number(cfg)
        n.SEMANTICS={c:v+'_origin_frozen_edge_lag_correlation' for c,v in semantics.items()}
        def code_hashes():
            parent=Path(os.environ['GOFA_WORKSPACE'])/'outputs/stage54-source-topology-audit-v1'
            if sha(parent/'report.json')!=STAGE54_REPORT or not p.read(parent/'final_verification.json')['passed']:
                raise ValueError('Verified fixed-weight topology audit required')
            return dict(hashes(),source_edge_lag_probe=sha(Path(__file__)),
                per_update_allnode_trainer=sha(Path(history.__file__)),stage54_report=STAGE54_REPORT)
        n.code_hashes=code_hashes
        try:yield
        finally:p.model_for,p.fit,n.PROTOCOL,n.stage_number,n.code_hashes,n.SEMANTICS=old


if __name__=='__main__':
    with edge_lag_runtime():n.main()
