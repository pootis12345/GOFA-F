import copy
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from gofa_f import source_graph_probe as p
from gofa_f import source_neighborhood_probe as n
from gofa_f import source_origin_history_probe as origin
from gofa_f.graph_probe_data import pack
from test_source_graph_probe import fixture,random_model,model_input
from test_source_allnode_probe import test_allnode_optimizer_rng_cursor_exact_resume as replay_check


def test_full_history_attention_matches_explicit_reference():
    model=random_model();old=model.exchanges[0];block=origin.OriginHistoryExchange(old)
    h=torch.randn(2,3,12,8);mask=torch.rand(2,3,12)>.2
    a=torch.zeros(2,3,3,dtype=torch.bool);a[:,0,1]=True;a[:,1,2]=True
    edge=torch.randn(2,3,3,2);actual=block(h,mask,a,edge)
    z=block.norm(h);expected=h.clone();present=mask.any(-1)
    for b in range(2):
        for t in range(12):
            for i in range(3):
                pairs=[(j,k) for j in range(3) for k in range(12) if j!=i and a[b,i,j] and present[b,i] and present[b,j] and mask[b,j,k]]
                if not pairs:continue
                scores=torch.stack([(block.query(z[b,i,t])*block.key(z[b,j,k])).sum()/(block.rank**.5)+block.edge_score(edge[b,i,j]).squeeze() for j,k in pairs])
                values=torch.stack([block.value(z[b,j,k])+block.edge_value(edge[b,i,j]) for j,k in pairs])
                context=(scores.softmax(0)[:,None]*values).sum(0)
                expected[b,i,t]+=block.up(F.gelu(context))*block.gate(z[b,i,t]).sigmoid()
    torch.testing.assert_close(actual,expected*present[:,:,None,None],rtol=0,atol=1e-6)


def test_later_observed_history_access_is_intentional_not_future_access():
    block=origin.OriginHistoryExchange(random_model().exchanges[0])
    with torch.no_grad():
        block.query.weight.zero_();block.key.weight.zero_();block.value.weight.copy_(torch.eye(8));block.up.weight.copy_(torch.eye(8));block.edge_value.weight.zero_()
    h=torch.randn(1,2,12,8);mask=torch.ones(1,2,12,dtype=torch.bool)
    a=torch.zeros(1,2,2,dtype=torch.bool);a[:,0,1]=True;edge=torch.zeros(1,2,2,2)
    before=block(h,mask,a,edge);changed=h.clone();changed[:,1,-1,0]+=20
    after=block(changed,mask,a,edge)
    assert not torch.equal(before[:,0,0],after[:,0,0])
    # The last historical slot is legal at the forecast origin; no future label input exists.
    d=fixture();r=dict(query=dict(source='pems04_flow',variable=2,origin=100,length=12,horizon=12),block=0)
    x=pack(d,r,list(range(8)),labels=False);d.values[100:]=1e12;d.mask[100:]=False
    y=pack(d,r,list(range(8)),labels=False)
    for key in x:np.testing.assert_array_equal(x[key],y[key])


def test_no_edge_exact_old_self_and_empty_history_finite():
    original=random_model();model=origin.add_origin_history(copy.deepcopy(original));x,a,edge=model_input();a[:]=False
    y=original(x,a,edge);z=model(x,a,edge)
    assert torch.equal(y,z)
    y.square().sum().backward();z.square().sum().backward()
    old=dict(original.named_parameters())
    for name,value in model.named_parameters():
        before=old[name].grad
        assert (before is None and value.grad is None) or torch.equal(before,value.grad)
    x[...,1]=0;out=model(x,a,edge);assert torch.equal(out,torch.zeros_like(out))
    assert all('lag_bias' not in name for name,_ in model.named_parameters())


def test_node_permutation_batch_isolation_and_nonzero_gradients():
    model=origin.add_origin_history(random_model());x,a,edge=model_input()
    out=model(x,a,edge);order=torch.tensor([2,0,3,1]);inverse=torch.argsort(order)
    perm=model(x[:,order],a[:,order][:,:,order],edge[:,order][:,:,order])[:,inverse]
    torch.testing.assert_close(out,perm,rtol=0,atol=1e-6)
    torch.testing.assert_close(out,torch.cat([model(x[i:i+1],a[i:i+1],edge[i:i+1]) for i in range(2)]),rtol=0,atol=1e-6)
    out.square().sum().backward()
    assert all(v.grad is None or v.grad.isfinite().all() for v in model.parameters())
    assert all(block.up.weight.grad.abs().sum()>0 for block in model.exchanges)


def test_actual_origin_model_optimizer_rng_cursor_replay(tmp_path,monkeypatch):
    factory=p.model_for
    monkeypatch.setattr(p,'model_for',lambda cfg,seed:origin.add_origin_history(factory(cfg,seed)))
    # Existing exact transaction replay runs on the modified model, not an unmodified surrogate.
    replay_check(tmp_path,monkeypatch)


def test_runtime_parameter_accounting_restore_and_fixed_configuration(monkeypatch):
    monkeypatch.setenv('GOFA_WORKSPACE','/tmp/source');monkeypatch.setenv('GOFA_AUTOMATION_DEADLINE_UNIX','9999999999')
    old=(p.model_for,p.fit,n.PROTOCOL,n.stage_number,n.SEMANTICS)
    cfg=dict(model=dict(width=32,rank=8,lags=[0,1,2,3]))
    base=p.model_for(cfg,42).state_dict()
    with origin.origin_history_runtime():
        fixed=n.configuration(Path(__file__).parents[1]/'configs/stage53_source_origin_history.yaml')
        assert n.stage_number(fixed)==53 and p.fit is origin.fit_origin_allnode
        model=p.model_for(fixed,42)
        assert sum(v.numel() for v in model.parameters())==10823
        for name,value in model.state_dict().items():assert torch.equal(value,base[name])
    assert (p.model_for,p.fit,n.PROTOCOL,n.stage_number,n.SEMANTICS)==old
