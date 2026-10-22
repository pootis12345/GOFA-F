import copy
from pathlib import Path
import numpy as np
import torch
from gofa_f import source_graph_probe as p
from gofa_f import source_neighborhood_probe as n
from gofa_f import source_normalized_probe as bounded
from gofa_f import source_allnode_probe as allnode
from gofa_f import source_origin_history_probe as history
from gofa_f import source_edge_lag_probe as lag
from gofa_f.graph_probe_data import pack
from test_source_graph_probe import fixture,random_model,model_input
from test_source_allnode_probe import test_allnode_optimizer_rng_cursor_exact_resume as replay_check


def test_signed_receiver_sender_lag_alignment_and_missing_constant():
    torch.manual_seed(3);steps=torch.randn(11);receiver=torch.cat((torch.randn(2),steps[:-2]))
    x=torch.zeros(1,2,12,2);x[...,1]=1
    x[0,0,1:,0]=receiver.cumsum(0);x[0,1,1:,0]=steps.cumsum(0)
    corr=lag.lag_correlations(x)
    assert abs(corr[0,0,1,2].item()-1)<1e-6 and corr[0,0,1].argmax()==2
    x.zero_();assert torch.equal(lag.lag_correlations(x),torch.zeros(1,2,2,4))
    x[...,1]=1;assert torch.equal(lag.lag_correlations(x),torch.zeros(1,2,2,4))


def test_zero_coefficients_exact_old_nonzero_model_and_no_edge():
    base=bounded.add_normalization(random_model());model=lag.add_edge_lag(copy.deepcopy(base));x,a,e=model_input()
    assert torch.equal(base(x,a,e),model(x,a,e))
    a[:]=False;assert torch.equal(base(x,a,e),model(x,a,e))
    for block in model.exchanges:assert torch.equal(block.correlation_weight,torch.zeros(4))


def test_nonzero_correlation_parameters_learn_and_permutation_batch_isolation():
    model=lag.add_edge_lag(random_model());x,a,e=model_input();out=model(x,a,e)
    out.square().sum().backward()
    assert all(b.correlation_weight.grad.abs().sum()>0 and b.correlation_weight.grad.isfinite().all() for b in model.exchanges)
    with torch.no_grad():
        for block in model.exchanges:block.correlation_weight.copy_(torch.tensor([.2,-.4,.1,.3]))
    out=model(x,a,e);order=torch.tensor([2,0,3,1]);back=torch.argsort(order)
    perm=model(x[:,order],a[:,order][:,:,order],e[:,order][:,:,order])[:,back]
    torch.testing.assert_close(out,perm,rtol=0,atol=1e-6)
    torch.testing.assert_close(out,torch.cat([model(x[i:i+1],a[i:i+1],e[i:i+1]) for i in range(2)]),rtol=0,atol=1e-6)


def test_future_values_and_masks_cannot_change_edge_lag_features():
    data=fixture();r=dict(query=dict(source='pems04_flow',variable=2,origin=100,length=12,horizon=12),block=0)
    a=pack(data,r,list(range(8)),labels=False);before=lag.lag_correlations(torch.from_numpy(a['x'][None]))
    data.values[100:]=1e10;data.mask[100:]=False;b=pack(data,r,list(range(8)),labels=False)
    for key in a:np.testing.assert_array_equal(a[key],b[key])
    assert torch.equal(before,lag.lag_correlations(torch.from_numpy(b['x'][None])))


def test_actual_new_model_and_per_update_trainer_exact_resume(tmp_path,monkeypatch):
    factory=p.model_for
    monkeypatch.setattr(p,'model_for',lambda cfg,seed:lag.add_edge_lag(factory(cfg,seed)))
    monkeypatch.setattr(allnode,'fit_allnode',history.fit_origin_allnode)
    replay_check(tmp_path,monkeypatch)


def test_fixed_runtime_restore_and_only_eight_new_parameters(monkeypatch):
    monkeypatch.setenv('GOFA_WORKSPACE','/tmp/source');monkeypatch.setenv('GOFA_AUTOMATION_DEADLINE_UNIX','9999999999')
    old=(p.model_for,p.fit,n.PROTOCOL,n.stage_number,n.SEMANTICS)
    with lag.edge_lag_runtime():
        cfg=n.configuration(Path(__file__).parents[1]/'configs/stage55_source_edge_lag.yaml')
        assert n.stage_number(cfg)==55
        assert sum(v.numel() for v in p.model_for(cfg,42).parameters())==10839
    assert (p.model_for,p.fit,n.PROTOCOL,n.stage_number,n.SEMANTICS)==old
