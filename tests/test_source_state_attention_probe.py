import copy
from pathlib import Path

import pytest
import torch

from gofa_f import source_graph_probe as p
from gofa_f import source_neighborhood_probe as n
from gofa_f.source_state_attention_probe import add_state_attention,state_features,state_attention_runtime
from test_source_graph_probe import random_model,model_input


def test_shapes_last_value_trend_missing_and_permutation():
    x=torch.zeros(1,3,12,2);x[...,1]=1
    x[:,0,:,0]=torch.arange(12)/11;x[:,1,:,0]=1-torch.arange(12)/11;x[:,2,:,1]=0
    feature=state_features(x)
    assert feature.shape==(1,3,3,2) and feature.isfinite().all()
    assert feature[0,0,1,0]==1 and feature[0,0,1,1]>0
    assert torch.equal(torch.diagonal(feature,dim1=1,dim2=2),torch.zeros(1,2,3))
    perm=torch.tensor([2,0,1])
    torch.testing.assert_close(state_features(x[:,perm]),feature[:,perm][:,:,perm],rtol=0,atol=0)
    assert torch.equal(state_features(torch.zeros_like(x)),torch.zeros_like(feature))


def test_zero_initial_compatibility_replays_old_nonzero_model_and_self_exact():
    model=random_model();other=add_state_attention(copy.deepcopy(model));x,a,edge=model_input()
    torch.testing.assert_close(model(x,a,edge),other(x,a,edge),rtol=1e-6,atol=1e-6)
    a[:]=False
    assert torch.equal(model(x,a,edge),other(x,a,edge))
    with pytest.raises(ValueError):add_state_attention(other)


def test_new_coefficients_receive_nonzero_finite_source_style_gradients():
    model=add_state_attention(random_model());x,a,edge=model_input()
    loss=model(x,a,edge).square().mean();loss.backward()
    for block in model.exchanges:
        assert block.edge_score.weight.grad[:,2:].abs().sum()>0
    assert all(q.grad is None or q.grad.isfinite().all() for q in model.parameters())


def test_batch_node_isolation_and_optimizer_exact_replay():
    model=add_state_attention(random_model());x,a,edge=model_input();perm=torch.tensor([2,0,3,1])
    y=model(x,a,edge)
    torch.testing.assert_close(y[:,perm],model(x[:,perm],a[:,perm][:,:,perm],edge[:,perm][:,:,perm]),rtol=1e-6,atol=1e-6)
    torch.testing.assert_close(y[:1],model(x[:1],a[:1],edge[:1]),rtol=1e-6,atol=1e-6)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    def step():
        optimizer.zero_grad(set_to_none=True);loss=model(x,a,edge).square().mean();loss.backward();optimizer.step()
        return p.state_digest(model)
    step();saved=copy.deepcopy((model.state_dict(),optimizer.state_dict()));expected=step()
    model.load_state_dict(saved[0]);optimizer.load_state_dict(saved[1]);assert step()==expected


def test_runtime_scope_new_parameter_budget_and_rng_preservation(monkeypatch):
    monkeypatch.setenv('GOFA_WORKSPACE','/tmp/source')
    cfg=n.configuration(Path(__file__).parents[1]/'configs/stage40_source_normalized_probe.yaml')
    before=p.model_for;baseline=before(cfg,42);rng=torch.get_rng_state().clone()
    with state_attention_runtime():
        fixed=n.configuration(Path(__file__).parents[1]/'configs/stage43_source_state_attention_probe.yaml')
        model=p.model_for(fixed,42)
        assert sum(q.numel() for q in model.parameters())==sum(q.numel() for q in baseline.parameters())+4
        assert torch.equal(torch.get_rng_state(),rng)
        assert all(torch.equal(b.edge_score.weight[:,2:],torch.zeros(1,2)) for b in model.exchanges)
    assert p.model_for is before
