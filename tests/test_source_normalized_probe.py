import copy
from pathlib import Path

import pytest
import torch

from gofa_f import source_graph_probe as p
from gofa_f import source_neighborhood_probe as n
from gofa_f.lag_graph_probe import LagExchange,state_digest
from gofa_f.source_normalized_probe import add_normalization,bounded_exchange,normalized_runtime
from test_source_graph_probe import model_input,random_model


def test_nonzero_message_bound_and_finite_backward_even_at_zero_hidden():
    e=LagExchange(8,4)
    with torch.no_grad():e.up.weight.normal_(0,.5)
    h=torch.randn(2,3,12,8,requires_grad=True)
    observed=torch.ones(2,3,12,dtype=torch.bool);a=torch.ones(2,3,3,dtype=torch.bool)
    edge=torch.randn(2,3,3,2)
    for hidden in (h,torch.zeros_like(h,requires_grad=True)):
        raw=e(hidden,observed,a,edge);result=bounded_exchange(e,(hidden,observed,a,edge),raw)
        assert ((result-hidden).norm(dim=-1)<=hidden.norm(dim=-1)+1e-6).all()
        result.square().sum().backward()
        assert hidden.grad is not None and hidden.grad.isfinite().all()
    assert all(par.grad is None or par.grad.isfinite().all() for par in e.parameters())


def test_self_control_values_and_all_parameter_gradients_exactly_unchanged():
    model=random_model();other=add_normalization(copy.deepcopy(model))
    x,a,edge=model_input();a[:]=False
    y=model(x,a,edge);z=other(x,a,edge)
    assert torch.equal(y,z)
    y.square().sum().backward();z.square().sum().backward()
    for p1,p2 in zip(model.parameters(),other.parameters()):
        assert (p1.grad is None and p2.grad is None) or torch.equal(p1.grad,p2.grad)


def test_runtime_restores_factory_and_exact_initialization_and_fixed_config(monkeypatch):
    monkeypatch.setenv('GOFA_WORKSPACE','/tmp/source')
    old=p.model_for;old_protocol=n.PROTOCOL
    cfg=n.configuration(Path(__file__).parents[1]/'configs/stage38_source_neighborhood_probe.yaml')
    before=state_digest(old(cfg,42))
    with normalized_runtime():
        fixed=n.configuration(Path(__file__).parents[1]/'configs/stage40_source_normalized_probe.yaml')
        model=p.model_for(fixed,42)
        assert state_digest(model)==before
        assert len(list(model.parameters()))==len(list(old(cfg,42).parameters()))
        with pytest.raises(ValueError):add_normalization(model)
    assert p.model_for is old and n.PROTOCOL==old_protocol


def test_nonzero_normalized_model_permutation_batch_and_optimizer_replay():
    model=add_normalization(random_model());x,a,edge=model_input()
    y=model(x,a,edge);perm=torch.tensor([2,0,3,1])
    torch.testing.assert_close(y[:,perm],model(x[:,perm],a[:,perm][:,:,perm],edge[:,perm][:,:,perm]),rtol=1e-6,atol=1e-6)
    torch.testing.assert_close(y[:1],model(x[:1],a[:1],edge[:1]),rtol=1e-6,atol=1e-6)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    def step():
        optimizer.zero_grad(set_to_none=True);loss=model(x,a,edge).square().mean();loss.backward();optimizer.step()
        assert all(p.grad is None or p.grad.isfinite().all() for p in model.parameters())
        return state_digest(model)
    step();saved=copy.deepcopy((model.state_dict(),optimizer.state_dict()))
    expected=step();model.load_state_dict(saved[0]);optimizer.load_state_dict(saved[1]);assert step()==expected


def test_missing_nodes_and_future_lag_causality_preserved():
    e=LagExchange(8,4)
    with torch.no_grad():e.up.weight.normal_(0,.5)
    e.register_forward_hook(bounded_exchange)
    h=torch.randn(1,3,12,8);obs=torch.ones(1,3,12,dtype=torch.bool)
    a=torch.zeros(1,3,3,dtype=torch.bool);a[:,0,1]=True;edge=torch.zeros(1,3,3,2)
    y=e(h,obs,a,edge);changed=h.clone();changed[:,1,7:]+=100
    torch.testing.assert_close(y[:,0,:7],e(changed,obs,a,edge)[:,0,:7],rtol=1e-6,atol=1e-6)
    obs[:]=False
    assert torch.equal(e(h,obs,a,edge),torch.zeros_like(h))
