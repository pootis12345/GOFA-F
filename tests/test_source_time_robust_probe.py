import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from gofa_f import source_graph_probe as p
from gofa_f import source_neighborhood_probe as n
from gofa_f.source_time_robust_probe import TimeWeights,robust_loss,robust_runtime,temporal_group
from test_source_graph_probe import model_input


def test_uniform_groups_loss_view_balance_and_hard_group_weight_increases():
    state=TimeWeights();values=torch.tensor([1.,2.,3.,4.,10.,20.,30.,40.],requires_grad=True)
    loss=state.objective(values,torch.arange(8))
    assert float(loss)==pytest.approx((2.5+25)/2)
    loss.backward();torch.testing.assert_close(values.grad,torch.full((8,),.125))
    weights=state.logits.softmax(-1)
    assert weights[0,3]>weights[0,0] and weights[1,3]>weights[1,0]
    torch.testing.assert_close(weights.sum(1),torch.ones(2,dtype=torch.float64))
    assert state.updates==1 and list(state.parameters())==[]


def test_absent_groups_zero_error_and_missing_targets_are_finite():
    state=TimeWeights()
    b=dict(scale=torch.ones(2),delta=torch.tensor([[1.,100.],[2.,100.]]),
        observed=torch.tensor([[True,False],[True,False]]),weight=torch.ones(2),time_group=torch.tensor([0,4]))
    x=torch.zeros(2,2,requires_grad=True)
    loss=robust_loss(x,b,state);assert float(loss)==pytest.approx(1.5)
    loss.backward();assert torch.equal(x.grad[:,1],torch.zeros(2))
    assert state.logits.isfinite().all()
    with pytest.raises(ValueError,match='Both views'):state.objective(torch.ones(2),torch.tensor([0,1]))
    with pytest.raises(ValueError,match='Finite'):state.objective(torch.tensor([float('nan')]),torch.tensor([0]))


def test_groups_are_origin_only_and_reject_validation_or_other_sources():
    def row(origin,source='pems04_flow'):
        return dict(query=dict(source=source,origin=origin,variable=1,length=12,horizon=12))
    assert temporal_group(row(12))==0 and temporal_group(row(2600))==1
    assert temporal_group(row(5200,'pems04_speed'))==6
    a=row(7800);a['future_target']=[1e20];assert temporal_group(a)==3
    for r in (row(11000),row(12,'target'),row(0)):
        with pytest.raises(ValueError,match='training'):temporal_group(r)


def test_model_optimizer_weights_and_cursor_exact_replay_and_runtime_restoration(monkeypatch):
    monkeypatch.setenv('GOFA_WORKSPACE','/tmp/source')
    old=(p.model_for,p.tensors,p.loss_value,n.PROTOCOL)
    with robust_runtime():
        cfg=n.configuration(Path(__file__).parents[1]/'configs/stage41_source_time_robust_probe.yaml')
        model=p.model_for(cfg,42);optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        x,a,edge=model_input();b=dict(scale=torch.ones(2),delta=torch.full((2,12),.3),
            observed=torch.ones(2,12,dtype=torch.bool),weight=torch.ones(2)/12,time_group=torch.tensor([0,4]))
        rng=np.random.default_rng(42);cursor={v:dict(order=list(range(9)),position=0,epochs=0) for v in p.VIEWS}
        def advance():
            order=[p.next_indices(cursor,v,5,rng).tolist() for v in p.VIEWS]
            optimizer.zero_grad(set_to_none=True)
            residual=model(x,a,edge,return_residual=True)[:,0]
            loss=p.loss_value(residual,b);loss.backward();optimizer.step()
            assert all(q.grad is None or q.grad.isfinite().all() for q in model.parameters())
            return p.state_digest(model),order
        advance();saved=copy.deepcopy((model.state_dict(),optimizer.state_dict(),rng.bit_generator.state,cursor))
        expected=advance();model.load_state_dict(saved[0]);optimizer.load_state_dict(saved[1])
        rng.bit_generator.state=saved[2];cursor=saved[3]
        assert expected==advance()
    assert (p.model_for,p.tensors,p.loss_value,n.PROTOCOL)==old


def test_forecast_does_not_update_weights_and_training_metadata_omitted_from_eval(monkeypatch):
    monkeypatch.setenv('GOFA_WORKSPACE','/tmp/source')
    with robust_runtime():
        cfg=n.configuration(Path(__file__).parents[1]/'configs/stage41_source_time_robust_probe.yaml')
        model=p.model_for(cfg,42);x,a,edge=model_input();before=model.time_weights.logits.clone()
        arrays=dict(x=x.numpy(),adjacency=a.numpy(),edge_features=edge.numpy(),
            nodes=np.tile(np.arange(4),(2,1)),scale=np.ones(2))
        rows=[dict(query=dict(source=v,variable=0,origin=11000,length=12,horizon=12)) for v in p.VIEWS]
        assert 'time_group' not in p.tensors(arrays,np.arange(2),rows,'physical')
        with torch.no_grad():model(x,a,edge)
        assert torch.equal(model.time_weights.logits,before) and model.time_weights.updates==0
