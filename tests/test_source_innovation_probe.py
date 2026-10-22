import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from gofa_f import source_graph_probe as p
from gofa_f import source_neighborhood_probe as n
from gofa_f import source_innovation_probe as innovation
from gofa_f.source_innovation_helpers import centered_history,baseline_delta,self_features,self_test
from gofa_f.source_normalized_probe import normalized_runtime
from gofa_f.lag_graph_probe import state_digest
from test_source_graph_probe import random_model,model_input


def model():return dict(mean=[0.]*24,std=[1.]*24,coefficients=np.zeros((24,12)).tolist(),intercept=[.25]*12)


def test_causal_features_missing_constants_and_exact_frozen_formula():
    self_test();x=np.zeros((2,9,12,2),np.float32);x[:,0,:,1]=1
    x[:,0,:,0]=np.arange(12)/12;x[:,1,:,0]=np.nan;original=x.copy()
    centered=centered_history(x)
    assert np.isfinite(centered).all() and not centered[:,0,-1,0].any()
    np.testing.assert_array_equal(centered[...,1],x[...,1]);np.testing.assert_array_equal(x,original)
    np.testing.assert_array_equal(baseline_delta(x,model()),np.full((2,12),.25))


def test_residual_target_subtraction_and_original_labels_unchanged():
    a=dict(x=np.zeros((2,9,12,2),np.float32),adjacency=np.zeros((2,9,9),bool),
        edge_features=np.zeros((2,9,9,2),np.float32),nodes=np.tile(np.arange(9),(2,1)),
        scale=np.array([2.,4.]),persistence=np.array([10.,20.]),raw_target=np.array([[16.]*12,[32.]*12]),
        observed=np.ones((2,12),bool),baseline_delta_normalized=np.full((2,12),.25))
    rows=[dict(query=dict(source='pems04_flow',variable=i,origin=100,length=12,horizon=12)) for i in range(2)]
    old=copy.deepcopy(a);ix=np.array([0,1]);weight=np.ones(2,np.float32)
    b=innovation.residual_training_batch(p.tensors,a,ix,rows,'physical',weight)
    expected=np.array([[5.5]*12,[11.]*12],np.float32);np.testing.assert_array_equal(b['delta'],expected)
    for k in a:np.testing.assert_array_equal(a[k],old[k])
    out=innovation.residual_training_batch(p.tensors,a,ix,rows,'self');assert 'delta' not in out


def test_runtime_zero_correction_returns_exact_fixed_baseline_and_restores(monkeypatch):
    monkeypatch.setattr(innovation,'frozen_models',lambda:({v:model() for v in p.VIEWS},{}))
    monkeypatch.setattr(p,'forecast',lambda model,a,rows,c:np.broadcast_to(a['persistence'][:,None],(2,12)).copy())
    old=(p.tensors,p.forecast,p.model_for,n.caches,n.PROTOCOL)
    a=dict(persistence=np.array([10.,20.]),scale=np.array([2.,4.]),baseline_delta_normalized=np.full((2,12),.25))
    with innovation.innovation_runtime():
        np.testing.assert_array_equal(p.forecast(None,a,None,'self'),np.array([[10.5]*12,[21.]*12]))
    assert (p.tensors,p.forecast,p.model_for,n.caches,n.PROTOCOL)==old


def test_same_initialization_parameter_count_and_fixed_config(monkeypatch):
    monkeypatch.setenv('GOFA_WORKSPACE','/tmp/source');root=Path(__file__).parents[1]
    monkeypatch.setattr(innovation,'frozen_models',lambda:({v:model() for v in p.VIEWS},{}))
    with normalized_runtime():
        cfg=n.configuration(root/'configs/stage40_source_normalized_probe.yaml');expected=state_digest(p.model_for(cfg,42))
    with innovation.innovation_runtime():
        cfg=n.configuration(root/'configs/stage49_source_innovation_probe.yaml');own=p.model_for(cfg,42)
        assert state_digest(own)==expected and sum(v.numel() for v in own.parameters())==10831
        assert n.stage_number(cfg)==49


def test_nonzero_centered_forward_gradients_and_optimizer_replay():
    network=random_model();x=torch.randn(2,9,12,2);x[...,1]=1
    a=torch.ones(2,9,9,dtype=torch.bool);edge=torch.randn(2,9,9,2)
    x=torch.from_numpy(centered_history(x.numpy())).float()
    optimizer=torch.optim.AdamW(network.parameters(),lr=.001)
    def step():
        optimizer.zero_grad(set_to_none=True);loss=(network(x,a,edge)-.25).square().mean();loss.backward();optimizer.step()
        assert all(v.grad is None or v.grad.isfinite().all() for v in network.parameters());return state_digest(network)
    step();saved=copy.deepcopy((network.state_dict(),optimizer.state_dict()));expected=step()
    network.load_state_dict(saved[0]);optimizer.load_state_dict(saved[1]);assert step()==expected
