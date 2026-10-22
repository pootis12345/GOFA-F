import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from gofa_f import source_graph_probe as p
from gofa_f import source_neighborhood_probe as n
from gofa_f.source_normalized_probe import normalized_runtime
from gofa_f.source_distance_prior_probe import (DistanceScore,shortest_paths,road_bias,add_distance_prior,distance_prior_runtime)
from gofa_f.lag_graph_probe import state_digest
from test_source_graph_probe import random_model,model_input


def test_published_direction_partition_and_inverse_hops():
    a=np.zeros((6,6),bool)
    for i in range(5):a[i,i+1]=True
    d=shortest_paths(a,[0,1,2,3]);assert d[0,3]==3 and np.isinf(d[3,0]) and np.isinf(d[0,4])
    d=shortest_paths(a|a.T,[0,1,2,3]);b=road_bias(d,np.array([0,1,2,3,-1]))
    np.testing.assert_allclose(np.exp(b[0,:4]),[1,1,.5,1/3],rtol=1e-6,atol=1e-6)
    assert not b[4].any()
    with pytest.raises(ValueError):road_bias(d,np.array([0,4]))


def test_static_target_fixed_permutation_preserves_distance_multiset():
    a=np.eye(9,k=1,dtype=bool);d=shortest_paths(a|a.T,list(range(9)));nodes=np.arange(9)
    real=road_bias(d,nodes);shuffled=road_bias(d,nodes,True)
    assert not np.array_equal(real,shuffled)
    np.testing.assert_array_equal(np.sort(real[0]),np.sort(shuffled[0]))
    np.testing.assert_array_equal(np.sort(real.ravel()),np.sort(shuffled.ravel()))
    np.testing.assert_array_equal(shuffled,road_bias(d,nodes,True))
    assert np.all(np.diag(shuffled)==0)


def test_score_exact_fixed_addition_no_added_parameters_and_gradients():
    original=torch.nn.Linear(2,1,bias=False);score=DistanceScore(original);edge=torch.randn(2,3,3,3,requires_grad=True)
    torch.testing.assert_close(score(edge),original(edge[...,:2])+edge[...,2:3],rtol=0,atol=0)
    assert score.weight is original.weight and sum(p.numel() for p in score.parameters())==2
    score(edge).sum().backward();assert torch.isfinite(edge.grad).all()


def test_exact_self_neutral_prior_and_nonzero_optimizer_replay():
    base=random_model();model=add_distance_prior(copy.deepcopy(base));x,a,edge=model_input()
    assert torch.equal(base(x,a,edge),model(x,a,edge))
    augmented=torch.cat((edge,torch.randn(*edge.shape[:-1],1)),-1);a[:]=False
    assert torch.equal(base(x,a,edge),model(x,a,augmented))
    a[:]=True;optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    def step():
        optimizer.zero_grad(set_to_none=True);loss=model(x,a,augmented).square().mean();loss.backward();optimizer.step()
        assert all(v.grad is None or v.grad.isfinite().all() for v in model.parameters());return state_digest(model)
    step();saved=copy.deepcopy((model.state_dict(),optimizer.state_dict()));expected=step()
    model.load_state_dict(saved[0]);optimizer.load_state_dict(saved[1]);assert step()==expected


def test_runtime_exact_initialization_config_parameter_count_and_restore(monkeypatch):
    monkeypatch.setenv('GOFA_WORKSPACE','/tmp/source');root=Path(__file__).parents[1];factory=p.model_for;semantics=n.SEMANTICS
    with normalized_runtime():
        cfg=n.configuration(root/'configs/stage40_source_normalized_probe.yaml');expected=state_digest(p.model_for(cfg,42))
    with distance_prior_runtime():
        cfg=n.configuration(root/'configs/stage47_source_distance_prior_probe.yaml');model=p.model_for(cfg,42)
        assert state_digest(model)==expected and sum(v.numel() for v in model.parameters())==10831
        assert n.stage_number(cfg)==47 and 'permuted_road_distances' in n.SEMANTICS['rewired']
    assert p.model_for is factory and n.SEMANTICS is semantics
