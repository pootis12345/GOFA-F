from pathlib import Path

import numpy as np
import torch

from gofa_f import source_graph_probe as p
from gofa_f import source_neighborhood_probe as n
from gofa_f import source_sparse_normalized_probe as sparse
from test_source_graph_probe import random_model,model_input


def test_static_permutation_target_degrees_topology_and_padding():
    external=np.eye(9,k=1,dtype=bool);external|=external.T;nodes=np.arange(9)
    a=sparse.road_adjacency(external,nodes);b=sparse.road_adjacency(external,nodes,True)
    assert a.sum()==b.sum()==16 and not np.array_equal(a,b)
    assert a[0].sum()==b[0].sum() and a[:,0].sum()==b[:,0].sum()
    np.testing.assert_array_equal(np.sort(a.sum(0)),np.sort(b.sum(0)))
    np.testing.assert_array_equal(b,sparse.road_adjacency(external,nodes,True))
    padded=sparse.road_adjacency(external,np.array([0,1,2,-1,-1]))
    assert not padded[3:].any() and not padded[:,3:].any()


def test_control_not_double_permuted_and_exact_self_mapping(monkeypatch):
    monkeypatch.setenv('GOFA_AUTOMATION_DEADLINE_UNIX','9999999999')
    calls=[];monkeypatch.setattr(p,'tensors',lambda a,i,r,c,*x,**k:calls.append(c) or c)
    with sparse.sparse_runtime():
        assert p.tensors(None,None,None,'rewired')=='physical'
        assert p.tensors(None,None,None,'physical')=='physical'
        assert p.tensors(None,None,None,'self')=='self'
    assert calls==['physical','physical','self']


def test_self_forward_and_gradients_exactly_independent_of_road():
    model=random_model();x,a,edge=model_input();a[:]=False;copy_state={k:v.clone() for k,v in model.state_dict().items()}
    y=model(x,a,edge);y.square().sum().backward();grads=[None if v.grad is None else v.grad.clone() for v in model.parameters()]
    model.load_state_dict(copy_state);model.zero_grad();z=model(x,a,edge);z.square().sum().backward()
    assert torch.equal(y,z)
    for expected,value in zip(grads,model.parameters()):assert (expected is None and value.grad is None) or torch.equal(expected,value.grad)


def test_fixed_config_and_runtime_restore(monkeypatch):
    monkeypatch.setenv('GOFA_WORKSPACE','/tmp/source');monkeypatch.setenv('GOFA_AUTOMATION_DEADLINE_UNIX','9999999999')
    old=(p.tensors,p.loss_value,n.run,n.PROTOCOL,n.SEMANTICS,p.STOP[0])
    with sparse.sparse_runtime():
        cfg=n.configuration(Path(__file__).parents[1]/'configs/stage51_source_sparse_normalized.yaml')
        assert cfg['training']['max_updates']==1000 and n.stage_number(cfg)==51
        assert sum(x.numel() for x in p.model_for(cfg,42).parameters())==10831
    assert (p.tensors,p.loss_value,n.run,n.PROTOCOL,n.SEMANTICS,p.STOP[0])==old


def test_cooperative_deadline_does_not_modify_loss(monkeypatch):
    monkeypatch.setenv('GOFA_AUTOMATION_DEADLINE_UNIX','150');monkeypatch.setattr(sparse.time,'time',lambda:100.)
    monkeypatch.setattr(p,'loss_value',lambda x:x+1);original=p.STOP[0]
    with sparse.sparse_runtime():assert p.loss_value(2)==3 and p.STOP[0]
    assert p.STOP[0]==original
