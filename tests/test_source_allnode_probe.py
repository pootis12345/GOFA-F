import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from gofa_f import source_graph_probe as p
from gofa_f import source_neighborhood_probe as n
from gofa_f import source_sparse_normalized_probe as sparse
from gofa_f import source_allnode_probe as allnode
from gofa_f.graph_probe_data import pack
from test_source_graph_probe import fixture,random_model,model_input


def sample():
    d=fixture();rows=[dict(query=dict(source='pems04_flow',variable=2,origin=o,length=12,horizon=12),block=0) for o in (80,100)]
    packed=[pack(d,r,list(range(8))) for r in rows]
    a={k:np.stack([x[k] for x in packed]) for k in packed[0]}
    return d,rows,a


def test_allnode_persistence_loss_matches_variable_macro_and_point_weights():
    d,rows,a=sample();extra,proof=allnode.auxiliary_labels(d,a,rows,list(range(8)))
    # Zero normalized residual is exactly the train-only persistence reference.
    b=allnode.training_batch(dict(a,**extra),np.arange(2),rows,'physical')
    assert abs(allnode.allnode_loss(torch.zeros(2,9,12),b).item()-1)<1e-6
    sums={v:0. for v in range(8)}
    for i,nodes in enumerate(a['nodes']):
        for j,v in enumerate(nodes):
            if v>=0:sums[v]+=extra['node_weight'][i,j]*extra['node_observed'][i,j].sum()/2*proof['persistence_macro_mae']
    np.testing.assert_allclose(list(sums.values()),np.full(8,1/8),rtol=0,atol=1e-12)
    assert proof['valid_points']==2*8*12


def test_auxiliary_future_changes_no_input_scale_or_graph():
    d,rows,a=sample();rows=rows[1:];a={k:v[1:] for k,v in a.items()}
    extra,_=allnode.auxiliary_labels(d,a,rows,list(range(8)))
    d.values[100:]+=100;d.mask[100:,3]=False
    changed=pack(d,rows[0],list(range(8)),labels=False)
    for k in changed:np.testing.assert_array_equal(a[k][0],changed[k])
    newer,_=allnode.auxiliary_labels(d,a,rows,list(range(8)))
    np.testing.assert_array_equal(extra['node_scale'],newer['node_scale'])
    assert not np.array_equal(extra['node_delta'],newer['node_delta'])


def test_missing_padding_constant_and_train_boundary():
    d,rows,a=sample();d.values[:112,2]=4.;d.mask[:,3]=False
    packed=[pack(d,r,list(range(8))) for r in rows];a={k:np.stack([x[k] for x in packed]) for k in packed[0]}
    e,_=allnode.auxiliary_labels(d,a,rows,list(range(8)))
    assert (e['node_scale'][:,0]==1).all() and not e['node_observed'][:,:,][a['nodes']==3].any()
    assert not e['node_observed'][a['nodes']<0].any() and not e['node_weight'][a['nodes']<0].any()
    invalid=copy.deepcopy(rows);invalid[0]['query']['origin']=175
    with pytest.raises(ValueError,match='first60'):allnode.auxiliary_labels(d,a,invalid,list(range(8)))
    with pytest.raises(ValueError,match='partition'):allnode.auxiliary_labels(d,a,rows,[2])


def test_auxiliary_slots_drive_gradients_and_full_shape_required():
    model=random_model();x,adj,edge=model_input();y=model(x,adj,edge,12,return_residual=True)
    observed=torch.zeros_like(y,dtype=torch.bool);observed[:,1]=True
    b=dict(node_scale=torch.ones(2,4),node_delta=torch.ones_like(y),node_observed=observed,node_weight=torch.ones(2,4))
    allnode.allnode_loss(y,b).backward()
    assert any(v.grad is not None and v.grad.abs().sum()>0 for v in model.parameters())
    with pytest.raises(ValueError,match='Complete'):allnode.allnode_loss(y[:,0],b)


def test_runtime_disables_self_reuse_and_restores_without_evaluation_aux(monkeypatch):
    monkeypatch.setenv('GOFA_AUTOMATION_DEADLINE_UNIX','9999999999');monkeypatch.setenv('GOFA_WORKSPACE','/tmp/source')
    old=(p.fit,sparse.reuse_self,n.caches,n.stage_number)
    with allnode.allnode_runtime():
        assert p.fit is allnode.fit_allnode and sparse.reuse_self(None,None) is None
        cfg=n.configuration(Path(__file__).parents[1]/'configs/stage52_source_allnode_supervision.yaml')
        assert n.stage_number(cfg)==52 and cfg['training']['max_updates']==1000
        d,rows,a=sample();b=p.tensors(a,np.arange(2),rows,'physical')
        assert not set(allnode.AUX_KEYS)&set(b)
    assert (p.fit,sparse.reuse_self,n.caches,n.stage_number)==old


def test_allnode_optimizer_rng_cursor_exact_resume(tmp_path,monkeypatch):
    monkeypatch.setenv('GOFA_AUTOMATION_DEADLINE_UNIX','9999999999')
    d,rows,a=sample();e,_=allnode.auxiliary_labels(d,a,rows,list(range(8)));train=dict(a,**e)
    packed=dict(train={v:train for v in p.VIEWS},validation={v:a for v in p.VIEWS})
    m=dict(queries={s:{v:rows for v in p.VIEWS} for s in ('train','validation')})
    cfg=dict(model=dict(width=8,rank=4,lags=[0,1,2,3]),training=dict(max_updates=50))
    speed=dict(seconds_per_update=.1)
    whole=tmp_path/'whole';resume=tmp_path/'resume'
    for folder in (whole,resume):folder.mkdir();p.immutable(folder/'manifest.json',dict(test='identical'))
    old=p.STOP[0]
    try:
        p.STOP[0]=False;allnode.fit_allnode(cfg,whole,m,packed,42,'physical',lambda:None,speed)
        def stop_after_one():
            if (resume/'models/seed-42/physical/last.pt').exists():
                saved=torch.load(resume/'models/seed-42/physical/last.pt',weights_only=False)
                if saved['update']==1:p.STOP[0]=True
        p.STOP[0]=False;assert allnode.fit_allnode(cfg,resume,m,packed,42,'physical',stop_after_one,speed) is None
        p.STOP[0]=False;allnode.fit_allnode(cfg,resume,m,packed,42,'physical',lambda:None,speed)
        # Both executions reach the same completed optimizer transaction at50.
        for folder in (whole,resume):
            saved=torch.load(folder/'models/seed-42/physical/last.pt',weights_only=False)
            assert saved['update']==50
        left=torch.load(whole/'models/seed-42/physical/last.pt',weights_only=False)
        right=torch.load(resume/'models/seed-42/physical/last.pt',weights_only=False)
        assert left['cursor']==right['cursor'] and left['rng']==right['rng']
        for key in left['model']:assert torch.equal(left['model'][key],right['model'][key])
        for slot,state in left['optimizer']['state'].items():
            for key,value in state.items():assert torch.equal(value,right['optimizer']['state'][slot][key])
        assert torch.equal(left['torch_rng'],right['torch_rng'])
    finally:p.STOP[0]=old
