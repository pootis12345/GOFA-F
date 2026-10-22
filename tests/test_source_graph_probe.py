import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gofa_f.graph_probe_data import (CONDITIONS,VIEWS,component_partition,controls,
    node_roster,pack,road_adjacency,select_training)
from gofa_f.lag_graph_probe import LagExchange,LagGraphProbe,state_digest
from gofa_f.source_graph_probe import (configuration,confirmation,digest,gate,
    immutable,loss_value,metrics,next_indices,train_weights,verify)


def fixture():
    values=(np.sin(np.arange(300)[:,None]/10+np.arange(8)[None])
            +np.arange(8)[None]*2).astype(np.float32)
    adjacency=np.zeros((8,8),bool)
    for i in range(7):adjacency[i+1,i]=True
    return SimpleNamespace(values=values,mask=np.ones_like(values,bool),
        external=adjacency,segments=np.zeros(300),ids=list(map(str,range(8))))


def random_model():
    torch.manual_seed(42)
    model=LagGraphProbe(width=8,rank=4)
    with torch.no_grad():
        model.head[-1].weight.normal_(0,.1)
        for e in model.exchanges:e.up.weight.normal_(0,.1)
    return model


def model_input():
    x=torch.randn(2,4,12,2);x[...,1]=1
    a=torch.zeros(2,4,4,dtype=torch.bool);a[:,1,0]=True;a[:,2,1]=True;a[:,0,3]=True
    edge=torch.randn(2,4,4,2)
    return x,a,edge


def test_components_wholly_excluded_from_training():
    adjacency=np.zeros((48,48),bool)
    start=0
    for count in (24,16,4,4):
        for i in range(start,start+count-1):adjacency[i+1,i]=True
        start+=count
    split=component_partition(adjacency,list(map(str,range(48))))
    assert [len(split[k]) for k in ('train','validation','confirmation')]==[24,8,16]
    assert set(split['train'])==set(range(24))
    assert set(split['confirmation'])==set(range(24,40))
    with pytest.raises(ValueError):component_partition(np.eye(3),['a','b','c'])


def test_roster_direction_and_node_partition_cannot_leak():
    d=fixture()
    r=dict(query=dict(source='pems04_flow',variable=2,origin=100,length=12,horizon=12),block=-1)
    a=pack(d,r,[0,1,2,3])
    assert set(a['nodes'][a['nodes']>=0])=={0,1,2,3}
    ids=a['nodes'].tolist()
    assert a['adjacency'][ids.index(2),ids.index(1)]
    assert not a['adjacency'][ids.index(1),ids.index(2)]
    d.values[:,4:]=1e10;d.mask[:,4:]=False
    b=pack(d,r,[0,1,2,3])
    for key in a:np.testing.assert_array_equal(a[key],b[key])
    with pytest.raises(ValueError):node_roster(d.external,2,[0,1],d.ids)


def test_every_future_value_missingness_and_auxiliary_label_are_input_independent():
    d=fixture()
    r=dict(query=dict(source='pems04_flow',variable=2,origin=100,length=12,horizon=12),block=-1)
    a=pack(d,r,list(range(8)),labels=False)
    d.values[100:]=1e8;d.mask[100:]=False
    b=pack(d,r,list(range(8)),labels=False)
    for key in a:np.testing.assert_array_equal(a[key],b[key])
    d=fixture();a=pack(d,r,list(range(8)))
    others=[i for i in range(8) if i!=2]
    d.values[100:,others]=9e7;d.mask[100:,others]=False
    b=pack(d,r,list(range(8)))
    for key in a:np.testing.assert_array_equal(a[key],b[key])


def test_constant_missing_neighbor_and_source_boundaries():
    d=fixture();d.values[:100,2]=4.;d.mask[:100,3]=False
    r=dict(query=dict(source='pems04_speed',variable=2,origin=100,length=12,horizon=12),block=-1)
    a=pack(d,r,list(range(8)))
    assert a['scale']==1 and np.isfinite(a['x']).all() and np.isfinite(a['edge_features']).all()
    with pytest.raises(ValueError,match='split'):
        pack(d,dict(r,query=dict(r['query'],origin=175)),list(range(8)))
    confirm=dict(r,query=dict(r['query'],origin=260))
    with pytest.raises(ValueError,match='Confirmation'):
        pack(d,confirm,list(range(8)),split='confirmation')
    c=pack(d,confirm,list(range(8)),labels=False,split='confirmation')
    assert not {'y','observed','raw_target'}&set(c)


def test_zero_cross_edge_exact_identity_and_missing_nodes_no_nan_gradient():
    e=LagExchange(8,4)
    with torch.no_grad():e.up.weight.normal_()
    h=torch.randn(2,3,12,8,requires_grad=True)
    observed=torch.ones(2,3,12,dtype=torch.bool)
    a=torch.eye(3,dtype=torch.bool).expand(2,-1,-1)
    y=e(h,observed,a,torch.randn(2,3,3,2))
    assert torch.equal(h,y)
    observed[:]=False
    y=e(h,observed,a,torch.randn(2,3,3,2))
    assert torch.equal(y,torch.zeros_like(y))
    y.sum().backward();assert h.grad.isfinite().all()
    assert all(p.grad is None or p.grad.isfinite().all() for p in e.parameters())


def test_lag_exchange_never_reads_later_hidden_and_preserves_road_direction():
    torch.manual_seed(42)
    e=LagExchange(8,4)
    with torch.no_grad():e.up.weight.normal_(0,.1)
    h=torch.randn(1,3,12,8)
    observed=torch.ones(1,3,12,dtype=torch.bool)
    a=torch.zeros(1,3,3,dtype=torch.bool);a[:,0,1]=True
    edge=torch.zeros(1,3,3,2)
    y=e(h,observed,a,edge)
    changed=h.clone();changed[:,1,7:]+=100*torch.randn_like(changed[:,1,7:])
    other=e(changed,observed,a,edge)
    torch.testing.assert_close(y[:,0,:7],other[:,0,:7],rtol=1e-6,atol=1e-6)
    assert not torch.allclose(y[:,0,7:],other[:,0,7:])
    assert torch.equal(y[:,1],h[:,1])
    assert torch.equal(y[:,2],h[:,2])


def test_model_node_permutation_batch_isolation_and_self_ignores_neighbors():
    model=random_model();x,a,edge=model_input()
    y=model(x,a,edge)
    perm=torch.tensor([2,0,3,1])
    other=model(x[:,perm],a[:,perm][:,:,perm],edge[:,perm][:,:,perm])
    torch.testing.assert_close(y[:,perm],other,rtol=1e-6,atol=1e-6)
    torch.testing.assert_close(y[:1],model(x[:1],a[:1],edge[:1]),rtol=1e-6,atol=1e-6)
    zero=torch.zeros_like(a)
    self_prediction=model(x,zero,edge)[:,0]
    changed=x.clone();changed[:,1:,:,0]+=100
    torch.testing.assert_close(self_prediction,model(changed,zero,edge)[:,0],rtol=0,atol=0)
    assert model(x,a,edge,horizon=24).shape==(2,4,24)


def test_exact_seed_matched_initialization_and_control_degree_invariance():
    initial=[]
    for _ in CONDITIONS:
        torch.manual_seed(42);initial.append(state_digest(LagGraphProbe(8,4)))
    assert len(set(initial))==1
    d=fixture();r=dict(query=dict(source='pems04_flow',variable=2,origin=100,length=12,horizon=12),block=-1)
    a=pack(d,r,list(range(8)))
    original=a['adjacency'][None];nodes=a['nodes'][None]
    scrambled=controls(original,nodes,[r],'rewired')
    assert np.count_nonzero(original)==np.count_nonzero(scrambled)
    assert np.count_nonzero(original[0,0])==np.count_nonzero(scrambled[0,0])
    assert not np.array_equal(original,scrambled)
    np.testing.assert_array_equal(scrambled,controls(original,nodes,[r],'rewired'))
    assert not controls(original,nodes,[r],'self').any()


def test_original_unit_variable_equal_weighted_loss():
    records=[dict(query=dict(variable=v)) for v in (1,1,2)]
    truth=np.array([[2.,9.],[4.,6.],[20.,99.]])
    observed=np.array([[1,0],[1,1],[1,0]],bool)
    a=dict(raw_target=truth,observed=observed,persistence=np.zeros(3))
    weights,norm=train_weights(a,records)
    b=dict(scale=torch.ones(3),delta=torch.tensor(truth,dtype=torch.float32),
        observed=torch.from_numpy(observed),weight=torch.from_numpy(weights/norm))
    # Both variable-equal raw MAEs are(4+20)/2=12; normalized persistence loss1.
    assert norm==12.
    assert float(loss_value(torch.zeros(3,2),b))==pytest.approx(1.,abs=1e-6)
    output=metrics(np.zeros_like(truth),a,[dict(r,block=0) for r in records])
    assert output['macro_mae']==12.


def test_optimizer_model_and_cursor_replay():
    torch.manual_seed(42)
    model=LagGraphProbe(8,4);optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    x,a,edge=model_input()
    rng=np.random.default_rng(42)
    cursor={name:dict(order=rng.permutation(11).tolist(),position=0,epochs=0) for name in VIEWS}
    saved=copy.deepcopy((model.state_dict(),optimizer.state_dict(),rng.bit_generator.state,cursor))
    def advance():
        order=[next_indices(cursor,n,8,rng).tolist() for n in VIEWS]
        optimizer.zero_grad(set_to_none=True)
        pred=model(x,a,edge,return_residual=True)
        loss=(pred-.3).square().mean();loss.backward();optimizer.step()
        assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())
        return state_digest(model),order
    first=advance()
    model.load_state_dict(saved[0]);optimizer.load_state_dict(saved[1]);rng.bit_generator.state=saved[2];cursor=saved[3]
    assert first==advance()


def test_nonzero_graph_branches_receive_finite_gradients():
    model=random_model();x,a,edge=model_input()
    loss=model(x,a,edge,return_residual=True).square().mean()
    loss.backward()
    for exchange in model.exchanges:
        assert exchange.up.weight.grad.abs().sum()>0
        assert exchange.value.weight.grad.abs().sum()>0
        assert exchange.query.weight.grad.abs().sum()>0
    assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())


def test_gates_reject_ties_bad_quarter_and_bad_repeat():
    cfg=dict(gate=dict(minimum_view_improvement=.01,maximum_time_block_degradation=.02))
    def selections(real):
        result={}
        for seed in (42,43,44):
            result[seed]={}
            for c in CONDITIONS:
                error=real if c=='physical' else 1.
                result[seed][c]=dict(validation={v:dict(macro_mae=error,
                    time_blocks={str(i):error for i in range(4)},
                    variables={str(i):dict(mae=error) for i in range(8)}) for v in VIEWS})
        return result
    assert gate(selections(.9),cfg,repeated=True)['passed']
    assert not gate(selections(1.),cfg)['passed']
    old=selections(.9);old[43]['physical']['validation'][VIEWS[0]]['macro_mae']=1.01
    assert not gate(old,cfg,repeated=True)['passed']
    old=selections(.9);old[42]['physical']['validation'][VIEWS[0]]['time_blocks']['2']=1.2
    assert not gate({42:old[42]},cfg)['passed']
    zero=selections(0.)
    for seed in zero:
        for c in CONDITIONS:
            for v in VIEWS:zero[seed][c]['validation'][v]['macro_mae']=0.
    assert not gate(zero,cfg)['passed']


def test_confirmation_requires_gate_and_model_lock(tmp_path):
    with pytest.raises(ValueError,match='locked'):
        confirmation({},tmp_path,{},None,None,lambda:None)


def test_portable_fixed_policy_and_immutable_wrong_binding(tmp_path,monkeypatch):
    monkeypatch.setenv('GOFA_WORKSPACE','/tmp/audited-source')
    cfg=configuration(Path(__file__).parents[1]/'configs/stage36_source_graph_probe.yaml')
    assert not cfg['policy']['target_training_enabled'] and not cfg['policy']['gpu_enabled']
    assert cfg['source_output'].startswith('/tmp/audited-source/')
    assert digest(cfg)==digest(copy.deepcopy(cfg))
    immutable(tmp_path/'record.json',dict(binding='a'))
    with pytest.raises(ValueError):immutable(tmp_path/'record.json',dict(binding='b'))
    immutable(tmp_path/'manifest.json',dict(configuration=cfg,code={}))
    with pytest.raises(ValueError,match='binding'):verify(cfg,tmp_path)


def test_training_pool_selection_keeps_all_variables_and_time_strata():
    pool=[dict(source='pems04_flow',variable=v,origin=20+i*20,length=12,horizon=12,chunk=0)
          for v in range(4) for i in range(20)]
    result=select_training(pool,[0,1,2,3],32,0)
    assert result==select_training(pool,[0,1,2,3],32,0)
    assert {r['query']['variable'] for r in result}=={0,1,2,3}
    assert len({digest(r) for r in result})==32


def test_bidirectional_contract_only_changes_support_not_features_labels_or_roster():
    d=fixture();other=copy.deepcopy(d)
    original=d.external.copy()
    other.external=road_adjacency(other.external,'publisher_bidirectional')
    np.testing.assert_array_equal(d.external,original)
    np.testing.assert_array_equal(other.external,(original|original.T).astype(np.float32))
    r=dict(query=dict(source='pems04_flow',variable=2,origin=100,length=12,horizon=12),block=-1)
    a=pack(d,r,list(range(8)));b=pack(other,r,list(range(8)))
    for key in a:
        if key!='adjacency':np.testing.assert_array_equal(a[key],b[key])
    assert np.count_nonzero(b['adjacency'])==2*np.count_nonzero(a['adjacency'])
    np.testing.assert_array_equal(controls(a['adjacency'][None],a['nodes'][None],[r],'self'),
        controls(b['adjacency'][None],b['nodes'][None],[r],'self'))
    with pytest.raises(ValueError):road_adjacency(original,'unregistered_search')


def test_road_study_pins_official_contract_without_retrofitting_old_protocol(tmp_path,monkeypatch):
    import yaml
    monkeypatch.setenv('GOFA_WORKSPACE','/tmp/audited-source')
    root=Path(__file__).parents[1]
    cfg=configuration(root/'configs/stage37_source_road_probe.yaml')
    assert cfg['graph_contract']=='publisher_bidirectional'
    assert cfg['protocol']=='stage37-source-road-probe-v1'
    old=configuration(root/'configs/stage36_source_graph_probe.yaml')
    old['graph_contract']='publisher_bidirectional'
    p=tmp_path/'bad.yaml';p.write_text(yaml.safe_dump(old))
    with pytest.raises(ValueError,match='immutable'):configuration(p)
