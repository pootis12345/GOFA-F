from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gofa_f.dual_memory_data import GROUPS,memory_input
from gofa_f.dual_memory_policy import (SourceSchedule,accumulate,capture,restore,query_loss,
    priority_rows,summarize,update_selection)
from gofa_f.dual_memory_store import SourceStore
from gofa_f.dual_memory_fusion import features,blend
from gofa_f.generative_data import Query


def queries(count=64):
    return {name:{shape:{'train':[dict(source=name,variable=i%8,origin=60+(i//8)*30,
        length=int(shape.split('-')[0]),horizon=int(shape.split('-')[1]),chunk=0) for i in range(count)],
        'development':[],'confirmation':[]} for shape in ('12-12','48-24')} for name in GROUPS}


@pytest.mark.parametrize('held',['traffic','weather','wind'])
def test_source_group_holdout_equal_domains_and_cursor_replay(held):
    schedule=SourceSchedule(queries(),held_group=held,initial=32)
    domains=[];rows=[]
    for _ in range(96):
        row,state=schedule.peek();schedule.commit(state);rows.append(row);domains.append(GROUPS[row['source']])
    assert held not in domains and all(domains.count(g)==48 for g in set(domains))
    if held=='traffic':assert not any(q['source'].startswith('pems04') for q in rows)
    saved=deepcopy(schedule.state);future=[]
    for _ in range(20):
        q,s=schedule.peek();schedule.commit(s);future.append(q)
    again=SourceSchedule(queries(),held_group=held,initial=32);again.commit(saved)
    actual=[]
    for _ in range(20):q,s=again.peek();again.commit(s);actual.append(q)
    assert actual==future


def test_nested_pool_is_variable_balanced_and_spans_time():
    rows=[dict(variable=v,origin=t) for t in range(256) for v in range(16)]
    initial=priority_rows(rows)[:512]
    assert {q['variable'] for q in initial}==set(range(16))
    assert all(sum(q['variable']==v for q in initial)==32 for v in range(16))
    assert {q['origin']//64 for q in initial}=={0,1,2,3}


def test_partial_gradients_optimizer_rng_exactly_resume_next_update():
    import random
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    model=torch.nn.Sequential(torch.nn.Linear(4,8),torch.nn.Dropout(.2),torch.nn.Linear(8,1))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    def micro():
        x=torch.randn(3,4);accumulate(model(x).square().mean()/8,list(model.parameters()))
    for _ in range(3):micro()
    value=capture(model,optimizer,dict(pending=3,cursor={'total':3},selection={'stale':2}),'locked')
    for _ in range(5):micro()
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    expected=deepcopy(model.state_dict());rng=torch.rand(4)
    state=restore(model,optimizer,value,'locked')
    assert state==dict(pending=3,cursor={'total':3},selection={'stale':2})
    for _ in range(5):micro()
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in expected.items())
    assert torch.equal(rng,torch.rand(4))
    with pytest.raises(ValueError,match='binding'):restore(model,optimizer,value,'other')


def test_failed_micro_backward_cannot_corrupt_previous_accumulation():
    class Fault(torch.autograd.Function):
        @staticmethod
        def forward(ctx,x):return x
        @staticmethod
        def backward(ctx,g):raise RuntimeError('simulated failed micro')
    model=torch.nn.Linear(3,1);parameters=list(model.parameters())
    accumulate(model(torch.ones(2,3)).square().sum(),parameters)
    saved=[p.grad.clone() for p in parameters]
    with pytest.raises(RuntimeError,match='failed micro'):
        accumulate(Fault.apply(model(torch.ones(2,3))).sum(),parameters)
    assert all(torch.equal(a,p.grad) for a,p in zip(saved,parameters))


def test_all_node_raw_loss_masks_missing_origins_and_query_reduction():
    prediction=torch.tensor([[2.,5.],[100.,200.],[3.,7.]],requires_grad=True)
    target=torch.tensor([[1.,1.],[0.,0.],[2.,4.]])
    mask=torch.tensor([[True,False],[True,True],[True,True]])
    present=torch.tensor([True,False,True])
    loss=query_loss(prediction,target,mask,present,2.)
    assert float(loss)==.75
    loss.backward();assert prediction.grad[1].eq(0).all() and prediction.grad[0,1]==0


def test_variable_macro_counts_points_not_query_means_and_selection_idempotence():
    q=lambda v:dict(source='pems04_flow',variable=v)
    records=[dict(query=q(0),prediction=[10.,0.],target=[0.,0.],observed=[True,True]),
        dict(query=q(0),prediction=[0.,0.],target=[0.,0.],observed=[True,False]),
        dict(query=q(1),prediction=[2.,2.],target=[0.,0.],observed=[True,True])]
    result=summarize(records,{'pems04_flow':1.})
    assert result['views']['pems04_flow']['macro_mae']==pytest.approx((10/3+2)/2)
    state=update_selection({},1.,0,400,5)
    state=update_selection(state,.9995,100,400,5)
    assert state['best_update']==100 and state['stale']==1
    assert update_selection(state,.9995,100,400,5)==state


def synthetic():
    t=np.arange(400)[:,None];v=np.arange(4)[None]
    return SimpleNamespace(values=(t*.1+v).astype('float32'),mask=np.ones((400,4),bool),ids=list('abcd'),
        external=np.ones((4,4))-np.eye(4),segments=np.zeros(400,int))


def test_source_cache_is_identical_for_both_runners_and_rejects_corruption(tmp_path):
    d=synthetic();allq=queries(4)
    for name in GROUPS:
        for values in allq[name].values():
            for q in values['train']:q['variable']%=4;q['origin']=200
    m=dict(source_hashes={n:n for n in GROUPS},metadata_hashes={},sampler='pinned',queries=allq,
        partitions={n:{k:list(range(4)) for k in ('train','validation','confirmation')} for n in GROUPS})
    store=SourceStore(m,tmp_path/'inputs');store.source=lambda _:d
    q=allq['pems04_flow']['12-12']['train'][0]
    expected=store.get(q,'train');store.build()
    loaded=SourceStore(m,tmp_path/'inputs');actual=loaded.get(q,'train')
    assert actual[0].identity==expected[0].identity
    assert torch.equal(actual[0].history,expected[0].history) and torch.equal(actual[1],expected[1])
    shard=next((tmp_path/'inputs').glob('inputs-*.pt'));shard.write_bytes(shard.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='changed'):SourceStore(m,tmp_path/'inputs').get(q,'train')


def test_out_of_fit_and_confirmation_read_barriers(tmp_path):
    rows=queries(1)
    for name in GROUPS:
        for v in rows[name].values():v['development']=deepcopy(v['train']);v['confirmation']=deepcopy(v['train'])
    m=dict(source_hashes={},metadata_hashes={},sampler='pinned',queries=rows,partitions={})
    store=SourceStore(m,tmp_path/'inputs',held_group='traffic');store.source=lambda _:synthetic()
    q=rows['pems04_flow']['12-12']['development'][0]
    with pytest.raises(ValueError,match='Held source'):store.get(q,'development')
    with pytest.raises(ValueError,match='locked'):store.get(q,'development','out_of_fit')
    seal=tmp_path/'selection.json';seal.write_text(json.dumps(dict(excluded_group='traffic',held_source_used_for_checkpoint_selection=False,target_labels_used=False)))
    store.permit_out_of_fit(seal)
    # Still cannot use confirmation labels under an out-of-fit permission.
    with pytest.raises(ValueError,match='confirmation'):store.get(q,'confirmation','out_of_fit')


def test_fusion_features_are_history_only_and_no_neighbor_returns_native_exactly():
    d=synthetic();q=Query('pems04_flow',0,200,48,24);item=memory_input(d,q)
    native=np.arange(24,dtype=float);graph=native+2
    first=features(item,native,graph)
    d.values[200:]=1e10;d.mask[200:]=False
    other=memory_input(d,q);second=features(other,native,graph)
    assert np.array_equal(first[0],second[0])
    item.adjacency[0,1:]=False
    prediction,alpha=blend(dict(parameters=[100.]*9),item,native,graph)
    assert np.array_equal(prediction,native) and not alpha.any()


def test_complete_fit_checkpoint_selection_and_resume_do_not_repeat_updates(tmp_path):
    from gofa_f.dual_memory_fitting import fit
    class Engine:
        cuda=False;microbatch=3
        def __init__(self):torch.manual_seed(42);self.state_model=torch.nn.Linear(1,1)
        def warmup(self,_):pass
        def loss(self,rows):
            assert not any(r['source'].startswith('pems04') for r in rows)
            return self.state_model(torch.ones(len(rows),1)).square().sum()
        def predict(self,row,split,role='fit'):
            assert not row['source'].startswith('pems04')
            with torch.no_grad():p=np.full(row['horizon'],float(self.state_model(torch.ones(1,1))))
            return p,np.zeros_like(p),np.ones_like(p,bool),'input'
    rows=queries(16)
    for name in rows:
        for values in rows[name].values():values['development']=[values['train'][0]]
    settings=dict(max_updates=2,effective_batch=8,learning_rate=.001,weight_decay=.0001,
        interface_only_updates=0,warmup_updates=1,gradient_clip=1.,validation_every=1,
        early_min_updates=2,early_patience=5)
    manifest=dict(queries=rows,configuration={'training':settings,'queries':{'initial_training_per_view':16}},
        training_unit_normalizers={n:1. for n in GROUPS})
    engine=Engine();best=fit(engine,manifest,{},tmp_path,'hold-traffic',3,'fit-binding',lambda:None,lambda **_:None)
    saved=torch.load(tmp_path/'last.pt',weights_only=False)
    assert saved['state']['updates']==2 and saved['state']['cursor']['total']==16 and saved['state']['completed']
    before={k:v.clone() for k,v in engine.state_model.state_dict().items()}
    again=fit(engine,manifest,{},tmp_path,'hold-traffic',3,'fit-binding',lambda:None,lambda **_:None)
    assert again['model_sha256']==best['model_sha256']
    assert all(torch.equal(v,engine.state_model.state_dict()[k]) for k,v in before.items())


def test_formal_cpu_proof_rejects_another_manifest(tmp_path):
    from gofa_f.dual_memory_training import require_cpu_proof
    path=tmp_path/'formal_cpu_preflight.json';path.write_text(json.dumps(dict(passed=True,manifest_sha256='other')))
    with pytest.raises(ValueError,match='manifest'):require_cpu_proof(tmp_path,'current')
    path.write_text(json.dumps(dict(passed=True,manifest_sha256='current')))
    require_cpu_proof(tmp_path,'current')
