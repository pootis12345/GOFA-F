from dataclasses import asdict
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from gofa_f.dual_memory_data import memory_input
from gofa_f.dual_memory_policy import capture,restore,accumulate,query_loss
from gofa_f.generative_data import Query
from gofa_f.variable_graph_data import variable_input,VariableSchedule,TRAIN_SHAPES,interval_seconds
from gofa_f.variable_graph_expert import VariableGraphExpert,pack_items,safe_softmax


def series(n=7):
    r=np.random.default_rng(42);x=r.normal(size=(700,n))
    return SimpleNamespace(values=x,mask=np.ones_like(x,dtype=bool),segments=np.zeros(700,dtype=int),
        times=np.arange(700,dtype=np.int64)*300_000_000_000,external=np.ones((n,n)),ids=[str(i) for i in range(n)])


def item(length=48,horizon=24,n=7):
    return variable_input(series(n),Query('fake',0,350,length,horizon))


def test_inputs_preserve_primary_arrays_and_future_independence():
    s=series()
    for l,h in ((12,12),(48,24)):
        q=Query('fake',0,350,l,h);a=variable_input(s,q);b=memory_input(s,q)
        assert a.identity==b.identity
        for k,v in vars(a).items():
            if isinstance(v,torch.Tensor):assert torch.equal(v,getattr(b,k))
    q=Query('fake',0,350,97,25);a=variable_input(s,q)
    s.values[350:]=1e20;s.mask[350:]=False
    b=variable_input(s,q);assert a.identity==b.identity
    assert interval_seconds(s,q)==300
    c=variable_input(s,Query('fake',0,350,97,143))
    assert torch.equal(b.history,c.history) and torch.equal(b.adjacency,c.adjacency)


@pytest.mark.parametrize('length,horizon',[(10,1),(14,7),(48,24),(97,25),(288,144)])
def test_all_shapes_finite_gradient_no_new_parameters(length,horizon):
    torch.manual_seed(42);m=VariableGraphExpert();x=item(length,horizon,n=31 if length==288 else 4)
    params=[(k,tuple(v.shape)) for k,v in m.named_parameters()]
    p=m(**pack_items([x]),interval=torch.tensor([300.]),horizon=horizon)
    assert p.shape==(1,len(x.node_ids),horizon) and torch.isfinite(p).all()
    p.sum().backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters())
    assert sum(float(v.grad.abs().sum()) for v in m.blocks.parameters())>0
    assert params==[(k,tuple(v.shape)) for k,v in m.named_parameters()]


def test_horizon_prefix_and_batch_isolation_and_node_permutation():
    torch.manual_seed(9);m=VariableGraphExpert().eval();x=item(49,24,n=4);p=pack_items([x])
    a=m(**p,interval=torch.tensor([300.]),horizon=7)
    b=m(**p,interval=torch.tensor([300.]),horizon=144)
    torch.testing.assert_close(a,b[:,:,:7],rtol=1e-6,atol=1e-6)
    batch=pack_items([x,item(49,24,n=7)])
    batch['normalized'][1]*=30
    c=m(**batch,interval=torch.tensor([300.,3600.]),horizon=7)
    torch.testing.assert_close(a,c[:1,:4],rtol=1e-6,atol=1e-6)
    order=torch.tensor([0,3,1,2]);permuted={}
    for name,v in p.items():
        v=v[:,order]
        if name in ('adjacency','edges'):v=v[:,:,order]
        permuted[name]=v
    d=m(**permuted,interval=torch.tensor([300.]),horizon=7)
    torch.testing.assert_close(a[:,order],d,rtol=1e-6,atol=1e-6)


def test_missing_constant_empty_neighbors_and_partial_patch():
    s=series(4);s.values[:350,1]=4;s.mask[:350,2]=False
    s.mask[340:349,0]=False;s.values[349,0]=-3
    x=variable_input(s,Query('fake',0,350,10,144));assert x.extent[1]==1 and not x.present[2]
    x.adjacency.zero_()
    m=VariableGraphExpert();out=m(**pack_items([x]),interval=torch.tensor([300.]),horizon=144)
    assert torch.isfinite(out).all();out.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())
    assert safe_softmax(torch.ones(2,3),torch.zeros(2,3,dtype=torch.bool)).eq(0).all()
    s.mask[340:350,0]=False
    with pytest.raises(ValueError):variable_input(s,Query('fake',0,350,10,1))


def test_schedule_length_domain_balance_and_exact_cursor():
    queries={n:{f'{l}-{h}':dict(train=[dict(source=n,variable=v,origin=400+v,length=l,horizon=h,chunk=0)
        for v in range(3)]) for l,h in TRAIN_SHAPES} for n in ('pems04_flow','pems04_speed','boston2017','penmanshiel2017')}
    for d in queries.values():d['14-7']=dict(train=[],development=[],confirmation=[dict(variable=0)])
    s=VariableSchedule(queries);seen=[]
    for _ in range(72):q,state=s.peek();s.commit(state);seen.append(q)
    assert len({(q['length'],q['horizon']) for q in seen})==6
    assert [sum(q['source']==n for q in seen) for n in queries]==[12,12,24,24]
    t=VariableSchedule(queries);t.commit(s.state);assert s.peek()==t.peek()


def test_partial_gradient_optimizer_rng_resume_exact():
    torch.manual_seed(5);m=VariableGraphExpert(width=16,blocks=1,heads=2);x=item(12,12,3)
    opt=torch.optim.AdamW(m.parameters(),lr=.001);args=pack_items([x]);rng=torch.get_rng_state()
    def loss(model):return model(**args,interval=torch.tensor([300.]),horizon=12).square().mean()
    accumulate(loss(m),list(m.parameters()));opt.step();opt.zero_grad(set_to_none=True)
    accumulate(loss(m)/2,list(m.parameters()));saved=capture(m,opt,dict(pending=1,cursor=13),'binding')
    accumulate(loss(m)/2,list(m.parameters()));opt.step();expected={k:v.detach().clone() for k,v in m.state_dict().items()}
    restored=VariableGraphExpert(width=16,blocks=1,heads=2);other=torch.optim.AdamW(restored.parameters(),lr=.001)
    state=restore(restored,other,saved,'binding');assert state==dict(pending=1,cursor=13)
    accumulate(loss(restored)/2,list(restored.parameters()));other.step()
    for k,v in restored.state_dict().items():assert torch.equal(v,expected[k])
    with pytest.raises(ValueError):restore(restored,other,saved,'changed')
