from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import torch
import pytest

from gofa_f.ufg_shared import SharedRosterUFG,SharedUFGEngine
from gofa_f.ufg_memory_roster import pack_roster
from gofa_f.ufg_subgraph import pack_history
from gofa_f.dual_memory_engines import model_digest
from gofa_f.dual_memory_policy import capture,restore,accumulate

ROOT=Path('/home/dcg/dcg_data/GOFA-F/external/ufgtime')


def history(length,nodes=7,batch=1):
    generator=torch.Generator().manual_seed(42+length+nodes+batch)
    x=torch.randn(batch,length,nodes,generator=generator,requires_grad=True)
    mask=torch.ones_like(x,dtype=torch.bool)
    if nodes>1:mask[:,1::3,1]=False
    return x,mask


def test_canonical_forward_and_parameter_gradients_match_pinned_official_single_query():
    torch.manual_seed(42);model=SharedRosterUFG(ROOT)
    x,mask=history(48);packed=pack_history(x,mask,31)
    actual=model.normalized_forecast(packed.normalized,packed.observed)
    expected=model.core(packed.normalized,packed.observed)
    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
    params=[p for p in model.parameters() if p.requires_grad]
    ga=torch.autograd.grad(actual.square().mean(),params,retain_graph=True,allow_unused=True)
    ge=torch.autograd.grad(expected.square().mean(),params,allow_unused=True)
    for a,e in zip(ga,ge):
        assert a is not None and e is not None
        torch.testing.assert_close(a,e,rtol=1e-6,atol=1e-6)


@pytest.mark.parametrize('length,horizon,nodes',[(12,12,1),(12,12,31),(48,24,7),(48,24,31)])
def test_one_model_two_tasks_complete_history_masks_gradients_and_reload(length,horizon,nodes,tmp_path):
    torch.manual_seed(42);model=SharedRosterUFG(ROOT)
    keys=tuple(model.state_dict());ids=tuple(id(p) for p in model.parameters())
    x,mask=history(length,nodes);packed=pack_history(x,mask,31)
    y=model(packed,horizon,all_nodes=True)
    assert y.shape==(1,horizon,nodes) and torch.isfinite(y).all()
    y.square().mean().backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters() if p.requires_grad)
    assert torch.equal(x.grad[~mask],torch.zeros_like(x.grad[~mask]))
    assert tuple(model.state_dict())==keys and tuple(id(p) for p in model.parameters())==ids
    path=tmp_path/'one-model.pt';torch.save(model.state_dict(),path)
    copy=SharedRosterUFG(ROOT);copy.load_state_dict(torch.load(path,map_location='cpu'))
    with torch.no_grad():torch.testing.assert_close(y,copy(packed,horizon,all_nodes=True),rtol=0,atol=0)
    assert not model.provenance['shape_specific_parameters']


@pytest.mark.parametrize('length,horizon',[(12,12),(48,24)])
def test_batched_queries_cannot_exchange_history_with_other_origins(length,horizon):
    torch.manual_seed(42);model=SharedRosterUFG(ROOT).eval();x,mask=history(length,7,batch=2)
    with torch.no_grad():
        actual=model(pack_history(x,mask,31),horizon)
        alone=model(pack_history(x[:1],mask[:1],31),horizon)
        changed=x.detach().clone();changed[1]=torch.sin(torch.arange(length)[:,None]*.37)
        other=model(pack_history(changed,mask,31),horizon)
    torch.testing.assert_close(actual[:1],alone,rtol=1e-6,atol=1e-6)
    torch.testing.assert_close(actual[:1],other[:1],rtol=1e-6,atol=1e-6)


def test_same_checkpoint_and_output_head_serves_both_shapes_without_mutation():
    torch.manual_seed(42);model=SharedRosterUFG(ROOT).eval();before=model_digest(model)
    with torch.no_grad():
        for length,horizon in [(12,12),(48,24),(12,12)]:
            x,mask=history(length,7);packed=pack_history(x,mask,31)
            full=model.normalized_forecast(packed.normalized,packed.observed)
            p=model(packed,horizon)
            expected=(full[:,:horizon,0].double()+.5)*packed.extent[:,0,None]+packed.minimum[:,0,None]
            torch.testing.assert_close(p,expected,rtol=0,atol=0)
            assert model_digest(model)==before


def test_mixed_shape_engine_loss_and_partial_optimizer_replay_exact():
    class Store:
        def __init__(self):
            self.items={}
            for length in (12,48):
                x,mask=history(length,4);packed=pack_history(x.detach(),mask,31);h=12 if length==12 else 24
                self.items[length]=SimpleNamespace(history=x[0].T.detach(),observed=mask[0].T,
                    node_ids=tuple(map(str,range(4))),horizon=h,identity='input'+str(length))
        def get(self,row,*_):
            item=self.items[row['length']];return item,torch.ones(4,row['horizon']),torch.ones(4,row['horizon'],dtype=torch.bool)
    torch.manual_seed(42);store=Store();m=dict(training_unit_normalizers=dict(a=1.,b=1.))
    engine=SharedUFGEngine(ROOT,m,store);q=[dict(source='a',length=12,horizon=12),dict(source='b',length=48,horizon=24)]
    total=engine.loss(q);separate=engine.loss(q[:1])+engine.loss(q[1:])
    torch.testing.assert_close(total,separate,rtol=1e-6,atol=1e-6)
    params=list(engine.model.parameters());opt=torch.optim.AdamW(params,lr=.001,weight_decay=0.)
    accumulate(engine.loss(q[:1])/2,params);saved=capture(engine.model,opt,dict(pending=1),'shared')
    clone=SharedUFGEngine(ROOT,m,store);other=torch.optim.AdamW(clone.model.parameters(),lr=.001,weight_decay=0.)
    assert restore(clone.model,other,saved,'shared')['pending']==1
    accumulate(engine.loss(q[1:])/2,params);accumulate(clone.loss(q[1:])/2,list(clone.model.parameters()))
    opt.step();other.step()
    assert model_digest(engine.model)==model_digest(clone.model)
