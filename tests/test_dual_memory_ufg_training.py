from pathlib import Path
import numpy as np
import torch

from gofa_f.ufg_memory_roster import MemoryRosterUFG
from gofa_f.ufg_subgraph import pack_history
from gofa_f.dual_memory_policy import accumulate,query_loss,capture,restore

ROOT=Path('/home/dcg/dcg_data/GOFA-F/external/ufgtime')


def test_official31_batched_training_equals_independent_query_accumulation():
    torch.manual_seed(42);model=MemoryRosterUFG(ROOT,12,12)
    history=torch.randn(4,12,31);observed=torch.ones_like(history,dtype=torch.bool)
    observed[1,:,16:]=False;observed[2,::3,4]=False
    packed=pack_history(history,observed,maximum_nodes=31)
    target=torch.randn(4,31,12,dtype=torch.float64);valid=observed.any(1)[:,:,None].expand(-1,-1,12)
    batch=model(packed,all_nodes=True).transpose(1,2)
    losses=sum(query_loss(batch[i],target[i],valid[i],packed.present[i],1.) for i in range(4))/4
    accumulate(losses,list(model.parameters()))
    expected={n:None if p.grad is None else p.grad.clone() for n,p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    parts=[]
    for i in range(4):
        own=pack_history(history[i:i+1],observed[i:i+1],maximum_nodes=31)
        actual=model(own,all_nodes=True).transpose(1,2)
        parts.append(actual[0].detach())
        accumulate(query_loss(actual[0],target[i],valid[i],own.present[0],1.)/4,list(model.parameters()))
    torch.testing.assert_close(torch.stack(parts),batch,rtol=1e-6,atol=1e-6)
    for name,p in model.named_parameters():
        if expected[name] is None:assert p.grad is None
        else:torch.testing.assert_close(p.grad,expected[name],rtol=1e-6,atol=1e-6)


def test_ufg_complex_adam_optimizer_partial_state_exact_resume(tmp_path):
    torch.manual_seed(42);model=MemoryRosterUFG(ROOT,12,12)
    packed=pack_history(torch.randn(1,12,31),torch.ones(1,12,31,dtype=torch.bool),maximum_nodes=31)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
    def micro():
        prediction=model(packed);accumulate(prediction.square().mean()/2,list(model.parameters()))
    micro();value=capture(model,optimizer,dict(pending=1,cursor={'total':1}),'ufg')
    path=tmp_path/'state.pt';torch.save(value,path)
    micro();optimizer.step();expected={k:v.detach().clone() for k,v in model.state_dict().items()}
    restore(model,optimizer,torch.load(path,map_location='cpu',weights_only=False),'ufg')
    micro();optimizer.step()
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in expected.items())
