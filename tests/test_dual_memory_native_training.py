from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import LlamaConfig,LlamaForCausalLM

from gofa_f.allnode_codec import AllNodeTokens
from gofa_f.dual_memory_data import memory_input
from gofa_f.dual_memory_model import DualMemoryGOFA
from gofa_f.dual_memory_policy import accumulate,capture,restore
from gofa_f.dual_memory_store import HistoryCache
from gofa_f.generative_data import Query


def fixture():
    torch.manual_seed(42)
    cfg=LlamaConfig(hidden_size=16,intermediate_size=24,num_hidden_layers=4,num_attention_heads=2,
        num_key_value_heads=2,vocab_size=160,attention_dropout=0.)
    cfg._attn_implementation='sdpa'
    model=DualMemoryGOFA(LlamaForCausalLM(cfg),depth=3,width=8,stable_fp32=True,weight_binding='tiny-weights')
    t=np.arange(400)[:,None];v=np.arange(4)[None]
    d=SimpleNamespace(values=(t*.1+v).astype('float32'),mask=np.ones((400,4),bool),ids=list('abcd'),
        external=np.ones((4,4))-np.eye(4),segments=np.zeros(400,int))
    item=memory_input(d,Query('pems04_flow',0,200,12,12))
    ids=torch.arange(48).reshape(4,12)+3;mask=torch.ones_like(ids,dtype=torch.bool)
    tokens=AllNodeTokens(ids,mask,torch.full_like(ids,-100),torch.full((4,),11),
        tuple(torch.arange(12) for _ in range(4)),tuple('abcd'),ids.clone())
    with torch.no_grad():
        model.adapter.head[-1].weight.normal_(std=.01)
        for e in model.adapter.exchanges:e.gate.fill_(.1);e.ff_gate.fill_(.1)
    return model,item,tokens


def test_real_memory_autograd_grad_partial_restore_matches_uninterrupted_step():
    model,item,tokens=fixture();prefix=model.frozen_history(tokens)
    optimizer=torch.optim.AdamW(model.adapter.parameters(),lr=.001)
    def micro():
        prediction,_=model.enhanced(tokens,item,torch.ones(16),prefix=prefix,optimized=True,checkpoint_tail=True)
        accumulate(prediction.square().mean()/4,list(model.adapter.parameters()))
    micro();state=capture(model.adapter,optimizer,dict(pending=1,cursor={'total':1}),'memory')
    for _ in range(3):micro()
    optimizer.step();expected=deepcopy(model.adapter.state_dict())
    restore(model.adapter,optimizer,state,'memory')
    for _ in range(3):micro()
    optimizer.step()
    assert all(torch.equal(value,model.adapter.state_dict()[name]) for name,value in expected.items())
    assert all(p.grad is None for p in model.native.parameters())


def test_frozen_prefix_cache_roundtrip_and_content_binding(tmp_path):
    model,item,tokens=fixture();cache=HistoryCache(tmp_path,ram_bytes=0,disk_bytes=2**20)
    first=cache.get(model,tokens);second=cache.get(model,tokens)
    assert torch.equal(first.hidden,second.hidden) and cache.hits==1 and model.prefix_executions==1
    prefix=next(tmp_path.glob('*.pt'));prefix.write_bytes(prefix.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='content changed'):HistoryCache(tmp_path,ram_bytes=0).get(model,tokens)
