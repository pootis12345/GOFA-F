from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import LlamaConfig,LlamaForCausalLM,LlamaTokenizer

from gofa_f.generative_data import Query
from gofa_f.graph_memory import GraphMemoryExchange
from gofa_f.graph_memory_data import memory_graph
from gofa_f.graph_memory_model import GraphMemoryGOFA
from gofa_f.short_codec import ShortCodec,numeric_loss
from gofa_f.short_training import compact_forward,compact_loss,teacher_prefix


@pytest.fixture(scope='module')
def codec():
    path=Path('/home/dcg/dcg_data/GOFA-F/data/weights/chattime-7b-20baa88')
    if not path.exists():pytest.skip('Pinned official tokenizer available on C only')
    return ShortCodec(LlamaTokenizer.from_pretrained(path,local_files_only=True))


def data(nodes=4):
    t=np.arange(400)[:,None];v=np.arange(nodes)[None]
    values=(10+v*3+np.sin(t/8+v)).astype(np.float32);external=np.zeros((nodes,nodes),np.float32)
    for i in range(nodes-1):external[i,i+1]=external[i+1,i]=1
    return SimpleNamespace(values=values,mask=np.ones_like(values,bool),external=external,ids=[str(i) for i in range(nodes)])


def tiny(codec):
    torch.manual_seed(42)
    cfg=LlamaConfig(hidden_size=16,intermediate_size=24,num_hidden_layers=4,num_attention_heads=2,
        num_key_value_heads=2,vocab_size=42001,max_position_embeddings=2048,attention_dropout=0.)
    cfg._attn_implementation='sdpa'
    return GraphMemoryGOFA(LlamaForCausalLM(cfg).eval(),codec,stable_fp32=True)


def q(length=12,horizon=12):return Query('pems04_flow',0,200,length,horizon)


def nonzero(model):
    with torch.no_grad():
        for block in model.adapter.exchanges:
            block.up.weight.normal_(std=.02);block.time_query.weight.normal_(std=.1)
            block.time_score.copy_(torch.tensor([.1,-.3]))


def test_graph_rosters_direction_scale_controls_and_future_independence():
    d=data(9);query=q();a=memory_graph(d,query);b=memory_graph(d,query,control='rewired');s=memory_graph(d,query,control='self')
    assert a.variable_ids==b.variable_ids==s.variable_ids and len(a.variable_ids)==9
    assert torch.equal(a.graph.history,b.graph.history) and torch.equal(a.graph.edges,b.graph.edges)
    assert torch.equal(a.graph.base>0,(torch.from_numpy(d.external)>0)|torch.eye(9,dtype=torch.bool))
    assert torch.equal(s.graph.base,torch.eye(9))
    first=a.identity;d.values[200:]=1e9;d.mask[200:]=False
    assert memory_graph(d,query).identity==first


def test_memory_empty_support_missing_nodes_no_nan_and_time_reads():
    d=data();d.mask[188:200,2]=False;graph=memory_graph(d,q()).graph
    block=GraphMemoryExchange(16);hidden=torch.randn(4,12,16)
    with torch.no_grad():block.up.weight.normal_(std=.03);block.time_query.weight.normal_(std=.2);block.time_score[1]=-.5
    bank=block.build(hidden,graph);queries=torch.randn(2,16);queries[1]=queries[0]
    assert torch.equal(block.history_delta(hidden,bank,graph)[2],torch.zeros(12,16))
    out=block.read(queries,torch.tensor([0.,2.]),torch.zeros(2,dtype=torch.long),bank,graph)
    assert torch.isfinite(out).all() and not torch.equal(out[0],out[1])
    graph.base[:]=0
    empty=block.read(queries,torch.tensor([0.,2.]),torch.zeros(2,dtype=torch.long),bank,graph)
    assert torch.equal(empty,torch.zeros_like(empty))
    empty.sum().backward();assert all(p.grad is None or p.grad.isfinite().all() for p in block.parameters())


def test_fixed_row_graph_reads_equal_chunking_with_nonzero_gradients():
    graph=memory_graph(data(),q()).graph;block=GraphMemoryExchange(16)
    with torch.no_grad():block.up.weight.normal_(std=.1)
    hidden=torch.randn(4,12,16,requires_grad=True);bank=block.build(hidden,graph)
    queries=torch.randn(5,16,requires_grad=True);times=torch.arange(5).float()/12;receiver=torch.zeros(5,dtype=torch.long)
    all_at_once=block.read(queries,times,receiver,bank,graph)
    chunks=torch.cat([block.read(queries[i:i+1],times[i:i+1],receiver[i:i+1],bank,graph) for i in range(5)])
    torch.testing.assert_close(chunks,all_at_once,rtol=0,atol=1e-6)
    all_at_once.square().sum().backward()
    assert hidden.grad[1:].abs().sum()>0 and block.time_query.weight.grad.abs().sum()>0
    assert (all_at_once.norm(dim=-1)<=queries.norm(dim=-1)+1e-6).all()


def test_initial_native_fallback_and_all_nodes_shared_frozen_llm(codec):
    d=data(9);model=tiny(codec);query=q();history=d.values[188:200,0]
    tokens=codec.batch([codec.record(history,d.values[200:212,0])]);ctx=model.encode_origins([memory_graph(d,query)],codec)
    assert ctx[0].nodes==9 and model.origin_executions[-1]['layers']==4
    assert torch.equal(model(tokens,contexts=ctx).logits,model(tokens).logits)
    loss=numeric_loss(model(tokens,contexts=ctx).logits,tokens.labels,codec)[0];loss.backward()
    assert all(p.grad is None for p in model.native.parameters())
    assert all(block.up.weight.grad.abs().sum()>0 for block in model.adapter.exchanges)
    nonzero(model);model.mark_updated();before=model(tokens).logits
    ctx=model.encode_origins([memory_graph(d,query)],codec)
    assert not torch.equal(model(tokens,contexts=ctx).logits,before)
    assert torch.equal(model(tokens).logits,before)  # Explicit native/off path after nonzero adaptation.


@pytest.mark.parametrize('length,horizon',[(12,12),(48,24)])
def test_nonzero_full_prefix_compact_incremental_and_gradients(codec,length,horizon):
    d=data();model=tiny(codec);nonzero(model);query=q(length,horizon)
    tokens=codec.batch([codec.record(d.values[200-length:200,0],d.values[200:200+horizon,0])])
    graph=memory_graph(d,query);ctx=model.encode_origins([graph],codec)
    full=model(tokens,contexts=ctx).logits;numeric_loss(full,tokens.labels,codec)[0].backward()
    expected={n:None if p.grad is None else p.grad.clone() for n,p in model.adapter.named_parameters()}
    model.zero_grad();ctx=model.encode_origins([graph],codec)
    cached=model(tokens,contexts=ctx,prefix=model.prefix(tokens)).logits
    torch.testing.assert_close(cached,full,rtol=1e-6,atol=1e-6)
    model.zero_grad();ctx=model.encode_origins([graph],codec)
    compact=compact_forward(model,codec,tokens,ctx,teacher_prefix(model,tokens));compact_loss(compact)[0].backward()
    for name,p in model.adapter.named_parameters():
        if expected[name] is None:assert p.grad is None
        else:torch.testing.assert_close(p.grad,expected[name],rtol=1e-6,atol=1e-6)
    model.zero_grad();ctx=model.encode_origins([graph],codec);cut=int(tokens.prompt_ends[0])+1
    part=tokens.first(cut);out=model(part,contexts=ctx,use_cache=True);pieces=[out.logits];kv=out.past_key_values
    with torch.no_grad():
        for index in range(cut,tokens.input_ids.shape[1]):
            part=part.append(tokens.input_ids[:,index:index+1]);out=model(part,contexts=ctx,cache=kv,use_cache=True)
            kv=out.past_key_values;pieces.append(out.logits)
    torch.testing.assert_close(torch.cat(pieces,1),full,rtol=1e-6,atol=1e-6)
    model.mark_updated()
    with pytest.raises(ValueError,match='Stale'):model(part,contexts=ctx,cache=kv,use_cache=True)


def test_teacher_future_tokens_cannot_change_earlier_logits_and_graph(codec):
    d=data();model=tiny(codec);nonzero(model);query=q();graph=memory_graph(d,query)
    a=codec.batch([codec.record(d.values[188:200,0],d.values[200:212,0])])
    changed=d.values[200:212,0].copy();changed[5:]+=100
    b=codec.batch([codec.record(d.values[188:200,0],changed)])
    ctx=model.encode_origins([graph],codec);x=model(a,contexts=ctx).logits;y=model(b,contexts=ctx).logits
    position=codec.record(d.values[188:200,0],d.values[200:212,0])['answer_positions'][5]
    assert torch.equal(x[:,:position],y[:,:position])


def test_context_isolation_and_numeric_slot_not_separator_count(codec):
    d=data();model=tiny(codec);nonzero(model);query=q();history=d.values[188:200,0]
    first=codec.record(history,d.values[200:212,0]);tokens=codec.batch([first]);ctx=model.encode_origins([memory_graph(d,query)],codec)
    seen=[];block=model.adapter.exchanges[0];original=block.read
    def capture(h,t,r,*args):seen.append(t.detach().clone());return original(h,t,r,*args)
    block.read=capture
    model(tokens,contexts=ctx);block.read=original
    clock=seen[-1]*12
    assert int(clock[0])==0 and clock.max()==12
    # Both numeric and separator positions after one generated number use slot1.
    assert int((clock==1).sum())==1+len(codec.separator)


def test_trainable_checkpoint_optimizer_and_rng_replay(codec):
    d=data();model=tiny(codec);nonzero(model);query=q();tokens=codec.batch([codec.record(d.values[188:200,0],d.values[200:212,0])])
    graph=memory_graph(d,query);optimizer=torch.optim.AdamW(model.adapter.parameters(),lr=1e-4)
    def update():
        optimizer.zero_grad();ctx=model.encode_origins([graph],codec)
        loss=compact_loss(compact_forward(model,codec,tokens,ctx,teacher_prefix(model,tokens)))[0]
        loss.backward();optimizer.step();model.mark_updated();return float(loss.detach())
    update();saved=deepcopy(model.trainable_state());state=deepcopy(optimizer.state_dict());rng=torch.get_rng_state()
    expected_loss=update();expected=model.trainable_state()
    model.load_trainable(saved);optimizer.load_state_dict(state);torch.set_rng_state(rng)
    assert update()==expected_loss
    for name,value in model.trainable_state().items():assert torch.equal(value,expected[name])


@pytest.mark.parametrize('length,horizon',[(12,12),(48,24)])
def test_nonzero_actual_sampled_paths_match_and_local_rng_isolated(codec,length,horizon):
    from gofa_f.graph_memory_stage import verify_sampled_paths
    d=data();model=tiny(codec);nonzero(model);query=q(length,horizon)
    ctx=model.encode_origins([memory_graph(d,query)],codec)
    before=torch.get_rng_state().clone()
    value=verify_sampled_paths(model,codec,d.values[200-length:200,0],horizon,ctx,seed=42,rtol=1e-6,atol=1e-6)
    assert value['sampled_tokens_equal'] and len(value['numeric_ids'])==horizon
    assert torch.equal(before,torch.get_rng_state())
