from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import LlamaConfig,LlamaForCausalLM,LlamaTokenizer

from gofa_f.cross_innovation import CrossInnovationExchange,CrossInnovationGOFA
from gofa_f.graph_memory_data import memory_graph
from gofa_f.graph_memory_diagnostic import scoring_tokens
from gofa_f.graph_memory_pilot import checkpoint,restore_checkpoint
from gofa_f.graph_memory_stage import verify_sampled_paths
from gofa_f.graph_proposal_probe import coupled_draw
from gofa_f.short_codec import ShortCodec,numeric_loss
from gofa_f.short_sampling_diagnostic import generate
from gofa_f.short_training import compact_forward,compact_loss,teacher_prefix
from gofa_f.source_cross_innovation import acceptance_factory
from test_graph_memory import data,nonzero,q


@pytest.fixture(scope='module')
def codec():
    path=Path('/home/dcg/dcg_data/GOFA-F/data/weights/chattime-7b-20baa88')
    if not path.exists():pytest.skip('Pinned official tokenizer on C')
    return ShortCodec(LlamaTokenizer.from_pretrained(path,local_files_only=True))


def tiny(codec):
    torch.manual_seed(42);cfg=LlamaConfig(hidden_size=16,intermediate_size=24,num_hidden_layers=4,num_attention_heads=2,
        num_key_value_heads=2,vocab_size=42001,max_position_embeddings=2048,attention_dropout=0.)
    cfg._attn_implementation='sdpa'
    return CrossInnovationGOFA(LlamaForCausalLM(cfg).eval(),codec,stable_fp32=True)


def test_self_only_exact_zero_missing_safe_and_temporal_baseline_learnable():
    graph=memory_graph(data(),q(),control='self').graph;block=CrossInnovationExchange(16)
    with torch.no_grad():block.up.weight.normal_(std=.1)
    hidden=torch.randn(4,12,16,requires_grad=True);bank=block.build(hidden,graph)
    delta=block.history_delta(hidden,bank,graph)
    assert torch.equal(delta,torch.zeros_like(delta));delta.sum().backward()
    assert all(p.grad is None or p.grad.isfinite().all() for p in block.parameters())
    block.innovation=False;block.zero_grad();bank=block.build(hidden,graph)
    temporal=block.history_delta(hidden,bank,graph)
    assert temporal.abs().sum()>0;temporal.square().sum().backward();assert block.up.weight.grad.abs().sum()>0
    block.innovation=True;graph.base[:]=0;bank=block.build(hidden,graph)
    assert torch.equal(block.history_delta(hidden,bank,graph),torch.zeros_like(hidden))


def test_actual_auxiliary_innovation_gradient_and_constant_value_cancellation():
    graph=memory_graph(data(),q()).graph;block=CrossInnovationExchange(16)
    with torch.no_grad():block.up.weight.normal_(std=.1)
    hidden=torch.randn(4,12,16,requires_grad=True);bank=block.build(hidden,graph)
    delta=block.read(torch.randn(2,16),torch.tensor([0.,.5]),torch.zeros(2,dtype=torch.long),bank,graph)
    delta.square().sum().backward()
    assert hidden.grad[1:].abs().sum()>0 and block.up.weight.grad.abs().sum()>0
    with torch.no_grad():
        bank.values[:]=bank.values[0].clone()
        constant=block.read(torch.randn(2,16),torch.tensor([0.,.5]),torch.zeros(2,dtype=torch.long),bank,graph)
        torch.testing.assert_close(constant,torch.zeros_like(constant),rtol=0,atol=1e-6)


def test_initial_native_all_nodes_and_mode_context_isolation(codec):
    d=data(9);model=tiny(codec);query=q();tokens=codec.batch([codec.record(d.values[188:200,0],d.values[200:212,0])])
    ctx=model.encode_origins([memory_graph(d,query)],codec)
    assert ctx[0].nodes==9 and model.origin_executions[-1]['layers']==4
    assert torch.equal(model(tokens,contexts=ctx).logits,model(tokens).logits)
    loss=numeric_loss(model(tokens,contexts=ctx).logits,tokens.labels,codec)[0];loss.backward()
    assert all(p.grad is None for p in model.native.parameters())
    assert all(block.up.weight.grad.abs().sum()>0 for block in model.adapter.exchanges)
    model.set_condition('self')
    with pytest.raises(ValueError,match='Stale'):model(tokens,contexts=ctx)
    nonzero(model);ctx=model.encode_origins([memory_graph(d,query,control='self')],codec)
    assert not torch.equal(model(tokens,contexts=ctx).logits,model(tokens).logits)


@pytest.mark.parametrize('length,horizon',[(12,12),(48,24)])
def test_native_reference_loss_gradient_and_actual_full_incremental_tokens(codec,length,horizon):
    d=data();model=tiny(codec);nonzero(model);query=q(length,horizon);history=d.values[200-length:200,0]
    proposal=generate(model,codec,history,horizon,seed=42,cached=True)
    off=coupled_draw(model,codec,history,horizon,proposal)
    assert off['numeric_ids']==proposal['numeric_ids']
    target=d.values[200:200+horizon,0].copy();target[::4]=np.nan
    tokens,_=scoring_tokens(codec,history,target,proposal['generated_ids']);graph=memory_graph(d,query)
    ctx=model.encode_origins([graph],codec);full=model(tokens,contexts=ctx).logits
    numeric_loss(full,tokens.labels,codec)[0].backward()
    gradients={n:None if p.grad is None else p.grad.clone() for n,p in model.adapter.named_parameters()}
    model.zero_grad();ctx=model.encode_origins([graph],codec)
    compact_loss(compact_forward(model,codec,tokens,ctx,teacher_prefix(model,tokens)))[0].backward()
    for n,p in model.adapter.named_parameters():
        if gradients[n] is None:assert p.grad is None
        else:torch.testing.assert_close(p.grad,gradients[n],rtol=1e-6,atol=1e-6)
    with torch.no_grad():
        ctx=model.encode_origins([graph],codec)
        value=verify_sampled_paths(model,codec,history,horizon,ctx,42,rtol=1e-6,atol=1e-6)
        assert value['sampled_tokens_equal']


def test_native_reference_future_labels_and_missingness_do_not_enter_input(codec):
    history=np.arange(12,dtype=float);model=tiny(codec);proposal=generate(model,codec,history,12,seed=42)
    target=np.arange(12,dtype=float);target[3]=np.nan
    before,_=scoring_tokens(codec,history,target,proposal['generated_ids'])
    target[2:]=1e9;target[8]=np.nan
    after,_=scoring_tokens(codec,history,target,proposal['generated_ids'])
    assert torch.equal(before.full_ids,after.full_ids) and not torch.equal(before.labels,after.labels)


def test_partial_native_prefix_gradient_optimizer_and_rng_restore(codec,tmp_path):
    model=tiny(codec);nonzero(model);d=data();query=q();history=d.values[188:200,0]
    proposal=generate(model,codec,history,12,seed=42)
    tokens,_=scoring_tokens(codec,history,d.values[200:212,0],proposal['generated_ids']);graph=memory_graph(d,query)
    optimizer=torch.optim.AdamW(model.adapter.parameters(),lr=1e-4)
    def micro():
        ctx=model.encode_origins([graph],codec);loss=compact_loss(compact_forward(model,codec,tokens,ctx,teacher_prefix(model,tokens)))[0]/2
        loss.backward();return float(loss.detach())
    micro();optimizer.step();model.mark_updated();optimizer.zero_grad();micro()
    state=dict(binding='b',condition='physical',updates=1,pending=1,partial_losses=[1.],losses=[],training_seconds=1.,reference_hash='frozen')
    path=tmp_path/'last.pt';checkpoint(model,optimizer,state,path)
    expected=micro();optimizer.step();model.mark_updated();weights=model.trainable_state();random_after=torch.rand(4)
    restored=restore_checkpoint(model,optimizer,path,'b','physical')
    assert restored['pending']==1 and restored['reference_hash']=='frozen'
    assert micro()==expected;optimizer.step();model.mark_updated()
    for n,p in model.trainable_state().items():assert torch.equal(p,weights[n])
    assert torch.equal(torch.rand(4),random_after)


def test_acceptance_factory_restores_original_runner_on_exception():
    from gofa_f import graph_memory_stage as old
    previous=old.GraphMemoryGOFA,old.FILES
    with pytest.raises(RuntimeError):
        with acceptance_factory():
            assert old.GraphMemoryGOFA is CrossInnovationGOFA and 'cross_innovation.py' in old.FILES
            raise RuntimeError('restore')
    assert (old.GraphMemoryGOFA,old.FILES)==previous
