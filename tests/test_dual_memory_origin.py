from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import LlamaConfig,LlamaForCausalLM,LlamaTokenizer

from gofa_f.dual_memory_data import memory_input
from gofa_f.dual_memory_model import DualMemoryGOFA
from gofa_f.dual_memory_origin import NativeOrigin,SharedPrefixBuilder,select_rows
from gofa_f.dual_memory_run import make_tokens
from gofa_f.generative_data import Query
from gofa_f.short_codec import ShortCodec
from gofa_f.short_sampling_diagnostic import generate


@pytest.fixture(scope='module')
def codec():
    path=Path('/home/dcg/dcg_data/GOFA-F/data/weights/chattime-7b-20baa88')
    if not path.exists():pytest.skip('Official local tokenizer required')
    return ShortCodec(LlamaTokenizer.from_pretrained(path,local_files_only=True))


def fixture(codec,length,horizon,depth):
    torch.manual_seed(42)
    cfg=LlamaConfig(hidden_size=16,intermediate_size=24,num_hidden_layers=8,num_attention_heads=2,
        num_key_value_heads=2,vocab_size=42001,max_position_embeddings=2048,attention_dropout=0.)
    cfg._attn_implementation='sdpa'
    model=DualMemoryGOFA(LlamaForCausalLM(cfg),depth=depth,width=8,stable_fp32=True,weight_binding='tiny')
    with torch.no_grad():
        for block in model.adapter.exchanges:block.gate.fill_(.1);block.ff_gate.fill_(.1)
        model.adapter.head[-1].weight.normal_(std=.02)
    t=np.arange(400)[:,None];v=np.arange(6)[None]
    d=SimpleNamespace(values=(t*.03+np.sin(t/10+v)).astype('float32'),mask=np.ones((400,6),bool),
        ids=list('abcdef'),external=np.ones((6,6))-np.eye(6),segments=np.zeros(400,int))
    item=memory_input(d,Query('pems04_flow',0,200,length,horizon));tokens=make_tokens(codec,item,'cpu')
    return model,item,tokens,d.values[200-length:200,0]


@pytest.mark.parametrize('depth',[3,6])
@pytest.mark.parametrize('length,horizon',[(12,12),(48,24)])
def test_shared_native_root_prefix_matches_independent_frozen_history(codec,depth,length,horizon):
    model,item,tokens,history=fixture(codec,length,horizon,depth)
    origin=NativeOrigin(model,select_rows(tokens,[0],crop=True),codec.first_separator)
    combined=SharedPrefixBuilder(model,origin).frozen_history(tokens)
    reference=model.frozen_history(tokens)
    torch.testing.assert_close(combined.hidden[tokens.attention_mask],reference.hidden[tokens.attention_mask],rtol=1e-6,atol=1e-6)
    with torch.no_grad():
        actual,_=model.enhanced(tokens,item,torch.ones(16),prefix=combined,optimized=True)
        expected,_=model.enhanced(tokens,item,torch.ones(16),prefix=reference,optimized=False)
    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)


@pytest.mark.parametrize('length,horizon',[(12,12),(48,24)])
def test_eight_independent_native_paths_exact_tokens_and_rng_isolation(codec,length,horizon):
    model,item,tokens,history=fixture(codec,length,horizon,3)
    origin=NativeOrigin(model,select_rows(tokens,[0],crop=True),codec.first_separator)
    rng=torch.get_rng_state().clone();origin.prepare()
    saved=[v.clone() for v in origin.kv.key_cache]
    for seed in range(8):
        reference=generate(model.native_view(),codec,history,horizon,seed=seed)
        actual=generate(origin,codec,history,horizon,seed=seed)
        assert actual['generated_ids']==reference['generated_ids'] and actual['prediction']==reference['prediction']
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(saved,origin.kv.key_cache))
    changed=deepcopy(origin.tokens);changed.input_ids[0,0]+=1;changed.full_ids=changed.input_ids
    with pytest.raises(ValueError,match='another prompt'):origin(changed,use_cache=True)
