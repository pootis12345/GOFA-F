from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import LlamaTokenizer

from gofa_f.graph_memory_data import memory_graph
from gofa_f.graph_proposal_probe import coupled_draw
from gofa_f.short_codec import ShortCodec
from gofa_f.short_sampling_diagnostic import generate
from test_graph_memory import data,tiny,nonzero,q


@pytest.fixture(scope='module')
def codec():
    path=Path('/home/dcg/dcg_data/GOFA-F/data/weights/chattime-7b-20baa88')
    if not path.exists():pytest.skip('Pinned tokenizer on C')
    return ShortCodec(LlamaTokenizer.from_pretrained(path,local_files_only=True))


@pytest.mark.parametrize('length,horizon',[(12,12),(48,24)])
def test_native_off_identity_full_path_serial_rng_and_one_forward(codec,length,horizon):
    d=data();history=d.values[200-length:200,0];model=tiny(codec)
    proposal=generate(model,codec,history,horizon,seed=42,cached=True)
    rng=torch.get_rng_state().clone();calls=[]
    original=model.forward
    def counted(*args,**kwargs):calls.append(args[0].full_ids.clone());return original(*args,**kwargs)
    model.forward=counted
    result=coupled_draw(model,codec,history,horizon,proposal)
    model.forward=original
    assert result['numeric_ids']==proposal['numeric_ids'] and result['generated_ids']==proposal['generated_ids']
    assert result['prediction']==proposal['prediction'] and len(calls)==1
    assert result['no_corrected_output_feedback'] and torch.equal(rng,torch.get_rng_state())


def test_future_values_and_missingness_not_used_and_nonzero_graph_context_isolated(codec):
    d=data();history=d.values[188:200,0].copy();model=tiny(codec);nonzero(model)
    proposal=generate(model,codec,history,12,seed=123,cached=True)
    ctx=model.encode_origins([memory_graph(d,q())],codec)
    before=coupled_draw(model,codec,history,12,proposal,ctx)
    d.values[200:]=1e10;d.mask[200:]=False
    after=coupled_draw(model,codec,d.values[188:200,0],12,proposal,ctx)
    assert before['numeric_ids']==after['numeric_ids'] and np.isfinite(before['prediction']).all()
    model.mark_updated()
    with pytest.raises(ValueError,match='Stale'):coupled_draw(model,codec,history,12,proposal,ctx)


def test_wrong_proposal_grammar_or_horizon_refused(codec):
    model=tiny(codec);history=np.arange(12,dtype=float)
    proposal=generate(model,codec,history,12,seed=42,cached=True)
    with pytest.raises(ValueError,match='grammar'):coupled_draw(model,codec,history,11,proposal)
    altered=dict(proposal,generated_ids=proposal['generated_ids']+[codec.nan_id])
    with pytest.raises(ValueError,match='grammar'):coupled_draw(model,codec,history,12,altered)
