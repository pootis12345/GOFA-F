from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import LlamaTokenizer

from gofa_f.graph_memory_data import memory_graph
from gofa_f.graph_memory_diagnostic import observe_reads,score_distribution,scoring_tokens,summarize_records,work_items
from gofa_f.short_codec import ShortCodec,numeric_loss
from test_graph_memory import data,tiny,nonzero,q


@pytest.fixture(scope='module')
def codec():
    path=Path('/home/dcg/dcg_data/GOFA-F/data/weights/chattime-7b-20baa88')
    if not path.exists():pytest.skip('Pinned official tokenizer on C')
    return ShortCodec(LlamaTokenizer.from_pretrained(path,local_files_only=True))


def test_generated_prefix_preserves_labels_and_does_not_see_future_truth_or_missingness(codec):
    history=np.arange(12,dtype=float);target=np.array([12.,np.nan,14.,15.])
    native=codec.record(history,np.array([8.,9.,10.,9.]))
    saved=native['ids'][native['prompt_end']+1:]
    tokens,record=scoring_tokens(codec,history,target,saved)
    changed=target.copy();changed[0]=1e4;changed[1]=3.;changed[3]=np.nan
    other,_=scoring_tokens(codec,history,changed,saved)
    assert torch.equal(tokens.input_ids,other.input_ids) and torch.equal(tokens.full_ids,other.full_ids)
    assert not torch.equal(tokens.labels,other.labels)
    assert torch.equal(tokens.labels,codec.batch([codec.record(history,target)]).labels)
    with pytest.raises(ValueError,match='grammar'):scoring_tokens(codec,history,target,saved[:-1])
    with pytest.raises(ValueError,match='grammar'):scoring_tokens(codec,history,target,saved+[codec.nan_id])


def test_score_shift_full_vocabulary_mask_and_median_matches_reference(codec):
    history=np.ones(12)*3;target=np.array([3.,np.nan,4.])
    tokens,record=scoring_tokens(codec,history,target)
    torch.manual_seed(42);logits=torch.randn(1,tokens.input_ids.shape[1],42001)
    value=score_distribution(logits,tokens,codec,record,target)
    assert [row['step'] for row in value['steps']]==[1,3] and value['numeric_vocabulary']==10000
    actual=np.mean([row['ce'] for row in value['steps']]);expected=float(numeric_loss(logits,tokens.labels,codec)[0])
    assert abs(actual-expected)<1e-6
    positions=torch.nonzero(tokens.labels[0]!=-100).flatten()
    scores=logits[0,positions[0]-1,codec.alphabet('cpu')].double();p=scores.softmax(-1)
    median=int(torch.searchsorted(p.cumsum(-1),torch.tensor(.5,dtype=torch.float64)))
    raw=(codec.id_centers[codec.finite_ids[median]]+.5-record['scaling'][1])/record['scaling'][0]
    assert abs(value['steps'][0]['median_prediction']-raw)<1e-6


@pytest.mark.parametrize('bin_index',[0,9999])
def test_boundary_labels_have_zero_cdf_and_ce_for_point_mass(codec,bin_index):
    history=np.array([-1.,1.]);center=codec.id_centers[codec.finite_ids[bin_index]];target=np.array([2*center])
    tokens,record=scoring_tokens(codec,history,target);position=int(torch.nonzero(tokens.labels[0]!=-100)[0])
    logits=torch.full((1,tokens.input_ids.shape[1],42001),-1e4);logits[0,position-1,codec.finite_ids[bin_index]]=0
    value=score_distribution(logits,tokens,codec,record,target)['steps'][0]
    assert value['ce']==0 and value['cdf_score']==0 and value['median_error']<1e-6 and value['endpoint_probability']==1


def test_observation_wrapper_preserves_logits_rng_and_restores_on_failure(codec):
    model=tiny(codec);nonzero(model);d=data();query=q()
    tokens=codec.batch([codec.record(d.values[188:200,0],d.values[200:212,0])]);graph=memory_graph(d,query)
    with torch.no_grad():
        expected=model(tokens,contexts=model.encode_origins([graph],codec)).logits
        before=torch.get_rng_state().clone();rows=[];phase=['source_history']
        with observe_reads(model,rows,phase):
            contexts=model.encode_origins([graph],codec);phase[0]='target_decoder';actual=model(tokens,contexts=contexts).logits
        assert torch.equal(expected,actual) and torch.equal(before,torch.get_rng_state())
        assert len(rows)==4 and {v['phase'] for v in rows}=={'source_history','target_decoder'}
        assert all(v['all_positions']['maximum']<=1 for v in rows)
        assert any(v['auxiliary_positions'] is not None for v in rows)
        assert all('read' not in block.__dict__ for block in model.adapter.exchanges)
        with pytest.raises(RuntimeError,match='forced'):
            with observe_reads(model,[],phase):raise RuntimeError('forced')
        assert all('read' not in block.__dict__ for block in model.adapter.exchanges)


def test_work_budget_has_no_optimizer_or_new_forecasts_and_summary_keeps_source_units():
    work=work_items([{} for _ in range(16)])
    assert len(work)==160 and sum(v['label'].startswith('physical_support_') for v in work)==32
    assert sum(v['prefix']=='native_sample0' for v in work)==64
    rows=[dict(work=dict(label='native',prefix='teacher'),query=dict(source=name),scores=dict(steps=[dict(step=1,
        **{key:value for key in ('ce','cdf_score','median_error','argmax_error','top100_median_error','entropy','endpoint_probability','true_label_top100')})]))
        for name,value in [('flow',2.),('weather',200.)]]
    result=summarize_records(rows)['native/teacher']
    assert result['flow']['all_steps']['median_error']==2 and result['weather']['all_steps']['median_error']==200
