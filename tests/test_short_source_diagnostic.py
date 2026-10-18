from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import LlamaTokenizer

from gofa_f.allnode_graph import build_graph
from gofa_f.generative_data import Query, SOURCES
from gofa_f.short_codec import ShortCodec
from gofa_f.short_data import TEMPLATES
from gofa_f.short_forecast import forecast_batch
from gofa_f.short_source_diagnostic import (Trace, compare_scores, distribution, exchange_measurements,
    inspect_query, run, select_queries, self_only, verify_source)
from test_allnode import tiny, source


@pytest.fixture(scope='module')
def codec():
    root = Path('/home/dcg/dcg_data/GOFA-F/data/weights/chattime-7b-20baa88')
    if not root.exists():
        pytest.skip('Official tokenizer unavailable on this CPU host')
    return ShortCodec(LlamaTokenizer.from_pretrained(root,local_files_only=True))


def test_fixed_source_queries_complete_and_cases_separate():
    queries = [dict(source=name,variable=v,origin=600,length=length,horizon=horizon,chunk=0)
        for name,count in SOURCES.items() for v in range(count) for length,horizon in TEMPLATES]
    selected = select_queries(queries)
    assert len(selected) == len({r['index'] for r in selected}) == 34
    assert sum(r['group'] == 'id_stratified' for r in selected) == 32
    assert [(r['query']['source'],r['query']['variable']) for r in selected[-2:]] == [('nn5',16),('nn5',92)]
    for name in SOURCES:
        assert sum(r['query']['source'] == name and r['group'] == 'id_stratified' for r in selected) == 8
    with pytest.raises(ValueError,match='Exactly one'):
        select_queries([q for q in queries if not (q['source']=='solar' and q['variable']==45)])


def test_self_only_graph_does_not_mutate_original_or_time_context():
    data = source(9)
    graph = build_graph(data,Query('pems04',0,500,12,12))
    prior,base = graph.graph.prior.clone(),graph.graph.base.clone()
    changed = self_only(graph)
    assert changed.identity != graph.identity
    assert changed.query_key == graph.query_key and changed.variable_ids == graph.variable_ids
    assert changed.graph.history is graph.graph.history and changed.graph.observed is graph.graph.observed
    torch.testing.assert_close(graph.graph.prior,prior,rtol=0,atol=0)
    torch.testing.assert_close(graph.graph.base,base,rtol=0,atol=0)
    assert torch.equal(changed.graph.prior,torch.diag(graph.graph.observed.any(0).float()))
    assert torch.equal(changed.graph.base,changed.graph.prior)


def test_logits_tolerance_does_not_replace_greedy_equality():
    a,b = [torch.tensor([[1.,1.0001]])],[torch.tensor([[1.0001,1.]])]
    check = compare_scores(a,b)
    assert check['violations'] == 0 and not check['argmax_equal']
    assert compare_scores([torch.zeros(1,2)],[torch.ones(1,2)])['violations'] == 2


def test_telemetry_observational_and_finite_at_zero_and_nonzero():
    data,model = source(9),tiny()
    graph = build_graph(data,Query('pems04',0,500,12,12)).graph
    hidden = torch.randn(9,12,16)
    module = model.adapter.exchanges[0]
    original_hidden = hidden.clone()
    zero = module(hidden,graph)
    telemetry = exchange_measurements(module,(hidden,graph),zero)
    assert telemetry['target_delta_to_hidden_norm'] == 0.
    assert telemetry['target_gate_mean'] == .5
    with torch.no_grad():
        module.up.weight.normal_(std=.01)
    expected = module(hidden,graph).detach()
    telemetry = exchange_measurements(module,(hidden,graph),expected)
    assert telemetry['target_delta_to_hidden_norm'] > 0
    assert 0 <= telemetry['target_attention_self_mean'] <= 1
    torch.testing.assert_close(module(hidden,graph).detach(),expected,rtol=0,atol=0)
    torch.testing.assert_close(hidden,original_hidden,rtol=0,atol=0)


def test_endpoint_probability_is_over_finite_alphabet(codec):
    item = distribution(torch.zeros(2,10000),codec)
    assert item['upper_probability'] == pytest.approx([1/10000]*2)
    assert item['lower_probability'] == pytest.approx([1/10000]*2)
    assert item['argmax_ids'] == [codec.finite_ids[0]]*2


@pytest.mark.parametrize('cached',[True,False])
def test_trace_keeps_original_tokens_and_logits(codec,cached):
    model,data = tiny(),source(9)
    with torch.no_grad():
        for module in model.adapter.exchanges:
            module.up.weight.normal_(std=.001)
        context = model.encode_origins([build_graph(data,Query('pems04',0,500,12,3))],codec)
        history = data.values[488:500,0]
        expected = forecast_batch(model,codec,[history],3,context,cached=cached)
        observer = Trace(model,codec,cached)
        actual = forecast_batch(observer,codec,[history],3,context,cached=cached)
    assert actual['generated_ids'] == expected['generated_ids']
    assert len(observer.scores) == 3
    assert all(s.shape == (1,10000) and not s.requires_grad for s in observer.scores)


@pytest.mark.parametrize('mode',['native','full','self_only'])
def test_teacher_free_diagnostic_no_gradients_or_parameter_changes(codec,mode):
    model,data = tiny(),source(9)
    with torch.no_grad():
        for module in model.adapter.exchanges:
            module.up.weight.normal_(std=.001)
    data.mask[501,0] = False
    before = model.trainable_state()
    record = inspect_query(model,codec,{'pems04':data},
        dict(source='pems04',variable=0,origin=500,length=12,horizon=3,chunk=0),mode,lambda:None,True)
    assert record['teacher']['observed_slots'] == [True,False,True]
    assert np.isfinite(record['teacher']['conditional_ce'])
    assert record['cache_comparison']['argmax_equal'] and record['cache_comparison']['raw_tokens_equal']
    assert len(record['exchanges']) == (0 if mode == 'native' else 2)
    assert all(p.grad is None for p in model.parameters())
    for name,p in model.trainable_state().items():
        torch.testing.assert_close(p,before[name],rtol=0,atol=0)


def test_diagnostic_refuses_gpu_execution_without_admission(monkeypatch,tmp_path):
    monkeypatch.delenv('GOFA_STAGE19_ADMITTED',raising=False)
    with pytest.raises(RuntimeError,match='fully-idle'):
        run({},tmp_path)


@pytest.mark.parametrize('change',[{'diagnostic_updates':1},{'resume_training_after_diagnostic':True},
                                  {'data':{'target_evaluation_enabled':True}}])
def test_diagnostic_cannot_train_or_evaluate_targets(change):
    cfg = dict(diagnostic_updates=0,resume_training_after_diagnostic=False,data={'target_evaluation_enabled':False})
    cfg.update(change)
    with pytest.raises(ValueError,match='read-only source'):
        verify_source(cfg)
