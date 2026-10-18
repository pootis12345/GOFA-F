from pathlib import Path

import pytest
import torch
from transformers import LlamaTokenizer

from gofa_f.allnode_graph import build_graph
from gofa_f.generative_data import Query, SOURCES
from gofa_f.short_codec import ShortCodec
from gofa_f.short_data import TEMPLATES
from gofa_f.short_forecast import forecast_batch
from gofa_f.short_sampling_diagnostic import choose, generate, measures, run, sample_seed, summarize
from test_allnode import tiny, source


@pytest.fixture(scope='module')
def codec():
    root = Path('/home/dcg/dcg_data/GOFA-F/data/weights/chattime-7b-20baa88')
    if not root.exists():
        pytest.skip('Official tokenizer unavailable on this CPU host')
    return ShortCodec(LlamaTokenizer.from_pretrained(root, local_files_only=True))


def test_sampler_top100_and_local_rng_without_mutating_logits():
    scores = torch.arange(10000, dtype=torch.float32)[None]/10000
    original, rng = scores.clone(), torch.get_rng_state().clone()
    draws = []
    for _ in range(2):
        gen = torch.Generator().manual_seed(42)
        draws.append([choose(scores, gen).item() for i in range(20)])
    assert draws[0] == draws[1] and min(draws[0]) >= 9900
    assert len(set(draws[0])) > 1
    assert torch.equal(scores, original) and torch.equal(torch.get_rng_state(), rng)
    assert choose(scores, None).item() == 9999
    with pytest.raises(FloatingPointError, match='Nonfinite'):
        choose(scores*float('nan'), torch.Generator())


def test_seeds_query_sample_specific_and_model_independent():
    q = dict(source='nn5', variable=92, origin=568, length=48, horizon=24, chunk=0)
    seeds = [sample_seed(q, i) for i in range(8)]
    assert seeds == [sample_seed(dict(q), i) for i in range(8)] and len(set(seeds)) == 8
    assert sample_seed(dict(q, origin=569), 0) not in seeds


@pytest.mark.parametrize('horizon', [1, 3, 24])
def test_decoder_greedy_equals_production_with_nonzero_graph(codec, horizon):
    model, data = tiny(), source(3)
    with torch.no_grad():
        for module in model.adapter.exchanges:
            module.up.weight.normal_(std=.002)
        contexts = model.encode_origins([build_graph(data, Query('pems04', 0, 500, 12, horizon))], codec)
    history = data.values[488:500, 0]
    before = model.trainable_state()
    expected = forecast_batch(model, codec, [history], horizon, contexts)
    actual = generate(model, codec, history, horizon, contexts)
    assert actual['numeric_ids'] == expected['numeric_ids'][0]
    assert actual['generated_ids'] == expected['generated_ids'][0]
    assert actual['prediction'] == expected['predictions'][0].tolist()
    assert len(actual['generated_ids']) == len(codec.first_separator)+horizon+(horizon-1)*len(codec.separator)
    for key, value in model.trainable_state().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert all(p.grad is None for p in model.parameters())


def test_sampled_incremental_full_equivalence_and_fresh_kv_each_sample(codec):
    model, data = tiny(), source(3)
    with torch.no_grad():
        for module in model.adapter.exchanges:
            module.up.weight.normal_(std=.003)
        contexts = model.encode_origins([build_graph(data, Query('pems04', 0, 500, 12, 3))], codec)
    history = data.values[488:500, 0]
    cached, full = [], []
    a = generate(model, codec, history, 3, contexts, seed=42, trace=cached)
    b = generate(model, codec, history, 3, contexts, seed=42, cached=False, trace=full)
    assert a['generated_ids'] == b['generated_ids']
    for x, y in zip(cached, full):
        torch.testing.assert_close(x, y, rtol=1e-6, atol=1e-6)
    assert generate(model, codec, history, 3, contexts, seed=42) == a
    model.mark_updated()
    with pytest.raises(ValueError, match='Stale'):
        generate(model, codec, history, 3, contexts, seed=42)


def test_missing_truth_is_metrics_only_and_uses_float64():
    result = measures([2., 4., 8.], [1., 1e10, 5.], [True, False, True])
    assert result['mae'] == 2 and result['observed_targets'] == 2
    with pytest.raises(ValueError):
        measures([1.], [2.], [False])


def test_subset_aggregation_excludes_known_cases_and_preserves_source_hierarchy():
    records = []
    for s, count in SOURCES.items():
        for v in (count//3, 2*count//3):
            for length, horizon in TEMPLATES:
                for model in ('native', '1000'):
                    error = 2 if model == '1000' else 1
                    item = dict(mae=error, upper_slots=0, repetition={'constant_complete': horizon>1})
                    records.append(dict(group='id_stratified', model=model, index=len(records)//2,
                        query=dict(source=s, variable=v, length=length, horizon=horizon), greedy=item, sample8=item))
    result = summarize(records+[dict(group='known_symptom')], {s:[1e-12]*n for s,n in SOURCES.items()})
    assert result['subset_S'] == {'greedy':2., 'sample8':2.}
    with pytest.raises(ValueError, match='Complete paired'):
        summarize(records[:-1], {})


def test_refuses_unadmitted_gpu_work(monkeypatch, tmp_path):
    monkeypatch.delenv('GOFA_STAGE19_ADMITTED', raising=False)
    with pytest.raises(RuntimeError, match='fully-idle'):
        run({}, tmp_path)
