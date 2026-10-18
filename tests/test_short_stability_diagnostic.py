from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import LlamaTokenizer

from gofa_f.allnode_graph import build_graph
from gofa_f.generative_data import Query
from gofa_f.short_codec import ShortCodec
from gofa_f.short_sampling_diagnostic import generate
from gofa_f.short_stability_diagnostic import amplitude, report, run, shard, teacher
from gofa_f.short_stability_metrics import quantiles, teacher_distribution
from gofa_f.prefix_cache import atomic_json
from test_allnode import tiny, source


@pytest.fixture(scope='module')
def codec():
    root = Path('/home/dcg/dcg_data/GOFA-F/data/weights/chattime-7b-20baa88')
    if not root.exists():
        pytest.skip('Official tokenizer unavailable on this CPU host')
    return ShortCodec(LlamaTokenizer.from_pretrained(root, local_files_only=True))


def initialized():
    model = tiny()
    with torch.no_grad():
        for module in model.adapter.exchanges:
            module.up.weight.normal_(std=.01)
    return model


def test_quarter_scales_all_nodes_and_recomputes_second_layer(codec):
    model, data = initialized(), source(3)
    graph = build_graph(data, Query('pems04', 0, 500, 12, 3))
    state = model.trainable_state()
    first_raw, second_input = [], []
    handle1 = model.adapter.exchanges[0].register_forward_hook(lambda m,a,o:first_raw.append(o.detach().clone()))
    handle2 = model.adapter.exchanges[1].register_forward_pre_hook(lambda m,a:second_input.append(a[0].detach().clone()))
    try:
        with amplitude(model, 1.):
            full = model.encode_origins([graph], codec)[0]
        telemetry = []
        with amplitude(model, .25, telemetry):
            small = model.encode_origins([graph], codec)[0]
    finally:
        handle1.remove()
        handle2.remove()
    torch.testing.assert_close(first_raw[0], first_raw[1], rtol=0, atol=0)
    torch.testing.assert_close(small.deltas[0], full.deltas[0]*.25, rtol=0, atol=0)
    assert not torch.equal(second_input[0], second_input[1])
    assert not torch.equal(small.deltas[1], full.deltas[1]*.25)
    assert len(telemetry) == 2 and all(r['scaled_all_node_position_ratios']['count']>12 for r in telemetry)
    for r in telemetry:
        assert r['scaled_target_delta_to_hidden_norm'] == pytest.approx(r['raw']['target_delta_to_hidden_norm']*.25)
    assert all(p.grad is None for p in model.parameters())
    for key, value in model.trainable_state().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)


def test_scope_invalidates_context_and_restores_hooks_on_error(codec):
    model, data = initialized(), source(3)
    counts = [len(m._forward_hooks) for m in model.adapter.exchanges]
    history = data.values[488:500, 0]
    with pytest.raises(RuntimeError, match='intentional'):
        with amplitude(model, .25):
            contexts = model.encode_origins([build_graph(data, Query('pems04', 0, 500, 12, 3))], codec)
            with pytest.raises(ValueError, match='nested'):
                with amplitude(model, 1.):
                    pass
            raise RuntimeError('intentional')
    assert [len(m._forward_hooks) for m in model.adapter.exchanges] == counts
    with pytest.raises(ValueError, match='Stale'):
        generate(model, codec, history, 3, contexts, seed=42)


@pytest.mark.parametrize('alpha', [0., .25, 1.])
def test_nonzero_cached_full_sampling_and_teacher_missing_mask(codec, alpha):
    model, data = initialized(), source(3)
    history = data.values[488:500, 0]
    target = np.array([1., np.nan, 2.])
    graph = build_graph(data, Query('pems04', 0, 500, 12, 3))
    with amplitude(model, alpha):
        contexts = model.encode_origins([graph], codec)
        a, b = [], []
        cached = generate(model, codec, history, 3, contexts, seed=42, trace=a)
        full = generate(model, codec, history, 3, contexts, seed=42, cached=False, trace=b)
        assert cached['generated_ids'] == full['generated_ids']
        for x,y in zip(a,b):
            torch.testing.assert_close(x,y,rtol=1e-6,atol=1e-6)
        observed = teacher(model, codec, history, target, contexts)
        assert observed['observed_slots'] == [True, False, True]
        assert observed['concentration']['true_ce'][1] is None
        assert all(np.isfinite(observed[k]) for k in ('conditional_ce','entropy_observed_mean'))
        # Target labels never enter generate() or change immutable graph origins.
        assert generate(model, codec, history, 3, contexts, seed=42)['generated_ids'] == cached['generated_ids']
    assert all(p.grad is None for p in model.parameters())
    if alpha == 0:
        assert generate(model, codec, history, 3, seed=42)['generated_ids'] == cached['generated_ids']


def test_teacher_numeric_median_uses_value_order_and_full_ce_not_topk():
    size = 10000
    class Codec:
        finite_ids = tuple(range(size))
        def lookup(self, count, device):return torch.arange(count,device=device)
        def alphabet(self, device):return torch.arange(size,device=device)
        def decode_ids(self, ids, scale, offset, last):
            values=np.array(ids,np.float32)
            return values,np.ones(len(ids),bool),np.zeros(len(ids),bool)
    scores = torch.full((2,size),-50.)
    scores[:,0],scores[:,8000],scores[:,9000]=np.log(.4),np.log(.35),np.log(.25)
    scores[:,9999] = -100.
    record=dict(answer_positions=[1,2],labels=[-100,9999,-100],scaling=(1.,0.),last=0.)
    result=teacher_distribution(scores,Codec(),record,[8000.,np.nan])
    assert result['predictions']['argmax']['prediction'][0] == 0
    assert result['predictions']['full_median']['prediction'][0] == 8000
    assert result['predictions']['top100_median']['prediction'][0] == 8000
    assert result['top100_truth_coverage'] == 0
    assert np.isfinite(result['conditional_ce']) and result['conditional_ce'] > 40


def test_empty_quantiles_and_checkpoint_shards_not_silently_rebound(tmp_path):
    assert quantiles(torch.tensor([]))['mean'] is None
    path=tmp_path/'sample.json'
    atomic_json(path,dict(binding='a',condition='alpha025',seed=42))
    assert shard(path,'a',condition='alpha025',seed=42)['seed']==42
    with pytest.raises(ValueError,match='binding'):
        shard(path,'a',condition='alpha025',seed=43)
    with pytest.raises(ValueError,match='binding'):
        shard(path,'b',condition='alpha025',seed=42)


def test_partial_report_cannot_invent_completed_statistics(tmp_path):
    (tmp_path/'queries').mkdir()
    atomic_json(tmp_path/'stability_manifest.json',dict(queries=[dict(index=0,query={'source':'nn5'})]))
    result=report({},tmp_path)
    assert not result['complete'] and len(result['missing'])==3 and 'summary' not in result


def test_no_unadmitted_execution(monkeypatch,tmp_path):
    monkeypatch.delenv('GOFA_STAGE19_ADMITTED',raising=False)
    with pytest.raises(RuntimeError,match='fully-idle'):
        run({},tmp_path)


def test_amplitude_cannot_be_arbitrarily_searched_or_used_during_forward():
    model=SimpleNamespace(_active=None,_kernel_active=False)
    with pytest.raises(ValueError):
        with amplitude(model,.5):
            pass
