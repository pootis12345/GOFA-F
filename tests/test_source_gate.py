from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gofa_f.source_gate import (FEATURES, SOURCES, TEMPLATES, features,
                              fit_and_cross_validate, fit_tree, select, source_score)


def observed(nodes=6, length=12):
    x = np.arange(length, dtype=np.float64)
    history = np.stack([x+i for i in range(nodes)], 1)
    weights = np.ones((nodes, nodes))/nodes
    graph = SimpleNamespace(history=history, observed=np.ones_like(history, bool),
        base=weights, target=0, edges=np.ones((nodes, nodes, 2)))
    return history[:, 0], graph


def rows():
    records = []
    for source in SOURCES:
        for v in range(8):
            for length, horizon in TEMPLATES:
                f = dict(values=[0.]*len(FEATURES), eligible=True, fallback_reason=None)
                records.append(dict(query=dict(source=source, variable=v, length=length, horizon=horizon, origin=100),
                    features=f, errors=[1., .9, .8, .7], risks=[-.1, -.2, -.3]))
    return records


def test_gate_has_no_ids_and_supports_six_node_graph_and_unit_changes():
    history, graph = observed()
    a = features(history, np.ones(12, bool), graph, 12, 12)
    assert a['eligible'] and a['values'][6] == 5/8
    other = deepcopy(graph)
    other.history = other.history*100+50
    b = features(history*100+50, np.ones(12, bool), other, 12, 12)
    np.testing.assert_allclose(a['values'], b['values'], rtol=1e-12, atol=1e-12)
    assert len(a['values']) == len(FEATURES) == 11


def test_empty_missing_constant_neighbors_and_origin_mismatch():
    x, g = observed()
    g.observed[:, 0] = False
    g.history[:, 0] = 0
    item = features(np.full(12, np.nan), np.zeros(12, bool), g, 12, 12)
    assert item['fallback_reason'] == 'empty_history'
    assert np.isfinite(item['values']).all()
    x, g = observed()
    g.base = np.eye(6)
    assert features(x, np.ones(12, bool), g, 12, 12)['fallback_reason'] == 'no_neighbors'
    g.history[:] = 3.
    item = features(np.full(12, 3.), np.ones(12, bool), g, 12, 12)
    assert np.isfinite(item['values']).all()
    with pytest.raises(ValueError, match='differs'):
        features(x+1, np.ones(12, bool), g, 12, 12)


def test_permutation_does_not_change_features():
    x, graph = observed(9)
    a = features(x, np.ones(12, bool), graph, 12, 12)
    p = np.array([4, 3, 0, 8, 1, 7, 2, 6, 5])
    changed = SimpleNamespace(history=graph.history[:, p], observed=graph.observed[:, p],
        base=graph.base[p][:, p], edges=graph.edges[p][:, p], target=2)
    np.testing.assert_allclose(a['values'], features(x, np.ones(12, bool), changed, 12, 12)['values'])


def test_exported_tree_and_pre_registered_margin_tie_policy():
    records = rows()
    tree = fit_tree(records)
    assert select(tree, records[0]['features'])['alpha'] == 1.
    for i in range(len(tree['risks'])):
        tree['risks'][i] = [-.02, -.02, -.01]
    assert select(tree, records[0]['features'])['alpha'] == .25
    tree['risks'][0] = [-.005, 0., .01]
    assert select(tree, records[0]['features'])['alpha'] == 0.
    item = dict(values=[0.]*11, eligible=False, fallback_reason='few_observations')
    assert select(tree, item)['reason'] == 'few_observations'
    tree['margin'] = 0
    with pytest.raises(ValueError, match='policy'):
        select(tree, item)


def test_leave_source_out_and_hierarchical_score_coverage_near_zero():
    records = rows()
    eps = {s: [1e-6]*8 for s in SOURCES}
    cv, gate = fit_and_cross_validate(records, eps)
    assert cv['validation']['passed'] and cv['validation']['score'] == pytest.approx(.7)
    assert len(cv['folds']) == 4 and all(t['fitted_queries'] == 96 for t in cv['folds'].values())
    assert gate['fitted_queries'] == 128
    score_rows = [dict(query=r['query'], native_mae=0., gated_mae=0.) for r in records]
    assert source_score(score_rows, eps)['score'] == 1.
    score_rows[0]['gated_mae'] = 1.
    assert source_score(score_rows, eps)['score'] > 1.
    with pytest.raises(ValueError, match='Incomplete'):
        source_score(score_rows[:-1], eps)


def test_no_target_outcomes_or_dataset_specific_gate_selection(tmp_path):
    from gofa_f.graph_benchmark import atomic
    from gofa_f.source_gate_run import locked_gate, target_barrier
    records = rows()
    eps = {s: [1e-6]*8 for s in SOURCES}
    cv, gate = fit_and_cross_validate(records, eps)
    atomic(tmp_path/'source_fit.json', {'records': records})
    from gofa_f.graph_benchmark import sha
    gate.update(binding='b', target_labels_used=False, fitting_records_sha256=sha(tmp_path/'source_fit.json'))
    cv['binding'] = 'b'
    atomic(tmp_path/'gate.json', gate)
    atomic(tmp_path/'source_cross_validation.json', cv)
    assert locked_gate(tmp_path, 'b')['fitted_queries'] == 128
    with pytest.raises(FileNotFoundError):
        target_barrier(tmp_path, 'b')
    atomic(tmp_path/'source_confirmation.json', dict(binding='b', complete=True,
        gate_sha256=sha(tmp_path/'gate.json'), validation={'passed': False}))
    with pytest.raises(ValueError, match='blocked'):
        target_barrier(tmp_path, 'b')


def test_confirmation_audit_blocks_any_recorded_source_holdout_exposure(tmp_path):
    from gofa_f.graph_benchmark import atomic
    from gofa_f.source_gate_state import holdout_audit
    root = tmp_path/'stage19-example'
    atomic(root/'pools/queries.json', {'train': {'solar': [dict(source='solar', origin=50, horizon=1)]}})
    data = {'solar': SimpleNamespace(values=np.zeros((100, 1)))}
    assert holdout_audit(tmp_path, data)['passed']
    atomic(root/'example_manifest.json', {'queries': [{'query': dict(source='solar', origin=85, horizon=1)}]})
    with pytest.raises(ValueError, match='isolation failed'):
        holdout_audit(tmp_path, data)


def test_source_pools_are_fixed_and_confirmation_label_windows_disjoint():
    from gofa_f.source_gate_state import make_pools
    n = 2000
    data, validation = {}, []
    for s in SOURCES:
        nodes = 11
        mask = np.ones((n, nodes), bool)
        data[s] = SimpleNamespace(values=np.zeros((n, nodes)), mask=mask, segments=np.zeros(n, int),
            prefix_counts=np.r_[np.zeros((1, nodes), int), mask.cumsum(0)])
    # The fixed original pool has 566 variables, not a flexible synthetic budget.
    from gofa_f.generative_data import SOURCES as COUNTS
    for s, count in COUNTS.items():
        mask = np.ones((n, count), bool)
        data[s] = SimpleNamespace(values=np.zeros((n, count)), mask=mask, segments=np.zeros(n, int),
            prefix_counts=np.r_[np.zeros((1, count), int), mask.cumsum(0)])
        for variable in range(count):
            for length, horizon in TEMPLATES:
                validation.append(dict(source=s, variable=variable, origin=1300, length=length, horizon=horizon, chunk=0))
    pools = make_pools(data, {'validation': validation})
    assert len(pools['fit']) == len(pools['confirmation']) == 128
    assert pools == make_pools(data, {'validation': validation})
    windows = {}
    for row in pools['confirmation']:
        q = row['query']
        assert 1600 <= q['origin'] <= n-q['horizon']
        key = q['source'], q['variable']
        for a, b in windows.setdefault(key, []):
            assert q['origin']+q['horizon'] <= a or q['origin'] >= b
        windows[key].append((q['origin'], q['origin']+q['horizon']))


@pytest.mark.parametrize('alpha', [.25, .5, 1.])
def test_nonzero_all_node_amplitude_cache_and_forward_equivalence(alpha):
    import torch
    from transformers import LlamaTokenizer
    from gofa_f.short_codec import ShortCodec
    from gofa_f.allnode_graph import build_graph
    from gofa_f.generative_data import Query
    from gofa_f.source_gate_inference import amplitude
    from gofa_f.short_sampling_diagnostic import generate
    from test_allnode import source, tiny
    root = Path('/home/dcg/dcg_data/GOFA-F/data/weights/chattime-7b-20baa88')
    if not root.exists():
        pytest.skip('Official tokenizer unavailable')
    codec = ShortCodec(LlamaTokenizer.from_pretrained(root, local_files_only=True))
    model, data = tiny(), source(6)
    for module in model.adapter.exchanges:
        with torch.no_grad():
            module.up.weight.normal_(std=.01)
    graph = build_graph(data, Query('pems04', 0, 500, 12, 12))
    initial = model.trainable_state()
    h = data.values[488:500, 0]
    with torch.no_grad():
        with amplitude(model, 1.):
            full_context = model.encode_origins([graph], codec)[0]
        with amplitude(model, alpha):
            context = model.encode_origins([graph], codec)
            torch.testing.assert_close(context[0].deltas[0], full_context.deltas[0]*alpha, rtol=0, atol=0)
            a, b = [], []
            cached = generate(model, codec, h, 3, context, seed=42, trace=a)
            full = generate(model, codec, h, 3, context, seed=42, cached=False, trace=b)
            assert cached['generated_ids'] == full['generated_ids']
            for x, y in zip(a, b):
                torch.testing.assert_close(x, y, rtol=1e-6, atol=1e-6)
            if alpha != 1:
                assert not torch.equal(context[0].deltas[1], full_context.deltas[1]*alpha)
        with pytest.raises(ValueError, match='Stale'):
            generate(model, codec, h, 3, context, seed=42)
    from gofa_f.short_migration import assert_equal
    assert_equal(initial, model.trainable_state())
    assert all(p.grad is None for p in model.parameters())


def test_prediction_requires_supervised_idle_admission(monkeypatch, tmp_path):
    from gofa_f.source_gate_run import run
    monkeypatch.delenv('GOFA_STAGE19_ADMITTED', raising=False)
    with pytest.raises(RuntimeError, match='fully-idle'):
        run({}, tmp_path)


def test_future_changes_do_not_change_history_features():
    from test_allnode import source
    from gofa_f.source_gate_run import observation
    data = source(9)
    query = dict(source='pems04', variable=0, origin=500, length=12, horizon=12, chunk=0)
    a = observation(data, query)
    data.values[500:] = 1e6
    data.mask[500:] = False
    assert observation(data, query) == a


def test_scope_rejects_nested_execution_and_cleans_up_after_error():
    from test_allnode import tiny
    from gofa_f.source_gate_inference import amplitude
    model = tiny()
    generation = model.cache_generation
    counts = [len(module._forward_hooks) for module in model.adapter.exchanges]
    with pytest.raises(RuntimeError, match='intentional'):
        with amplitude(model, .5):
            with pytest.raises(ValueError, match='nested'):
                with amplitude(model, .25):
                    pass
            raise RuntimeError('intentional')
    assert model.cache_generation == generation+2
    assert not model._source_gate_scope
    assert [len(module._forward_hooks) for module in model.adapter.exchanges] == counts


def test_gate_fit_rejects_target_or_unsealed_source_queries(monkeypatch, tmp_path):
    from gofa_f.graph_benchmark import atomic, sha
    from gofa_f.source_reliability_gate import fit_gate
    from gofa_f import source_gate_state
    records = rows()
    atomic(tmp_path/'manifest.json', {'pools': {'fit': [dict(query=r['query']) for r in records]}})
    monkeypatch.setattr(source_gate_state, 'verify', lambda cfg, output: __import__('json').loads(
        (output/'manifest.json').read_text()))
    records[0]['query'] = dict(source='engrad')
    atomic(tmp_path/'source_fit.json', dict(binding=sha(tmp_path/'manifest.json'), complete=True,
        action_order=[0., .25, .5, 1.], records=records))
    with pytest.raises(ValueError, match='sealed source'):
        fit_gate({}, tmp_path)
