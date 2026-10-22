import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gofa_f.generative_graph import origin_graph
from gofa_f.source_graph_audit import (CONDITIONS, VIEWS, audit_sample, dual_support,
    fresh_queries, immutable, predict, rewire, summarize, variable_errors)
from gofa_f.source_dcrnn_model import DCRNN


def fixture():
    values = np.sin(np.arange(300)[:, None]/10+np.arange(4)[None]).astype(np.float32)
    d = SimpleNamespace(values=values, mask=np.ones_like(values, bool),
        segments=np.zeros(300), external=np.array([[0,1,0,0],[0,0,1,0],
            [0,0,0,1],[1,0,0,0]]), ids=['A','B','C','D'])
    d.graph = lambda q: origin_graph(d.values[q.origin-q.length:q.origin],
        d.mask[q.origin-q.length:q.origin], q.variable, d.external, d.ids)
    return d


def test_physical_direction_padding_and_self_control():
    d = fixture()
    q = dict(source='pems04_flow', variable=0, origin=180, length=12, horizon=12)
    a, supports, proof = audit_sample(d, q)
    assert supports['physical'][0, 0, 1] > 0
    assert supports['physical'][0, 0, 2] == 0
    assert supports['physical'][1, 1, 0] > 0
    assert supports['physical'][:, 4:].sum() == 0
    np.testing.assert_array_equal(supports['self'][:, :4, :4], np.tile(np.eye(4), (2,1,1)))
    assert proof['permutation'][0] == 0
    assert proof['changed_physical']
    np.testing.assert_array_equal(supports['mixed'], a['supports'])


def test_no_future_value_or_missingness_affects_inputs_or_graph_controls():
    d = fixture()
    q = dict(source='pems04_flow', variable=0, origin=180, length=12, horizon=12)
    a, sa, pa = audit_sample(d, q)
    d.values[180:] = 99999
    d.mask[180:] = False
    b, sb, pb = audit_sample(d, q)
    for key in ('x', 'present', 'supports', 'low', 'scale', 'empty_target'):
        np.testing.assert_array_equal(a[key], b[key])
    for c in sa:
        np.testing.assert_array_equal(sa[c], sb[c])
    assert pa == pb


def test_rewiring_preserves_support_and_degree_multiset_without_relabeling_target():
    p = np.array([1,1,1,1,0], bool)
    a = np.array([[1,2,0,0,0],[0,1,3,0,0],[0,0,1,4,0],[5,0,0,1,0],[0,0,0,0,0]], np.float32)
    s = dual_support(a, p)
    q = dict(source='boston2017', variable=0, origin=250, length=48, horizon=24)
    changed, perm = rewire(s, p, q)
    np.testing.assert_array_equal(changed, s[:, perm][:, :, perm])
    np.testing.assert_array_equal(np.sort(changed.flatten()), np.sort(s.flatten()))
    assert perm[0] == 0 and perm[4] == 4
    np.testing.assert_array_equal(changed[:,4], 0)
    assert np.count_nonzero(changed[0,0]) == np.count_nonzero(s[0,0])


def test_prediction_receives_no_teacher_and_consumes_no_rng():
    d = fixture()
    q = dict(source='pems04_flow', variable=0, origin=180, length=12, horizon=12)
    row, supports, _ = audit_sample(d, q)
    torch.manual_seed(42)
    model = DCRNN(hidden=8).eval()
    before = torch.get_rng_state().clone()
    a = predict(model, [row], [supports['physical']], 12)
    other = copy.deepcopy(row)
    other['teacher'][:] = 100000
    other['y'][:] = 100000
    other['observed'][:] = False
    b = predict(model, [other], [supports['physical']], 12)
    np.testing.assert_array_equal(a, b)
    assert torch.equal(before, torch.get_rng_state())


def test_metric_uses_valid_points_within_variable_then_equal_variables():
    rows = [dict(query=dict(variable=1), target=[2,99], observed=[True,False],
                 predictions=dict(physical=[0,0])),
            dict(query=dict(variable=1), target=[4,6], observed=[True,True],
                 predictions=dict(physical=[0,0])),
            dict(query=dict(variable=2), target=[20,99], observed=[True,False],
                 predictions=dict(physical=[0,0]))]
    result = variable_errors(rows, 'physical')
    assert result == {'1': 4., '2': 20.}
    assert np.mean(list(result.values())) == 12.


def test_immutable_evidence_rejects_changed_binding(tmp_path):
    p = tmp_path/'record.json'
    immutable(p, dict(binding='a', prediction=[1.]))
    immutable(p, dict(binding='a', prediction=[1.]))
    with pytest.raises(ValueError, match='Immutable'):
        immutable(p, dict(binding='b', prediction=[1.]))


def test_query_selection_no_overlap_and_reproducible_source_only():
    class Data:
        def candidates(self, *args):
            assert args[-1] == 'test'
            return np.arange(1000, 2000)
    excluded = [dict(source='pems04_flow', variable=0, origin=1200, horizon=12)]
    rows = fresh_queries(Data(), 'pems04_flow', [0,1], excluded, count=16)
    assert rows == fresh_queries(Data(), 'pems04_flow', [0,1], excluded, count=16)
    for i, a in enumerate(rows):
        for b in rows[i+1:]+excluded:
            if a['variable'] == b['variable']:
                assert a['origin']+12 <= b['origin'] or a['origin'] >= b['origin']+12
    with pytest.raises(ValueError):
        fresh_queries(Data(), 'sdwpf', [0], [], count=16)


def test_gate_requires_real_graph_gain_and_informative_rewiring():
    cfg = dict(bootstrap_draws=2000, gate=dict(minimum_improvement=.005,
        minimum_domains=2, maximum_other_domain_degradation=.02, minimum_supported_domains=1))
    def records(physical, changed=True):
        return {name: [dict(query=dict(variable=i % 8), target=[0.], observed=[True],
            graph=dict(changed_physical=changed, changed_mixed=True, physical_cross_edges=3),
            predictions={c: [physical if c == 'physical' else 1.] for c in CONDITIONS})
            for i in range(64)] for name in VIEWS}
    report = summarize(records(.9), cfg)
    assert report['gate']['passed']
    assert not summarize(records(1.), cfg)['gate']['passed']
    assert not summarize(records(.9, changed=False), cfg)['gate']['passed']
    regressed = records(.9)
    for r in regressed['penmanshiel2017']:
        r['predictions']['physical'] = [1.03]
    assert not summarize(regressed, cfg)['gate']['passed']


def test_missing_neighbor_and_constant_history_remain_finite():
    d = fixture()
    d.values[:180, 0] = 4.
    d.mask[:180, 3] = False
    q = dict(source='pems04_flow', variable=0, origin=180, length=12, horizon=12)
    packed, conditions, _ = audit_sample(d, q)
    assert packed['scale'] == 1.
    assert np.isfinite(packed['x']).all()
    for supports in conditions.values():
        assert np.isfinite(supports).all()
        np.testing.assert_array_equal(supports.sum(-1) > 0, np.tile(packed['present'] > 0, (2,1)))
