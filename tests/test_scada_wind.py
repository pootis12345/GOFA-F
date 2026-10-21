import copy
from types import SimpleNamespace

import numpy as np
import pytest

from gofa_f.scada_wind_models import (CHANNELS, RECIPES, arrays, features, folds,
                                    fresh_confirmation, load_scada, summary)
from gofa_f.generative_data import sha


def fixture():
    v = np.arange(1000*3, dtype=np.float32).reshape(1000, 3)/10-30
    return SimpleNamespace(values=v, mask=np.ones_like(v, bool),
        times=np.arange(1000, dtype=np.int64)*600*10**9,
        segments=np.zeros(1000), external=np.ones((3, 3))-np.eye(3), ids=['A', 'B', 'C'],
        prefix_counts=np.vstack([np.zeros((1, 3), np.int64), np.ones_like(v, np.int64).cumsum(0)]))


def test_all_scada_features_ignore_every_future_channel_and_mask():
    d = fixture()
    s = np.repeat(d.values[:, :, None], len(CHANNELS), axis=2)
    other = copy.deepcopy(d)
    changed = s.copy()
    other.values[100:] = 1e9
    other.mask[100:] = False
    other.times[100:] = 0
    changed[100:] = np.nan
    q = dict(source='sdwpf', variable=0, origin=100, length=48, horizon=24)
    assert len(RECIPES) == 4
    for recipe in RECIPES:
        a, b = features(d, s, q, recipe)
        x, y = features(other, changed, q, recipe)
        np.testing.assert_array_equal(a, x)
        np.testing.assert_array_equal(b, y)


def test_scada_changes_can_distinguish_identical_power_and_empty_neighbors_are_zero():
    d = fixture()
    s = np.zeros((*d.values.shape, len(CHANNELS)), np.float32)
    q = dict(source='sdwpf', variable=0, origin=100, length=48, horizon=24)
    a, _ = features(d, s, q, RECIPES[0])
    s[52:100, 0, 0] = 8.
    b, _ = features(d, s, q, RECIPES[0])
    assert not np.array_equal(a, b)
    s[:, 1:] = np.nan
    c, _ = features(d, s, q, 'residual_neighbor')
    assert np.all(c[:, -199:] == 0)
    assert not np.isinf(c).any()


def test_raw_weight_and_absolute_or_residual_targets_match():
    d = fixture()
    d.mask[100:104, 0] = False
    s = np.zeros((*d.values.shape, len(CHANNELS)), np.float32)
    q = dict(source='sdwpf', variable=0, origin=100, length=48, horizon=24)
    other = dict(q, variable=1, origin=200)
    _, residual, w = arrays(d, s, [q, other], 'residual_local')
    _, level, w2 = arrays(d, s, [q, other], 'level_local')
    assert np.isclose(w[:20].sum(), w[20:].sum())
    np.testing.assert_array_equal(w, w2)
    np.testing.assert_allclose(level[:20]-residual[:20], d.values[99, 0], atol=1e-6)
    with pytest.raises(ValueError, match='First60'):
        arrays(d, s, [dict(q, origin=590)], RECIPES[0])


def test_scada_summary_missing_constant_and_slope_alignment():
    a = summary(np.column_stack((np.ones(48), np.full(48, np.nan), np.arange(48))))
    assert a[0] == 1 and a[1] == 1
    assert np.isnan(a[11]) and a[12] == 0
    assert a[22] == 47
    assert a[26] == 1 and a[29] == 1 and a[32] == 1


def test_forward_folds_never_fit_later_or_cross_blocks():
    d = SimpleNamespace(values=np.zeros((10000, 3)))
    records = [dict(query=dict(origin=6000+400*b+20+i, horizon=24))
               for b in range(5) for i in range(128 if b == 4 else 32)]
    for train, valid in folds(d, records):
        assert max(records[i]['query']['origin'] for i in train)+24 \
            < min(records[i]['query']['origin'] for i in valid)
    records[0]['query']['origin'] = 6390
    with pytest.raises(ValueError, match='crossed'):
        folds(d, records)


def test_new_confirmation_does_not_replace_identity_if_capacity_exhausted():
    d = fixture()
    q = dict(source='sdwpf', variable=0, origin=770, length=48, horizon=24, chunk=0)
    with pytest.raises(ValueError, match='new disjoint'):
        fresh_confirmation(d, dict(queries={'sdwpf': [q]*128}), [dict(query=q)])


def test_cache_rejects_mutation_even_after_memory_hit(tmp_path):
    path = tmp_path/'scada.npz'
    np.savez_compressed(path, values=np.zeros((2, 3, 9), np.float32))
    expected = sha(path)
    assert load_scada(str(tmp_path), expected).shape == (2, 3, 9)
    np.savez_compressed(path, values=np.ones((2, 3, 9), np.float32))
    with pytest.raises(ValueError, match='changed'):
        load_scada(str(tmp_path), expected)
