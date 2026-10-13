import numpy as np
import pytest

from gofa_f.data import Series, Windows, correlation_graph


def sample(tmp_path, segments=None):
    path = tmp_path / "sample.npz"
    values = np.arange(120, dtype=np.float32).reshape(40, 3)
    np.savez(path, values=values, mask=np.ones_like(values, bool), train_end=24, val_end=32,
             segments=np.zeros(40, np.int64) if segments is None else segments)
    return Series.load(path)


def test_scaling_only_uses_training_data(tmp_path):
    series = sample(tmp_path)
    z1, mean1, std1 = series.standardize()
    series.values[series.train_end:] += 100000
    z2, mean2, std2 = series.standardize()
    np.testing.assert_array_equal(mean1, mean2)
    np.testing.assert_array_equal(std1, std2)
    np.testing.assert_array_equal(z1[:24], z2[:24])


def test_target_splits_are_disjoint(tmp_path):
    series = sample(tmp_path)
    z, _, _ = series.standardize()
    for split, lo, hi in (("train", 0, 24), ("val", 24, 32), ("test", 32, 40)):
        windows = Windows(series, z, split, 4, 2)
        assert all(lo <= origin and origin + 2 <= hi for origin in windows.origins)
        assert windows[0]["x"].shape == (4, 3)
    assert Windows(series, z, "val", 4, 2).origins[0] == 24


def test_windows_do_not_cross_segment_boundaries(tmp_path):
    series = sample(tmp_path, np.repeat([0, 1], 20))
    windows = Windows(series, series.values, "train", 4, 2)
    assert all(not (t-4 < 20 <= t+1) for t in windows.origins)


def test_empty_split_raises(tmp_path):
    series = sample(tmp_path)
    with pytest.raises(ValueError, match="No valid"):
        Windows(series, series.values, "test", 4, 20)


def test_pairwise_graph_ignores_unobserved_values():
    values = np.arange(120, dtype=np.float32).reshape(40, 3)
    mask = np.ones_like(values, bool)
    mask[:10, 0] = False
    a = correlation_graph(values, mask, k=1)
    values[:10, 0] = 1e9
    b = correlation_graph(values, mask, k=1)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, a.T)
    assert np.all(np.diag(a) == 1)


def test_cold_start_scaler_uses_pooled_training_statistics(tmp_path):
    series = sample(tmp_path)
    series.mask[:series.train_end, 0] = False
    z, mean, std = series.standardize()
    observed = series.values[:series.train_end][series.mask[:series.train_end]]
    assert mean[0] == pytest.approx(observed.mean())
    assert std[0] == pytest.approx(observed.std())
    series.values[series.train_end:, 0] += 1e9
    _, mean2, std2 = series.standardize()
    np.testing.assert_array_equal(mean, mean2)
    np.testing.assert_array_equal(std, std2)
    assert np.isfinite(z).all()
