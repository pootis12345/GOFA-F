"""Causal, label-free history centering and fixed Stage48 self prediction."""
import numpy as np


def centered_history(x):
    x = np.array(x, dtype=np.float64, copy=True)
    if x.ndim != 4 or x.shape[1:] != (9, 12, 2) or not len(x):
        raise ValueError('Expected nonempty [Q,9,12,2] historical inputs')
    mask = x[..., 1]
    if not np.isfinite(mask).all() or not np.isin(mask, [0., 1.]).all():
        raise ValueError('Historical observation mask must be binary')
    observed = mask.astype(bool)
    if not observed[:, 0].any(1).all() or not np.isfinite(x[..., 0][observed]).all():
        raise ValueError('Finite observed values and nonempty target history required')
    values = np.where(observed, x[..., 0], 0.)
    last_index = np.where(observed, np.arange(12), -1).max(2).clip(min=0)
    last = np.take_along_axis(values, last_index[..., None], axis=2)[..., 0]
    x[..., 0] = np.where(observed, values - last[..., None], 0.)
    if not np.isfinite(x).all():
        raise ValueError('Historical centering overflow')
    return x


def self_features(x):
    centered = centered_history(x)
    return np.concatenate([centered[:, 0, :, 0], centered[:, 0, :, 1]], axis=1)


def baseline_delta(x, model):
    features = self_features(x)
    arrays = {key: np.asarray(model[key], dtype=np.float64)
              for key in ('mean', 'std', 'coefficients', 'intercept')}
    expected = {'mean': (24,), 'std': (24,), 'coefficients': (24, 12), 'intercept': (12,)}
    if any(a.shape != expected[k] or not np.isfinite(a).all() for k, a in arrays.items()):
        raise ValueError('Invalid fixed self-model shapes or values')
    if not (arrays['std'] > 0.).all():
        raise ValueError('Fixed self-model standard deviations must be positive')
    result = ((features - arrays['mean']) / arrays['std']) @ arrays['coefficients']
    result = result + arrays['intercept']
    if result.shape != (len(features), 12) or not np.isfinite(result).all():
        raise ValueError('Fixed self-model arithmetic produced nonfinite output')
    return result


def self_test():
    x = np.zeros((2, 9, 12, 2), dtype=np.float64)
    x[:, 0, :, 1] = 1.; x[:, 0, :, 0] = np.arange(12) / 11 - .5
    x[:, 1, -2:, 1] = 1.; x[:, 1, -2:, 0] = [-.5, .5]
    x[:, 2:, :, 0] = np.nan
    before = x.copy(); result = centered_history(x)
    assert not np.shares_memory(result, x) and np.array_equal(x, before, equal_nan=True)
    assert np.array_equal(result[..., 1], x[..., 1]) and np.isfinite(result).all()
    assert np.array_equal(result[:, 1, :, 0], np.tile([0.] * 10 + [-1., 0.], (2, 1)))
    replacement = x.copy(); replacement[:, 2:, :, 0] = -1e6
    assert np.array_equal(result, centered_history(replacement))
    model = dict(mean=np.arange(24) / 24., std=np.arange(24) + 1.,
                 coefficients=np.arange(288).reshape(24, 12) / 288., intercept=np.arange(12))
    f = self_features(x)
    assert f.dtype == np.float64 and f.shape == (2, 24)
    expected = (f - model['mean']) / model['std'] @ model['coefficients'] + model['intercept']
    np.testing.assert_array_equal(baseline_delta(x, model), expected)
    model['raw_target'] = np.full((2, 12), np.nan)
    model['scale'] = np.nan
    np.testing.assert_array_equal(baseline_delta(x, model), expected)
    for invalid in (np.zeros_like(x), np.where(np.indices(x.shape)[-1] == 1, 2., x)):
        try:
            centered_history(invalid)
        except ValueError:
            continue
        raise AssertionError('Invalid history was accepted')
    return dict(passed=True, missing_payload_ignored=True, copy_mask_preserved=True,
                empty_target_rejected=True, fixed_formula_exact=True, labels_scale_ignored=True)
