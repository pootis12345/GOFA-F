"""Fixed source-only, float64 ridge information diagnostic; no search or file I/O."""
import numpy as np


def features(a):
    """Causal inputs only; ignore all cached labels, targets, scales and edges."""
    x = np.asarray(a['x'], dtype=np.float64)
    if x.ndim != 4 or x.shape[1:] != (9, 12, 2) or not len(x):
        raise ValueError('Expected nonempty [Q,9,12,2] historical inputs')
    mask_values = x[..., 1]
    if not np.isfinite(mask_values).all() or not np.isin(mask_values, [0., 1.]).all():
        raise ValueError('Historical observation mask must be binary')
    observed = mask_values.astype(bool)
    if not np.isfinite(x[..., 0][observed]).all() or not observed[:, 0].any(1).all():
        raise ValueError('Finite observed values and nonempty target history required')
    # Invalid payloads are ignored before arithmetic, including padded/missing NaN.
    values = np.where(observed, x[..., 0], 0.)
    present = observed.any(2)
    last_index = np.where(observed, np.arange(12), -1).max(2).clip(min=0)
    last = np.take_along_axis(values, last_index[..., None], axis=2)[..., 0]
    centered = np.where(observed, values - last[..., None], 0.)
    target = np.concatenate([centered[:, 0], observed[:, 0]], axis=1)
    neighbor_mask = observed[:, 1:] & present[:, 1:, None]
    count = neighbor_mask.sum(1)
    neighbor_mean = np.divide((centered[:, 1:] * neighbor_mask).sum(1), count,
                              out=np.zeros_like(count, dtype=np.float64), where=count > 0)
    denominator = present[:, 1:].sum(1, keepdims=True)
    fraction = np.divide(count, denominator, out=np.zeros_like(count, dtype=np.float64),
                         where=denominator > 0)
    augmented = np.concatenate([target, neighbor_mean, fraction], axis=1)
    if not np.isfinite(augmented).all():
        raise ValueError('Historical feature arithmetic overflow')
    return target.astype(np.float64), augmented


def fit_predict(trainX, validX, train_residual, observed, query_weights, ridge=1.):
    """Fit average weighted SSE + 1*||beta||² separately for each horizon slot."""
    x, v, y = (np.asarray(z, dtype=np.float64) for z in (trainX, validX, train_residual))
    raw_mask, w = np.asarray(observed), np.asarray(query_weights, dtype=np.float64)
    if float(ridge) != 1.:
        raise ValueError('Only the preregistered fixed ridge=1 is permitted')
    if x.ndim != 2 or not x.shape[0] or not x.shape[1] or v.ndim != 2 or v.shape[1] != x.shape[1]:
        raise ValueError('Training/validation feature matrices disagree')
    if y.shape != (len(x), 12) or raw_mask.shape != y.shape or w.shape != (len(x),):
        raise ValueError('Expected [Q,12] labels/mask and [Q] query weights')
    if not np.isin(raw_mask, [0, 1]).all():
        raise ValueError('Target mask must be binary')
    mask = raw_mask.astype(bool)
    if not np.isfinite(x).all() or not np.isfinite(v).all() or not np.isfinite(y[mask]).all():
        raise ValueError('Nonfinite effective model inputs/labels')
    if not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0 or not np.isfinite(w.sum()):
        raise ValueError('Finite nonnegative query weights with positive sum required')
    w = w / w.sum()
    mean = (w[:, None] * x).sum(0)
    std = np.sqrt((w[:, None] * (x - mean)**2).sum(0))
    std = np.where(std < 1e-12, 1., std)
    design = np.column_stack([np.ones(len(x)), (x - mean) / std])
    evaluation = np.column_stack([np.ones(len(v)), (v - mean) / std])
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.
    coefficients = np.empty((design.shape[1], 12), dtype=np.float64)
    for h in range(12):
        use = mask[:, h] & (w > 0)
        if not use.any():
            raise ValueError('Each predicted horizon requires an effective training label')
        weight = w[use] / w[use].sum()
        d = design[use]
        coefficients[:, h] = np.linalg.solve(d.T @ (weight[:, None] * d) + penalty,
                                              d.T @ (weight * y[use, h]))
    prediction = evaluation @ coefficients
    if not np.isfinite(prediction).all() or not np.isfinite(coefficients).all():
        raise ValueError('Ridge arithmetic produced nonfinite results')
    metadata = dict(mean=mean.tolist(), std=std.tolist(), intercept=coefficients[0].tolist(),
                    coefficients=coefficients[1:].tolist(), ridge=1., dtype='float64',
                    objective='per_horizon_average_query_weighted_sse_plus_ridge_l2',
                    standardization='training_only_query_weighted_population_mean_std')
    return prediction, metadata


def self_test():
    x = np.zeros((3, 9, 12, 2), dtype=np.float64)
    x[:, 0, :, 1] = 1
    x[:, 0, :, 0] = np.arange(12) / 11 - .5
    x[:, 1, -2:, 1] = 1
    x[:, 1, -2:, 0] = [-.5, .5]
    x[:, 2:, :, 0] = np.nan
    a = {'x': x, 'raw_target': np.ones((3, 12))}
    s, g = features(a)
    a['raw_target'][:] = np.nan
    assert np.array_equal(s, features(a)[0]) and np.array_equal(g, features(a)[1])
    assert s.shape == (3, 24) and g.shape == (3, 48) and np.isfinite(g).all()
    assert np.array_equal(g[:, 24:36], np.tile([0.]*10 + [-1., 0.], (3, 1)))
    assert np.array_equal(g[:, 36:48], np.tile([0.]*10 + [1., 1.], (3, 1)))
    constant = np.zeros_like(x); constant[:, 0, :, 1] = 1
    assert np.array_equal(features({'x': constant})[1][:, 24:], np.zeros((3, 24)))
    # With unit feature variance and ridge=1, analytic slope is true slope/2.
    tx = np.array([[-1.], [1.]])
    truth = np.tile((3. + 4. * tx), (1, 12))
    pred, info = fit_predict(tx, tx, truth, np.ones_like(truth, bool), np.ones(2))
    np.testing.assert_allclose(pred, np.tile((3. + 2. * tx), (1, 12)), atol=1e-12)
    masked = truth.copy(); mask = np.ones_like(truth, bool)
    masked[0, ::2] = np.nan; mask[0, ::2] = False
    pred2, info2 = fit_predict(tx, tx, masked, mask, np.array([1., 3.]))
    assert np.isfinite(pred2).all()
    _, info3 = fit_predict(tx, tx * 100., masked, mask, np.array([1., 3.]))
    assert info2 == info3
    return dict(passed=True, causal_feature_labels_ignored=True,
                missing_constant_features_finite=True, ridge_analytic_solution=True,
                weighted_missing_labels_finite=True, validation_cannot_affect_fit=True)
