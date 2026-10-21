"""Bounded target-supervised CPU candidates; no future values in features."""
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

from gofa_f.generative_data import Query
from gofa_f.supervised_ensemble_data import features

RECIPES = ('raw150', 'norm150', 'season150', 'normseason150', 'norm300',
           'normseason300', 'normextra64', 'normseasonextra64')


def design(data, query, recipe):
    if recipe not in RECIPES:
        raise ValueError('Unregistered CPU candidate')
    x, last = features(data, query)
    q = Query(**query)
    values = np.asarray(data.values[q.origin-q.length:q.origin, q.variable], np.float64)
    mask = np.asarray(data.mask[q.origin-q.length:q.origin, q.variable], bool) & np.isfinite(values)
    scale = float(values[mask].max()-values[mask].min()) or 1.
    base = np.full(q.horizon, last, np.float64)
    if 'season' in recipe and q.source in ('meteonet_nw2018', 'engrad'):
        indices = q.length-24+np.arange(q.horizon)
        base = np.where(mask[indices], values[indices], last)
    normal = recipe.startswith('norm')
    if normal:
        x = x.astype(np.float64)
        x[:, :q.length] = (x[:, :q.length]-last)/scale
        offset = 2*q.length
        for _ in sorted({3, 6, 12, q.length}):
            x[:, offset] = (x[:, offset]-last)/scale
            x[:, offset+1:offset+3] /= scale
            offset += 4
        x[:, offset] = 0.
        x[:, offset+2:offset+4] = (x[:, offset+2:offset+4]-last)/scale
    else:
        scale = 1.
    x = np.column_stack((x, (base-last)/scale)).astype(np.float32)
    if np.isinf(x).any() or not np.isfinite(base).all() or scale <= 0:
        raise ValueError('Invalid causal normalized CPU design')
    return x, base, scale


def arrays(data, queries, recipe):
    xs, ys, variables = [], [], []
    for query in queries:
        q = Query(**query)
        if q.origin+q.horizon > int(.6*len(data.values)) \
                or data.segments[q.origin-q.length] != data.segments[q.origin+q.horizon-1]:
            raise ValueError('Target training label boundary violation')
        x, base, scale = design(data, query, recipe)
        y = np.asarray(data.values[q.origin:q.origin+q.horizon, q.variable], np.float64)
        mask = np.asarray(data.mask[q.origin:q.origin+q.horizon, q.variable], bool) & np.isfinite(y)
        if not mask.any():
            raise ValueError('No effective training labels')
        xs.append(x[mask])
        ys.extend(((y-base)/scale)[mask])
        variables.extend([q.variable]*int(mask.sum()))
    _, inverse, counts = np.unique(variables, return_inverse=True, return_counts=True)
    weights = len(variables)/(len(counts)*counts[inverse].astype(np.float64))
    return np.concatenate(xs), np.asarray(ys), weights


def estimator(recipe):
    if 'extra' in recipe:
        return ExtraTreesRegressor(n_estimators=64, max_depth=10, min_samples_leaf=32,
            max_features=1., criterion='squared_error', random_state=42, n_jobs=2)
    return HistGradientBoostingRegressor(loss='absolute_error', max_iter=300 if '300' in recipe else 150,
        learning_rate=.05, max_leaf_nodes=31 if '300' in recipe else 15, max_depth=5 if '300' in recipe else 4,
        min_samples_leaf=40, l2_regularization=1., max_bins=63, early_stopping=False,
        random_state=42, categorical_features=None)


def predict(model, data, queries, recipe):
    rows = []
    for q in queries:
        if recipe == 'raw150':
            x, last = features(data, q)
            own = last+model.predict(x)
        else:
            x, base, scale = design(data, q, recipe)
            # ExtraTrees1.5 supports NaN only on some criteria; explicit causal imputation.
            if 'extra' in recipe:
                x = np.nan_to_num(x, nan=0.)
            own = base+scale*model.predict(x)
        if not np.isfinite(own).all():
            raise ValueError('Nonfinite CPU candidate forecast')
        rows.append(own)
    return rows


def save_model(model, path):
    path = Path(path)
    temporary = path.with_suffix('.part')
    joblib.dump(model, temporary)
    temporary.replace(path)
