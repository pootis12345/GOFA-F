"""Stage33 fixed causal SCADA CPU experts; target supervision is explicit."""
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from gofa_f.autonomous_iteration_models import design
from gofa_f.generative_data import Query, sha
from gofa_f.short_data import origins

RECIPES = ('residual_local', 'residual_neighbor', 'level_local', 'level_neighbor')
CHANNELS = ('Wspd', 'Etmp', 'Itmp', 'Pab1', 'Prtv', 'Wdir_sin', 'Wdir_cos',
            'Ndir_sin', 'Ndir_cos')


def load_scada(directory, expected_sha):
    path = Path(directory)/'scada.npz'
    if sha(path) != expected_sha:
        raise ValueError('SCADA cache changed')
    return _read_scada(str(path), expected_sha)


@lru_cache(maxsize=1)
def _read_scada(path, expected_sha):
    with np.load(path, allow_pickle=False) as packed:
        result = packed['values']
    result.setflags(write=False)
    return result


def summary(values):
    """Fixed summaries, no inferred future state or target missingness."""
    result = []
    for column in np.asarray(values, np.float64).T:
        mask = np.isfinite(column)
        ix = np.flatnonzero(mask)
        last = float(column[ix[-1]]) if len(ix) else np.nan
        result.extend([last, len(ix)/len(column)])
        for k in (3, 12, 48):
            indices = np.flatnonzero(mask[-k:])+len(column)-k
            if len(indices):
                y = column[indices]
                slope = 0.
                if len(indices) > 1:
                    t = indices.astype(float)-indices.mean()
                    slope = float(t@(y-y.mean())/(t@t))
                result.extend([float(y.mean()), float(y.std(ddof=0)), slope])
            else:
                result.extend([np.nan, np.nan, 0.])
    return np.asarray(result, np.float64)


def features(data, scada, query, recipe):
    if recipe not in RECIPES or query['source'] != 'sdwpf':
        raise ValueError('Fixed Stage33 SDWPF recipe required')
    q = Query(**query)
    if (q.length, q.horizon) != (48, 24) or scada.shape[:2] != data.values.shape \
            or scada.shape[2] != len(CHANNELS):
        raise ValueError('Aligned historical SCADA shape required')
    raw, base, _ = design(data, query, 'season150')
    normal, _, _ = design(data, query, 'norm150')
    own = summary(scada[q.origin-48:q.origin, q.variable])
    # Retain absolute level/range alongside normalized power history: no scale invariance assumption.
    parts = [raw, normal, np.repeat(own[None], 24, axis=0)]
    if recipe.endswith('neighbor'):
        if data.external is None:
            raise ValueError('Static audited topology required')
        ids = sorted((int(j) for j in np.flatnonzero(data.external[q.variable] > 0)
                      if j != q.variable), key=lambda j: str(data.ids[j]))[:8]
        signals = [summary(scada[q.origin-48:q.origin, j]) for j in ids
                   if np.isfinite(scada[q.origin-48:q.origin, j]).any()]
        if signals:
            a = np.asarray(signals)
            counts = np.isfinite(a).sum(0)
            mean = np.divide(np.nansum(a, axis=0), counts, out=np.zeros(a.shape[1]), where=counts > 0)
            var = np.divide(np.nansum((a-mean)**2, axis=0), counts,
                            out=np.zeros(a.shape[1]), where=counts > 0)
            aggregate = np.r_[mean, np.sqrt(var), len(signals)/8]
        else:
            aggregate = np.zeros(2*len(CHANNELS)*11+1)
        parts.append(np.repeat(aggregate[None], 24, axis=0))
    x = np.concatenate(parts, axis=1).astype(np.float32)
    if np.isinf(x).any():
        raise ValueError('Infinite causal SCADA feature')
    if recipe.startswith('level'):
        base = np.zeros(24, np.float64)
    return x, base


def arrays(data, scada, queries, recipe):
    xs, ys, variables = [], [], []
    for query in queries:
        q = Query(**query)
        if q.origin+24 > int(.6*len(data.values)) \
                or data.segments[q.origin-48] != data.segments[q.origin+23]:
            raise ValueError('First60 percent contiguous training labels required')
        x, base = features(data, scada, query, recipe)
        y = np.asarray(data.values[q.origin:q.origin+24, q.variable], np.float64)
        mask = np.asarray(data.mask[q.origin:q.origin+24, q.variable], bool)&np.isfinite(y)
        if not mask.any():
            raise ValueError('No valid training slots')
        xs.append(x[mask])
        ys.extend((y-base)[mask])
        variables.extend([q.variable]*int(mask.sum()))
    _, inverse, counts = np.unique(variables, return_inverse=True, return_counts=True)
    w = 1./counts[inverse]
    w /= w.mean()
    return np.concatenate(xs), np.asarray(ys), w


def estimator():
    return HistGradientBoostingRegressor(loss='absolute_error', max_iter=300, learning_rate=.05,
        max_leaf_nodes=31, max_depth=5, min_samples_leaf=64, l2_regularization=1., max_bins=63,
        early_stopping=False, categorical_features=None, random_state=42)


def predict(model, data, scada, queries, recipe):
    result = []
    for q in queries:
        x, base = features(data, scada, q, recipe)
        p = base+model.predict(x)
        if not np.isfinite(p).all():
            raise ValueError('Nonfinite SCADA forecast')
        result.append(p)
    return result


def fresh_confirmation(data, old_manifest, development):
    """Same target IDs, fourth disjoint set, sealed before fitting any new expert."""
    count = len(data.values)
    begin = int(.6*count)+(int(.8*count)-int(.6*count))*4//5
    end = int(.8*count)
    known = [r['query'] for r in development]
    result = []
    for index, query in enumerate(old_manifest['queries']['sdwpf'][64:]):
        q = Query(**query)
        pool = origins(data, q.variable, 48, 24, 'validation')
        pool = pool[(pool >= begin)&(pool+24 <= end)]
        for other in known+result:
            if other['variable'] == q.variable:
                pool = pool[np.abs(pool-other['origin']) >= 24]
        if not len(pool):
            raise ValueError('No new disjoint confirmation for fixed variable')
        rng = np.random.default_rng(np.random.SeedSequence([42, 33, index, q.variable]))
        result.append(asdict(Query('sdwpf', q.variable, int(rng.choice(pool)), 48, 24)))
    if len(result) != 64:
        raise ValueError('Exactly64 new confirmation queries required')
    return result


def folds(data, records):
    begin, end = int(.6*len(data.values)), int(.8*len(data.values))
    boundaries = [begin+(end-begin)*i//5 for i in range(6)]
    blocks = []
    for r in records:
        q = r['query']
        eligible = [b for b in range(5) if boundaries[b] <= q['origin']
                    and q['origin']+24 <= boundaries[b+1]]
        if len(eligible) != 1:
            raise ValueError('Development label crossed a time block')
        blocks.append(eligible[0])
    if [blocks.count(i) for i in range(5)] != [32, 32, 32, 32, 128]:
        raise ValueError('Expected exposed256 development queries in five time blocks')
    return [(np.flatnonzero(np.asarray(blocks) < b), np.flatnonzero(np.asarray(blocks) == b))
            for b in (2, 3, 4)]
