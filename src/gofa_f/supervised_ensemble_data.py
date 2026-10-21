"""Stage29 causal univariate features and capacity-balanced target training pools."""
from dataclasses import asdict
import time

import numpy as np

from gofa_f.generative_data import Query
from gofa_f.short_data import origins

CADENCE_SECONDS = dict(pems03=300, pems_bay=300, meteonet_nw2018=3600,
                       engrad=3600, sdwpf=600, kelmarsh2019=600)


def feature_names(length):
    names = [f'history_{i-length}' for i in range(length)]
    names += [f'observed_{i-length}' for i in range(length)]
    for k in sorted({3, 6, 12, length}):
        names += [f'{stat}_{k}' for stat in ('mean', 'std', 'slope', 'coverage')]
    return names+['last', 'last_age', 'minimum', 'maximum', 'horizon',
                  'day_sin', 'day_cos', 'week_sin', 'week_cos']


def features(data, query):
    """Never inspect target values, target masks, or actual future timestamps."""
    q = Query(**query)
    length, horizon, origin, variable = q.length, q.horizon, q.origin, q.variable
    if q.source not in CADENCE_SECONDS or (length, horizon) not in ((12, 12), (48, 24)) or origin < length:
        raise ValueError('Fixed Stage29 dimensions/source required')
    values = np.asarray(data.values[origin-length:origin, variable], np.float64)
    observed = np.asarray(data.mask[origin-length:origin, variable], bool) & np.isfinite(values)
    times = np.asarray(data.times[origin-length:origin], np.int64)
    cadence = CADENCE_SECONDS[q.source]*10**9
    if len(times) != length or not np.all(np.diff(times) == cadence):
        raise ValueError('Audited contiguous fixed-cadence history required')
    indices = np.flatnonzero(observed)
    if not len(indices):
        raise ValueError('Training/evaluation requires at least one historical observation')
    last = float(values[indices[-1]])
    raw = np.where(observed, values, np.nan).tolist()+observed.astype(float).tolist()
    for k in sorted({3, 6, 12, length}):
        ix = np.flatnonzero(observed[-k:])+length-k
        if len(ix):
            own = values[ix]
            slope = 0.
            if len(ix) >= 2:
                t = ix.astype(np.float64)-ix.mean()
                slope = float(t @ (own-own.mean())/(t @ t))
            raw.extend([float(own.mean()), float(own.std(ddof=0)), slope, len(ix)/k])
        else:
            raw.extend([np.nan, np.nan, 0., 0.])
    raw.extend([last, float(length-1-indices[-1]), float(values[indices].min()), float(values[indices].max())])
    x = np.repeat(np.asarray(raw, np.float64)[None], horizon, axis=0)
    # Forecast clocks are calculated from the last PAST timestamp, never looked up ahead.
    seconds = (times[-1]//10**9)+(np.arange(horizon, dtype=np.int64)+1)*CADENCE_SECONDS[q.source]
    day = (seconds % 86400)/86400.*2*np.pi
    week = (seconds % (7*86400))/(7*86400.)*2*np.pi
    x = np.column_stack((x, np.arange(horizon, dtype=np.float64)/(horizon-1),
                         np.sin(day), np.cos(day), np.sin(week), np.cos(week))).astype(np.float32)
    if x.shape != (horizon, len(feature_names(length))) or np.isinf(x).any():
        raise ValueError('Feature shape/nonfinite infinity error')
    return x, last


def disjoint_chain(points, horizon):
    result, next_origin = [], -1
    for p in points:
        p = int(p)
        if p >= next_origin:
            result.append(p)
            next_origin = p+horizon
    return np.asarray(result, np.int64)


def quotas(capacities, total, rng):
    ids = sorted(capacities)
    if not ids or sum(capacities.values()) < total or total < len(ids):
        raise ValueError('Insufficient capacity to retain all eligible variables')
    allocation = {v: min(total//len(ids), capacities[v]) for v in ids}
    order = rng.permutation(ids).tolist()
    remaining = total-sum(allocation.values())
    while remaining:
        advanced = False
        for v in order:
            if allocation[v] < capacities[v]:
                allocation[v] += 1
                remaining -= 1
                advanced = True
                if not remaining:
                    break
        if not advanced:
            raise ValueError('Capacity-balanced allocation exhausted')
    return allocation


def training_queries(data, source, length, horizon, total=4096, notify=lambda *_: None):
    chains = {}
    for v in range(data.values.shape[1]):
        pool = origins(data, v, length, horizon, 'train', require_history=True)
        chain = disjoint_chain(pool, horizon)
        if len(chain):
            chains[v] = chain
        notify(v+1, data.values.shape[1])
    rng = np.random.default_rng(np.random.SeedSequence([42, 29, list(CADENCE_SECONDS).index(source)]))
    allocation = quotas({v: len(p) for v, p in chains.items()}, total, rng)
    rows = []
    for v in sorted(chains):
        own_rng = np.random.default_rng(np.random.SeedSequence([42, 29, list(CADENCE_SECONDS).index(source), v]))
        for part in np.array_split(chains[v], allocation[v]):
            rows.append(asdict(Query(source, v, int(own_rng.choice(part)), length, horizon)))
    if len(rows) != total or len({Query(**q).key for q in rows}) != total:
        raise ValueError('Fixed training query budget/identity mismatch')
    end = int(.6*len(data.values))
    for v in chains:
        own = sorted(q['origin'] for q in rows if q['variable'] == v)
        if any(b-a < horizon for a, b in zip(own, own[1:])):
            raise ValueError('Overlapping same-variable training labels')
    if any(q['origin']+horizon > end for q in rows):
        raise ValueError('Training labels crossed the first60 percent')
    return rows, dict(eligible_variables=len(chains), all_variables=data.values.shape[1],
        variable_quotas={str(k): v for k, v in allocation.items()},
        minimum_disjoint_capacity=min(map(len, chains.values())), queries=total,
        label_end_exclusive=end, selection='seed42 ID/capacity/time only, earliest-disjoint chains',
        same_variable_labels_disjoint=True)


def training_arrays(data, queries, notify=lambda *_: None):
    x, y, variables = [], [], []
    end = int(.6*len(data.values))
    for index, query in enumerate(queries):
        q = Query(**query)
        if q.origin+q.horizon > end or data.segments[q.origin-q.length] != data.segments[q.origin+q.horizon-1]:
            raise ValueError('CPU target training boundary/segment violation')
        own, last = features(data, query)
        values = np.asarray(data.values[q.origin:q.origin+q.horizon, q.variable], np.float64)
        mask = np.asarray(data.mask[q.origin:q.origin+q.horizon, q.variable], bool) & np.isfinite(values)
        if not mask.any():
            raise ValueError('Sealed training query has no labels')
        x.append(own[mask])
        y.extend((values[mask]-last).tolist())
        variables.extend([q.variable]*int(mask.sum()))
        notify(index+1, len(queries))
    _, inverse, counts = np.unique(variables, return_inverse=True, return_counts=True)
    # Mean weight1 preserves the declared L2 scale; each variable still has equal total weight.
    weights = len(variables)/(len(counts)*counts[inverse].astype(np.float64))
    return np.concatenate(x), np.asarray(y, np.float64), weights, dict(
        rows=len(variables), variables=len(counts), weight_mean=float(weights.mean()),
        weight_normalization='mean1; equal total weight per variable',
        features=feature_names(queries[0]['length']))


class FitLog:
    """Observe sklearn1.5.2 iteration logging without changing the fitting algorithm."""
    def __init__(self, original, callback):
        self.original, self.callback, self.buffer = original, callback, ''

    def write(self, text):
        import re
        self.original.write(text)
        self.buffer += text
        for match in re.finditer(r'\[(\d+)/(\d+)\]', self.buffer):
            self.callback(int(match[1]), int(match[2]), time.monotonic())
        if '\n' in self.buffer:
            self.buffer = self.buffer.rsplit('\n', 1)[-1]
        return len(text)

    def flush(self):
        self.original.flush()
