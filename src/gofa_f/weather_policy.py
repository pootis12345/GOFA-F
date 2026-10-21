"""Stage25 fixed replay pools, deterministic mixed-prefix schedule and MAE gates."""
from functools import lru_cache

import numpy as np

from gofa_f.generative_data import Query
from gofa_f.short_data import stratified
from gofa_f.source_retraining_data import VIEWS

COUNTS = dict(pems04_flow=128, pems04_speed=128, boston2017=512, penmanshiel2017=256)
PER_BATCH = dict(pems04_flow=8, pems04_speed=8, boston2017=32, penmanshiel2017=16)
PROTECTED = ('pems04_flow', 'pems04_speed', 'penmanshiel2017')


def subset(original):
    """Only IDs/times in the prior training pool; no forecast-error access."""
    traffic = np.random.default_rng(np.random.SeedSequence([42, 25, 0])).permutation(307)[:128].tolist()
    wind = np.random.default_rng(np.random.SeedSequence([42, 25, 1])).permutation(14).tolist()
    result = {}
    for s, name in enumerate(VIEWS):
        variables = traffic if s < 2 else list(range(16)) if s == 2 else wind
        rows = []
        for j, variable in enumerate(variables):
            count = 1 if s < 2 else 32 if s == 2 else 18+int(j < 4)
            available = {q['origin']: q for q in original[name] if q['variable'] == int(variable)}
            points = np.array(sorted(available), dtype=np.int64)
            rng = np.random.default_rng(np.random.SeedSequence([42, 25, s, int(variable), 2]))
            origins = stratified(points, count, VIEWS[name][2], rng, disjoint=s == 2)
            rows.extend(dict(available[int(t)]) for t in origins)
        if len(rows) != COUNTS[name]:
            raise ValueError('Required fixed training coverage unavailable: '+name)
        result[name] = rows
    return result


@lru_cache(maxsize=128)
def order(name, epoch):
    # Keep weather positions stable and flip their prefix style each epoch.
    e = 0 if name == 'boston2017' else epoch
    return np.random.default_rng(np.random.SeedSequence([42, 25, list(VIEWS).index(name), e, 3])).permutation(COUNTS[name])


def scheduled(pool, presentation):
    update, slot = divmod(presentation, 64)
    if slot < 32:
        name, local = 'boston2017', slot
    elif slot < 40:
        name, local = 'pems04_flow', slot-32
    elif slot < 48:
        name, local = 'pems04_speed', slot-40
    else:
        name, local = 'penmanshiel2017', slot-48
    cursor = update*PER_BATCH[name]+local
    epoch, offset = divmod(cursor, COUNTS[name])
    query = pool[name][int(order(name, epoch)[offset])]
    kind = 'weather_true' if name == 'boston2017' and (local+epoch) % 2 == 0 else \
        'weather_generated' if name == 'boston2017' else 'keep'
    return dict(query=query, kind=kind, weight=2/3 if name == 'boston2017' else 4/3)


def macro(rows, queries):
    """Pool observed points within variable, then equally average variables."""
    if len(rows) != len(queries):
        raise ValueError('Complete paired query coverage required')
    totals = {}
    for row, query in zip(rows, queries):
        if row['query'] != query:
            raise ValueError('MAE query identity differs')
        y, p, seen = np.asarray(row['target'], np.float64), np.asarray(row['prediction'], np.float64), \
            np.asarray(row['observed'], bool)
        if y.shape != p.shape or seen.shape != y.shape or not np.isfinite(y[seen]).all() \
                or not np.isfinite(p).all() or not seen.any():
            raise ValueError('Finite observed source targets and predictions required')
        key = query['source'], query['variable']
        v = totals.setdefault(key, [0., 0])
        v[0] += float(np.abs(y[seen]-p[seen]).sum(dtype=np.float64))
        v[1] += int(seen.sum())
    variables = {}
    for (name, variable), (error, count) in totals.items():
        variables.setdefault(name, {})[str(variable)] = dict(mae=error/count, observations=count)
    if set(variables) != set(VIEWS):
        raise ValueError('All four source views required')
    return dict(views={n: float(np.mean([v['mae'] for v in own.values()], dtype=np.float64))
                       for n, own in variables.items()}, variables=variables,
                aggregation='observed_points_within_variable_then_equal_variables')


def decision(update, current, native, parent):
    c, n, p = current['views'], native['views'], parent['views']
    if update == 25:
        return 'stop' if any(c[v] > 1.02*p[v] for v in VIEWS) else 'continue'
    if update not in (50, 100):
        raise ValueError('Only precommitted update gates are allowed')
    if any(c[v] > 1.005*p[v] for v in PROTECTED):
        return 'stop'
    if c['boston2017'] <= .99*n['boston2017']:
        return 'lock'
    return 'continue' if update == 50 and c['boston2017'] <= .99*p['boston2017'] else 'stop'


def confirmation_pass(current, native, parent):
    return current['views']['boston2017'] <= .99*native['views']['boston2017'] \
        and all(current['views'][v] <= 1.005*parent['views'][v] for v in VIEWS)


def query_object(event):
    return Query(**event['query'])
