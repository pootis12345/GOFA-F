"""Stage23 fixed source query pools, hierarchical metrics and best/early-stop state."""
from dataclasses import asdict
from functools import lru_cache

import numpy as np

from gofa_f.generative_data import Query
from gofa_f.short_data import origins, stratified
from gofa_f.source_retraining_data import VIEWS

DOMAINS = {'traffic': ('pems04_flow', 'pems04_speed'), 'weather': ('boston2017',),
           'wind': ('penmanshiel2017',)}


def epsilon(data):
    result = []
    for variable in range(data.values.shape[1]):
        x = data.values[:int(.6*len(data.values)), variable].astype(np.float64)
        x = x[data.mask[:len(x), variable]]
        if not len(x):
            raise ValueError('Every training variable requires observations')
        result.append(max(1e-12, 1e-6*max(float(np.subtract(*np.percentile(x, [75, 25]))), float(x.std()))))
    return result


def pools(series, confirmation_exclusions=None):
    train, validations, confirmations, diagnostics = {}, [], [], []
    traffic_variables = np.random.default_rng(np.random.SeedSequence([42, 23, 0])).permutation(307)[:32].tolist()
    for s, (name, (nodes, length, horizon)) in enumerate(VIEWS.items()):
        data = series[name]
        if data.values.shape[1] != nodes:
            raise ValueError('Fixed source node count changed: '+name)
        rng = np.random.default_rng(np.random.SeedSequence([42, 23, s, 1]))
        order = rng.permutation(nodes)
        legal = {int(v): origins(data, int(v), length, horizon, 'train') for v in order}
        rows = []
        for j, v in enumerate(order):
            count = 8192//nodes+int(j < 8192 % nodes)
            points = stratified(legal[int(v)], count, horizon, rng)
            rows.extend(asdict(Query(name, int(v), p, length, horizon)) for p in points)
        train[name] = [rows[int(i)] for i in rng.permutation(len(rows))]
        variables = traffic_variables if s < 2 else list(map(int, order))
        counts = [2]*32 if s < 2 else [8]*16 if s == 2 else [10]*2+[9]*12
        for split, target in [('validation', validations), ('test', confirmations)]:
            for variable, count in zip(variables, counts):
                points = origins(data, variable, length, horizon, split)
                if s < 2:
                    # Paired traffic views share both sensors and origins.
                    other = series['pems04_speed' if s == 0 else 'pems04_flow']
                    points = np.intersect1d(points, origins(other, variable, length, horizon, split))
                if split == 'test':
                    for start, end in (confirmation_exclusions or {}).get(name, {}).get(str(variable), []):
                        points = points[(points+horizon <= start) | (points >= end)]
                seed_index = 0 if s < 2 else s
                prng = np.random.default_rng(np.random.SeedSequence([42, 23, seed_index, variable,
                                                                       2 if split == 'validation' else 3]))
                target.extend(asdict(Query(name, variable, p, length, horizon))
                              for p in stratified(points, count, horizon, prng, disjoint=True))
        candidates = [q for q in validations if q['source'] == name]
        # Independent precommitted source-only graph controls, never selection.
        drng = np.random.default_rng(np.random.SeedSequence([42, 23, s, 4]))
        diagnostics.extend(candidates[int(i)] for i in drng.permutation(len(candidates))[:16])
    if len(validations) != 384 or len(confirmations) != 384 or len(diagnostics) != 64:
        raise ValueError('Required fixed source coverage not met')
    return dict(train=train, validation=validations, confirmation=confirmations, diagnostic=diagnostics,
                epsilons={n: epsilon(d) for n, d in series.items()}, traffic_variables=traffic_variables)


@lru_cache(maxsize=128)
def permutation(view, epoch):
    return np.random.default_rng(np.random.SeedSequence([42, 23, view, epoch, 5])).permutation(8192)


def scheduled(pool, presentation):
    """Continuous three-domain cycle; traffic alternates views, never batch-reset."""
    cycle, domain = divmod(presentation, 3)
    if domain == 0:
        v, offset = cycle % 2, cycle//2
    else:
        v, offset = domain+1, cycle
    epoch, index = divmod(offset, 8192)
    name = list(VIEWS)[v]
    return Query(**pool[name][int(permutation(v, epoch)[index])])


def score(errors, reference, queries, epsilons):
    if len(errors) != len(reference) or len(errors) != len(queries):
        raise ValueError('Complete paired validation required')
    grouped = {}
    for e, r, q in zip(errors, reference, queries):
        if not np.isfinite([e, r]).all() or min(e, r) < 0:
            raise ValueError('Finite nonnegative source MAE required')
        grouped.setdefault((q['source'], q['variable']), [[], []])
        grouped[q['source'], q['variable']][0].append(e)
        grouped[q['source'], q['variable']][1].append(r)
    variables, views = {}, {}
    for (name, variable), (a, b) in grouped.items():
        left, right = float(np.mean(a, dtype=np.float64)), float(np.mean(b, dtype=np.float64))
        threshold = epsilons[name][variable]
        ratio = max(left, threshold)/max(right, threshold)
        variables.setdefault(name, {})[str(variable)] = dict(mae=left, reference_mae=right,
            epsilon=threshold, ratio=ratio, queries=len(a), near_zero=right < threshold)
    if set(variables) != set(VIEWS):
        raise ValueError('All four views required')
    for name, rows in variables.items():
        views[name] = float(np.exp(np.mean([np.log(v['ratio']) for v in rows.values()], dtype=np.float64)))
    domains = {d: float(np.exp(np.mean([np.log(views[v]) for v in names], dtype=np.float64)))
               for d, names in DOMAINS.items()}
    return dict(S=float(np.exp(np.mean(np.log(list(domains.values())), dtype=np.float64))),
                W=max(domains.values()), domains=domains, views=views, variables=variables)


def initial_selection(parent_score, digest):
    return dict(best_update=0, best_sha256=digest, best_score=parent_score,
                early_reference_W=parent_score['W'], bad_validations=0, last_validated_update=0, history=[])


def commit_selection(state, update, digest, native_score, parent_score):
    if update <= state['last_validated_update']:
        old = [r for r in state['history'] if r['update'] == update]
        if len(old) != 1 or old[0]['sha256'] != digest or old[0]['native_score'] != native_score \
                or old[0]['parent_score'] != parent_score:
            raise ValueError('Repeated validation differs; refusing early-stop double count')
        return False
    eligible = all(v <= 1.02 for v in parent_score['domains'].values())
    best = state['best_score']
    selected = eligible and (native_score['W'], native_score['S']) < (best['W'], best['S'])
    if selected:
        state.update(best_update=update, best_sha256=digest, best_score=native_score)
    significant = eligible and native_score['W'] <= state['early_reference_W']*(1-.001)
    if significant:
        state.update(early_reference_W=native_score['W'], bad_validations=0)
    else:
        state['bad_validations'] += 1
    state['last_validated_update'] = update
    state['history'].append(dict(update=update, sha256=digest, eligible=eligible, selected=selected,
        significant=significant, native_score=native_score, parent_score=parent_score))
    return selected


def confirmation_pass(native_score, parent_score):
    return native_score['S'] <= .99 and all(v <= 1.02 for v in native_score['domains'].values()) \
        and all(v <= 1.02 for v in parent_score['domains'].values())
