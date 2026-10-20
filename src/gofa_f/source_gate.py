"""Stage22 history-only amplitude policy and source-only CPU fitting.

This module neither imports torch nor accepts target dataset labels for fitting.
The exported tree is evaluated without pickle or a sklearn runtime dependency.
"""
import numpy as np


AMPLITUDES = (0., .25, .5, 1.)
SOURCES = ('solar', 'nn5', 'pems04', 'aq_shunyi')
TEMPLATES = ((10, 1), (12, 12), (14, 7), (48, 24))
FEATURES = ('log_history', 'log_horizon', 'missing_fraction', 'std_over_range',
    'quarter_trend_over_range', 'adjacent_abs_difference_over_range',
    'neighbors_over_eight', 'weighted_absolute_correlation', 'weighted_signed_correlation',
    'weighted_overlap', 'weighted_trend_agreement')
TREE_PARAMETERS = dict(max_depth=2, min_samples_leaf=16, criterion='squared_error', random_state=42)
MARGIN = .01


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def features(history, mask, graph, length, horizon):
    """Only origin history and its existing sampled graph; no series/future access."""
    x, seen = np.asarray(history, np.float64), np.asarray(mask, bool)
    if x.shape != (length,) or seen.shape != x.shape or (length, horizon) not in TEMPLATES:
        raise ValueError('Invalid origin-only history/template')
    seen = seen & np.isfinite(x)
    g = graph.graph if hasattr(graph, 'graph') else graph
    gx, gm = array(g.history).astype(np.float64), array(g.observed).astype(bool)
    target = int(g.target)
    if gx.shape != gm.shape or gx.shape[0] != length or not 0 <= target < gx.shape[1]:
        raise ValueError('Invalid sampled graph history')
    if not np.array_equal(gm[:, target], seen) or not np.array_equal(gx[seen, target], x[seen]):
        raise ValueError('Graph/target origin history differs')
    valid = x[seen]
    extent = float(np.ptp(valid)) if len(valid) else 0.
    extent = 1. if extent < 10*np.finfo(np.float64).eps else extent
    quarter = length//4
    def trend(v, m):
        a, b = v[:quarter][m[:quarter]], v[-quarter:][m[-quarter:]]
        return float(b.mean()-a.mean()) if len(a) and len(b) else 0.
    delta = trend(x, seen)
    adjacent = seen[1:] & seen[:-1]
    difference = np.abs(x[1:]-x[:-1])[adjacent]
    weights = array(g.base)[target].astype(np.float64).copy()
    weights[target] = 0.
    weights *= gm.any(0)
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError('Nonfinite/negative graph support')
    neighbors = int((weights > 0).sum())
    weights = weights/weights.sum() if weights.sum() else np.zeros_like(weights)
    edge = array(g.edges)[target].astype(np.float64)
    agreement = np.array([float(np.sign(delta) == np.sign(trend(gx[:, j], gm[:, j])))
                          for j in range(gx.shape[1])])
    values = [np.log(length), np.log(horizon), 1.-seen.mean(),
        float(valid.std()/extent) if len(valid) else 0., delta/extent,
        float(difference.mean()/extent) if len(difference) else 0., neighbors/8.,
        float(weights @ np.abs(edge[:, 0])), float(weights @ edge[:, 0]),
        float(weights @ edge[:, 1]), float(weights @ agreement)]
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite gate features')
    reason = ('empty_history' if not len(valid) else 'few_observations' if len(valid) < 8
              else 'no_neighbors' if not neighbors else None)
    return dict(values=values, eligible=reason is None, fallback_reason=reason)


def fit_tree(records):
    from sklearn.tree import DecisionTreeRegressor
    if len(records) < 16 or any(len(r['features']['values']) != len(FEATURES) for r in records):
        raise ValueError('Insufficient/malformed source gate fitting data')
    x = np.asarray([r['features']['values'] for r in records], np.float32)
    y = np.asarray([r['risks'] for r in records], np.float64)
    if y.shape != (len(records), 3) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid free-forecast risk labels')
    tree = DecisionTreeRegressor(**TREE_PARAMETERS).fit(x, y).tree_
    return dict(feature_names=list(FEATURES), amplitudes=list(AMPLITUDES), margin=MARGIN,
        learner=TREE_PARAMETERS, features=tree.feature.tolist(), thresholds=tree.threshold.tolist(),
        left=tree.children_left.tolist(), right=tree.children_right.tolist(),
        risks=tree.value[:, :, 0].tolist(), fitted_queries=len(records))


def select(gate, observation):
    if gate['feature_names'] != list(FEATURES) or gate['amplitudes'] != list(AMPLITUDES) \
            or gate['margin'] != MARGIN or gate['learner'] != TREE_PARAMETERS:
        raise ValueError('Gate schema/policy changed')
    if not observation['eligible']:
        return dict(alpha=0., reason=observation['fallback_reason'], predicted_risks=None)
    x = np.asarray(observation['values'], np.float32)
    if x.shape != (len(FEATURES),) or not np.isfinite(x).all():
        raise ValueError('Invalid gate features')
    node = 0
    while gate['left'][node] != -1:
        feature = gate['features'][node]
        node = gate['left'][node] if x[feature] <= gate['thresholds'][node] else gate['right'][node]
    risks = np.asarray(gate['risks'][node], np.float64)
    if risks.shape != (3,) or not np.isfinite(risks).all():
        raise ValueError('Invalid exported gate risks')
    best = int(np.argmin(risks))
    alpha = AMPLITUDES[best+1] if risks[best] <= -MARGIN else 0.
    return dict(alpha=alpha, reason='predicted_gain' if alpha else 'insufficient_predicted_gain',
                predicted_risks=risks.tolist())


def select_amplitude(gate, history, mask, graph, length, horizon):
    observation = features(history, mask, graph, length, horizon)
    return dict(**select(gate, observation), features=observation)


def source_score(records, epsilons):
    """Small subset only: four templates/variable, source geometric averages."""
    grouped, ratios, means = {}, {}, {}
    for row in records:
        q = row['query']
        key = q['source'], q['variable']
        template = q['length'], q['horizon']
        own = grouped.setdefault(key, {})
        if template in own or template not in TEMPLATES:
            raise ValueError('Duplicate/unknown source variable template')
        own[template] = (row['gated_mae'], row['native_mae'])
    for (source, variable), own in grouped.items():
        if set(own) != set(TEMPLATES) or source not in SOURCES:
            raise ValueError('Incomplete four-template/source coverage')
        a, b = np.asarray(list(own.values()), np.float64).mean(0)
        if not np.isfinite([a, b]).all() or min(a, b) < 0:
            raise ValueError('Invalid source MAE')
        eps = epsilons[source][variable]
        ratios.setdefault(source, {})[str(variable)] = float(max(a, eps)/max(b, eps))
        means.setdefault(source, {})[str(variable)] = dict(gate=float(a), native=float(b), epsilon=eps)
    if set(ratios) != set(SOURCES) or any(len(v) != 8 for v in ratios.values()):
        raise ValueError('Expected exactly eight variables in each of four sources')
    scores = {s: float(np.exp(np.log(list(v.values())).mean())) for s, v in ratios.items()}
    score = float(np.exp(np.log(list(scores.values())).mean()))
    return dict(score=score, source_scores=scores, variable_ratios=ratios, variable_errors=means,
        passed=bool(score <= .99 and max(scores.values()) <= 1.02),
        criterion=dict(maximum_score=.99, maximum_source_ratio=1.02),
        subset_only=True, replaces_full_source_validation=False)


def fit_and_cross_validate(records, epsilons):
    if len(records) != 128:
        raise ValueError('Fixed source pilot requires 128 fitting queries')
    rows, folds = [], {}
    for source in SOURCES:
        tree = fit_tree([r for r in records if r['query']['source'] != source])
        folds[source] = tree
        for row in records:
            if row['query']['source'] != source:
                continue
            choice = select(tree, row['features'])
            rows.append(dict(query=row['query'], native_mae=row['errors'][0],
                gated_mae=row['errors'][AMPLITUDES.index(choice['alpha'])], choice=choice))
    return dict(complete=True, gate_only_leave_source_out=True,
        base_adapter_trained_on_all_four_sources=True, folds=folds, records=rows,
        validation=source_score(rows, epsilons)), fit_tree(records)
