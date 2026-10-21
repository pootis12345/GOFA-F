"""Stage29 fixed convex three-expert L1 fitting and honest five-system reports."""
import copy

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack, vstack

from gofa_f.short_forecast import summarize
from gofa_f.target_calibration_math import improves, paired


def prediction(native, graph, cpu, weights):
    weights = np.asarray(weights, np.float64)
    if weights.shape != (3,) or not np.isfinite(weights).all() or weights.min() < 0 \
            or abs(weights.sum()-1.) > 1e-10:
        raise ValueError('Three finite convex weights required')
    if graph is None and weights[1] != 0:
        raise ValueError('Nonzero graph weight requires actual graph predictions')
    n, c = np.asarray(native, np.float64), np.asarray(cpu, np.float64)
    g = None if graph is None else np.asarray(graph, np.float64)
    if n.shape != c.shape or not np.isfinite([n, c]).all() \
            or g is not None and (g.shape != n.shape or not np.isfinite(g).all()):
        raise ValueError('Finite equal-shape expert forecasts required')
    for index, own in enumerate((n, g, c)):
        if weights[index] == 1. and np.count_nonzero(weights) == 1:
            return own.copy()
    result = weights[0]*n+weights[2]*c
    if weights[1] != 0:
        result = result+weights[1]*g
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite ensemble; no clipping or fallback')
    return result


def fit(native, graph, cpu, native_only=False):
    paired(native, graph)
    if len(cpu) != len(native):
        raise ValueError('Complete CPU development predictions required')
    n, g, c, y, variables = [], [], [], [], []
    for a, b, own in zip(native, graph, cpu):
        mask = np.asarray(a['observed'], bool)
        n.extend(np.asarray(a['prediction'], np.float64)[mask])
        g.extend(np.asarray(b['prediction'], np.float64)[mask])
        c.extend(np.asarray(own, np.float64)[mask])
        y.extend(np.asarray(a['target'], np.float64)[mask])
        variables.extend([a['query']['variable']]*int(mask.sum()))
    n, g, c, y = map(lambda v: np.asarray(v, np.float64), (n, g, c, y))
    if native_only:
        g = n.copy()  # The control's objective normalization cannot depend on graph forecasts.
    if not np.isfinite([n, g, c, y]).all():
        raise ValueError('Finite expert forecasts and observed targets required')
    _, inverse, counts = np.unique(variables, return_inverse=True, return_counts=True)
    weights = 1./(len(counts)*counts[inverse].astype(np.float64))
    count = len(y)
    normalization = max(float(np.abs(y-n).max()), float(np.abs(g-n).max()), float(np.abs(c-n).max()), 1e-12)
    x = np.column_stack((np.zeros(count), (g-n)/normalization, (c-n)/normalization))
    main = csr_matrix(x)
    inequalities = vstack((hstack((main, -eye(count))), hstack((-main, -eye(count))))).tocsr()
    rhs = np.r_[(y-n)/normalization, (n-y)/normalization]
    size = count+3
    objectives = [np.r_[np.zeros(3), weights], -np.eye(1, size, 0)[0], np.eye(1, size, 1)[0]]
    equalities, values, runs = [np.r_[np.ones(3), np.zeros(count)]], [1.], []
    for cost in objectives:
        result = linprog(cost, A_ub=inequalities, b_ub=rhs,
            A_eq=csr_matrix(np.asarray(equalities)), b_eq=np.asarray(values),
            bounds=[(0., 1.), (0., 0. if native_only else 1.), (0., 1.)]+[(0., None)]*count,
            method='highs', options=dict(primal_feasibility_tolerance=1e-10, dual_feasibility_tolerance=1e-10))
        if not result.success or not np.isfinite(result.x).all():
            raise ValueError('Three-expert lexicographic LP failed: '+result.message)
        runs.append(float(result.fun))
        equalities.append(cost)
        values.append(float(result.fun))
    raw = result.x[:3]
    if raw.min() < -1e-10 or abs(raw.sum()-1.) > 1e-10:
        raise ValueError('Solver simplex violation')
    actual = np.maximum(raw, 0.)
    actual /= actual.sum()
    loss = float(weights @ np.abs(prediction(n, g, c, actual)-y))
    if abs(loss/normalization-runs[0]) > 1e-8:
        raise ValueError('Tie-breaking changed primary MAE')
    return dict(weights=actual.tolist(), experts=['native', 'gofa', 'cpu'], fit_mae=loss,
        native_fit_mae=float(weights @ np.abs(n-y)), cpu_fit_mae=float(weights @ np.abs(c-y)),
        native_only=native_only, solver='float64 scipy/HiGHS lexicographic LP',
        normalization=normalization, solver_objectives=runs, feasibility_tolerance=1e-10,
        tie_order=['MAE', '-native', 'gofa'], stored_parameters=3)


def derived(native, graph, cpu, weights):
    if graph is not None:
        paired(native, graph)
    if len(native) != len(cpu):
        raise ValueError('CPU query coverage mismatch')
    rows, outside, fallback, total = [], 0, 0, 0
    for i, (a, c) in enumerate(zip(native, cpu)):
        b = graph[i] if graph is not None else None
        own = prediction(a['prediction'], None if b is None else b['prediction'], c, weights)
        valid = np.ones(len(own), bool)
        if weights[0] != 0:
            valid &= np.asarray(a['inverse_valid'], bool)
        if weights[1] != 0:
            valid &= np.asarray(b['inverse_valid'], bool)
        fallback += int((~valid).sum())
        low, high = a['representable_range']
        outside += int(((own < low) | (own > high)).sum())
        total += len(own)
        row = copy.deepcopy(a)
        row.update(prediction=own.tolist(), inverse_valid=np.isfinite(own).tolist(),
            boundary=[False]*len(own), format_complete=False, generated_token_count=0,
            forward_calls=0, generated_text=False)
        for key in ('generated_ids', 'numeric_ids'):
            row.pop(key, None)
        rows.append(row)
    metric = summarize(rows)
    metric.pop('formatting')
    metric['derived_prediction'] = dict(generation_coverage=None, generated_text=False,
        requested_slots=total, effective_base_fallback_fraction=fallback/total,
        outside_original_range_fraction=outside/total, clipped=False, rescaled=False)
    return rows, metric


def comparison(native, graph, cpu, parameters, bootstrap=False):
    systems = dict(native=(native, summarize(native)))
    if graph is not None:
        systems['gofa'] = graph, summarize(graph)
    systems['cpu'] = derived(native, graph, cpu, [0., 0., 1.])
    systems['native_cpu'] = derived(native, graph, cpu, parameters['native_cpu']['weights'])
    systems['ensemble'] = derived(native, graph, cpu, parameters['full']['weights'])
    result = dict(metrics={name: value[1] for name, value in systems.items()},
        strictly_improved=improves(systems['ensemble'][1]['macro']['mae'], systems['native'][1]['macro']['mae']),
        extra_graph_gain_demonstrated=parameters['full']['weights'][1] > 0 and improves(
            systems['ensemble'][1]['macro']['mae'], systems['native_cpu'][1]['macro']['mae']),
        changes={}, paired_query_counts={}, graph_confirmation_executed=graph is not None)
    if graph is None:
        result['metrics']['gofa'] = None
        result['gofa_missing_reason'] = 'not_executed_zero_deployment_weight'
    for name, (rows, metric) in systems.items():
        result['changes'][name] = {ref: dict(mae_difference=metric['macro']['mae']-systems[ref][1]['macro']['mae'],
            mae_ratio=metric['macro']['mae']/systems[ref][1]['macro']['mae'] if systems[ref][1]['macro']['mae'] else None)
            for ref in ('native', 'gofa', 'native_cpu') if ref in systems}
        differences = []
        for row, base in zip(rows, native):
            mask, y = np.asarray(base['observed'], bool), np.asarray(base['target'], np.float64)
            differences.append(float((np.abs(np.asarray(row['prediction'])-y)[mask]
                -np.abs(np.asarray(base['prediction'])-y)[mask]).mean()))
        d = np.asarray(differences)
        result['paired_query_counts'][name] = dict(improved=int((d < 0).sum()), equal=int((d == 0).sum()), degraded=int((d > 0).sum()))
    if bootstrap:
        metric = systems['ensemble'][1]
        variables = sorted(metric['variables'])
        ix = np.random.default_rng(4226).integers(0, len(variables), (2000, len(variables)))
        intervals = {}
        for ref in ('native', 'gofa', 'native_cpu', 'cpu'):
            if ref not in systems:
                continue
            d = np.asarray([metric['variables'][v]['mae']-systems[ref][1]['variables'][v]['mae'] for v in variables])
            intervals[ref] = np.quantile(d[ix].mean(1), [.025, .975]).tolist()
        result['paired_variable_cluster_bootstrap'] = dict(draws=2000, seed=4226, clusters=len(variables),
            mae_difference_95pct=intervals, used_as_gate=False,
            limitation='Only six turbine clusters' if len(variables) == 6 else 'Fixed selected variables')
    return result
