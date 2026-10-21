"""Float64 variable-equal L1 calibration; no neural model operations."""
import copy

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack, vstack

from gofa_f.short_forecast import summarize
from gofa_f.target_calibration_math import blend, improves, objective_arrays, paired


def prediction(native, graph, coefficient, bias):
    if not np.isfinite(bias):
        raise ValueError('Finite shared original-unit bias required')
    result = blend(native, graph, coefficient)
    if bias != 0:
        result = result + np.float64(bias)
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite derived prediction')
    return result


def fit(native, graph, native_only=False):
    """Four lexicographic LPs, fixing earlier objectives by equality, no grid."""
    n, g, y, weights = objective_arrays(native, graph)
    delta, target = g-n, y-n
    scale = max(float(np.max(np.abs(target))), float(np.max(np.abs(delta))), 1e-12)
    d, target = delta/scale, target/scale
    count = len(n)
    # Variables: lambda, normalized bias, absolute bias, absolute residuals.
    main = hstack((csr_matrix(d[:, None]), csr_matrix(np.ones((count, 1))),
                   csr_matrix((count, 1))))
    constraints = vstack((hstack((main, -eye(count))),
                          hstack((-main, -eye(count))),
                          csr_matrix(([1., -1., -1., -1.],
                                      ([0, 0, 1, 1], [1, 2, 1, 2])), shape=(2, count+3))))
    rhs = np.r_[target, -target, 0., 0.]
    bounds = [(0., 0. if native_only else 1.), (None, None), (0., None)]+[(0., None)]*count
    objective = np.r_[0., 0., 0., weights]
    equalities, values, records = [], [], []
    objectives = [objective, np.eye(1, count+3, 0)[0],
                  np.eye(1, count+3, 2)[0], np.eye(1, count+3, 1)[0]]
    for cost in objectives:
        result = linprog(cost, A_ub=constraints.tocsr(), b_ub=rhs,
            A_eq=csr_matrix(np.asarray(equalities)) if equalities else None,
            b_eq=np.asarray(values) if values else None, bounds=bounds, method='highs',
            options={'primal_feasibility_tolerance': 1e-10, 'dual_feasibility_tolerance': 1e-10})
        if not result.success or not np.isfinite(result.x).all():
            raise ValueError('HiGHS calibration failed: '+result.message)
        records.append(dict(objective=float(result.fun), status=int(result.status)))
        equalities.append(cost)
        values.append(float(result.fun))
    raw_coefficient = float(result.x[0])
    if raw_coefficient < -1e-10 or raw_coefficient > 1+1e-10:
        raise ValueError('LP violated convex coefficient bounds')
    # Canonicalize only solver roundoff outside the declared [0,1] bound.
    coefficient = max(0., min(1., raw_coefficient))
    bias = float(result.x[1]*scale)
    loss = float(weights @ np.abs(prediction(n, g, coefficient, bias)-y))
    if abs(loss/scale-records[0]['objective']) > 1e-8:
        raise ValueError('Lexicographic solution changed primary MAE')
    return dict(coefficient=coefficient, bias=bias, fit_mae=loss,
        native_mae=float(weights @ np.abs(n-y)), gofa_mae=float(weights @ np.abs(g-y)),
        solver='scipy.optimize.linprog(method=highs), float64, four equality-fixed lexicographic LPs',
        tie_order=['MAE', 'lambda', 'abs(bias)', 'bias'], solver_runs=records,
        normalization_scale=scale, numerical_feasibility_tolerance=1e-10,
        raw_solver_coefficient=raw_coefficient,
        native_only=native_only, variables=len({r['query']['variable'] for r in native}))


def derived(native, graph, coefficient, bias):
    paired(native, graph)
    entries, fallback, outside, total = [], 0, 0, 0
    for a, b in zip(native, graph):
        row = copy.deepcopy(a)
        p = prediction(a['prediction'], b['prediction'], coefficient, bias)
        va, vb = np.asarray(a['inverse_valid'], bool), np.asarray(b['inverse_valid'], bool)
        valid = va if coefficient == 0 else vb if coefficient == 1 else va & vb
        fallback += int((~valid).sum())
        low, high = a['representable_range']
        outside += int(((p < low) | (p > high)).sum())
        total += len(p)
        row.update(prediction=p.tolist(), inverse_valid=np.isfinite(p).tolist(),
            format_complete=False, generated_token_count=0, forward_calls=0,
            boundary=[False]*len(p), generated_text=False)
        row.pop('generated_ids', None)
        row.pop('numeric_ids', None)
        entries.append(row)
    metrics = summarize(entries)
    metrics.pop('formatting')
    metrics['derived_prediction'] = dict(generation_coverage=None, generated_text=False,
        finite_slots=total, effective_base_fallback_slots=fallback,
        effective_base_fallback_fraction=fallback/total, outside_original_range_slots=outside,
        outside_original_range_fraction=outside/total, rescaled=False, clipped=False)
    return entries, metrics


def comparison(native, graph, fitted, bootstrap=False):
    paired(native, graph)
    systems = dict(native=(native, summarize(native)), gofa=(graph, summarize(graph)))
    for key, params in (('native_bias', fitted['native_bias']), ('blend_bias', fitted['blend_bias'])):
        systems[key] = derived(native, graph, params['coefficient'], params['bias'])
    baseline = systems['native'][1]['macro']['mae']
    fused = systems['blend_bias'][1]['macro']['mae']
    control = systems['native_bias'][1]['macro']['mae']
    result = dict(metrics={key: value[1] for key, value in systems.items()},
        strictly_improved=improves(fused, baseline),
        extra_graph_gain_demonstrated=fitted['blend_bias']['coefficient'] > 0 and improves(fused, control),
        changes={}, paired_query_counts={})
    for key, (rows, metric) in systems.items():
        result['changes'][key] = {ref: dict(mae_difference=metric['macro']['mae']-systems[ref][1]['macro']['mae'],
            mae_ratio=metric['macro']['mae']/systems[ref][1]['macro']['mae']
                if systems[ref][1]['macro']['mae'] else None) for ref in ('native', 'gofa', 'native_bias')}
        errors = np.asarray([np.abs(np.asarray(r['prediction'])-np.asarray(r['target']))[
            np.asarray(r['observed'], bool)].mean() for r in rows])
        base_errors = np.asarray([np.abs(np.asarray(r['prediction'])-np.asarray(r['target']))[
            np.asarray(r['observed'], bool)].mean() for r in native])
        difference = errors-base_errors
        result['paired_query_counts'][key] = dict(improved=int((difference < 0).sum()),
            equal=int((difference == 0).sum()), degraded=int((difference > 0).sum()))
    if bootstrap:
        variables = sorted(systems['native'][1]['variables'])
        rng = np.random.default_rng(4226)
        indices = rng.integers(0, len(variables), size=(2000, len(variables)))
        intervals = {}
        for ref in ('native', 'gofa', 'native_bias'):
            differences = np.asarray([systems['blend_bias'][1]['variables'][v]['mae']-
                systems[ref][1]['variables'][v]['mae'] for v in variables])
            estimates = differences[indices].mean(1)
            intervals[ref] = np.quantile(estimates, [.025, .975]).tolist()
        result['paired_variable_cluster_bootstrap'] = dict(draws=2000, seed=4226,
            clusters=len(variables), mae_difference_95pct=intervals,
            limitation='Only six turbine clusters' if len(variables) == 6 else 'Fixed selected variable clusters',
            used_as_gate=False)
    return result
