"""Small causal expert calibrators; float64 L1/HiGHS, no neural training."""
import copy

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack, vstack

from gofa_f.short_forecast import summarize
from gofa_f.target_calibration_math import improves, paired

WEATHER = ('meteonet_nw2018', 'engrad')
TAUS = (0.01, 0.1, 1.)


def frame(native, graph, data, native_only=False):
    """No values or missingness at/after origin are inspected here."""
    q = native['query']
    if q != graph['query'] or native['scaling'] != graph['scaling']:
        raise ValueError('Paired query/scaling required')
    origin, length, horizon, variable = (q[k] for k in ('origin', 'length', 'horizon', 'variable'))
    if origin < length or horizon < 2 or length < 12:
        raise ValueError('Existing short-query dimensions required')
    values = np.asarray(data.values[origin-length:origin, variable], np.float64)
    mask = np.asarray(data.mask[origin-length:origin, variable], bool) & np.isfinite(values)
    if not mask.any():
        raise ValueError('Sealed queries require observed history')
    observed = np.flatnonzero(mask)
    last_index, last = int(observed[-1]), float(values[observed[-1]])
    recent = np.flatnonzero(mask[-12:])+length-12
    mean = float(values[recent].mean()) if len(recent) else last
    slope = 0.
    if len(recent) >= 2:
        centered = recent.astype(np.float64)-recent.mean()
        slope = float(centered @ (values[recent]-values[recent].mean())/(centered @ centered))
    future_index = length+np.arange(horizon, dtype=np.float64)
    trend = last+(future_index-last_index)*slope
    names = ['native']+([] if native_only else ['gofa'])+['persistence', 'mean', 'trend']
    columns = [np.asarray(native['prediction'], np.float64)]
    if not native_only:
        columns.append(np.asarray(graph['prediction'], np.float64))
    columns += [np.full(horizon, last), np.full(horizon, mean), trend]
    if q['source'] in WEATHER:
        if length != 48 or horizon != 24 or not np.all(np.diff(data.times[origin-length:origin]) == 3600*10**9):
            raise ValueError('Daily expert requires audited hourly48->24 history')
        indices = length-24+np.arange(horizon)
        columns.append(np.where(mask[indices], values[indices], last))
        names.append('daily')
    scale = float(native['scaling'][0])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Positive finite official origin scale required')
    inverse_scale = 1./scale
    matrix = np.column_stack(columns)
    if matrix.shape != (horizon, len(names)) or not np.isfinite(matrix).all() or not np.isfinite(inverse_scale):
        raise FloatingPointError('Nonfinite causal expert')
    return dict(names=names, experts=matrix, scale=inverse_scale,
                u=np.arange(horizon, dtype=np.float64)/(horizon-1))


def arrays(native, graph, frames):
    paired(native, graph)
    if not frames or len(frames) != len(native) or any(f['names'] != frames[0]['names'] for f in frames):
        raise ValueError('Consistent paired frames required')
    experts, bias, labels, variables = [], [], [], []
    for n, f in zip(native, frames):
        mask = np.asarray(n['observed'], bool)
        experts.extend(f['experts'][mask])
        bias.extend(np.column_stack((np.full(len(mask), f['scale']), f['scale']*f['u']))[mask])
        labels.extend(np.asarray(n['target'], np.float64)[mask])
        variables.extend([n['query']['variable']]*int(mask.sum()))
    _, indices, counts = np.unique(variables, return_inverse=True, return_counts=True)
    weights = 1./(len(counts)*counts[indices].astype(np.float64))
    return np.asarray(experts), np.asarray(bias), np.asarray(labels), weights


def predict(f, parameters):
    if f['names'] != parameters['experts']:
        raise ValueError('Calibrator expert identity/order changed')
    weights = np.asarray([parameters['weights'][n] for n in f['names']], np.float64)
    b0, b1 = parameters['bias']
    if not np.isfinite(np.r_[weights, b0, b1]).all() or weights.min() < 0 \
            or abs(weights.sum()-1.) > 1e-10 or max(abs(b0), abs(b0+b1)) > .25+1e-10:
        raise ValueError('Convex weights/bias constraints violated')
    # Preserve the zero-correction native endpoint exactly.
    if weights[0] == 1. and not weights[1:].any() and b0 == b1 == 0.:
        return f['experts'][:, 0].copy()
    result = f['experts'] @ weights + f['scale']*(b0+b1*f['u'])
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite derived prediction; never clip/fallback')
    return result


def fit(native, graph, frames, tau):
    if tau not in TAUS:
        raise ValueError('Only the three preregistered strengths are permitted')
    x, b, y, weights = arrays(native, graph, frames)
    names = frames[0]['names']
    count, e = x.shape
    baseline = float(weights @ np.abs(x[:, 0]-y))
    identity = dict(experts=names, weights={n: float(i == 0) for i, n in enumerate(names)},
        bias=[0., 0.], tau=tau, native_fit_mae=baseline)
    if baseline == 0:
        return dict(identity, fit_mae=0., penalty=0., objective=0., solver='analytic nonnegative zero objective',
                    solver_objectives=[], tie_order=[], stored_parameters=e+2)
    # Variables: expert weights, b0/b1, |b0|/|b1|, absolute normalized residuals.
    size = e+4+count
    normalization = max(float(np.abs(x-x[:, :1]).max()), float(np.abs(y-x[:, 0]).max()),
                        float(np.abs(b).max()), 1e-12)
    features = np.column_stack(((x-x[:, :1])/normalization, b/normalization, np.zeros((count, 2))))
    rhs = (y-x[:, 0])/normalization
    constraints = vstack((hstack((csr_matrix(features), -eye(count))),
                          hstack((-csr_matrix(features), -eye(count)))))
    bound_rows = np.zeros((6, size))
    for i in range(2):
        bound_rows[2*i, e+i], bound_rows[2*i, e+2+i] = 1., -1.
        bound_rows[2*i+1, e+i], bound_rows[2*i+1, e+2+i] = -1., -1.
    bound_rows[4, e:e+2], bound_rows[5, e:e+2] = 1., -1.
    constraints = vstack((constraints, csr_matrix(bound_rows))).tocsr()
    rhs = np.r_[rhs, -rhs, 0., 0., 0., 0., .25, .25]
    bounds = [(0., 1.)]*e+[(-.25, .25), (-.5, .5), (0., None), (0., None)]+[(0., None)]*count
    penalty = np.zeros(size)
    penalty[0], penalty[e+2:e+4] = -1., 1.
    objective = tau*penalty
    objective[e+4:] = weights*normalization/max(baseline, 1e-12)
    sum_weights = np.r_[np.ones(e), np.zeros(size-e)]
    equalities, values = [sum_weights], [1.]
    tie_order = ['regularized_objective', 'penalty', '-native']+names[1:]+['abs_b0', 'abs_b1', 'b0', 'b1']
    objectives = [objective, penalty, -np.eye(1, size, 0)[0]]
    objectives += [np.eye(1, size, i)[0] for i in range(1, e)]
    objectives += [np.eye(1, size, i)[0] for i in (e+2, e+3, e, e+1)]
    runs = []
    for cost in objectives:
        result = linprog(cost, A_ub=constraints, b_ub=rhs, A_eq=csr_matrix(np.asarray(equalities)),
            b_eq=np.asarray(values), bounds=bounds, method='highs',
            options=dict(primal_feasibility_tolerance=1e-10, dual_feasibility_tolerance=1e-10))
        if not result.success or not np.isfinite(result.x).all():
            raise ValueError('Lexicographic HiGHS failure: '+result.message)
        runs.append(float(result.fun))
        equalities.append(cost)
        values.append(float(result.fun))
    raw = result.x[:e]
    if raw.min() < -1e-10 or abs(raw.sum()-1.) > 1e-10:
        raise ValueError('Solver violated simplex')
    w = np.maximum(raw, 0.)
    w /= w.sum()  # Only machine roundoff from equality/bounds is canonicalized.
    b0, b1 = map(float, result.x[e:e+2])
    value = dict(identity, weights=dict(zip(names, map(float, w))), bias=[b0, b1],
        solver='float64 scipy/HiGHS lexicographic LP', solver_objectives=runs, tie_order=tie_order,
        feasibility_tolerance=1e-10, normalization=normalization, stored_parameters=e+2)
    actual = np.concatenate([predict(f, value)[np.asarray(r['observed'], bool)] for f, r in zip(frames, native)])
    loss = float(weights @ np.abs(actual-y))
    regularizer = float(1-w[0]+abs(b0)+abs(b1))
    actual_objective = loss/max(baseline, 1e-12)+tau*regularizer
    if abs(actual_objective-(runs[0]+tau)) > 1e-8*(1+abs(actual_objective)):
        raise ValueError('Later tie-breaking changed primary objective')
    return dict(value, fit_mae=loss, penalty=regularizer, objective=actual_objective)


def derived(native, graph, frames, parameters):
    paired(native, graph)
    entries, fallback, outside, total = [], 0, 0, 0
    for n, g, f in zip(native, graph, frames):
        row, p = copy.deepcopy(n), predict(f, parameters)
        valid = np.ones(len(p), bool)
        for key, base in (('native', n), ('gofa', g)):
            if parameters['weights'].get(key, 0.) > 0:
                valid &= np.asarray(base['inverse_valid'], bool)
        fallback += int((~valid).sum())
        low, high = n['representable_range']
        outside += int(((p < low) | (p > high)).sum())
        total += len(p)
        row.update(prediction=p.tolist(), inverse_valid=np.isfinite(p).tolist(), boundary=[False]*len(p),
                   format_complete=False, generated_token_count=0, forward_calls=0, generated_text=False)
        row.pop('generated_ids', None)
        row.pop('numeric_ids', None)
        entries.append(row)
    metrics = summarize(entries)
    metrics.pop('formatting')
    metrics['derived_prediction'] = dict(generation_coverage=None, generated_text=False,
        requested_slots=total, effective_base_fallback_slots=fallback,
        effective_base_fallback_fraction=fallback/total, outside_original_range_slots=outside,
        outside_original_range_fraction=outside/total, clipped=False, rescaled=False)
    return entries, metrics


def comparison(native, graph, frames, control_frames, parameters, control, bootstrap=False):
    new_rows, new = derived(native, graph, frames, parameters)
    control_rows, other = derived(native, graph, control_frames, control)
    systems = dict(native=(native, summarize(native)), gofa=(graph, summarize(graph)),
                   native_history=(control_rows, other), calibrated=(new_rows, new))
    baseline = systems['native'][1]['macro']['mae']
    mae = new['macro']['mae']
    result = dict(metrics={k: v[1] for k, v in systems.items()},
        strictly_improved=improves(mae, baseline),
        extra_graph_gain_demonstrated=parameters['weights'].get('gofa', 0.) > 0 and improves(mae, other['macro']['mae']),
        changes={}, paired_query_counts={})
    for key, (rows, metric) in systems.items():
        result['changes'][key] = {ref: dict(mae_difference=metric['macro']['mae']-systems[ref][1]['macro']['mae'],
            mae_ratio=metric['macro']['mae']/systems[ref][1]['macro']['mae'] if systems[ref][1]['macro']['mae'] else None)
            for ref in ('native', 'gofa', 'native_history')}
        differences = []
        for row, n in zip(rows, native):
            mask = np.asarray(n['observed'], bool)
            y = np.asarray(n['target'])
            differences.append(float(np.abs(np.asarray(row['prediction'])-y)[mask].mean()-
                                     np.abs(np.asarray(n['prediction'])-y)[mask].mean()))
        differences = np.asarray(differences)
        result['paired_query_counts'][key] = dict(improved=int((differences < 0).sum()),
            equal=int((differences == 0).sum()), degraded=int((differences > 0).sum()))
    if bootstrap:
        variables = sorted(new['variables'])
        indices = np.random.default_rng(4226).integers(0, len(variables), (2000, len(variables)))
        intervals = {}
        for ref in ('native', 'gofa', 'native_history'):
            differences = np.asarray([new['variables'][v]['mae']-systems[ref][1]['variables'][v]['mae'] for v in variables])
            intervals[ref] = np.quantile(differences[indices].mean(1), [.025, .975]).tolist()
        result['paired_variable_cluster_bootstrap'] = dict(draws=2000, seed=4226, clusters=len(variables),
            mae_difference_95pct=intervals, used_as_gate=False,
            limitation='Only six turbine clusters' if len(variables) == 6 else 'Fixed sampled variables')
    return result
