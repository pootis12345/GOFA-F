"""Stage28 bounded eight-coefficient residual calibration; no neural training."""
import copy

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack, vstack

from gofa_f.allnode_graph import build_graph
from gofa_f.generative_data import Query
from gofa_f.short_forecast import summarize
from gofa_f.target_calibration_math import paired, improves

FEATURES = ('intercept', 'horizon', 'graph_difference', 'persistence_difference',
            'trend', 'horizon_trend', 'horizon_neighbor_trend', 'relative_level_difference')
TAUS = (.01, .1, 1.)


def history_state(data, query):
    q = Query(**query)
    if q.source != 'sdwpf' or (q.length, q.horizon) != (48, 24):
        raise ValueError('Fixed SDWPF48->24 required')
    graph = build_graph(data, q)
    lookup = {str(v): i for i, v in enumerate(data.ids)}
    if len(lookup) != len(data.ids):
        raise ValueError('Unique variable IDs required')
    indices = [lookup[v] for v in graph.variable_ids]
    if indices[0] != q.variable:
        raise ValueError('Target-first accepted sampled subgraph required')
    values = np.asarray(data.values[q.origin-q.length:q.origin], np.float64)
    mask = np.asarray(data.mask[q.origin-q.length:q.origin], bool) & np.isfinite(values)

    def node(index):
        own = np.flatnonzero(mask[:, index])
        if not len(own):
            return None
        observed = values[own, index]
        low, high = float(observed.min()), float(observed.max())
        # Match official MinMaxScaler's constant-history inverse scale.
        scale = high-low if high != low else 1.
        recent = own[own >= q.length-12]
        slope = 0.
        if len(recent) >= 2:
            t = recent.astype(np.float64)-recent.mean()
            slope = float(t @ (values[recent, index]-values[recent, index].mean())/(t @ t))
        last = float(observed[-1])
        return dict(last=last, scale=scale, trend=float(np.clip(q.horizon*slope/scale, -1., 1.)),
                    level=(last-low)/scale)

    target = node(q.variable)
    if target is None:
        raise ValueError('Sealed query requires observed target history')
    neighbors = []
    if data.external is not None:
        for index in indices[1:]:
            if data.external[q.variable, index] > 0:
                state = node(index)
                if state is not None:
                    neighbors.append(state)
    target.update(neighbor_trend=float(np.mean([x['trend'] for x in neighbors])) if neighbors else 0.,
        level_difference=target['level']-float(np.mean([x['level'] for x in neighbors])) if neighbors else 0.,
        selected_ids=list(graph.variable_ids), neighbors=len(neighbors), graph_identity=graph.identity)
    return target


def frame(native, graph, data, native_only=False):
    paired([native], [graph])
    if native['scaling'] != graph['scaling']:
        raise ValueError('Paired official scales required')
    state = history_state(data, native['query'])
    scale = 1./float(native['scaling'][0])
    if not np.isfinite(scale) or scale <= 0 or not np.isclose(scale, state['scale'], rtol=1e-6, atol=1e-9):
        raise ValueError('Independent target history/official inverse scale mismatch')
    n, g = np.asarray(native['prediction'], np.float64), np.asarray(graph['prediction'], np.float64)
    u = np.arange(len(n), dtype=np.float64)/(len(n)-1)
    x = np.column_stack((np.ones(len(n)), u, np.clip((g-n)/scale, -1., 1.),
        np.clip((state['last']-n)/scale, -1., 1.), np.full(len(n), state['trend']),
        u*state['trend'], u*state['neighbor_trend'], np.full(len(n), state['level_difference'])))
    names = list(FEATURES)
    if native_only:
        x = np.delete(x, 2, axis=1)
        names.pop(2)
    if not np.isfinite(x).all() or np.abs(x).max() > 1.+1e-12:
        raise ValueError('Bounded finite causal features required')
    return dict(features=names, matrix=x, scale=scale, native=n, history=state)


def predict(frame, parameters):
    beta = np.asarray(parameters['beta'], np.float64)
    if frame['features'] != parameters['features'] or beta.shape != (len(frame['features']),) \
            or not np.isfinite(beta).all() or np.abs(beta).sum() > .25+1e-10:
        raise ValueError('Residual coefficient binding/L1 constraint changed')
    if not beta.any():
        return frame['native'].copy()
    result = frame['native']+frame['scale']*(frame['matrix'] @ beta)
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite residual result; no clipping/fallback')
    return result


def arrays(native, graph, frames):
    paired(native, graph)
    if len(frames) != len(native) or not frames or any(f['features'] != frames[0]['features'] for f in frames):
        raise ValueError('Consistent paired residual frames required')
    x, y, baseline, variables = [], [], [], []
    for row, f in zip(native, frames):
        mask = np.asarray(row['observed'], bool)
        if not mask.any():
            raise ValueError('At least one target observation required')
        x.extend(f['scale']*f['matrix'][mask])
        baseline.extend(f['native'][mask])
        y.extend(np.asarray(row['target'], np.float64)[mask])
        variables.extend([row['query']['variable']]*int(mask.sum()))
    _, inverse, counts = np.unique(variables, return_inverse=True, return_counts=True)
    weights = 1./(len(counts)*counts[inverse].astype(np.float64))
    return np.asarray(x), np.asarray(y), np.asarray(baseline), weights


def fit(native, graph, frames, tau):
    if tau not in TAUS:
        raise ValueError('Only preregistered strengths permitted')
    x, y, base, weights = arrays(native, graph, frames)
    count, width = x.shape
    mae = float(weights @ np.abs(base-y))
    initial = dict(features=frames[0]['features'], beta=[0.]*width, tau=tau,
        native_fit_mae=mae, stored_parameters=width, l1_limit=.25)
    if mae == 0.:
        return dict(initial, fit_mae=0., penalty=0., objective=0., solver='analytic_zero_baseline')
    normalization = max(float(np.abs(x).max()), float(np.abs(y-base).max()), 1e-12)
    size = 2*width+count
    features = hstack((csr_matrix(x/normalization), csr_matrix((count, width))))
    inequalities = vstack((hstack((features, -eye(count))), hstack((-features, -eye(count))),
        hstack((eye(width), -eye(width), csr_matrix((width, count)))),
        hstack((-eye(width), -eye(width), csr_matrix((width, count)))),
        csr_matrix(np.r_[np.zeros(width), np.ones(width), np.zeros(count)][None]))).tocsr()
    rhs = np.r_[(y-base)/normalization, (base-y)/normalization, np.zeros(2*width), .25]
    objective = np.r_[np.zeros(width), np.full(width, tau), weights*normalization/max(mae, 1e-12)]
    penalty = np.r_[np.zeros(width), np.ones(width), np.zeros(count)]
    objectives = [objective, penalty]+[np.eye(1, size, width+i)[0] for i in range(width)] \
        +[np.eye(1, size, i)[0] for i in range(width)]
    equalities, values, runs = [], [], []
    for cost in objectives:
        result = linprog(cost, A_ub=inequalities, b_ub=rhs,
            A_eq=csr_matrix(np.asarray(equalities)) if equalities else None,
            b_eq=np.asarray(values) if equalities else None,
            bounds=[(-.25, .25)]*width+[(0., .25)]*width+[(0., None)]*count,
            method='highs', options=dict(primal_feasibility_tolerance=1e-10, dual_feasibility_tolerance=1e-10))
        if not result.success or not np.isfinite(result.x).all():
            raise ValueError('Residual lexicographic LP failed: '+result.message)
        equalities.append(cost)
        values.append(float(result.fun))
        runs.append(float(result.fun))
    beta = result.x[:width]
    predicted = base+x @ beta
    loss = float(weights @ np.abs(predicted-y))
    regularizer = float(np.abs(beta).sum())
    actual = loss/max(mae, 1e-12)+tau*regularizer
    if regularizer > .25+1e-10 or abs(actual-runs[0]) > 1e-8*(1+abs(actual)):
        raise ValueError('Tie-breaking changed primary residual objective/constraint')
    return dict(initial, beta=beta.tolist(), fit_mae=loss, penalty=regularizer, objective=actual,
        solver='float64 scipy/HiGHS lexicographic LP', normalization=normalization,
        solver_objectives=runs, feasibility_tolerance=1e-10,
        tie_order=['objective', 'l1']+['abs_'+n for n in initial['features']]+initial['features'])


def derived(native, graph, frames, parameters):
    paired(native, graph)
    entries, outside, fallback, total = [], 0, 0, 0
    for n, g, f in zip(native, graph, frames):
        row, prediction = copy.deepcopy(n), predict(f, parameters)
        low, high = n['representable_range']
        outside += int(((prediction < low) | (prediction > high)).sum())
        valid = np.asarray(n['inverse_valid'], bool)
        if 'graph_difference' in parameters['features'] \
                and parameters['beta'][parameters['features'].index('graph_difference')] != 0:
            valid &= np.asarray(g['inverse_valid'], bool)
        fallback += int((~valid).sum())
        total += len(prediction)
        row.update(prediction=prediction.tolist(), inverse_valid=np.isfinite(prediction).tolist(),
            boundary=[False]*len(prediction), format_complete=False, generated_token_count=0,
            forward_calls=0, generated_text=False)
        for key in ('generated_ids', 'numeric_ids'):
            row.pop(key, None)
        entries.append(row)
    metric = summarize(entries)
    metric.pop('formatting')
    metric['derived_prediction'] = dict(generation_coverage=None, generated_text=False,
        outside_original_range_fraction=outside/total, effective_base_fallback_fraction=fallback/total,
        requested_slots=total, clipped=False, rescaled=False)
    return entries, metric


def compare(native, graph, new_rows, new_metric, control_rows, control_metric, graph_used, bootstrap=False):
    systems = dict(native=(native, summarize(native)), gofa=(graph, summarize(graph)),
        native_history=(control_rows, control_metric), calibrated=(new_rows, new_metric))
    base = systems['native'][1]['macro']['mae']
    mae = new_metric['macro']['mae']
    result = dict(metrics={k: x[1] for k, x in systems.items()}, strictly_improved=improves(mae, base),
        extra_graph_gain_demonstrated=graph_used and improves(mae, control_metric['macro']['mae']),
        changes={}, paired_query_counts={})
    for name, (rows, metric) in systems.items():
        result['changes'][name] = {k: dict(mae_difference=metric['macro']['mae']-systems[k][1]['macro']['mae'],
            mae_ratio=metric['macro']['mae']/systems[k][1]['macro']['mae'] if systems[k][1]['macro']['mae'] else None)
            for k in ('native', 'gofa', 'native_history')}
        differences = []
        for row, original in zip(rows, native):
            mask = np.asarray(original['observed'], bool)
            y = np.asarray(original['target'])
            differences.append(float((np.abs(np.asarray(row['prediction'])-y)[mask]
                -np.abs(np.asarray(original['prediction'])-y)[mask]).mean()))
        differences = np.asarray(differences)
        result['paired_query_counts'][name] = dict(improved=int((differences < 0).sum()),
            equal=int((differences == 0).sum()), degraded=int((differences > 0).sum()))
    if bootstrap:
        variables = sorted(new_metric['variables'])
        ix = np.random.default_rng(4226).integers(0, len(variables), (2000, len(variables)))
        intervals = {}
        for name in ('native', 'gofa', 'native_history'):
            d = np.asarray([new_metric['variables'][v]['mae']-systems[name][1]['variables'][v]['mae'] for v in variables])
            intervals[name] = np.quantile(d[ix].mean(1), [.025, .975]).tolist()
        result['paired_variable_cluster_bootstrap'] = dict(draws=2000, seed=4226, clusters=len(variables),
            mae_difference_95pct=intervals, used_as_gate=False,
            limitation='Only six turbine clusters' if len(variables) == 6 else 'Fixed sampled variables')
    return result
