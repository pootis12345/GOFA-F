"""Float64 post-median calibration; no model, sampling or optimizer operations."""
import copy

import numpy as np

from gofa_f.short_forecast import summarize


def tolerance(baseline):
    return max(1e-12, 1e-9*float(baseline))


def improves(current, baseline):
    return float(current) < float(baseline)-tolerance(baseline)


def blend(native, graph, coefficient):
    if not np.isfinite(coefficient) or not 0 <= coefficient <= 1:
        raise ValueError('One finite convex coefficient is required')
    native, graph = np.asarray(native, np.float64), np.asarray(graph, np.float64)
    if native.shape != graph.shape or not np.isfinite([native, graph]).all():
        raise ValueError('Paired finite predictions with identical shapes required')
    # Preserve endpoints exactly, including finite base fallbacks.
    if coefficient == 0:
        return native.copy()
    if coefficient == 1:
        return graph.copy()
    return (1-coefficient)*native+coefficient*graph


def paired(native, graph):
    if not native or len(native) != len(graph):
        raise ValueError('Complete paired query records required')
    for a, b in zip(native, graph):
        if a['query'] != b['query'] or a['scaling'] != b['scaling'] \
                or a['representable_range'] != b['representable_range']:
            raise ValueError('Query or original-unit scaling differs')
        y, other = np.asarray(a['target'], np.float64), np.asarray(b['target'], np.float64)
        mask, seen = np.asarray(a['observed'], bool), np.asarray(b['observed'], bool)
        size = a['query']['horizon']
        if y.shape != (size,) or mask.shape != y.shape or not np.array_equal(mask, seen) \
                or not np.array_equal(y, other, equal_nan=True) or not mask.any() \
                or not np.isfinite(y[mask]).all():
            raise ValueError('Paired original targets/masks/slots differ or are invalid')
        if blend(a['prediction'], b['prediction'], 0).shape != y.shape:
            raise ValueError('Prediction slots differ from horizon')


def objective_arrays(native, graph):
    """Each observed point receives 1/(variables * points_of_its_variable)."""
    paired(native, graph)
    n, g, truth, variables = [], [], [], []
    for a, b in zip(native, graph):
        mask = np.asarray(a['observed'], bool)
        n.extend(np.asarray(a['prediction'], np.float64)[mask])
        g.extend(np.asarray(b['prediction'], np.float64)[mask])
        truth.extend(np.asarray(a['target'], np.float64)[mask])
        variables.extend([a['query']['variable']]*int(mask.sum()))
    _, inverse, counts = np.unique(variables, return_inverse=True, return_counts=True)
    weights = 1./(len(counts)*counts[inverse].astype(np.float64))
    return np.asarray(n), np.asarray(g), np.asarray(truth), weights


def macro_mae(native, graph, coefficient):
    n, g, y, weights = objective_arrays(native, graph)
    return float(np.dot(np.abs(blend(n, g, coefficient)-y), weights))


def fit_coefficient(native, graph):
    n, g, y, weights = objective_arrays(native, graph)
    delta = g-n
    if not np.isfinite(delta).all():
        raise FloatingPointError('Nonfinite prediction difference cannot be calibrated')
    different = delta != 0
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        breaks = (y[different]-n[different])/delta[different]
    candidates = np.unique(np.r_[0., 1., breaks[np.isfinite(breaks) & (breaks >= 0) & (breaks <= 1)]])
    losses = np.asarray([np.dot(np.abs(blend(n, g, float(c))-y), weights) for c in candidates])
    # Sorted candidates plus first argmin choose the smallest minimizer in float64.
    index = int(np.argmin(losses))
    return dict(candidate=float(candidates[index]),candidate_mae=float(losses[index]),
        native_mae=float(losses[0]),graph_mae=float(losses[-1]),breakpoints_checked=len(candidates),
        solver='float64 endpoints and all finite absolute-loss breakpoints; smallest minimizing coefficient')


def calibrate_dataset(native, graph):
    if len(native) != 128 or len(graph) != 128:
        raise ValueError('Exactly 96 fit and 32 independent confirmation queries required')
    fitted = fit_coefficient(native[:96], graph[:96])
    c = fitted['candidate']
    confirmation = dict(native_mae=macro_mae(native[96:], graph[96:], 0),
        graph_mae=macro_mae(native[96:], graph[96:], 1),
        candidate_mae=macro_mae(native[96:], graph[96:], c))
    accepted = improves(confirmation['candidate_mae'], confirmation['native_mae'])
    return dict(coefficient=c if accepted else 0.,candidate=c,confirmation_supported=accepted,
        fit=fitted,confirmation=confirmation,fit_queries=96,confirmation_queries=32,
        refit_on_all_validation=False,
        reason='confirmed_improvement' if accepted else 'independent_confirmation_did_not_support_improvement')


def fused_metrics(native, graph, coefficient):
    paired(native, graph)
    entries = []
    base_fallback, finite = 0, 0
    for a, b in zip(native, graph):
        entry = copy.deepcopy(a)
        p = blend(a['prediction'], b['prediction'], coefficient)
        if not np.isfinite(p).all():
            raise FloatingPointError('Nonfinite derived prediction')
        valid_a, valid_b = np.asarray(a['inverse_valid'], bool), np.asarray(b['inverse_valid'], bool)
        valid = valid_a if coefficient == 0 else valid_b if coefficient == 1 else valid_a & valid_b
        base_fallback += int((~valid).sum())
        finite += len(p)
        entry.update(prediction=p.tolist(),inverse_valid=np.isfinite(p).tolist(),
            format_complete=False,generated_token_count=0,forward_calls=0,
            boundary=[False]*len(p),coefficient=coefficient,generated_text=False)
        # Prediction records carry no sampled text or token-completion claim.
        entry.pop('generated_ids', None)
        entry.pop('numeric_ids', None)
        entries.append(entry)
    metrics = summarize(entries)
    metrics.pop('formatting')
    metrics['derived_prediction'] = dict(finite_slots=finite,requested_slots=finite,
        finite_fraction=1.,effective_base_fallback_slots=base_fallback,
        effective_base_fallback_fraction=base_fallback/finite,
        generation_coverage=None,generated_text=False,
        definition='original-unit convex fusion after each model sample8 pointwise median; no rescaling or clipping')
    return metrics
