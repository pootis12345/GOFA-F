"""Distance/probability diagnostics; teacher-conditioned values are not forecasts."""
import numpy as np
import torch

from gofa_f.short_sampling_diagnostic import measures


def quantiles(values):
    values = values.detach().float().reshape(-1)
    if not len(values):
        return dict(count=0, mean=None, p50=None, p95=None, maximum=None)
    return dict(count=len(values), mean=float(values.mean()), p50=float(values.median()),
                p95=float(torch.quantile(values, .95)), maximum=float(values.max()))


def teacher_distribution(scores, codec, record, target):
    """Full 10000-class CE and numeric-order medians, never top-k CE."""
    scores = scores.float()
    if scores.shape != (len(target), len(codec.finite_ids)) or not torch.isfinite(scores).all():
        raise ValueError('Finite H x numeric-vocabulary logits required')
    positions = record['answer_positions']
    labels = torch.tensor([record['labels'][i] for i in positions], device=scores.device)
    observed = labels != -100
    if not observed.any():
        raise ValueError('Teacher query needs an observed target')
    classes = codec.lookup(max(codec.finite_ids)+1, scores.device)
    truth = classes[labels[observed]]
    if (truth < 0).any():
        raise ValueError('Observed teacher label outside finite alphabet')
    logp = scores.log_softmax(-1)
    p = logp.exp()
    values, indices = scores.topk(100, dim=-1, sorted=True)
    top = torch.zeros_like(p).scatter(-1, indices, values.softmax(-1))
    full_median = (p.double().cumsum(-1) < .5).sum(-1).clamp_max(p.shape[-1]-1)
    top_median = (top.double().cumsum(-1) < .5).sum(-1).clamp_max(p.shape[-1]-1)
    predictions = {}
    alphabet = codec.alphabet(scores.device)
    observed_np = observed.cpu().numpy()
    for name, chosen in [('argmax', scores.argmax(-1)), ('full_median', full_median), ('top100_median', top_median)]:
        ids = alphabet[chosen].tolist()
        raw, valid, boundary = codec.decode_ids(ids, *record['scaling'], record['last'])
        if not valid.all():
            raise FloatingPointError('Teacher numeric decision cannot use inverse fallback')
        predictions[name] = dict(prediction=raw.tolist(), numeric_ids=ids,
            boundary=boundary.tolist(), **measures(raw, target, observed_np))
    chosen_scores = scores[observed].gather(-1, truth[:, None])
    ranks = (scores[observed] > chosen_scores).sum(-1)+1
    membership = (indices[observed] == truth[:, None]).any(-1)
    true_ranks, true_coverage, true_ce = [], [], []
    cursor = 0
    for valid in observed.tolist():
        true_ranks.append(int(ranks[cursor]) if valid else None)
        true_coverage.append(bool(membership[cursor]) if valid else None)
        true_ce.append(float(-logp[observed][cursor, truth[cursor]]) if valid else None)
        cursor += int(valid)
    concentration = dict(entropy=(-(p*logp).sum(-1)).tolist(),
        maximum_probability=p.max(-1).values.tolist(), lower_probability=p[:, 0].tolist(),
        upper_probability=p[:, -1].tolist(), top100_lower_probability=top[:, 0].tolist(),
        top100_upper_probability=top[:, -1].tolist(), true_rank_min=true_ranks,
        true_in_actual_top100=true_coverage, true_ce=true_ce)
    return dict(conditional_ce=float(-logp[observed].gather(1, truth[:, None]).mean()),
        predictions=predictions, concentration=concentration, observed_slots=observed.tolist(),
        top100_truth_coverage=float(membership.float().mean()),
        entropy_observed_mean=float((-(p*logp).sum(-1))[observed].mean()),
        first_slot_max_probability=float(p[0].max()),
        label_shift='answer position j is scored at logits j-1',
        interpretation='true earlier responses supplied; not free forecasting')


def point_errors(rows, phase):
    """Query-weighted MAE plus explicitly point-weighted horizon diagnostics."""
    def segment(predicate):
        errors = []
        for r in rows:
            p = np.asarray(r['free']['prediction'], np.float64)
            y, seen = np.asarray(r['target'], np.float64), np.asarray(r['observed'], bool)
            for i in np.flatnonzero(seen):
                if predicate(r['query']['horizon'], int(i)):
                    errors.append(abs(p[i]-y[i]))
        return dict(observations=len(errors), mae=float(np.mean(errors)) if errors else None)
    multi = [r for r in rows if r['query']['horizon'] > 1]
    slots = sum(r['query']['horizon'] for r in rows)
    return dict(condition=phase, queries=len(rows), query_mean_mae=float(np.mean([r['free']['mae'] for r in rows])),
        teacher_ce=float(np.mean([r['teacher']['conditional_ce'] for r in rows])),
        teacher_argmax_mae=float(np.mean([r['teacher']['predictions']['argmax']['mae'] for r in rows])),
        teacher_full_median_mae=float(np.mean([r['teacher']['predictions']['full_median']['mae'] for r in rows])),
        teacher_top100_median_mae=float(np.mean([r['teacher']['predictions']['top100_median']['mae'] for r in rows])),
        teacher_top100_coverage=float(np.mean([r['teacher']['top100_truth_coverage'] for r in rows])),
        constant_multistep=sum(r['free']['repetition']['constant_complete'] for r in multi),
        multistep_queries=len(multi), upper_fraction=sum(r['free']['upper_slots'] for r in rows)/slots,
        lower_fraction=sum(r['free']['lower_slots'] for r in rows)/slots,
        first_step=segment(lambda h, i: i == 0), subsequent_steps=segment(lambda h, i: i > 0),
        h24_steps_1_16=segment(lambda h, i: h == 24 and i < 16),
        h24_steps_17_24=segment(lambda h, i: h == 24 and i >= 16))


def summarize(rows):
    conditions = ('native', 'alpha1', 'alpha025')
    primary = [r for r in rows if r['group'] == 'id_stratified']
    if len(primary) != 96 or len({(r['condition'], r['index']) for r in primary}) != 96:
        raise ValueError('All 32 queries under three conditions required')
    result = dict(by_source={}, known_cases=[r for r in rows if r['group'] != 'id_stratified'],
                  no_selection=True, no_cross_source_raw_error_average=True)
    for source in sorted({r['query']['source'] for r in primary}):
        source_rows = [r for r in primary if r['query']['source'] == source]
        summary = {c: point_errors([r for r in source_rows if r['condition'] == c], c) for c in conditions}
        templates = sorted({(r['query']['length'], r['query']['horizon']) for r in source_rows})
        by_template = {f'{length}->{horizon}': {
            c: point_errors([r for r in source_rows if r['condition'] == c and
                            (r['query']['length'], r['query']['horizon']) == (length, horizon)], c)
            for c in conditions} for length, horizon in templates}
        paired = {r['index']: {} for r in source_rows}
        for r in source_rows:
            paired[r['index']][r['condition']] = r
        counts = dict(free_improved=0, free_equal=0, free_worse=0,
            ce_and_teacher_median_better_but_free_worse=0,
            ce_better_but_teacher_median_worse=0, shrink_teacher_and_free_better=0)
        for parts in paired.values():
            native, full, small = (parts[c] for c in conditions)
            ce_better = full['teacher']['conditional_ce'] < native['teacher']['conditional_ce']
            teacher_better = full['teacher']['predictions']['full_median']['mae'] < native['teacher']['predictions']['full_median']['mae']
            free_worse = full['free']['mae'] > native['free']['mae']
            counts['ce_and_teacher_median_better_but_free_worse'] += int(ce_better and teacher_better and free_worse)
            counts['ce_better_but_teacher_median_worse'] += int(ce_better and
                full['teacher']['predictions']['full_median']['mae'] > native['teacher']['predictions']['full_median']['mae'])
            delta = small['free']['mae']-full['free']['mae']
            counts['free_improved' if delta < 0 else 'free_worse' if delta > 0 else 'free_equal'] += 1
            counts['shrink_teacher_and_free_better'] += int(delta < 0 and
                small['teacher']['predictions']['full_median']['mae'] < full['teacher']['predictions']['full_median']['mae'])
        result['by_source'][source] = dict(conditions=summary, by_template=by_template, descriptive_patterns=counts)
    return result
