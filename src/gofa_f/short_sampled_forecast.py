"""Eight independent paths, pointwise median; shared source/future-test protocol."""
import time

import numpy as np
import torch

from gofa_f.allnode_graph import build_graph
from gofa_f.allnode_stage import read
from gofa_f.generative_data import Query
from gofa_f.prefix_cache import atomic_json
from gofa_f.short_forecast import summarize
from gofa_f.short_sampling_diagnostic import SETTINGS, generate, sample_seed


def checked(path, binding, query, **identity):
    item = read(path)
    if item is not None and (item.get('binding') != binding or item.get('query') != query or
            any(item.get(k) != v for k,v in identity.items())):
        raise ValueError('Sampled evaluation shard binding/query/seed differs')
    return item


@torch.no_grad()
def evaluate(model, codec, series, queries, output, label, binding, progress, check, native=False):
    """Atomically commit each path; missing paths rebuild KV and retain fixed seeds."""
    root = output/'evaluations'/label
    root.mkdir(parents=True, exist_ok=True)
    (root/'trajectories').mkdir(exist_ok=True)
    complete = read(root/'complete.json')
    if complete is not None:
        if complete['binding'] != binding or not complete['complete'] or complete['sampling'] != SETTINGS:
            raise ValueError('Complete sampled evaluation protocol differs')
        return complete
    entries = [checked(root/f'{i:06d}.json', binding, q) for i,q in enumerate(queries)]
    work_total = 8*sum(q['horizon'] for q in queries)
    # Include work in partially finished queries when estimating progress after resume.
    saved_paths = {}
    for i,q in enumerate(queries):
        for sample in range(8):
            path = root/'trajectories'/f'{i:06d}-{sample:02d}.json'
            value = checked(path, binding, q, seed=sample_seed(q,sample), sample=sample)
            if value is not None:
                saved_paths[i,sample] = value
    work = sum(v['query']['horizon'] for v in saved_paths.values())
    seconds = sum(v['active_seconds'] for v in saved_paths.values())
    progress.evaluation(sum(e is not None for e in entries), len(queries), work, work_total, seconds,
                        evaluation_kind=label, evaluation_sample=0)
    for i,query in enumerate(queries):
        if entries[i] is not None:
            if any((i,s) not in saved_paths for s in range(8)):
                raise ValueError('Committed median lacks all eight trajectory shards')
            continue
        check()
        q, data = Query(**query), series[query['source']]
        history = np.where(data.mask[q.origin-q.length:q.origin,q.variable],
                           data.values[q.origin-q.length:q.origin,q.variable], np.nan)
        base = codec.record(history)
        contexts, context_seconds = None, 0.
        if not native and base['present'] and any((i,s) not in saved_paths for s in range(8)):
            start = time.monotonic()
            contexts = model.encode_origins([build_graph(data,q)],codec,check=check)
            context_seconds = time.monotonic()-start
        samples = []
        for sample in range(8):
            check()
            item = saved_paths.get((i,sample))
            if item is None:
                start = time.monotonic()
                seed = sample_seed(query,sample)
                if base['present']:
                    item = generate(model,codec,history,q.horizon,contexts,seed=seed,check=check)
                    item.update(format_complete=True, inverse_valid=[True]*q.horizon)
                else:
                    # Preserve Stage19's explicit zero fallback for entirely absent history.
                    item = dict(prediction=[0.]*q.horizon,numeric_ids=[],generated_ids=[],
                        boundary=[False]*q.horizon,format_complete=False,inverse_valid=[False]*q.horizon)
                item.update(binding=binding,query=query,seed=seed,sample=sample,
                            active_seconds=time.monotonic()-start+context_seconds)
                context_seconds = 0.
                atomic_json(root/'trajectories'/f'{i:06d}-{sample:02d}.json',item)
                saved_paths[i,sample] = item
                work += q.horizon
                seconds += item['active_seconds']
            samples.append(item)
            progress.evaluation(sum(e is not None for e in entries),len(queries),work,work_total,seconds,
                                evaluation_kind=label,evaluation_sample=sample+1,
                                evaluation_current_horizon=q.horizon)
        p = np.median(np.asarray([v['prediction'] for v in samples],np.float64),axis=0)
        if not np.isfinite(p).all():
            raise FloatingPointError('Nonfinite sample median cannot be reported')
        y = data.values[q.origin:q.origin+q.horizon,q.variable].astype(np.float64)
        observed = data.mask[q.origin:q.origin+q.horizon,q.variable]
        if not observed.any():
            raise ValueError('Evaluation has no observed target')
        scale, offset = base['scaling']
        low, high = (-.9999+.5-offset)/scale, (.9999+.5-offset)/scale
        centers = (np.median([[codec.id_centers[t] for t in v['numeric_ids']] for v in samples],axis=0)
                   if base['present'] else np.full(q.horizon,np.nan))
        item = dict(binding=binding,query=query,prediction=p.tolist(),target=y.tolist(),observed=observed.tolist(),
            scaling=list(base['scaling']),representable_range=[low,high],
            true_outside_range=(((y<low)|(y>high))&observed).tolist(),
            range_only_error_lower_bound=np.where(observed,np.abs(y-np.clip(y,low,high)),0).tolist(),
            inverse_valid=np.asarray([v['inverse_valid'] for v in samples]).all(0).tolist(),
            boundary=((centers==-.9999)|(centers==.9999)).tolist(),
            format_complete=all(v['format_complete'] for v in samples),empty_history=not base['present'],
            generated_token_count=sum(len(v['generated_ids']) for v in samples),
            numeric_slots=q.horizon,forward_calls=8*q.horizon if base['present'] else 0,
            mae=float(np.abs(p-y)[observed].mean()),active_seconds=sum(v['active_seconds'] for v in samples),
            seeds=[v['seed'] for v in samples],sampling=SETTINGS)
        atomic_json(root/f'{i:06d}.json',item)
        entries[i] = item
    result = dict(binding=binding,complete=True,queries=len(queries),active_seconds=seconds,
        errors=[e['mae'] for e in entries],metrics=summarize(entries),sampling=SETTINGS,
        numeric_slots_in_eight_paths=work_total,protocol='single_block_24_sample8_median_v1')
    atomic_json(root/'complete.json',result)
    progress.evaluation(len(queries),len(queries),work_total,work_total,seconds,evaluation_kind=label,evaluation_sample=8)
    return result
