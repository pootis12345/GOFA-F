"""Stage24 full-alphabet CE + normalized discrete CRPS + fixed-parent KL."""
from contextlib import contextmanager

import numpy as np
import torch
from torch.nn import functional as F

from gofa_f.allnode_graph import build_graph
from gofa_f.allnode_stage import restore_rng, rng_state
from gofa_f.generative_model import GenerativePrefixLRU
from gofa_f.short_training import compact_forward, teacher_prefix


def distribution_loss(result, parent_scores):
    scores, targets, rows, counts = result
    if scores.ndim != 2 or scores.shape != parent_scores.shape or scores.shape[1] < 2 \
            or not torch.isfinite(scores).all() or not torch.isfinite(parent_scores).all() \
            or parent_scores.requires_grad or (counts <= 0).any():
        raise ValueError('Finite aligned detached teacher and observed query slots required')
    if len(targets) != len(scores) or (targets < 0).any() or (targets >= scores.shape[1]).any() \
            or len(rows) != len(scores) or not torch.equal(torch.bincount(rows, minlength=len(counts)), counts):
        raise ValueError('Observed labels/rows/counts differ')
    logp = scores.float().log_softmax(-1)
    parent_logp = parent_scores.detach().float().log_softmax(-1)
    cdf = logp.exp().cumsum(-1)[:, :-1]
    truth = torch.arange(scores.shape[1]-1, device=scores.device)[None] >= targets[:, None]
    slots = dict(ce=F.cross_entropy(scores.float(), targets, reduction='none'),
        cdf=(cdf-truth.float()).square().mean(-1),
        kl=(parent_logp.exp()*(parent_logp-logp)).sum(-1))
    parts = {n: (scores.new_zeros(len(counts), dtype=torch.float32).scatter_add(0, rows, v)/counts).sum()
             for n, v in slots.items()}
    return parts['ce']+parts['cdf']+.1*parts['kl'], parts


@contextmanager
def parent_adapter(model, teacher):
    """Reference swap only: never copy into live student parameters or gradients."""
    if model._active is not None or model._kernel_active:
        raise ValueError('Parent swap requires an inactive execution context')
    student, version, rng = model.adapter, model.parameter_version, rng_state()
    model.cache_generation += 1
    try:
        model.adapter = teacher
        with torch.no_grad():
            yield
    finally:
        model.adapter = student
        model.parameter_version = version
        model.cache_generation += 1
        restore_rng(rng)


def records(codec, series, queries):
    result, graphs = [], []
    for q in queries:
        data = series[q.source]
        history = np.where(data.mask[q.origin-q.length:q.origin, q.variable],
            data.values[q.origin-q.length:q.origin, q.variable], np.nan)
        target = np.where(data.mask[q.origin:q.origin+q.horizon, q.variable],
            data.values[q.origin:q.origin+q.horizon, q.variable], np.nan)
        result.append(codec.record(history, target))
        graphs.append(build_graph(data, q))
    return result, graphs


def batch(model, teacher, codec, series, queries, check=lambda: None, shared=True):
    check()
    rows, graphs = records(codec, series, queries)
    device = next(model.native.parameters()).device
    tokens = codec.batch(rows).to(device)
    # This cache is discarded at the end of each microbatch. No graph KV/logits
    # are cached. Teacher/student use exactly the same geometry and token IDs.
    cache = GenerativePrefixLRU('stage24_exact_microbatch', max_bytes=2*2**30) if shared else None
    prefix = teacher_prefix(model, tokens, cache)
    with parent_adapter(model, teacher):
        contexts = model.encode_origins(graphs, codec, cache, check)
        parent = compact_forward(model, codec, tokens, contexts, prefix)[0].detach()
    del contexts
    contexts = model.encode_origins(graphs, codec, cache, check)
    student = compact_forward(model, codec, tokens, contexts, prefix)
    if student[0].shape[1] != 10000:
        raise ValueError('Stage24 requires all 10000 finite numerical tokens')
    loss, parts = distribution_loss(student, parent)
    return loss, int(student[3].sum()), {n: float(v.detach()) for n, v in parts.items()}
