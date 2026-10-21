"""Fixed generated inputs, independent true labels and source-role loss reduction."""
import hashlib
import json

import numpy as np
import torch
from torch.nn import functional as F

from gofa_f.allnode_graph import build_graph
from gofa_f.generative_data import Query
from gofa_f.generative_model import GenerativePrefixLRU
from gofa_f.robust_loss import parent_adapter
from gofa_f.short_training import compact_forward, teacher_prefix


def prefix_key(query):
    return hashlib.sha256(json.dumps(query, sort_keys=True).encode()).hexdigest()


def generated_record(codec, history, horizon, generated_ids, target=None):
    base = codec.record(history)
    ids, positions, expected = list(base['ids']), [], []
    cursor = 0
    for i in range(horizon):
        sep = codec.first_separator if i == 0 else codec.separator
        if list(generated_ids[cursor:cursor+len(sep)]) != list(sep):
            raise ValueError('Generated prefix separator differs')
        expected.extend(sep)
        cursor += len(sep)
        if cursor >= len(generated_ids) or generated_ids[cursor] not in codec.finite_set:
            raise ValueError('Generated prefix needs H finite numeric tokens')
        positions.append(len(ids)+len(expected))
        expected.append(int(generated_ids[cursor]))
        cursor += 1
    if cursor != len(generated_ids):
        raise ValueError('Unexpected generated suffix')
    ids.extend(expected)
    if target is not None:
        truth = codec.record(history, target)
        if truth['answer_positions'] != positions or len(truth['ids']) != len(ids):
            raise ValueError('Generated inputs and true labels must have equal grammar/coordinates')
        # True labels deliberately differ from the generated tokens in input_ids.
        truth['ids'] = ids
        return truth
    labels = [-100]*len(ids)
    for position in positions:
        labels[position] = ids[position]  # projection selector only; anchors never use CE
    base.update(ids=ids, labels=labels, answer_positions=positions)
    return base


def records(codec, series, events, prefixes):
    rows, graphs = [], []
    for event in events:
        q = Query(**event['query'])
        data = series[q.source]
        history = np.where(data.mask[q.origin-q.length:q.origin, q.variable],
                           data.values[q.origin-q.length:q.origin, q.variable], np.nan)
        target = None
        if event['kind'].startswith('weather_'):
            if q.source != 'boston2017':
                raise ValueError('Only weather may read future labels in training')
            target = np.where(data.mask[q.origin:q.origin+q.horizon, q.variable],
                              data.values[q.origin:q.origin+q.horizon, q.variable], np.nan)
        if event['kind'] == 'weather_true':
            row = codec.record(history, target)
        elif event['kind'] in ('weather_generated', 'keep'):
            saved = prefixes[prefix_key(event['query'])]
            if saved['query'] != event['query']:
                raise ValueError('Parent prefix/query mismatch')
            row = generated_record(codec, history, q.horizon, saved['generated_ids'], target)
        else:
            raise ValueError('Unknown fixed training role')
        rows.append(row)
        graphs.append(build_graph(data, q))
    return rows, graphs


def loss(result, parent_scores, events):
    scores, target, rows, counts = result
    if scores.shape != parent_scores.shape or parent_scores.requires_grad or not torch.isfinite(scores).all() \
            or not torch.isfinite(parent_scores).all() or (counts <= 0).any() \
            or len(counts) != len(events) or not torch.equal(torch.bincount(rows, minlength=len(events)), counts):
        raise ValueError('Aligned finite detached parent/observed slots required')
    logp, logt = scores.float().log_softmax(-1), parent_scores.float().detach().log_softmax(-1)
    cdf = logp.exp().cumsum(-1)[:, :-1]
    indicator = torch.arange(scores.shape[1]-1, device=scores.device)[None] >= target[:, None]
    slots = dict(ce=F.cross_entropy(scores.float(), target, reduction='none'),
                 cdf=(cdf-indicator.float()).square().mean(-1), kl=(logt.exp()*(logt-logp)).sum(-1))
    query = {k: scores.new_zeros(len(events), dtype=torch.float32).scatter_add(0, rows, v)/counts
             for k, v in slots.items()}
    total = scores.sum()*0
    parts = dict(weather_ce=0., weather_cdf=0., traffic_kl=0., wind_kl=0.)
    for i, event in enumerate(events):
        if event['kind'].startswith('weather_'):
            term = query['ce'][i]+query['cdf'][i]
            for key in ('ce', 'cdf'):
                parts['weather_'+key] += float(query[key][i].detach())*event['weight']
        elif event['kind'] == 'keep':
            term = query['kl'][i]
            key = 'traffic_kl' if event['query']['source'].startswith('pems04') else 'wind_kl'
            parts[key] += float(term.detach())*event['weight']
        else:
            raise ValueError('Unknown training role')
        total = total+event['weight']*term
    return total, parts


def batch(model, teacher, codec, series, events, prefixes, check=lambda: None, shared=True, telemetry=None):
    check()
    rec, graphs = records(codec, series, events, prefixes)
    tokens = codec.batch(rec).to(next(model.native.parameters()).device)
    cache = GenerativePrefixLRU('stage25_microbatch', max_bytes=2*2**30) if shared else None
    prefix = teacher_prefix(model, tokens, cache)
    with parent_adapter(model, teacher):
        contexts = model.encode_origins(graphs, codec, cache, check)
        parent = compact_forward(model, codec, tokens, contexts, prefix)[0].detach()
    del contexts
    contexts = model.encode_origins(graphs, codec, cache, check)
    current = compact_forward(model, codec, tokens, contexts, prefix)
    if current[0].shape[1] != 10000:
        raise ValueError('Full finite numeric alphabet required')
    value, parts = loss(current, parent, events)
    if telemetry is not None and cache is not None:
        telemetry['hits'] += cache.hits
        telemetry['misses'] += cache.misses
        telemetry['peak_bytes'] = max(telemetry['peak_bytes'], cache.peak)
    return value, int(current[3].sum()), parts
