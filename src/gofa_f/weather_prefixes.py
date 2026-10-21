"""Atomic fixed-parent greedy inputs; training anchors never need future values."""
from pathlib import Path
import time

import numpy as np
import torch

from gofa_f.allnode_graph import build_graph
from gofa_f.allnode_stage import read, restore_rng, rng_state
from gofa_f.generative_data import Query, sha
from gofa_f.graph_benchmark import atomic
from gofa_f.short_forecast import forecast_batch
from gofa_f.weather_loss import generated_record, prefix_key


def prepare(model, codec, series, manifest, output, check, progress):
    root = Path(output)/'parent_prefixes'
    root.mkdir(exist_ok=True)
    binding = sha(Path(output)/'manifest.json')
    expected = [q for rows in manifest['pools']['train'].values() for q in rows]
    complete = read(root/'complete.json')
    if complete:
        if complete['binding'] != binding or complete['queries'] != len(expected) or not complete['complete'] \
                or set(complete['records']) != {prefix_key(q) for q in expected}:
            raise ValueError('Completed parent prefix plan changed')
        if any(not (root/(key+'.json')).exists() or sha(root/(key+'.json')) != digest
               for key, digest in complete['records'].items()):
            raise ValueError('Sealed parent prefix content changed')
    result, rng = {}, rng_state()
    try:
        with torch.no_grad():
            for i, query in enumerate(expected):
                check()
                key = prefix_key(query)
                path = root/(key+'.json')
                value = read(path)
                identity = dict(binding=binding, query=query, parent_sha256=manifest['configuration']['adapter_sha256'],
                                alphabet_sha256=codec.alphabet_sha, decoding='deterministic_greedy')
                if value is not None:
                    if any(value.get(k) != v for k, v in identity.items()):
                        raise ValueError('Parent prefix binding changed')
                else:
                    q, data = Query(**query), series[query['source']]
                    history = np.where(data.mask[q.origin-q.length:q.origin, q.variable],
                                       data.values[q.origin-q.length:q.origin, q.variable], np.nan)
                    started = time.monotonic()
                    contexts = model.encode_origins([build_graph(data, q)], codec, check=check)
                    generated = forecast_batch(model, codec, [history], q.horizon, contexts, check)
                    if not generated['format_complete'][0] or not generated['inverse_valid'][0].all():
                        raise ValueError('Fixed parent prefix must complete without fallback')
                    value = dict(**identity, generated_ids=generated['generated_ids'][0],
                                 numeric_ids=generated['numeric_ids'][0], active_seconds=time.monotonic()-started,
                                 future_values_read=False)
                    atomic(path, value)
                result[key] = value
                row = generated_record(codec, np.array([0.]), query['horizon'], value['generated_ids'])
                if [row['ids'][p] for p in row['answer_positions']] != value['numeric_ids']:
                    raise ValueError('Parent prefix numeric and serialized IDs differ')
                progress.tick('parent_prefixes', prefix_done=i+1, prefix_total=len(expected),
                              prefix_seconds=sum(v['active_seconds'] for v in result.values()), force=i % 16 == 0)
        atomic(root/'complete.json', dict(binding=binding, complete=True, queries=len(result),
            records={k: sha(root/(k+'.json')) for k in result},
            active_seconds=sum(v['active_seconds'] for v in result.values()), random_sampling=False))
        return result
    finally:
        restore_rng(rng)
