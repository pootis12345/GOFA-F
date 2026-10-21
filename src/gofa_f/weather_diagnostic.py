"""Paired source-only teacher/parent-prefix/candidate-prefix diagnostic; no generation."""
from pathlib import Path

import numpy as np
import torch

from gofa_f.allnode_graph import build_graph
from gofa_f.allnode_stage import read, restore_rng, rng_state
from gofa_f.generative_data import Query, sha
from gofa_f.graph_benchmark import atomic
from gofa_f.robust_diagnostic import cdf_distance, selected_scores
from gofa_f.short_sampled_state import temporary_adapter
from gofa_f.short_stability_metrics import teacher_distribution
from gofa_f.weather_loss import generated_record


def run(model, codec, series, manifest, output, check, progress):
    output, old = Path(output), Path(manifest['configuration']['stage24_output'])
    prior = Path(manifest['configuration']['stage23_output'])
    root = output/'diagnostics'
    root.mkdir(exist_ok=True)
    binding = sha(output/'manifest.json')
    complete = read(root/'complete.json')
    if complete:
        if complete['binding'] != binding or not complete['complete']:
            raise ValueError('Diagnostic complete binding differs')
        return complete
    original = read(old/'manifest.json')
    models = dict(parent=output/'parent.pt', update100=old/'candidate-000100.pt')
    done, reused, rng = 0, 0, rng_state()
    try:
        for name, weights in models.items():
            with temporary_adapter(model, torch.load(weights, map_location='cpu', weights_only=False)):
                for i, query in enumerate(manifest['pools']['diagnostic']):
                    q, data = Query(**query), series[query['source']]
                    base_index = original['pools']['diagnostic'].index(query)
                    reusable = old/'diagnostics'/f'parent-{base_index:03d}.json'
                    view = [x for x in original['pools']['validation'] if x['source'] == q.source]
                    index = view.index(query)
                    history = np.where(data.mask[q.origin-q.length:q.origin, q.variable],
                                       data.values[q.origin-q.length:q.origin, q.variable], np.nan)
                    target = np.where(data.mask[q.origin:q.origin+q.horizon, q.variable],
                                      data.values[q.origin:q.origin+q.horizon, q.variable], np.nan)
                    for prefix in ('true', 'parent', 'update100'):
                        check()
                        path = root/f'{name}-{prefix}-{i:03d}.json'
                        identity = dict(binding=binding, model=name, prefix=prefix, query=query)
                        saved = read(path)
                        if saved:
                            if any(saved.get(k) != v for k, v in identity.items()):
                                raise ValueError('Diagnostic row binding differs')
                        elif name == 'parent' and prefix in ('true', 'parent'):
                            previous = read(reusable)
                            field = 'teacher' if prefix == 'true' else 'common_parent_prefix'
                            atomic(path, dict(**identity, metrics=previous[field],
                                reused_from_sha256=sha(reusable), no_new_forward=True))
                        else:
                            with torch.no_grad():
                                row = codec.record(history, target)
                                if prefix != 'true':
                                    source, label = (prior, 'validation-parent') if prefix == 'parent' else \
                                        (old, 'validation-update-000100')
                                    sample = read(source/'evaluations'/(label+'-'+q.source)/'trajectories'/f'{index:06d}-00.json')
                                    if sample['query'] != query:
                                        raise ValueError('Saved diagnostic prefix differs')
                                    row = generated_record(codec, history, q.horizon, sample['generated_ids'], target)
                                tokens = codec.batch([row]).to(next(model.native.parameters()).device)
                                contexts = model.encode_origins([build_graph(data, q)], codec, check=check)
                                positions = torch.tensor(row['answer_positions'], device=tokens.input_ids.device)
                                scores = selected_scores(model, codec, tokens, contexts, positions)
                                value = teacher_distribution(scores, codec, row, target)
                                value['cdf'] = cdf_distance(scores, target, codec, row)
                            atomic(path, dict(**identity, metrics=value, no_new_forward=False))
                        done += 1
                        reused += int(read(path)['no_new_forward'])
                        progress.tick('source_diagnostic', diagnostic_done=done, diagnostic_total=192, force=done % 16 == 0)
        value = dict(binding=binding, complete=True, records=done, reused_records=reused,
                     new_forwards=done-reused, new_free_generations=0, optimizer_updates=0,
                     interpretation='conditional diagnostic, not a forecast score or recipe selection')
        atomic(root/'complete.json', value)
        return value
    finally:
        restore_rng(rng)
