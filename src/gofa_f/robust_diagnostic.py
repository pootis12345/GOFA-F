"""Source-only teacher/common-prefix diagnosis. No generation or optimizer steps."""
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import torch

from gofa_f.allnode_graph import build_graph
from gofa_f.allnode_stage import read, restore_rng, rng_state
from gofa_f.generative_data import Query, sha
from gofa_f.graph_benchmark import atomic
from gofa_f.robust_loss import records
from gofa_f.short_sampled_state import temporary_adapter
from gofa_f.short_stability_diagnostic import amplitude
from gofa_f.short_stability_metrics import teacher_distribution, quantiles
from gofa_f.source_retraining_data import VIEWS


@contextmanager
def observe_gates(model, telemetry):
    """Observe every node's internal gate without replacing any module output."""
    handles = []
    def hook(index):
        def observe(module, args, delta):
            hidden, graph = args
            with torch.no_grad(), torch.autocast(hidden.device.type, enabled=False):
                z = module.down(module.norm(hidden.float())).transpose(0, 1)
                gates = module.gate(z).sigmoid().squeeze(-1)
                valid = graph.observed
                telemetry.append(dict(layer=index+1, all_nodes=quantiles(gates[valid]),
                    target=quantiles(gates[:, 0][valid[:, 0]]),
                    by_node=[quantiles(gates[:, j][valid[:, j]]) for j in range(hidden.shape[0])]))
            return None
        return observe
    try:
        handles = [m.register_forward_hook(hook(i)) for i, m in enumerate(model.adapter.exchanges)]
        yield
    finally:
        for handle in handles:
            handle.remove()


def selected_scores(model, codec, tokens, contexts, positions):
    output = model(tokens, contexts=contexts)
    return output.logits[0].index_select(0, positions-1).float().index_select(-1, codec.alphabet(positions.device))


def cdf_distance(scores, target, codec, record):
    valid = torch.tensor([record['labels'][i] != -100 for i in record['answer_positions']], device=scores.device)
    labels = torch.tensor([record['labels'][i] for i in record['answer_positions']], device=scores.device)
    truth = codec.lookup(max(codec.finite_ids)+1, scores.device)[labels[valid]]
    cdf = scores[valid].float().softmax(-1).cumsum(-1)[:, :-1]
    return float((cdf-(torch.arange(9999, device=scores.device)[None] >= truth[:, None])).square().mean())


def run(model, codec, series, manifest, output, check, progress):
    output = Path(output)
    root = output/'diagnostics'
    root.mkdir(exist_ok=True)
    binding = sha(output/'manifest.json')
    done = read(root/'complete.json')
    if done:
        if done['binding'] != binding or not done['complete']:
            raise ValueError('Diagnostic binding changed')
        return done
    prior = Path(manifest['configuration']['stage23_output'])
    parent = torch.load(output/'parent.pt', map_location='cpu', weights_only=False)
    conditions = [('native', None), ('parent', parent)]+[(f'update{u}', torch.load(
        prior/f'candidate-{u:06d}.pt', map_location='cpu', weights_only=False)) for u in (250, 500)]
    original_rng = rng_state()
    rows = []
    for condition, weights in conditions:
        with temporary_adapter(model, weights):
            # Native is the original no-context route, not an alpha-zero approximation.
            for index, query in enumerate(manifest['pools']['diagnostic']):
                check()
                progress.tick('source_diagnostic', diagnostic_condition=condition, diagnostic_query=index,
                              diagnostic_total=64, force=index % 8 == 0)
                path = root/f'{condition}-{index:03d}.json'
                old = read(path)
                identity = dict(binding=binding, condition=condition, query=query)
                if old:
                    if any(old[k] != v for k, v in identity.items()):
                        raise ValueError('Diagnostic record identity differs')
                    rows.append(old)
                    continue
                q, data = Query(**query), series[query['source']]
                history = np.where(data.mask[q.origin-q.length:q.origin, q.variable],
                    data.values[q.origin-q.length:q.origin, q.variable], np.nan)
                target = np.where(data.mask[q.origin:q.origin+q.horizon, q.variable],
                    data.values[q.origin:q.origin+q.horizon, q.variable], np.nan)
                view = [x for x in manifest['pools']['validation'] if x['source'] == q.source]
                old_index = view.index(query)
                folder = prior/'evaluations'/('validation-parent-'+q.source)
                sample_path = folder/'trajectories'/f'{old_index:06d}-00.json'
                sampled = read(sample_path)
                prefix_ids = sampled['generated_ids']
                telemetry = []
                gates = []
                from contextlib import nullcontext
                with torch.no_grad(), (amplitude(model, 1., telemetry) if condition != 'native' else nullcontext()), \
                        (observe_gates(model, gates) if condition != 'native' else nullcontext()):
                    graphs = [build_graph(data, q)]
                    contexts = model.encode_origins(graphs, codec, check=check) if condition != 'native' else None
                    record = codec.record(history, target)
                    tokens = codec.batch([record]).to(next(model.native.parameters()).device)
                    positions = torch.tensor(record['answer_positions'], device=tokens.input_ids.device)
                    teacher_scores = selected_scores(model, codec, tokens, contexts, positions)
                    teacher = teacher_distribution(teacher_scores, codec, record, target)
                    teacher['cdf'] = cdf_distance(teacher_scores, target, codec, record)
                    # Copy exact saved token IDs, including canonical separators.
                    answer_start = record['prompt_end']+1
                    if len(prefix_ids) != len(record['ids'])-answer_start:
                        raise ValueError('Saved sample grammar differs from canonical teacher length')
                    common = dict(record, ids=record['ids'][:answer_start]+prefix_ids)
                    common_tokens = codec.batch([common]).to(tokens.input_ids.device)
                    scores = selected_scores(model, codec, common_tokens, contexts, positions)
                    common_result = teacher_distribution(scores, codec, record, target)
                    common_result['cdf'] = cdf_distance(scores, target, codec, record)
                value = dict(**identity, teacher=teacher, common_parent_prefix=common_result,
                    exchanges=telemetry, all_node_gates=gates, common_prefix_sha256=sha(sample_path),
                    free_prediction_reused=True, teacher_is_not_forecast=True)
                old_label = 'validation-native' if condition == 'native' else 'validation-parent' if condition == 'parent' \
                    else 'validation-update-000250' if condition == 'update250' else 'validation-update-000500'
                free_path = prior/'evaluations'/(old_label+'-'+q.source)/f'{old_index:06d}.json'
                free = read(free_path)
                if free['query'] != query:
                    raise ValueError('Existing free prediction differs from query')
                value.update(free_sample8=free, free_sample8_sha256=sha(free_path))
                atomic(path, value)
                rows.append(value)
    gradients = {}
    with temporary_adapter(model, parent):
        for name in VIEWS:
            check()
            saved = {n: p.grad.detach().clone() if p.grad is not None else None
                     for n, p in model.adapter.named_parameters()}
            model.zero_grad(set_to_none=True)
            from gofa_f.short_training import compact_forward, compact_loss
            try:
                for query in manifest['pools']['train'][name][:4]:
                    check()
                    rec, graphs = records(codec, series, [Query(**query)])
                    tokens = codec.batch(rec).to(next(model.native.parameters()).device)
                    context = model.encode_origins(graphs, codec, check=check)
                    (compact_loss(compact_forward(model, codec, tokens, context))[0]/4).backward()
                gradients[name] = torch.cat([p.grad.detach().float().flatten().cpu()
                                            for p in model.adapter.parameters()])
            finally:
                for n, p in model.adapter.named_parameters():
                    p.grad = saved[n]
    norms = {n: float(g.norm()) for n, g in gradients.items()}
    cosine = {a+'/'+b: float(torch.nn.functional.cosine_similarity(gradients[a], gradients[b], dim=0))
              for i, a in enumerate(VIEWS) for b in list(VIEWS)[i+1:]}
    restore_rng(original_rng)
    # The record files hold complete slot-level evidence, without the very large
    # full-vocabulary arrays; no future label is ever passed into graph creation.
    result = dict(binding=binding, complete=True, teacher_forwards=256, common_prefix_forwards=256,
        source_gradient_queries=16, gradient_norms=norms, gradient_cosines=cosine,
        optimizer_updates=0, coefficients_selected=False, new_free_trajectories=0,
        records=len(rows), limitations=['fixed small source sample; gradients are descriptive only'])
    atomic(root/'complete.json', result)
    return result
