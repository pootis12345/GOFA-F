"""Paired, source-only eight-sample diagnosis; production remains paused.

Stage19 BF16/prompt/immutable scale/finite grammar/single-block execution stay
fixed. Samples run serially so changing batch geometry cannot confound decoding.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
import yaml

from gofa_f.allnode_graph import build_graph
from gofa_f.allnode_stage import read, supervisor_alive
from gofa_f.generative_data import Query, SOURCES, Series, sha
from gofa_f.prefix_cache import atomic_json
from gofa_f.short_forecast import forecast_batch
from gofa_f.short_generative_stage import load_model
from gofa_f.short_native_case_probe import repetition
from gofa_f.short_source_diagnostic import compare_scores, prepare as source_prepare, verify_source


SETTINGS = dict(samples=8, top_k=100, temperature=1., top_p=1., seed=42,
                aggregate='pointwise_median', execution='serial_single_query_independent_kv')


def sample_seed(query, sample):
    # Model-independent seeds preserve paired draws across order and recovery.
    key = f'42:{Query(**query).key}:{sample}'.encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') % (2**63-1)


def choose(scores, generator):
    if not torch.isfinite(scores).all():
        raise FloatingPointError('Nonfinite numeric logits cannot be sampled or hidden by fallback')
    if generator is None:
        return scores.argmax(-1)
    values, indices = scores.topk(100, dim=-1, sorted=True)
    selected = torch.multinomial(values.softmax(-1), 1, generator=generator)
    return indices.gather(-1, selected).squeeze(-1)


@torch.no_grad()
def generate(model, codec, history, horizon, contexts=None, seed=None, cached=True,
             check=lambda: None, trace=None):
    """Same one-row forward/grammar as forecast_batch; only numeric choice changes."""
    if not 1 <= horizon <= 24 or not np.isfinite(history).any():
        raise ValueError('Pilot requires 1..24 slots and at least one historical observation')
    device = next(model.native.parameters()).device
    record = codec.record(history)
    tokens = codec.batch([record]).to(device)
    if codec.first_separator:
        tokens = tokens.append(torch.tensor(codec.first_separator, device=device)[None])
        tokens.input_ids = tokens.full_ids
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    alphabet, kv = codec.alphabet(device), None
    numbers, generated = [], list(codec.first_separator)
    endpoint_probability = []
    for slot in range(horizon):
        check()
        output = model(tokens, contexts=contexts, cache=kv, use_cache=cached)
        if kv is None:
            end = tokens.prompt_ends+len(codec.first_separator) if cached else \
                torch.full_like(tokens.prompt_ends, tokens.full_ids.shape[1]-1)
            scores = output.logits[torch.arange(1, device=device), end]
        else:
            scores = output.logits[:, -1]
        scores = scores.float().index_select(-1, alphabet)
        selected = choose(scores, generator)
        ids = alphabet[selected]
        endpoint_probability.append(scores.softmax(-1)[0, [0, -1]].tolist())
        if trace is not None:
            trace.append(scores.cpu())
        numbers.append(int(ids.item()))
        generated.append(int(ids.item()))
        if slot+1 < horizon:
            generated.extend(codec.separator)
            separator = torch.tensor(codec.separator, device=device)[None]
            tokens = tokens.append(torch.cat((ids[:, None], separator), 1))
            kv = output.past_key_values if cached else None
            if not cached:
                tokens.input_ids = tokens.full_ids
                tokens.labels = torch.full_like(tokens.full_ids, -100)
    prediction, valid, boundary = codec.decode_ids(numbers, *record['scaling'], record['last'])
    if not valid.all():
        raise FloatingPointError('Invalid inverse transform in fixed sampling pilot')
    return dict(numeric_ids=numbers, generated_ids=generated, prediction=prediction.tolist(),
        boundary=boundary.tolist(), endpoint_probability=endpoint_probability,
        repetition=repetition(prediction), seed=seed)


def measures(prediction, target, observed):
    prediction, target, observed = np.asarray(prediction, np.float64), np.asarray(target, np.float64), np.asarray(observed, bool)
    if not observed.any() or not np.isfinite(prediction).all():
        raise ValueError('Complete finite forecast and observed truth required')
    return dict(mae=float(np.abs(prediction-target)[observed].mean()),
                observed_targets=int(observed.sum()), repetition=repetition(prediction))


def prepare(cfg, output):
    if cfg['sampling'] != SETTINGS or list(cfg['checkpoints']) != ['1000']:
        raise ValueError('Fixed eight-sample / source-best-only protocol changed')
    manifest = source_prepare(cfg, output)
    source = Path(cfg['source_output'])
    if sha(source/'best.pt') != cfg['checkpoints']['1000']:
        raise ValueError('Actual source best differs from locked 1000-update candidate')
    references = {}
    for label in ('native', 'update-001000'):
        references[label] = {str(row['index']): sha(source/'evaluations'/label/f"{row['index']:06d}.json")
                             for row in manifest['queries']}
    manifest.update(reference_shards_sha256=references, source_best_sha256=sha(source/'best.pt'),
        epsilon_sha256=sha(source/'pools/metric_epsilons.json'), sampling_settings=SETTINGS,
        shared_seeds={str(row['index']): [sample_seed(row['query'], i) for i in range(8)]
                      for row in manifest['queries']},
        interpretation='32 ID-stratified queries; 2 known cases separate; no formal selection or target predictions')
    old = read(output/'sampling_manifest.json')
    if old is not None and old != manifest:
        raise ValueError('Sampling diagnostic binding changed')
    atomic_json(output/'sampling_manifest.json', manifest)
    return manifest


def summarize(records, epsilons):
    """Subset scores retain hierarchy but cannot replace full source validation."""
    primary = [r for r in records if r['group'] == 'id_stratified']
    if len(primary) != 64 or len({(r['model'], r['index']) for r in primary}) != 64:
        raise ValueError('Complete paired 32-query subset required')
    result = {'by_source': {}, 'subset_only_not_selection': True}
    for source in SOURCES:
        rows = [r for r in primary if r['query']['source'] == source]
        conditions = {}
        for model in ('native', '1000'):
            subset = [r for r in rows if r['model'] == model]
            for decoding in ('greedy', 'sample8'):
                items = [r[decoding] for r in subset]
                multistep = [i for i, r in zip(items, subset) if r['query']['horizon'] > 1]
                conditions[f'{model}_{decoding}'] = {
                    'query_mean_mae': float(np.mean([i['mae'] for i in items])),
                    'constant_multistep_queries': sum(i['repetition']['constant_complete'] for i in multistep),
                    'multistep_queries': len(multistep),
                    'upper_endpoint_fraction': sum(i['upper_slots'] for i in items)/sum(r['query']['horizon'] for r in subset)}
        ratios = {}
        for decoding in ('greedy', 'sample8'):
            by_variable = {}
            for row in rows:
                by_variable.setdefault(row['query']['variable'], {}).setdefault(row['model'], []).append(row[decoding]['mae'])
            ratios[decoding] = float(np.exp(np.mean([np.log(
                max(np.mean(v['1000']), epsilons[source][variable]) /
                max(np.mean(v['native']), epsilons[source][variable]))
                for variable, v in by_variable.items()])))
        result['by_source'][source] = dict(conditions=conditions, adapted_to_native=ratios)
    result['subset_S'] = {decoding: float(np.exp(np.mean([np.log(r['adapted_to_native'][decoding])
        for r in result['by_source'].values()]))) for decoding in ('greedy', 'sample8')}
    return result


def verify_inputs(cfg, manifest):
    source, base, _ = verify_source(cfg)
    if sha(source/'best.pt') != manifest['source_best_sha256'] or \
            sha(source/'manifest.json') != manifest['source_manifest_sha256'] or \
            sha(source/'pools/metric_epsilons.json') != manifest['epsilon_sha256']:
        raise ValueError('Paused source best/manifest/epsilon changed')
    for label, refs in manifest['reference_shards_sha256'].items():
        for index, expected in refs.items():
            if sha(source/'evaluations'/label/f'{int(index):06d}.json') != expected:
                raise ValueError('Fixed greedy reference changed')
    for name, expected in manifest['source_data_sha256'].items():
        if sha(Path(base['configuration']['data']['directory'])/'sources'/(name+'.npz')) != expected:
            raise ValueError('Source observations changed')
    return source, base


def run(cfg, output):
    if os.environ.get('GOFA_STAGE19_ADMITTED') != '1':
        raise RuntimeError('Sampling diagnostic requires fully-idle supervisor admission')
    proof, manifest = read(output/'cpu_preflight.json', {}), read(output/'sampling_manifest.json')
    if not proof.get('passed') or proof['manifest_sha256'] != sha(output/'sampling_manifest.json') or \
            manifest['configuration'] != cfg or manifest['code'] != \
            {p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}:
        raise ValueError('Code-bound CPU proof required')
    source, base = verify_inputs(cfg, manifest)
    stopped = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    def check():
        if stopped[0] or not supervisor_alive(os.environ['GOFA_STAGE19_SUPERVISOR_PID'],
                                              os.environ['GOFA_STAGE19_SUPERVISOR_TICKS']):
            raise InterruptedError('Cooperative sampling diagnostic pause')
    started, completed, records = time.monotonic(), 0, []
    binding = sha(output/'sampling_manifest.json')
    def status(state, **fields):
        seconds = time.monotonic()-started
        atomic_json(output/'status.json', dict(state=state, phase='sampling_diagnostic',
            done=completed, total=68, elapsed_worker_seconds=seconds,
            eta_seconds=seconds/completed*(68-completed) if completed else None,
            eta_excludes_unknown_wait=True, updated_at=time.time(), **fields))
    torch.cuda.set_per_process_memory_fraction(38*2**30/torch.cuda.get_device_properties(0).total_memory)
    model, codec = load_model(base['configuration'], torch.device('cuda:0'))
    model.adapter.requires_grad_(False)
    series = {name: Series(Path(base['configuration']['data']['directory'])/'sources'/(name+'.npz')) for name in SOURCES}
    try:
        for label in ('native', '1000'):
            check()
            if label != 'native':
                model.load_trainable(torch.load(source/'candidates/update-001000.pt', map_location='cpu', weights_only=False))
            for row in manifest['queries']:
                check()
                index, q = row['index'], Query(**row['query'])
                path = output/'queries'/f'{label}-{index:06d}.json'
                old = read(path)
                if old is not None:
                    if old['binding'] != binding:
                        raise ValueError('Resumed diagnostic shard binding changed')
                    records.append(old)
                    completed += 1
                    continue
                data = series[q.source]
                history = np.where(data.mask[q.origin-q.length:q.origin, q.variable],
                    data.values[q.origin-q.length:q.origin, q.variable], np.nan)
                y, observed = data.values[q.origin:q.origin+q.horizon, q.variable], data.mask[q.origin:q.origin+q.horizon, q.variable]
                context = None if label == 'native' else model.encode_origins([build_graph(data, q)], codec, check=check)
                legacy = forecast_batch(model, codec, [history], q.horizon, context, check=check)
                greedy = generate(model, codec, history, q.horizon, context, check=check)
                original_label = 'native' if label == 'native' else 'update-001000'
                reference = read(source/'evaluations'/original_label/f'{index:06d}.json')
                if greedy['generated_ids'] != legacy['generated_ids'][0] or \
                        greedy['generated_ids'] != reference['generated_ids'] or greedy['prediction'] != reference['prediction']:
                    raise ValueError('New decoder greedy path differs from existing production prediction')
                greedy.update(measures(greedy['prediction'], y, observed))
                greedy['upper_slots'] = sum(i == codec.finite_ids[-1] for i in greedy['numeric_ids'])
                samples = []
                for sample, seed in enumerate(manifest['shared_seeds'][str(index)]):
                    status('running', model=label, query_index=index, sample=sample+1)
                    trace = []
                    item = generate(model, codec, history, q.horizon, context, seed=seed, check=check, trace=trace)
                    item.update(measures(item['prediction'], y, observed))
                    if sample == 0 and (q.source, q.variable, q.length, q.horizon) in \
                            (('nn5', 92, 48, 24), ('pems04', 102, 48, 24)):
                        other_trace = []
                        other = generate(model, codec, history, q.horizon, context, seed=seed,
                                         cached=False, check=check, trace=other_trace)
                        comparison = compare_scores(trace, other_trace)
                        comparison['raw_tokens_equal'] = item['generated_ids'] == other['generated_ids']
                        if comparison['violations'] or not comparison['argmax_equal'] or not comparison['raw_tokens_equal']:
                            raise ValueError('Actual sampled full/incremental equivalence failed: '+str(comparison))
                        item['cache_comparison'] = comparison
                    samples.append(item)
                prediction = np.median(np.array([s['prediction'] for s in samples], np.float64), axis=0)
                centers = np.median(np.array([[codec.id_centers[i] for i in s['numeric_ids']] for s in samples]), axis=0)
                median = dict(prediction=prediction.tolist(), **measures(prediction, y, observed),
                    upper_slots=int((centers == .9999).sum()), lower_slots=int((centers == -.9999).sum()))
                record = dict(binding=binding, index=index, query=row['query'], group=row['group'], model=label,
                    target=y.tolist(), observed=observed.tolist(), greedy=greedy, samples=samples, sample8=median,
                    production_greedy_tokens_reproduced=True, no_training_updates=True)
                atomic_json(path, record)
                records.append(record)
                completed += 1
        verify_inputs(cfg, manifest)
        if any(p.grad is not None for p in model.parameters()):
            raise RuntimeError('Sampling diagnostic created gradients')
        summary = summarize(records, read(source/'pools/metric_epsilons.json'))
        atomic_json(output/'result.json', dict(complete=True, records=records, summary=summary,
            optimizer_updates=0, source_unchanged=True, targets_evaluated=0,
            allocated_bytes=torch.cuda.max_memory_allocated(), reserved_bytes=torch.cuda.max_memory_reserved(),
            interpretation='fixed small source subset, one seed set; neither formal selection nor proof of generalization'))
        status('complete')
    except InterruptedError as exc:
        status('resource_paused', reason=str(exc))
    except Exception as exc:
        atomic_json(output/'failures.json', dict(type=type(exc).__name__, error=str(exc), done=completed))
        status('failed', error=str(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    cfg, output = yaml.safe_load(Path(args.config).read_text()), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'queries').mkdir(exist_ok=True)
    if args.prepare:
        print(json.dumps(prepare(cfg, output)))
    else:
        run(cfg, output)


if __name__ == '__main__':
    main()
