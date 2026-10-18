"""Read-only graph-amplitude and probabilistic/numeric objective diagnosis.

Existing native/alpha1 trajectories are referenced with their original hashes.
Only alpha=.25 trajectories and three teacher-conditioned passes are new.
"""
import argparse
from contextlib import contextmanager, nullcontext
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
from gofa_f.generative_stage import verify_weights
from gofa_f.idle_launcher import process_identity
from gofa_f.prefix_cache import atomic_json
from gofa_f.short_generative_stage import load_model
from gofa_f.short_sampling_diagnostic import (SETTINGS, generate, measures, prepare as sampling_prepare,
                                             verify_inputs as sampling_verify)
from gofa_f.short_source_diagnostic import compare_scores, exchange_measurements
from gofa_f.short_stability_metrics import quantiles, summarize, teacher_distribution

CONDITIONS = ('native', 'alpha1', 'alpha025')
REPLAYS = (('nn5', 92, 48, 24), ('pems04', 102, 48, 24))


def stopped_prior(root):
    root = Path(root)
    launch = read(root.with_suffix('.launch.json'))
    if read(root/'status.json')['state'] != 'complete' or read(root/'launcher.json')['state'] != 'complete' \
            or read(root/'activity.json')['worker'] is not None:
        raise ValueError('Prior diagnostic must be complete and stopped')
    if process_identity(launch['pid']) == str(launch['start_ticks']):
        raise ValueError('Prior diagnostic supervisor still alive')
    for line in (root/'lifecycle.jsonl').read_text().splitlines():
        event = json.loads(line)
        if event.get('worker_start_ticks') and process_identity(event['worker_pid']) == str(event['worker_start_ticks']):
            raise ValueError('Prior diagnostic worker still alive')


def references(cfg, manifest):
    sample_root, teacher_root = Path(cfg['prior_sampling_output']), Path(cfg['prior_teacher_output'])
    for root in (sample_root, teacher_root):
        stopped_prior(root)
    previous = read(sample_root/'sampling_manifest.json')
    if not read(sample_root/'result.json')['complete'] or previous['queries'] != manifest['queries'] \
            or previous['sampling_settings'] != SETTINGS or previous['shared_seeds'] != manifest['shared_seeds'] \
            or previous['source_manifest_sha256'] != manifest['source_manifest_sha256'] \
            or previous['source_data_sha256'] != manifest['source_data_sha256'] \
            or previous['source_best_sha256'] != manifest['source_best_sha256']:
        raise ValueError('Previous sampling inputs/weights/queries/seeds differ')
    for name in ('short_sampling_diagnostic.py', 'short_native_case_probe.py', 'short_source_diagnostic.py'):
        if previous['code'][name] != manifest['code'][name]:
            raise ValueError('Reused decoder/observation code changed: '+name)
    files = {str(sample_root/'sampling_manifest.json'): sha(sample_root/'sampling_manifest.json'),
        str(sample_root/'result.json'): sha(sample_root/'result.json'),
        str(teacher_root/'diagnostic_manifest.json'): sha(teacher_root/'diagnostic_manifest.json')}
    old_teacher = read(teacher_root/'diagnostic_manifest.json')
    if old_teacher['queries'] != manifest['queries'] or old_teacher['source_manifest_sha256'] != manifest['source_manifest_sha256']:
        raise ValueError('Historical teacher queries/source binding differ')
    candidate = Path(cfg['source_output'])/'candidates/update-000500.pt'
    if sha(candidate) != cfg['historical_checkpoint_500_sha256']:
        raise ValueError('Historical 500-update checkpoint changed')
    for row in manifest['queries']:
        index = row['index']
        for model in ('native', '1000'):
            path = sample_root/'queries'/f'{model}-{index:06d}.json'
            item = read(path)
            if item['binding'] != sha(sample_root/'sampling_manifest.json') or item['query'] != row['query'] \
                    or item['group'] != row['group'] or [s['seed'] for s in item['samples']] != manifest['shared_seeds'][str(index)]:
                raise ValueError('Old sample shard/query/seed mismatch')
            files[str(path)] = sha(path)
        for prefix in ('native-native', '500-full', '1000-full'):
            path = teacher_root/'queries'/f'{prefix}-{index:06d}.json'
            item = read(path)
            if item['query'] != row['query'] or item['sampling_group'] != row['group']:
                raise ValueError('Historical teacher shard differs')
            files[str(path)] = sha(path)
    return files


def prepare(cfg, output):
    if cfg['protocol'] != 'stage19-stability-diagnostic-v1' or cfg['intervention_alpha'] != .25:
        raise ValueError('Only the preregistered quarter-amplitude diagnostic is permitted')
    manifest = sampling_prepare(cfg, output)
    source = Path(cfg['source_output'])
    base = read(source/'manifest.json')
    if verify_weights(base['configuration']) != base['weights_sha256']:
        raise ValueError('Official backbone content differs')
    root = Path(base['configuration']['weights']['directory'])
    weights = read(root/'snapshot.json')
    manifest.update(reused_files=references(cfg, manifest), weights_snapshot_sha256=sha(root/'snapshot.json'),
        weight_stats={str(root/n): dict(size=(root/n).stat().st_size, mtime_ns=(root/n).stat().st_mtime_ns)
                      for n in weights['files']}, conditions=list(CONDITIONS), intervention_alpha=.25,
        teacher_queries=102, new_sample_trajectories=272, no_training_resume=True,
        future_policy='training teacher forcing; routine validation and final test sample8; neither resumed here')
    old = read(output/'stability_manifest.json')
    if old is not None and old != manifest:
        raise ValueError('Stability diagnostic manifest changed')
    atomic_json(output/'stability_manifest.json', manifest)
    return manifest


def verify(cfg, manifest):
    source, base = sampling_verify(cfg, manifest)
    for path, expected in manifest['reused_files'].items():
        if sha(path) != expected:
            raise ValueError('Reused diagnostic evidence changed: '+path)
    root = Path(base['configuration']['weights']['directory'])
    if sha(root/'snapshot.json') != manifest['weights_snapshot_sha256']:
        raise ValueError('Backbone snapshot binding changed')
    for name, expected in manifest['weight_stats'].items():
        actual = Path(name).stat()
        if actual.st_size != expected['size'] or actual.st_mtime_ns != expected['mtime_ns']:
            raise ValueError('Backbone file changed since CPU content-hash verification')
    return source, base


@contextmanager
def amplitude(model, alpha, telemetry=None):
    """Scale exchange outputs once, before all-node layer execution/context copy."""
    if alpha not in (0., .25, 1.) or model._active is not None or model._kernel_active \
            or getattr(model, '_stability_scope', False):
        raise ValueError('Invalid or nested amplitude execution scope')
    handles = []
    model.cache_generation += 1
    model._stability_scope = True
    def make_hook(index):
        def hook(module, args, raw):
            scaled = raw*alpha
            if telemetry is not None:
                hidden, graph = args
                h, d, small = hidden[0].float(), raw[0].float(), scaled[0].float()
                valid = graph.observed[:, 0]
                hnorm = h.norm(dim=-1).clamp_min(1e-30)
                all_norm = hidden.float().norm(dim=-1).clamp_min(1e-30)
                all_valid = graph.observed.transpose(0, 1)
                telemetry.append(dict(layer=index+1, alpha=alpha,
                    raw=exchange_measurements(module, args, raw),
                    scaled_target_delta_to_hidden_norm=float(small[valid].norm()/h[valid].norm().clamp_min(1e-30)),
                    raw_target_position_ratios=quantiles((d.norm(dim=-1)/hnorm)[valid]),
                    scaled_target_position_ratios=quantiles((small.norm(dim=-1)/hnorm)[valid]),
                    scaled_all_node_position_ratios=quantiles((scaled.norm(dim=-1)/all_norm)[all_valid])))
            return scaled
        return hook
    try:
        for index, module in enumerate(model.adapter.exchanges):
            handles.append(module.register_forward_hook(make_hook(index)))
        yield
    finally:
        for handle in handles:
            handle.remove()
        model.cache_generation += 1
        model._stability_scope = False


@torch.no_grad()
def teacher(model, codec, history, target, contexts):
    record = codec.record(history, target)
    tokens = codec.batch([record]).to(next(model.native.parameters()).device)
    output = model(tokens, contexts=contexts)
    positions = torch.tensor(record['answer_positions'], device=tokens.input_ids.device)-1
    scores = output.logits[0].index_select(0, positions).float().index_select(-1, codec.alphabet(positions.device))
    return teacher_distribution(scores, codec, record, target)


def old_sample(cfg, row, condition):
    model = 'native' if condition == 'native' else '1000'
    path = Path(cfg['prior_sampling_output'])/'queries'/f"{model}-{row['index']:06d}.json"
    return read(path), dict(path=str(path), sha256=sha(path), condition=condition, reused=True)


def analyze_existing(cfg, output):
    manifest = read(output/'stability_manifest.json')
    verify(cfg, manifest)
    records = []
    for row in manifest['queries']:
        index = row['index']
        native_reference, _ = old_sample(cfg, row, 'native')
        for model in ('native', '500', '1000'):
            prefix = 'native-native' if model == 'native' else model+'-full'
            path = Path(cfg['prior_teacher_output'])/'queries'/f'{prefix}-{index:06d}.json'
            previous = read(path)
            free8 = None
            if model != '500':
                sampled, _ = old_sample(cfg, row, 'native' if model == 'native' else 'alpha1')
                free8 = sampled['sample8']
            records.append(dict(index=index, query=row['query'], group=row['group'], model=model,
                teacher=previous['teacher'], free_greedy=previous['free'], exchanges=previous['exchanges'],
                free_sample8=free8, target=native_reference['target'], observed=native_reference['observed'],
                provenance=dict(path=str(path), sha256=sha(path))))
    def errors(rows, key, part):
        values = []
        for r in rows:
            if r[key] is None:
                return None
            for i, (a, b, seen) in enumerate(zip(r[key]['prediction'], r['target'], r['observed'])):
                if seen and (part == 'all' or part == 'first' and i == 0 or part == 'later' and i > 0 or
                             r['query']['horizon'] == 24 and (part == 'h24_1_16' and i < 16 or part == 'h24_17_24' and i >= 16)):
                    values.append(abs(float(a)-float(b)))
        return dict(observations=len(values), mae=float(np.mean(values)) if values else None)
    def group(rows):
        return dict(queries=len(rows), teacher_ce=float(np.mean([r['teacher']['conditional_ce'] for r in rows])),
            teacher_argmax_mae=float(np.mean([r['teacher']['conditional_mae'] for r in rows])),
            greedy_query_mean_mae=float(np.mean([r['free_greedy']['mae'] for r in rows])),
            sample8_query_mean_mae=float(np.mean([r['free_sample8']['mae'] for r in rows])) if rows[0]['model'] != '500' else None,
            step_errors={key:{part:errors(rows, key, part) for part in ('all', 'first', 'later', 'h24_1_16', 'h24_17_24')}
                         for key in ('free_greedy', 'free_sample8')},
            graph_layers=[dict(layer=i+1,
                mean_delta_ratio=float(np.mean([r['exchanges'][i]['target_delta_to_hidden_norm'] for r in rows])),
                mean_gate=float(np.mean([r['exchanges'][i]['target_gate_mean'] for r in rows])))
                          for i in range(2)] if rows[0]['model'] != 'native' else [])
    by_source = {}
    for source in SOURCES:
        by_source[source] = {}
        for model in ('native', '500', '1000'):
            rows = [r for r in records if r['group'] == 'id_stratified' and
                    r['query']['source'] == source and r['model'] == model]
            value = group(rows)
            value['by_template'] = {f'{length}->{horizon}':group([r for r in rows if
                (r['query']['length'], r['query']['horizon']) == (length, horizon)])
                for length, horizon in sorted({(r['query']['length'], r['query']['horizon']) for r in rows})}
            by_source[source][model] = value
    result = dict(complete=True, records=records, by_source=by_source, gpu_activity_seconds=0,
                  missing_historical_full_distribution=True, no_selection=True)
    atomic_json(output/'existing_analysis.json', result)
    return result


def shard(path, binding, **identity):
    value = read(path)
    if value is not None and (value.get('binding') != binding or any(value.get(k) != v for k, v in identity.items())):
        raise ValueError('Resumed stability shard binding/identity changed')
    return value


def report(cfg, output):
    manifest = read(output/'stability_manifest.json')
    binding, rows, missing = sha(output/'stability_manifest.json'), [], []
    for condition in CONDITIONS:
        for row in manifest['queries']:
            index = row['index']
            record = shard(output/'queries'/f'{condition}-{index:06d}.json', binding,
                           condition=condition, index=index, query=row['query'])
            if record is None:
                missing.append(dict(condition=condition, index=index))
            else:
                rows.append(record)
    result = dict(complete=not missing, records=rows, missing=missing, optimizer_updates=0,
        targets_evaluated=0, no_selection=True, source_unchanged=True,
        interpretation='fixed source diagnostic, single seed set, no formal validation or architecture selection')
    if not missing:
        result['summary'] = summarize(rows)
    atomic_json(output/'report.json', result)
    return result


@torch.no_grad()
def run(cfg, output, resume=False):
    if os.environ.get('GOFA_STAGE19_ADMITTED') != '1':
        raise RuntimeError('Stability diagnostic requires fully-idle CPU supervisor admission')
    manifest, proof = read(output/'stability_manifest.json'), read(output/'cpu_preflight.json', {})
    binding = sha(output/'stability_manifest.json')
    if not proof.get('passed') or proof['manifest_sha256'] != binding or manifest['configuration'] != cfg \
            or manifest['code'] != {p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}:
        raise ValueError('Code-bound CPU proof required')
    if not resume and (any((output/'teachers').glob('*.json')) or any((output/'samples').glob('*.json'))):
        raise ValueError('Existing diagnostic shards require explicit resume')
    source, base = verify(cfg, manifest)
    stopped = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    def check():
        if stopped[0] or not supervisor_alive(os.environ['GOFA_STAGE19_SUPERVISOR_PID'],
                                              os.environ['GOFA_STAGE19_SUPERVISOR_TICKS']):
            raise InterruptedError('Cooperative stability diagnostic pause')
    started, done = time.monotonic(), [0]
    def status(state, **fields):
        seconds = time.monotonic()-started
        atomic_json(output/'status.json', dict(state=state, phase='stability_diagnostic', done=done[0], total=102,
            worker_elapsed_seconds=seconds, eta_seconds=seconds/done[0]*(102-done[0]) if done[0] else None,
            eta_excludes_unknown_wait=True, updated_at=time.time(), **fields))
    torch.cuda.set_per_process_memory_fraction(38*2**30/torch.cuda.get_device_properties(0).total_memory)
    model, codec = load_model(base['configuration'], torch.device('cuda:0'))
    model.adapter.requires_grad_(False)
    series = {n: Series(Path(base['configuration']['data']['directory'])/'sources'/(n+'.npz')) for n in SOURCES}
    try:
        for condition in CONDITIONS:
            check()
            if condition == 'alpha1':
                model.load_trainable(torch.load(source/'candidates/update-001000.pt', map_location='cpu', weights_only=False))
            for row in manifest['queries']:
                check()
                index, q = row['index'], Query(**row['query'])
                identity = dict(condition=condition, index=index, query=row['query'])
                path = output/'queries'/f'{condition}-{index:06d}.json'
                if shard(path, binding, **identity) is not None:
                    done[0] += 1
                    continue
                data = series[q.source]
                history = np.where(data.mask[q.origin-q.length:q.origin, q.variable],
                    data.values[q.origin-q.length:q.origin, q.variable], np.nan)
                y, observed = data.values[q.origin:q.origin+q.horizon, q.variable], data.mask[q.origin:q.origin+q.horizon, q.variable]
                target = np.where(observed, y, np.nan)
                telemetry = []
                scope = nullcontext() if condition == 'native' else amplitude(model, 1. if condition == 'alpha1' else .25, telemetry)
                with scope:
                    contexts = None if condition == 'native' else model.encode_origins([build_graph(data, q)], codec, check=check)
                    teacher_path = output/'teachers'/f'{condition}-{index:06d}.json'
                    saved_teacher = shard(teacher_path, binding, **identity)
                    if saved_teacher is None:
                        status('running', **identity, operation='teacher')
                        observed_teacher = teacher(model, codec, history, target, contexts)
                        saved_teacher = dict(binding=binding, **identity, teacher=observed_teacher, exchanges=telemetry)
                        atomic_json(teacher_path, saved_teacher)
                    observed_teacher = saved_teacher['teacher']
                    if condition != 'alpha025':
                        previous, provenance = old_sample(cfg, row, condition)
                        free = previous['sample8']
                        if (q.source, q.variable, q.length, q.horizon) in REPLAYS:
                            replay = generate(model, codec, history, q.horizon, contexts,
                                seed=manifest['shared_seeds'][str(index)][0], check=check)
                            if replay['generated_ids'] != previous['samples'][0]['generated_ids']:
                                raise ValueError('Original-amplitude replay changed actual source tokens')
                            observed_teacher['actual_sample_replay_tokens_equal'] = True
                    else:
                        samples = []
                        for i, seed in enumerate(manifest['shared_seeds'][str(index)]):
                            sample_path = output/'samples'/f'{index:06d}-{i:02d}.json'
                            item = shard(sample_path, binding, **identity, seed=seed)
                            if item is None:
                                status('running', **identity, operation='sample', sample=i+1)
                                trace = []
                                item = generate(model, codec, history, q.horizon, contexts, seed=seed, check=check, trace=trace)
                                item.update(binding=binding, **identity, **measures(item['prediction'], y, observed))
                                if i == 0 and (q.source, q.variable, q.length, q.horizon) in REPLAYS:
                                    other_trace = []
                                    other = generate(model, codec, history, q.horizon, contexts, seed=seed,
                                                     cached=False, check=check, trace=other_trace)
                                    comparison = compare_scores(trace, other_trace)
                                    comparison['raw_tokens_equal'] = item['generated_ids'] == other['generated_ids']
                                    if comparison['violations'] or not comparison['argmax_equal'] or not comparison['raw_tokens_equal']:
                                        raise ValueError('Quarter-amplitude full/KV equivalence failed: '+str(comparison))
                                    item['cache_comparison'] = comparison
                                atomic_json(sample_path, item)
                            samples.append(item)
                        prediction = np.median(np.array([s['prediction'] for s in samples], np.float64), axis=0)
                        centers = np.median(np.array([[codec.id_centers[t] for t in s['numeric_ids']] for s in samples]), axis=0)
                        free = dict(prediction=prediction.tolist(), **measures(prediction, y, observed),
                            upper_slots=int((centers == .9999).sum()), lower_slots=int((centers == -.9999).sum()))
                        provenance = dict(reused=False, sample_paths=[str(output/'samples'/f'{index:06d}-{i:02d}.json') for i in range(8)])
                    if 'lower_slots' not in free:
                        previous, _ = old_sample(cfg, row, condition)
                        centers = np.median(np.array([[codec.id_centers[t] for t in s['numeric_ids']] for s in previous['samples']]), axis=0)
                        free = dict(free, lower_slots=int((centers == -.9999).sum()))
                    result = dict(binding=binding, **identity, group=row['group'], target=y.tolist(), observed=observed.tolist(),
                        teacher=observed_teacher, free=free, exchanges=saved_teacher['exchanges'], provenance=provenance)
                    atomic_json(path, result)
                done[0] += 1
        verify(cfg, manifest)
        if any(p.grad is not None for p in model.parameters()):
            raise RuntimeError('Read-only diagnostic created gradients')
        result = report(cfg, output)
        if not result['complete']:
            raise RuntimeError('Cannot complete a partial diagnostic')
        result.update(allocated_bytes=torch.cuda.max_memory_allocated(), reserved_bytes=torch.cuda.max_memory_reserved())
        atomic_json(output/'result.json', result)
        status('complete')
    except InterruptedError as exc:
        status('resource_paused', reason=str(exc))
    except Exception as exc:
        atomic_json(output/'failures.json', dict(type=type(exc).__name__, error=str(exc), done=done[0]))
        status('failed', error=str(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    modes = parser.add_mutually_exclusive_group()
    for mode in ('prepare', 'analyze-existing', 'report-only'):
        modes.add_argument('--'+mode, action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cfg, output = yaml.safe_load(Path(args.config).read_text()), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for name in ('teachers', 'samples', 'queries'):
        (output/name).mkdir(exist_ok=True)
    if args.prepare:
        print(json.dumps(prepare(cfg, output)))
    elif args.analyze_existing:
        print(json.dumps(analyze_existing(cfg, output)))
    elif args.report_only:
        verify(cfg, read(output/'stability_manifest.json'))
        print(json.dumps(report(cfg, output)))
    else:
        run(cfg, output, args.resume)


if __name__ == '__main__':
    main()
