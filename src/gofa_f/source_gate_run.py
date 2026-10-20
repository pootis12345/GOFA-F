"""Stage22 frozen neural inference, source-only gate fit and conditional targets."""
from contextlib import nullcontext
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from gofa_f.allnode_graph import build_graph
from gofa_f.allnode_stage import Paused, supervisor_alive
from gofa_f.generative_data import Query, Series
from gofa_f.graph_benchmark import ADAPTER_SHA, atomic, read, sha
from gofa_f.short_forecast import summarize
from gofa_f.short_generative_stage import load_model
from gofa_f.short_migration import assert_equal
from gofa_f.short_sampled_forecast import evaluate
from gofa_f.short_sampled_targets import mechanisms
from gofa_f.short_sampling_diagnostic import generate, sample_seed
from gofa_f.short_source_diagnostic import compare_scores
from gofa_f.source_gate import AMPLITUDES, SOURCES, features, select, source_score
from gofa_f.source_gate_inference import amplitude, label
from gofa_f.source_gate_progress import Progress
from gofa_f.source_gate_state import TARGETS, cached_reference, verify
from gofa_f.source_reliability_gate import fit_gate, report


def immutable(path, item):
    if path.exists() and read(path) != item:
        raise ValueError('Immutable Stage22 record changed: '+str(path))
    atomic(path, item)


def history(data, q):
    mask = data.mask[q.origin-q.length:q.origin, q.variable]
    x = np.where(mask, data.values[q.origin-q.length:q.origin, q.variable], np.nan)
    return x, mask


def observation(data, query):
    q = Query(**query)
    graph = build_graph(data, q)
    x, mask = history(data, q)
    return features(x, mask, graph, q.length, q.horizon)


def locked_gate(output, binding):
    gate = read(output/'gate.json')
    cv = read(output/'source_cross_validation.json')
    if gate['binding'] != binding or gate['target_labels_used'] or cv['binding'] != binding \
            or not cv['validation']['passed'] or sha(output/'source_fit.json') != gate['fitting_records_sha256']:
        raise ValueError('Source-only frozen gate or cross-validation acceptance missing')
    return gate


def target_barrier(output, binding):
    gate = locked_gate(output, binding)
    confirmation = read(output/'source_confirmation.json')
    if confirmation['binding'] != binding or confirmation['gate_sha256'] != sha(output/'gate.json') \
            or not confirmation['validation']['passed'] or not confirmation['complete']:
        raise ValueError('Target labels/predictions blocked until both source checks pass')
    return gate


@torch.no_grad()
def path_check(model, codec, data, query, alpha, output, name, binding, check):
    path = output/'path_mechanisms'/(name+'.json')
    expected = dict(binding=binding, query=query, alpha=alpha)
    if path.exists():
        old = read(path)
        if not old['passed'] or any(old[k] != v for k, v in expected.items()):
            raise ValueError('Amplitude path acceptance binding changed')
        return
    q = Query(**query)
    x, _ = history(data, q)
    with amplitude(model, alpha) if alpha else nullcontext():
        context = model.encode_origins([build_graph(data, q)], codec, check=check) if alpha else None
        traces, other = [], []
        kwargs = dict(contexts=context, seed=sample_seed(query, 0), check=check)
        untraced = generate(model, codec, x, q.horizon, **kwargs)
        cached = generate(model, codec, x, q.horizon, trace=traces, **kwargs)
        full = generate(model, codec, x, q.horizon, cached=False, trace=other, **kwargs)
    comparison = compare_scores(traces, other, 2e-3)
    passed = not comparison['violations'] and untraced['generated_ids'] == cached['generated_ids'] == full['generated_ids']
    atomic(path, dict(**expected, passed=passed, comparison=comparison,
        cached=cached, full=full, untraced=untraced, rtol=2e-3, atol=2e-3))
    if not passed:
        raise ValueError('Nonzero amplitude full/incremental/sample token mismatch: '+name)


@torch.no_grad()
def run(cfg, output):
    if os.environ.get('GOFA_STAGE19_ADMITTED') != '1':
        raise RuntimeError('Stage22 requires fully-idle supervised admission')
    output = Path(output)
    manifest = verify(cfg, output)
    binding = sha(output/'manifest.json')
    proof = read(output/'cpu_preflight.json')
    if not proof['passed'] or proof['manifest_sha256'] != binding:
        raise ValueError('Bound complete CPU proof required')
    stopped = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    def check():
        if stopped[0] or not supervisor_alive(os.environ['GOFA_STAGE19_SUPERVISOR_PID'],
                                               os.environ['GOFA_STAGE19_SUPERVISOR_TICKS']):
            raise Paused('cooperative_resource_or_user_pause')
    progress = Progress(output)
    phase = 'restore_frozen_3500'
    memory = read(output/'cuda_memory.json') if (output/'cuda_memory.json').exists() else {}
    def status(state, **fields):
        atomic(output/'status.json', dict(state=state, phase=phase, updated_at=time.time(),
            selected_update=3500, neural_optimizer_updates=0, **fields))
    def save_memory():
        if torch.cuda.is_initialized():
            prior = memory.setdefault(phase, dict(allocated_bytes=0, reserved_bytes=0))
            prior['allocated_bytes'] = max(prior['allocated_bytes'], torch.cuda.max_memory_allocated())
            prior['reserved_bytes'] = max(prior['reserved_bytes'], torch.cuda.max_memory_reserved())
            atomic(output/'cuda_memory.json', memory)
    def enter(name):
        nonlocal phase
        save_memory()
        phase = name
        torch.cuda.reset_peak_memory_stats()
        status('running')
        progress.phase(name)
        check()
    def result_path(name, i):
        return output/'evaluations'/name/f'{i:06d}.json'
    def eval_action(series, queries, name, alpha):
        action_binding = binding+':'+name+':'+label(alpha)
        with amplitude(model, alpha) if alpha else nullcontext():
            result = evaluate(model, codec, series, queries, output, name, action_binding,
                progress, check, native=alpha == 0.)
        progress.evaluation(len(queries), len(queries), result['numeric_slots_in_eight_paths'],
            result['numeric_slots_in_eight_paths'], result['active_seconds'], evaluation_kind=name)
        audit_model()
        return [read(result_path(name, i)) for i in range(len(queries))]
    def audit_model():
        assert_equal(adapter, model.trainable_state())
        if any(p.grad is not None for p in model.parameters()) or torch.is_grad_enabled():
            raise ValueError('Frozen Stage22 unexpectedly created gradients')
        if sha(output/'adapter.pt') != ADAPTER_SHA:
            raise ValueError('Paused original checkpoint changed')
    def finish(reason):
        audit_model()
        save_memory()
        status('complete', stop_reason=reason, target_evaluation_started=(output/'target_decisions.json').exists())
        progress.phase('complete', stop_reason=reason, phase_eta_seconds=0., overall_eta_upper_bound_seconds=0.)
        report(output)
    try:
        status('running')
        torch.cuda.set_per_process_memory_fraction(38*2**30/torch.cuda.get_device_properties(0).total_memory)
        model, codec = load_model(manifest['source_configuration'], torch.device('cuda:0'))
        adapter = torch.load(output/'adapter.pt', map_location='cpu', weights_only=False)
        model.load_trainable(adapter)
        model.parameter_version = 3500
        if codec.alphabet_sha != manifest['alphabet']['alphabet_sha256'] or not model.training \
                or model.native.training or not model.adapter.training \
                or any(p.requires_grad for p in model.native.parameters()) \
                or not all(p.requires_grad for p in model.adapter.parameters()):
            raise ValueError('Source-validation numerical execution flags changed')
        audit_model()
        data_root = Path(manifest['source_configuration']['data']['directory'])
        sources = {s: Series(data_root/'sources'/(s+'.npz')) for s in SOURCES}
        enter('source_replay_and_amplitude_mechanisms')
        if not (output/'mechanisms.json').exists():
            evidence = mechanisms(model, codec, manifest, check,
                audit=lambda item: atomic(output/'mechanism_diagnostics.json', item))
            evidence['binding'] = binding
            atomic(output/'mechanisms.json', evidence)
        evidence = read(output/'mechanisms.json')
        if not evidence['passed'] or evidence['binding'] != binding or evidence['gradients_created']:
            raise ValueError('Original four source replays failed')
        for probe in manifest['probes']:
            q = probe['query']
            for a in (.25, .5):
                path_check(model, codec, sources[q['source']], q, a, output,
                    q['source']+'-'+str(q['variable'])+'-'+label(a), binding, check)
        enter('source_fit_action_forecasts')
        fit_queries = [r['query'] for r in manifest['pools']['fit']]
        actions = {a: eval_action(sources, fit_queries, 'source-fit-'+label(a), a) for a in (.25, .5)}
        records = []
        for row in manifest['pools']['fit']:
            q, i = row['query'], row['index']
            reference = manifest['source_references'][str(i)]
            pair = {}
            for a in (0., 1.):
                ref = reference[str(a)]
                for p in [ref]+ref['trajectories']:
                    if sha(p['path']) != p['sha256']:
                        raise ValueError('Reused source trajectory changed')
                pair[a] = read(ref['path'])
            for a in (.25, .5):
                pair[a] = actions[a][i]
                if pair[a]['scaling'] != pair[0.]['scaling'] or pair[a]['target'] != pair[0.]['target'] \
                        or pair[a]['observed'] != pair[0.]['observed'] or pair[a]['seeds'] != pair[0.]['seeds']:
                    raise ValueError('Amplitude comparison scales/labels/seeds differ')
            extent = 1./abs(pair[0.]['scaling'][0])
            errors = [float(pair[a]['mae']) for a in AMPLITUDES]
            records.append(dict(query=q, features=observation(sources[q['source']], q),
                errors=errors, origin_range=extent, risks=[(e-errors[0])/extent for e in errors[1:]],
                original_references=reference))
        immutable(output/'source_fit.json', dict(binding=binding, complete=True, records=records,
            action_order=list(AMPLITUDES), target_labels_used=False))
        progress.phase('cpu_gate_fit', reused_queries=256)
        enter('cpu_gate_fit')
        cv = fit_gate(cfg, output)
        if not cv['validation']['passed']:
            finish('source_cross_validation_rejected')
            return
        gate = locked_gate(output, binding)
        gate_sha = sha(output/'gate.json')
        confirmation_rows = manifest['pools']['confirmation']
        decisions = [dict(query=r['query'], **select(gate, observation(sources[r['query']['source']], r['query'])))
                     for r in confirmation_rows]
        immutable(output/'source_confirmation_decisions.json', dict(binding=binding, gate_sha256=gate_sha,
            decisions=decisions, chosen_before_confirmation_predictions=True))
        enter('source_time_confirmation')
        queries = [r['query'] for r in confirmation_rows]
        native = eval_action(sources, queries, 'source-confirmation-native', 0.)
        adapted = {}
        for a in (.25, .5, 1.):
            indices = [i for i, choice in enumerate(decisions) if choice['alpha'] == a]
            if indices:
                values = eval_action(sources, [queries[i] for i in indices], 'source-confirmation-'+label(a), a)
                adapted.update(zip(indices, values))
        confirmation = []
        for i, q in enumerate(queries):
            record = adapted.get(i, native[i])
            confirmation.append(dict(query=q, native_mae=native[i]['mae'], gated_mae=record['mae'],
                choice=decisions[i], native=native[i], gate=record))
        eps = read(Path(cfg['source_output'])/'pools/metric_epsilons.json')
        confirmation = dict(binding=binding, gate_sha256=gate_sha, complete=True, records=confirmation,
            validation=source_score(confirmation, eps))
        immutable(output/'source_confirmation.json', confirmation)
        if not confirmation['validation']['passed']:
            finish('source_time_confirmation_rejected')
            return
        gate = target_barrier(output, binding)
        enter('target_history_only_decisions')
        target_root = Path(manifest['target_data_directory'])/'targets'
        target_decisions = {}
        for name in TARGETS:
            data = Series(target_root/(name+'.npz'))
            target_decisions[name] = [dict(query=q, **select(gate, observation(data, q)))
                                      for q in manifest['target_queries'][name]]
        immutable(output/'target_decisions.json', dict(binding=binding, gate_sha256=gate_sha,
            decisions=target_decisions, all_actions_sealed_before_target_outcome_access=True))
        results = {}
        reused_queries = 256
        for name in TARGETS:
            target_barrier(output, binding)
            enter('target-'+name)
            data = Series(target_root/(name+'.npz'))
            queries = manifest['target_queries'][name]
            decisions = target_decisions[name]
            adapted = {}
            for a in (.25, .5):
                indices = [i for i, choice in enumerate(decisions) if choice['alpha'] == a]
                if indices:
                    path_check(model, codec, data, queries[indices[0]], a, output,
                        name+'-'+label(a), binding+':'+gate_sha, check)
                    values = eval_action({name: data}, [queries[i] for i in indices], 'target-'+name+'-'+label(a), a)
                    adapted.update(zip(indices, values))
            entries, native_entries, original_entries = [], [], []
            for i, q in enumerate(queries):
                pair = {}
                for a, mode in ((0., 'native'), (1., 'gofa')):
                    _, value = cached_reference(cfg['stage21_output'], 'target-'+name+'-'+mode, i, q,
                        cfg['stage21_manifest_sha256'], manifest['weights_sha256'] if not a else ADAPTER_SHA,
                        codec.alphabet_sha)
                    pair[a] = value
                value = adapted.get(i, pair.get(decisions[i]['alpha']))
                if value is None:
                    raise ValueError('Selected target amplitude result missing')
                entries.append(value)
                native_entries.append(pair[0.])
                original_entries.append(pair[1.])
                reused_queries += int(decisions[i]['alpha'] in (0., 1.))
            # Existing summarize is the unchanged missing-aware macro/micro definition.
            metrics = {mode: summarize(rows) for mode, rows in
                (('native', native_entries), ('original_gofa', original_entries), ('gate', entries))}
            native_mae = metrics['native']['macro']['mae']
            gated_mae = metrics['gate']['macro']['mae']
            results[name] = dict(complete=True, queries=128, metrics=metrics,
                strictly_improved=bool(gated_mae < native_mae-max(1e-12, 1e-9*native_mae)),
                amplitude_counts={str(a): sum(d['alpha'] == a for d in decisions) for a in AMPLITUDES},
                choices=decisions)
            atomic(output/('target-'+name+'-result.json'), results[name])
            progress.phase(phase, reused_queries=reused_queries)
        atomic(output/'target_results.json', dict(binding=binding, gate_sha256=gate_sha, complete=True,
            datasets=results, all_six_strictly_improved=all(r['strictly_improved'] for r in results.values())))
        verify(cfg, output)
        finish('six_target_evaluation_complete')
    except Paused as exc:
        save_memory()
        status('resource_paused', reason=str(exc))
    except torch.cuda.OutOfMemoryError as exc:
        save_memory()
        # Keep atomically committed paths. Never repeat an identical OOM indefinitely.
        path = output/'oom_attempts.json'
        attempts = read(path) if path.exists() else {}
        attempts[phase] = attempts.get(phase, 0)+1
        atomic(path, attempts)
        if attempts[phase] < 3:
            status('resource_paused', reason='cuda_oom', attempts=attempts[phase])
        else:
            atomic(output/'failures.json', dict(type=type(exc).__name__, phase=phase, error=str(exc)))
            status('failed', error='Repeated OOM; no blind restart')
            raise
    except Exception as exc:
        save_memory()
        atomic(output/'failures.json', dict(type=type(exc).__name__, phase=phase, error=str(exc)))
        status('failed', error=str(exc))
        raise
