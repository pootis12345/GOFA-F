"""Stage24 warm-start adapter training; unique source selection then one confirmation."""
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch

from gofa_f.allnode_model import frozen_cache_binding
from gofa_f.allnode_stage import Paused, cpu_copy, restore_checkpoint, restore_rng, rng_state, supervisor_alive
from gofa_f.generative_data import Series, sha
from gofa_f.generative_model import GenerativePrefixLRU
from gofa_f.graph_benchmark import atomic, read
from gofa_f.short_generative_stage import load_model
from gofa_f.short_migration import assert_equal
from gofa_f.short_sampled_forecast import evaluate
from gofa_f.short_sampled_state import atomic_torch, save, temporary_adapter
from gofa_f.short_sampled_targets import mechanisms as parent_replay
from gofa_f.short_source_diagnostic import self_only
from gofa_f.robust_loss import batch as training_batch
from gofa_f.robust_checks import mechanisms
from gofa_f.source_retraining_data import VIEWS
from gofa_f.source_retraining_policy import scheduled, score
from gofa_f.robust_state import gate100, passed, select, validation_boundary
from gofa_f.robust_progress import Progress
from gofa_f.robust_state import verify


def apply_update(model, optimizer, state, progress):
    if state['pending'] != 64:
        raise ValueError('Effective batch must contain64 queries')
    adapter, moments, version = model.trainable_state(), cpu_copy(optimizer.state_dict()), model.parameter_version
    gradients = {n: cpu_copy(p.grad) for n, p in model.adapter.named_parameters()}
    before_rng = rng_state()
    try:
        for p in model.adapter.parameters():
            if p.grad is not None:
                p.grad.div_(64)
        norm = torch.nn.utils.clip_grad_norm_(model.adapter.parameters(), 1., error_if_nonfinite=True)
        lr = 2e-5*min(1., (state['updates']+1)/20)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.step()
    except Exception:
        model.load_trainable(adapter)
        model.parameter_version = version
        optimizer.state.clear()
        optimizer.load_state_dict(moments)
        for n, p in model.adapter.named_parameters():
            p.grad = gradients[n].to(p.device) if gradients[n] is not None else None
        restore_rng(before_rng)
        raise
    optimizer.zero_grad(set_to_none=True)
    model.mark_updated()
    loss = state['pending_loss']/64
    terms = {n: v/64 for n, v in state['pending_terms'].items()}
    state.update(updates=state['updates']+1, pending=0, pending_tokens=0, pending_loss=0.,
                 pending_terms=dict(ce=0., cdf=0., kl=0.))
    progress.state['measured_updates'] += 1
    progress.tick(updates=state['updates'], pending_queries=0, loss=loss, learning_rate=lr,
                  gradient_norm=float(norm), loss_terms=terms)


@contextmanager
def graph_control(enabled):
    """Only remove cross-node messages; retain selected nodes and shared LLM work."""
    import gofa_f.short_sampled_forecast as forecast
    original = forecast.build_graph
    try:
        if enabled:
            forecast.build_graph = lambda data, q: self_only(original(data, q))
        yield
    finally:
        forecast.build_graph = original


def run(cfg, output, resume=False):
    if os.environ.get('GOFA_STAGE19_ADMITTED') != '1':
        raise RuntimeError('Stage24 requires resource-supervised admission')
    output = Path(output)
    manifest = verify(cfg, output)
    binding = sha(output/'manifest.json')
    proof = read(output/'cpu_preflight.json')
    if not proof or not proof['passed'] or proof['manifest_sha256'] != binding:
        raise ValueError('Bound full CPU proof required')
    stopped = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    def check():
        if stopped[0] or not supervisor_alive(os.environ['GOFA_STAGE19_SUPERVISOR_PID'],
                                               os.environ['GOFA_STAGE19_SUPERVISOR_TICKS']):
            raise Paused('cooperative_resource_or_user_pause')
    model, optimizer, progress, ledger = None, None, None, None
    state = dict(updates=0, pending=0, pending_tokens=0, pending_loss=0., microbatch=2, pending_terms=dict(ce=0., cdf=0., kl=0.),
                 target_evaluations=0, peak_memory={}, cache_totals=dict(hits=0, misses=0, peak_bytes=0))
    phase = 'load_parent3500_new_optimizer'
    def status(mode, **fields):
        atomic(output/'status.json', dict(state=mode, phase=phase, parent_updates=3500,
            new_updates=state['updates'], pending_queries=state['pending'], target_evaluations=state['target_evaluations'],
            updated_at=time.time(), **fields))
    try:
        torch.cuda.set_per_process_memory_fraction(38*2**30/torch.cuda.get_device_properties(0).total_memory)
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        model, codec = load_model(manifest['source_configuration'], torch.device('cuda:0'))
        parent = torch.load(output/'parent.pt', map_location='cpu', weights_only=False)
        model.load_trainable(parent)
        model.parameter_version = 3500
        teacher = deepcopy(model.adapter).requires_grad_(False).train()
        optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=2e-5, weight_decay=1e-4)
        if not model.training or model.native.training or not model.adapter.training \
                or any(p.requires_grad for p in model.native.parameters()) \
                or not all(p.requires_grad and p.dtype == torch.float32 for p in model.adapter.parameters()):
            raise ValueError('Accepted training/source-validation execution flags changed')
        restored = None
        if (output/'last.pt').exists():
            if not resume:
                raise ValueError('Existing state requires explicit resume')
            saved = torch.load(output/'last.pt', map_location='cpu', weights_only=False)
            state, restored = restore_checkpoint(output/'last.pt', model, optimizer, binding)
            ledger = saved['sampling_ledger']
            assert_equal(model.trainable_state(), saved['adapter'])
            assert_equal(cpu_copy(optimizer.state_dict()), saved['optimizer'])
            assert_equal({n: cpu_copy(p.grad) for n, p in model.adapter.named_parameters()}, saved['gradients'])
            assert_equal(rng_state(), saved['rng'])
            del saved
        progress = Progress(output, restored)
        if not restored:
            prior_rates = [read(Path(cfg['stage23_output'])/(label+'-result.json')) for label in
                           ('validation-parent', 'validation-update-000500')]
            progress.state['evaluation_seconds_per_slot'] = [v['active_seconds']/v['numeric_slots_in_eight_paths']
                                                            for v in prior_rates]
            progress.state['eta_reference_costs_not_charged_to_new_activity'] = True
        sources = {n: Series(Path(cfg['data']['directory'])/'sources'/(n+'.npz')) for n in VIEWS}
        pools = manifest['pools']
        cache = GenerativePrefixLRU(frozen_cache_binding(model, manifest['weights_sha256']), max_bytes=2*2**30)
        cache_base = dict(state['cache_totals'])
        def memory():
            record = state['peak_memory'].setdefault(phase, dict(allocated_bytes=0, reserved_bytes=0))
            record['allocated_bytes'] = max(record['allocated_bytes'], torch.cuda.max_memory_allocated())
            record['reserved_bytes'] = max(record['reserved_bytes'], torch.cuda.max_memory_reserved())
            if record['reserved_bytes'] > 38*2**30:
                raise Paused('torch38GiB_cap')
            atomic(output/'cuda_memory.json', state['peak_memory'])
            state['cache_totals'] = dict(hits=cache_base['hits']+cache.hits, misses=cache_base['misses']+cache.misses,
                peak_bytes=max(cache_base['peak_bytes'], cache.peak))
            atomic(output/'cache.json', dict(**state['cache_totals'], current_bytes=cache.used,
                ram_limit_gib=2, disk_limit_gib=0, disk_enabled=False, prefix_layers=30,
                training_prefix_cache_scope='one_microbatch_teacher_student_only',
                graph_kv_cross_update=False))
        def checkpoint():
            memory()
            save(output/'last.pt', model, optimizer, state, progress, ledger, binding)
        def enter(name):
            nonlocal phase
            memory()
            phase = name
            torch.cuda.reset_peak_memory_stats()
            progress.tick(name, force=True)
            status('running')
            check()
        def evaluation(label, split, path, native=False, self_graph=False):
            enter(label)
            result_path = output/(label+'-result.json')
            digest = manifest['weights_sha256'] if native else sha(path)
            action_binding = binding+':sample8:'+label+':'+digest+':'+codec.alphabet_sha
            queries = pools[split]
            per_view, errors, durations, total_work = {}, [], 0., 0
            with temporary_adapter(model, None if native else torch.load(path, map_location='cpu', weights_only=False)):
                with graph_control(self_graph):
                    for name in VIEWS:
                        own = [q for q in queries if q['source'] == name]
                        progress.tick(current_view=name, current_view_queries=len(own))
                        result = evaluate(model, codec, sources, own, output, label+'-'+name,
                            action_binding+':'+name, progress, check, native)
                        per_view[name] = result
                        errors.extend(result['errors'])
                        durations += result['active_seconds']
                        total_work += result['numeric_slots_in_eight_paths']
            result = dict(binding=action_binding, complete=True, queries=len(queries), errors=errors,
                active_seconds=durations, numeric_slots_in_eight_paths=total_work, views=per_view,
                graph_control='self_only_same_sampled_nodes' if self_graph else 'published_graph',
                raw_units_not_pooled_across_views=True)
            if result_path.exists() and read(result_path) != result:
                raise ValueError('Complete source result changed')
            atomic(result_path, result)
            progress.completed(label, result)
            memory()
            return result
        enter('cuda_mechanisms')
        if not (output/'mechanisms.json').exists():
            if state['updates']:
                raise ValueError('Trained state without initial accepted CUDA mechanisms')
            with temporary_adapter(model, parent):
                with torch.no_grad():
                    replay = parent_replay(model, codec, manifest, check,
                        audit=lambda item: atomic(output/'parent_replay_diagnostics.json', item))
                acceptance = mechanisms(model, codec, sources, pools['validation'], 2e-3, output, check)
            atomic(output/'mechanisms.json', dict(passed=True, binding=binding, parent_replay=replay,
                                                 new_source_acceptance=acceptance))
        accepted = read(output/'mechanisms.json')
        if not accepted['passed'] or accepted['binding'] != binding:
            raise ValueError('CUDA acceptance binding changed')
        from gofa_f.robust_diagnostic import run as diagnose
        enter('source_diagnostic')
        diagnose(model, codec, sources, manifest, output, check, progress)
        baseline = read(Path(cfg['stage23_output'])/'validation-native-result.json')
        anchor = read(Path(cfg['stage23_output'])/'validation-parent-result.json')
        parent_reference = score(anchor['errors'], baseline['errors'], pools['validation'], pools['epsilons'])
        if ledger is None:
            ledger = dict(history=[], quick_done=False, parent_reference=parent_reference)
        def finish(reason):
            nonlocal phase
            state['stop_reason'] = reason
            checkpoint()
            phase = 'complete'
            status('complete', stop_reason=reason)
            progress.tick('complete', force=True)
            from gofa_f.robust_source_adaptation import report
            report(output)
        if state.get('stop_reason') and not (output/'final_selection.json').exists():
            finish(state['stop_reason'])
            return
        checkpoint()
        last_save = time.monotonic()
        while not (output/'final_selection.json').exists():
            check()
            update = state['updates']
            if validation_boundary(update, state['pending']):
                if state['pending']:
                    raise ValueError('Partial batch at validation boundary')
                candidate = output/f'candidate-{update:06d}.pt'
                if not candidate.exists():
                    atomic_torch(candidate, model.trainable_state())
                assert_equal(model.trainable_state(), torch.load(candidate, map_location='cpu', weights_only=False))
                if update == 50 and not ledger['quick_done']:
                    evaluation('quick-000050', 'quick', candidate)
                    ledger['quick_done'] = True
                    checkpoint()
                if update in (100, 250):
                    value = evaluation(f'validation-update-{update:06d}', 'validation', candidate)
                    current = score(value['errors'], baseline['errors'], pools['validation'], pools['epsilons'])
                    relative = score(value['errors'], anchor['errors'], pools['validation'], pools['epsilons'])
                    row = dict(update=update, sha256=sha(candidate), native_score=current, parent_score=relative)
                    existing = [x for x in ledger['history'] if x['update'] == update]
                    if existing and existing != [row]:
                        raise ValueError('Repeated full validation differs')
                    if not existing:
                        ledger['history'].append(row)
                    atomic(output/'selection_progress.json', ledger)
                    checkpoint()
                    if update == 100:
                        accepted = gate100(current, relative, parent_reference)
                        atomic(output/'update100_gate.json', dict(passed=accepted, binding=binding, **row))
                        if not accepted:
                            finish('failed_update100_source_gate')
                            return
                    if update == 250:
                        chosen = select(ledger['history'])
                        if chosen is None or not passed(chosen['native_score'], chosen['parent_score']):
                            finish('failed_final_source_gate')
                            return
                        path = output/f"candidate-{chosen['update']:06d}.pt"
                        if sha(path) != chosen['sha256']:
                            raise ValueError('Selected SHA differs')
                        atomic(output/'final_selection.json', dict(binding=binding, path=str(path),
                            checkpoint_sha256=chosen['sha256'], parent_update=3500, new_update=chosen['update'],
                            native_score=chosen['native_score'], parent_score=chosen['parent_score'],
                            full_validation_only=True, source_gate_passed=True))
                        checkpoint()
                        break
            enter('train') if phase != 'train' else None
            progress.tick(updates=state['updates'], pending_queries=state['pending'])
            size = min(state['microbatch'], 64-state['pending'])
            queries = [scheduled(pools['train'], update*64+state['pending']+i) for i in range(size)]
            before = {n: cpu_copy(p.grad) for n, p in model.adapter.named_parameters()}
            rng = rng_state()
            try:
                loss, count, terms = training_batch(model, teacher, codec, sources, queries, check)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite robust loss')
                loss.backward()
            except Exception as exc:
                for n, p in model.adapter.named_parameters():
                    p.grad = before[n].to(p.device) if before[n] is not None else None
                restore_rng(rng)
                if isinstance(exc, torch.cuda.OutOfMemoryError) and size == 2:
                    state['microbatch'] = 1
                    state.setdefault('microbatch_fallbacks', []).append(dict(update=update,
                        pending=state['pending'], reason='oom_complete_microtransaction_rollback', to=1))
                    torch.cuda.empty_cache()
                    continue
                raise
            state['pending'] += size
            state['pending_tokens'] += count
            state['pending_loss'] += float(loss.detach())
            for n, value in terms.items():
                state['pending_terms'][n] += value
            if state['pending'] == 64:
                apply_update(model, optimizer, state, progress)
                with (output/'training_history.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(parent_update=3500, new_update=state['updates'],
                        loss=progress.state['loss'], loss_terms=progress.state['loss_terms'],
                        learning_rate=progress.state['learning_rate'],
                        gradient_norm=progress.state['gradient_norm'], at=time.time()))+'\n')
            if time.monotonic()-last_save >= 30 or state['updates'] in (50, 100, 250):
                checkpoint()
                last_save = time.monotonic()
        locked = read(output/'final_selection.json')
        selected = Path(locked['path'])
        if locked['binding'] != binding or sha(selected) != locked['checkpoint_sha256']:
            raise ValueError('Locked confirmation model changed')
        native_confirm = evaluation('confirmation-native', 'confirmation', output/'parent.pt', native=True)
        parent_confirm = evaluation('confirmation-parent', 'confirmation', output/'parent.pt')
        selected_confirm = evaluation('confirmation-selected', 'confirmation', selected)
        confirm_native = score(selected_confirm['errors'], native_confirm['errors'], pools['confirmation'], pools['epsilons'])
        confirm_parent = score(selected_confirm['errors'], parent_confirm['errors'], pools['confirmation'], pools['epsilons'])
        accepted = passed(confirm_native, confirm_parent)
        atomic(output/'source_confirmation.json', dict(binding=binding, checkpoint_sha256=sha(selected),
            complete=True, native_score=confirm_native, parent_score=confirm_parent,
            passed=accepted, labels_used_for_selection=False))
        if not accepted:
            finish('failed_locked_source_confirmation')
            return
        from gofa_f.robust_targets import run as target_run
        enter('target_evaluation')
        with temporary_adapter(model, torch.load(selected, map_location='cpu', weights_only=False)):
            targets = target_run(model, codec, manifest, output, progress, check)
        state['target_evaluations'] = len(targets)*128
        finish('complete_six_target_evaluation')
    except Paused as exc:
        if model is not None and optimizer is not None and progress is not None:
            save(output/'last.pt', model, optimizer, state, progress, ledger, binding)
        status('resource_paused', reason=str(exc))
    except torch.cuda.OutOfMemoryError as exc:
        key = f'{phase}:{state["updates"]}:{state["pending"]}'
        attempts = state.setdefault('oom_attempts', {})
        attempts[key] = attempts.get(key, 0)+1
        if model is not None and optimizer is not None and progress is not None:
            save(output/'last.pt', model, optimizer, state, progress, ledger, binding)
        if attempts[key] < 3:
            status('resource_paused', reason='oom_transaction_preserved', attempts=attempts[key])
        else:
            atomic(output/'failures.json', dict(type=type(exc).__name__, error=str(exc), phase=phase))
            status('failed', error='Three identical OOMs after readmission; no blind restart')
            raise
    except Exception as exc:
        if model is not None and optimizer is not None and progress is not None:
            save(output/'last.pt', model, optimizer, state, progress, ledger, binding)
        atomic(output/'failures.json', dict(type=type(exc).__name__, error=str(exc), phase=phase))
        status('failed', error=str(exc))
        raise
