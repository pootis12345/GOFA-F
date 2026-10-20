"""Stage23 warm-start adapter training; unique source selection then one confirmation."""
from contextlib import contextmanager
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
from gofa_f.graph_benchmark import ADAPTER_SHA, atomic, read
from gofa_f.short_generative_stage import load_model
from gofa_f.short_migration import assert_equal
from gofa_f.short_sampled_forecast import evaluate
from gofa_f.short_sampled_state import atomic_torch, save, temporary_adapter
from gofa_f.short_sampled_targets import mechanisms as parent_replay
from gofa_f.short_source_diagnostic import self_only
from gofa_f.short_training import training_batch
from gofa_f.source_retraining_checks import mechanisms
from gofa_f.source_retraining_data import VIEWS
from gofa_f.source_retraining_policy import (commit_selection, confirmation_pass, initial_selection,
                                             scheduled, score)
from gofa_f.source_retraining_progress import Progress
from gofa_f.source_retraining_state import verify


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
        lr = 5e-5*min(1., (state['updates']+1)/50)
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
    state.update(updates=state['updates']+1, pending=0, pending_tokens=0, pending_loss=0.)
    progress.state['measured_updates'] += 1
    progress.tick(updates=state['updates'], pending_queries=0, loss=loss, learning_rate=lr,
                  gradient_norm=float(norm))


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
        raise RuntimeError('Stage23 requires fully-idle supervised admission')
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
    state = dict(updates=0, pending=0, pending_tokens=0, pending_loss=0., microbatch=2,
                 peak_memory={}, cache_totals=dict(hits=0, misses=0, peak_bytes=0))
    phase = 'load_parent3500_new_optimizer'
    def status(mode, **fields):
        atomic(output/'status.json', dict(state=mode, phase=phase, parent_updates=3500,
            new_updates=state['updates'], pending_queries=state['pending'], target_evaluations=0,
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
        optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=5e-5, weight_decay=1e-4)
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
        sources = {n: Series(Path(cfg['data']['directory'])/'sources'/(n+'.npz')) for n in VIEWS}
        pools = manifest['pools']
        cache = GenerativePrefixLRU(frozen_cache_binding(model, manifest['weights_sha256']), max_bytes=32*2**30)
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
                ram_limit_gib=32, disk_limit_gib=128, disk_enabled=False, prefix_layers=30,
                graph_kv_cross_update=False))
        def checkpoint():
            memory()
            save(output/'last.pt', model, optimizer, state, progress, ledger, binding)
        def enter(name):
            nonlocal phase
            memory()
            phase = name
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
        baseline = evaluation('validation-native', 'validation', output/'parent.pt', native=True)
        anchor = evaluation('validation-parent', 'validation', output/'parent.pt')
        if ledger is None:
            ledger = initial_selection(score(anchor['errors'], baseline['errors'], pools['validation'], pools['epsilons']),
                                       ADAPTER_SHA)
        checkpoint()
        last_save = time.monotonic()
        while not (output/'final_selection.json').exists():
            check()
            update = state['updates']
            if update and update % 250 == 0 and ledger['last_validated_update'] < update:
                if state['pending']:
                    raise ValueError('Partial batch at validation boundary')
                candidate = output/f'candidate-{update:06d}.pt'
                if not candidate.exists():
                    atomic_torch(candidate, model.trainable_state())
                assert_equal(model.trainable_state(), torch.load(candidate, map_location='cpu', weights_only=False))
                value = evaluation(f'validation-update-{update:06d}', 'validation', candidate)
                current = score(value['errors'], baseline['errors'], pools['validation'], pools['epsilons'])
                relative_parent = score(value['errors'], anchor['errors'], pools['validation'], pools['epsilons'])
                commit_selection(ledger, update, sha(candidate), current, relative_parent)
                atomic(output/'selection_progress.json', ledger)
                checkpoint()
            if update >= 1000 or update >= 500 and ledger['bad_validations'] >= 2:
                state['stop_reason'] = 'maximum1000_new_updates' if update >= 1000 else 'source_early_stopping'
                chosen = output/'parent.pt' if ledger['best_update'] == 0 else output/f'candidate-{ledger["best_update"]:06d}.pt'
                if sha(chosen) != ledger['best_sha256']:
                    raise ValueError('Unique selected adapter SHA changed')
                atomic(output/'final_selection.json', dict(binding=binding, path=str(chosen),
                    checkpoint_sha256=ledger['best_sha256'], parent_update=3500,
                    new_update=ledger['best_update'], score=ledger['best_score'],
                    new_adapter_improved=ledger['best_update'] != 0, stop_reason=state['stop_reason'],
                    full_validation_only=True, target_predictions=0))
                checkpoint()
                break
            enter('train') if phase != 'train' else None
            progress.tick(updates=state['updates'], pending_queries=state['pending'])
            if state['pending'] < 64:
                size = min(state['microbatch'], 64-state['pending'])
                queries = [scheduled(pools['train'], update*64+state['pending']+i) for i in range(size)]
                before = {n: cpu_copy(p.grad) for n, p in model.adapter.named_parameters()}
                rng = rng_state()
                try:
                    loss, count = training_batch(model, codec, sources, queries, cache, check)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite new-source CE')
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
            if state['pending'] == 64:
                apply_update(model, optimizer, state, progress)
                with (output/'training_history.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(parent_update=3500, new_update=state['updates'],
                        loss=progress.state['loss'], learning_rate=progress.state['learning_rate'],
                        gradient_norm=progress.state['gradient_norm'], at=time.time()))+'\n')
            if time.monotonic()-last_save >= 30 or state['updates'] % 250 == 0:
                checkpoint()
                last_save = time.monotonic()
        locked = read(output/'final_selection.json')
        selected = Path(locked['path'])
        if locked['binding'] != binding or sha(selected) != locked['checkpoint_sha256']:
            raise ValueError('Locked confirmation model changed')
        native_confirm = evaluation('confirmation-native', 'confirmation', output/'parent.pt', native=True)
        parent_confirm = evaluation('confirmation-parent', 'confirmation', output/'parent.pt')
        if locked['new_update']:
            selected_confirm = evaluation('confirmation-selected', 'confirmation', selected)
        else:
            selected_confirm = parent_confirm
        confirm_native = score(selected_confirm['errors'], native_confirm['errors'], pools['confirmation'], pools['epsilons'])
        confirm_parent = score(selected_confirm['errors'], parent_confirm['errors'], pools['confirmation'], pools['epsilons'])
        atomic(output/'source_confirmation.json', dict(binding=binding, checkpoint_sha256=sha(selected),
            complete=True, native_score=confirm_native, parent_score=confirm_parent,
            passed=confirmation_pass(confirm_native, confirm_parent), labels_used_for_selection=False))
        for label, path in [('parent', output/'parent.pt'), ('selected', selected)]:
            for graph_mode in ('real', 'self'):
                evaluation(f'diagnostic-{label}-{graph_mode}', 'diagnostic', path, self_graph=graph_mode == 'self')
        checkpoint()
        phase = 'complete'
        status('complete', stop_reason=state['stop_reason'], confirmation_passed=confirmation_pass(confirm_native, confirm_parent),
               selected_new_update=locked['new_update'])
        progress.tick('complete', force=True, overall_eta_seconds=0.)
        from gofa_f.source_retraining import report
        report(output)
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
