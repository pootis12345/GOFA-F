"""Read-only supervised sample8 inference, with separate pilot/confirm workers."""
import os
from pathlib import Path
import signal

import torch

from gofa_f.allnode_stage import Paused, read, supervisor_alive
from gofa_f.generative_data import Series, sha
from gofa_f.graph_benchmark_data import TARGETS
from gofa_f.graph_benchmark_run import target_path_acceptance
from gofa_f.prefix_cache import atomic_json
from gofa_f.short_generative_stage import load_model
from gofa_f.short_migration import assert_equal
from gofa_f.short_sampled_forecast import evaluate
from gofa_f.short_sampled_targets import mechanisms
from gofa_f.robust_wind_iteration import load_coefficients, confirm_dataset
from gofa_f.robust_wind_iteration import status
from gofa_f.six_calibration_state import immutable
from gofa_f.robust_wind_iteration_imports import import_confirmation
from gofa_f.robust_wind_iteration import inference_binding, verify_inference
from gofa_f.robust_wind_iteration_progress import Progress


@torch.no_grad()
def predict(cfg, output):
    if os.environ.get('GOFA_STAGE19_ADMITTED') != '1':
        raise RuntimeError('GPU inference requires cap-only supervised admission')
    manifest = verify_inference(cfg, output)
    binding = sha(output/'manifest.json')
    proof = read(output/'cpu_preflight.json', {})
    if not proof.get('passed') or proof['manifest_sha256'] != binding or not proof['cuda_disabled']:
        raise ValueError('Bound CPU proof required before GPU prediction')
    phase = 'confirm'
    _, parameters, _ = load_coefficients(cfg, output)
    import_confirmation(cfg, output, manifest, inference=True)
    if read(output/'status.json', {}).get('state') in ('complete', 'failed'):
        raise ValueError('Terminal Stage32 cannot automatically restart')
    stopped = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    def check():
        from gofa_f.robust_wind_iteration import check_time
        check_time()
        if stopped[0] or not supervisor_alive(os.environ['GOFA_STAGE19_SUPERVISOR_PID'],
                                              os.environ['GOFA_STAGE19_SUPERVISOR_TICKS']):
            raise Paused('cooperative_resource_or_user_pause')
    progress = Progress(output, manifest)
    memory = read(output/'cuda_memory.json', {})
    current = phase+'_restore'
    def save_memory():
        old = memory.setdefault(current, dict(allocated_bytes=0, reserved_bytes=0))
        old['allocated_bytes'] = max(old['allocated_bytes'], torch.cuda.max_memory_allocated())
        old['reserved_bytes'] = max(old['reserved_bytes'], torch.cuda.max_memory_reserved())
        atomic_json(output/'cuda_memory.json', memory)
    try:
        status(output, 'running', current)
        torch.cuda.set_per_process_memory_fraction(38*2**30/torch.cuda.get_device_properties(0).total_memory)
        model, codec = load_model(manifest['source_configuration'], torch.device('cuda:0'))
        adapter = torch.load(output/'adapter.pt', map_location='cpu', weights_only=False)
        model.load_trainable(adapter)
        model.parameter_version = 3500
        if codec.alphabet_sha != manifest['alphabet']['alphabet_sha256'] or not model.training \
                or model.native.training or not model.adapter.training \
                or any(p.requires_grad for p in model.native.parameters()) \
                or not all(p.requires_grad for p in model.adapter.parameters()):
            raise ValueError('Accepted source-validation execution flags/alphabet differ')
        assert_equal(adapter, model.trainable_state())
        current = 'source_replay_mechanisms'
        status(output, 'running', current)
        accepted = read(output/'mechanisms.json')
        if accepted is None:
            accepted = mechanisms(model, codec, manifest, check,
                audit=lambda v: atomic_json(output/'mechanism_diagnostics.json', v))
            accepted['binding'] = binding
            atomic_json(output/'mechanisms.json', accepted)
        if not accepted['passed'] or accepted['binding'] != binding or accepted['gradients_created']:
            raise ValueError('Four exact source replay cases must pass')
        save_memory()
        results = {}
        for name in TARGETS:
            data = {name: Series(Path(manifest['data_directory'])/'targets'/(name+'.npz'))}
            queries = manifest['queries'][name][64:128]
            for native in (True, False):
                if not native and parameters['coefficients'][name]['full']['weights'].get('gofa', 0.) == 0.:
                    continue
                mode = 'native' if native else 'gofa'
                current = f'{phase}-{name}-{mode}'
                check()
                status(output, 'running', current, dataset=name, model=mode)
                progress.tick(current, force=True, dataset=name, model=mode)
                torch.cuda.reset_peak_memory_stats()
                target_path_acceptance(model, codec, data[name], queries[0], native,
                    output, current, binding, check)
                result = evaluate(model, codec, data, queries, output, current,
                    inference_binding(manifest, binding, name, mode), progress, check, native)
                if not result['complete'] or result['queries'] != 64:
                    raise ValueError('Incomplete fixed prediction phase')
                atomic_json(output/(current+'-result.json'), result)
                progress.completed[current] = result
                progress.evaluation(64, 64, result['numeric_slots_in_eight_paths'],
                    result['numeric_slots_in_eight_paths'], result['active_seconds'],
                    evaluation_kind=current, evaluation_sample=8)
                results[current] = sha(output/(current+'-result.json'))
                assert_equal(adapter, model.trainable_state())
                if any(p.grad is not None for p in model.parameters()):
                    raise ValueError('Calibration inference created gradients')
                save_memory()
            check()
            accepted_dataset = confirm_dataset(cfg, output, name, inference=True)
            if not accepted_dataset['comparison']['strictly_improved']:
                save_memory()
                return  # No remaining dataset or test inference after a failed fixed check.
        verify_inference(cfg, output)
        immutable(output/f'{phase}_complete.json', dict(binding=binding, complete=True,
            results=results, model_queries=64*len(results),
            trajectories=512*len(results), optimizer_updates=0, target_supervised=True))
        status(output, phase+'_complete', phase, cpu_gate_pending=True)
    except Paused as exc:
        save_memory()
        status(output, 'resource_paused', current, reason=str(exc))
    except torch.cuda.OutOfMemoryError as exc:
        save_memory()
        attempts = read(output/'oom_attempts.json', {})
        key = current+':'+str(progress.state.get('evaluation_done'))+':'+str(progress.state.get('evaluation_sample'))
        attempts[key] = attempts.get(key, 0)+1
        atomic_json(output/'oom_attempts.json', attempts)
        if attempts[key] >= 3:
            atomic_json(output/'failures.json', dict(phase=current, error=str(exc), reason='repeated_identical_oom'))
            status(output, 'failed', current, reason='repeated_identical_oom')
            raise
        status(output, 'resource_paused', current, reason='cuda_oom; atomic paths retained')
    except Exception as exc:
        save_memory()
        atomic_json(output/'failures.json', dict(phase=current, type=type(exc).__name__, error=str(exc)))
        status(output, 'failed', current, error=str(exc))
        raise
