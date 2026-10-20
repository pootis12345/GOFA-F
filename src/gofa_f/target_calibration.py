"""Stage20: validation-only scalar fitting, then CPU replay of frozen test medians."""
import argparse
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
import yaml

from gofa_f.allnode_stage import Paused, read, supervisor_alive
from gofa_f.generative_data import Series, sha
from gofa_f.prefix_cache import atomic_json
from gofa_f.short_data import TARGETS
from gofa_f.short_generative_stage import load_model
from gofa_f.short_migration import assert_equal
from gofa_f.short_sampled_forecast import checked, evaluate
from gofa_f.short_sampled_targets import mechanisms
from gofa_f.short_sampling_diagnostic import SETTINGS, sample_seed
from gofa_f.short_target_progress import Progress
from gofa_f.target_calibration_math import calibrate_dataset, fused_metrics, improves, paired
from gofa_f.target_calibration_state import digest, own_manifest, prepare, result_binding, verify_inference


def cpu_only():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise RuntimeError('Calibration/test replay must run with CUDA disabled and no CUDA context')


def status(output, state, phase, **fields):
    atomic_json(output/'status.json', dict(state=state,phase=phase,updated_at=time.time(),
        selected_update=3500,optimizer_updates=0,strict_zero_shot=False,**fields))


def immutable(path, value):
    if read(path, value) != value:
        raise ValueError('Sealed artifact changed: '+str(path))
    atomic_json(path, value)


def validation_records(output, manifest):
    """Read only this run's validation records; never open inherited test data."""
    binding = sha(output/'manifest.json')
    complete = read(output/'validation_complete.json')
    if complete is None or complete['binding'] != binding or not complete['complete'] \
            or read(output/'activity.json', {}).get('worker') is not None:
        raise ValueError('Complete validation and exited GPU worker required before calibration')
    datasets, hashes = {}, {}
    for name, queries in manifest['queries'].items():
        datasets[name] = {}
        for mode in ('native','gofa'):
            label = f'validation-{name}-{mode}'
            expected = result_binding(manifest,binding,name,mode)
            result_path = output/(label+'-result.json')
            result = read(result_path)
            if sha(result_path) != complete['results'][label] or result['binding'] != expected \
                    or not result['complete'] or result['queries'] != 128 or result['sampling'] != SETTINGS \
                    or len(result['errors']) != 128 \
                    or result['numeric_slots_in_eight_paths'] != 8*sum(q['horizon'] for q in queries):
                raise ValueError('Incomplete or rebound validation result')
            hashes[str(result_path.relative_to(output))] = sha(result_path)
            root, entries = output/'evaluations'/label, []
            for i, query in enumerate(queries):
                path = root/f'{i:06d}.json'
                entry = checked(path,expected,query)
                if entry is None or entry['sampling'] != SETTINGS:
                    raise ValueError('Missing validation median query')
                samples = []
                for sample in range(8):
                    trajectory = root/'trajectories'/f'{i:06d}-{sample:02d}.json'
                    item = checked(trajectory,expected,query,seed=sample_seed(query,sample),sample=sample)
                    if item is None:
                        raise ValueError('Median query lacks eight complete sampled trajectories')
                    samples.append(item['prediction'])
                    hashes[str(trajectory.relative_to(output))] = sha(trajectory)
                if not np.array_equal(np.median(np.asarray(samples,np.float64),axis=0),entry['prediction']):
                    raise ValueError('Saved validation median differs from its eight trajectories')
                mask = np.asarray(entry['observed'],bool)
                error = float(np.abs(np.asarray(entry['prediction'],np.float64)-
                    np.asarray(entry['target'],np.float64))[mask].mean())
                if error != entry['mae'] or error != result['errors'][i]:
                    raise ValueError('Saved validation target/mask/error binding differs')
                hashes[str(path.relative_to(output))] = sha(path)
                entries.append(entry)
            datasets[name][mode] = entries
        paired(datasets[name]['native'],datasets[name]['gofa'])
    return datasets, digest(hashes)


def calibrate(cfg, output):
    cpu_only()
    manifest = own_manifest(cfg,output)
    datasets, inputs = validation_records(output,manifest)
    coefficients = {name: calibrate_dataset(pair['native'],pair['gofa']) for name,pair in datasets.items()}
    value = dict(binding=sha(output/'manifest.json'),validation_inputs_sha256=inputs,
        coefficients=coefficients,coefficient_count=9,optimizer_updates=0,
        graph_alpha=1.,refit_on_all_validation=False,test_records_used=False,strict_zero_shot=False)
    immutable(output/'coefficients.json',value)
    immutable(output/'coefficient_lock.json',dict(binding=value['binding'],
        coefficients_sha256=sha(output/'coefficients.json'),validation_inputs_sha256=inputs))
    if not (output/'test_replay.json').exists():
        status(output,'calibrated','cpu_calibration',coefficient_count=9)
    return value


def load_coefficients(cfg, output):
    manifest = own_manifest(cfg,output)
    value, lock = read(output/'coefficients.json'), read(output/'coefficient_lock.json')
    if value is None or lock is None or value['binding'] != sha(output/'manifest.json') \
            or lock['binding'] != value['binding'] or sha(output/'coefficients.json') != lock['coefficients_sha256'] \
            or set(value['coefficients']) != set(TARGETS) or value['coefficient_count'] != 9 \
            or value['test_records_used'] or value['strict_zero_shot']:
        raise ValueError('Nine immutable validation-only coefficients must be sealed before test access')
    return manifest, value, lock


def test_records(cfg, manifest, name, mode):
    """Only called after all nine validation coefficients have been sealed."""
    parent = Path(cfg['stage19_output'])
    label = f'target-{name}-{mode}'
    weight = manifest['weights_sha256'] if mode == 'native' else cfg['adapter_sha256']
    expected = f"{cfg['stage19_manifest_sha256']}:sample8:{label}:{weight}:"+manifest['alphabet']['alphabet_sha256']
    file = parent/(label+'-result.json')
    if sha(file) != manifest['test_result_hashes'][name][mode]:
        raise ValueError('Frozen original test result changed')
    result = read(file)
    queries = manifest['test_queries'][name]
    if not result['complete'] or result['binding'] != expected or result['sampling'] != SETTINGS \
            or result['queries'] != len(queries):
        raise ValueError('Original test model/query/sampling differs')
    entries = []
    for i, query in enumerate(queries):
        path = parent/'evaluations'/label/f'{i:06d}.json'
        if sha(path) != manifest['test_query_hashes'][name][mode][i]:
            raise ValueError('Frozen test prediction shard changed')
        entry = checked(path,expected,query)
        if entry is None or entry['sampling'] != SETTINGS \
                or entry['seeds'] != [sample_seed(query,s) for s in range(8)]:
            raise ValueError('Frozen median seed/sampling identity changed')
        entries.append(entry)
    return entries, result['metrics']


def replay_test(cfg, output):
    cpu_only()
    manifest, coefficients, lock = load_coefficients(cfg,output)
    # Even after a seal, changed validation inputs invalidate all coefficients.
    _, inputs = validation_records(output,manifest)
    if inputs != lock['validation_inputs_sha256']:
        raise ValueError('Validation changed after coefficient sealing')
    verify_inference(cfg,output)
    results = {}
    for name in TARGETS:
        native, native_metrics = test_records(cfg,manifest,name,'native')
        graph, graph_metrics = test_records(cfg,manifest,name,'gofa')
        coefficient = coefficients['coefficients'][name]['coefficient']
        fused = fused_metrics(native,graph,coefficient)
        baseline, current = native_metrics['macro']['mae'], fused['macro']['mae']
        value = dict(binding=coefficients['binding'],coefficient_seal_sha256=lock['coefficients_sha256'],
            coefficient=coefficient,queries=len(native),metrics=dict(native=native_metrics,gofa=graph_metrics,fused=fused),
            strictly_improved=improves(current,baseline),
            mae_ratio=current/baseline if baseline > 0 else None,
            test_gpu_predictions_generated=0,derived_prediction=True,strict_zero_shot=False)
        immutable(output/f'calibrated-{name}-result.json',value)
        results[name] = value
    value = dict(complete=True,binding=coefficients['binding'],
        coefficient_seal_sha256=lock['coefficients_sha256'],datasets=results,
        all_nine_strictly_improved=all(r['strictly_improved'] for r in results.values()),
        improved_datasets=sum(r['strictly_improved'] for r in results.values()),
        optimizer_updates=0,test_gpu_predictions_generated=0,strict_zero_shot=False,
        interpretation='target-validation calibrated system, not strict zero-shot; original test results already seen')
    immutable(output/'test_replay.json',value)
    status(output,'complete','cpu_test_replay',goal_achieved=value['all_nine_strictly_improved'],
        outcome='all_nine_improved' if value['all_nine_strictly_improved'] else 'not_all_nine_improved')
    return report(cfg,output)


def report(cfg, output):
    cpu_only()
    manifest = own_manifest(cfg,output)
    results = read(output/'test_replay.json')
    lock = read(output/'coefficient_lock.json')
    if results is not None:
        load_coefficients(cfg,output)
        if results['binding'] != sha(output/'manifest.json') \
                or results['coefficient_seal_sha256'] != lock['coefficients_sha256']:
            raise ValueError('Test replay binding differs from sealed coefficients')
        for name,value in results['datasets'].items():
            if read(output/f'calibrated-{name}-result.json') != value:
                raise ValueError('Completed fused result changed')
    value = dict(complete=results is not None,results=results,
        coefficients=read(output/'coefficients.json'),manifest_sha256=sha(output/'manifest.json'),
        selected_update=3500,adapter_sha256=cfg['adapter_sha256'],
        validation_coverage=manifest['coverage'],validation_queries=1152,validation_trajectories=18432,
        activity=read(output/'activity.json'),memory=read(output/'cuda_memory.json'),
        cpu_finalization=read(output/'cpu_finalization.json'),strict_zero_shot=False,
        optimizer_updates=0,hourly_followup_remains_paused=True,
        limitations=['target validation labels calibrate nine scalars; not strict zero-shot',
            'prior target test results already seen during development',
            'Electric/Traffic/PEMS08 validation covers only 128 selected variables each',
            'confirmation only 32 queries; no guarantee of test improvement',
            'original high-dimensional test has one origin per variable',
            'fused numeric output is not generated text; no new generation coverage claim',
            'Weather 48->24 custom; single seed; pretrained exposure not independently verifiable'])
    atomic_json(output/'report.json',value)
    return value


@torch.no_grad()
def predict_validation(cfg, output):
    if os.environ.get('GOFA_STAGE19_ADMITTED') != '1':
        raise RuntimeError('Fully-idle CPU supervisor admission required')
    manifest = verify_inference(cfg,output)
    binding = sha(output/'manifest.json')
    proof = read(output/'cpu_preflight.json',{})
    if not proof.get('passed') or proof['manifest_sha256'] != binding:
        raise ValueError('Bound CPU preflight required')
    stopped = [False]
    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    def check():
        if stopped[0] or not supervisor_alive(os.environ['GOFA_STAGE19_SUPERVISOR_PID'],
                                             os.environ['GOFA_STAGE19_SUPERVISOR_TICKS']):
            raise Paused('cooperative_resource_or_user_pause')
    progress = Progress(output,manifest['total_numeric_work'])
    phase, model = 'restore_frozen_3500', None
    memory = read(output/'cuda_memory.json',{})
    def save_memory():
        if model is not None:
            old = memory.setdefault(phase,dict(allocated_bytes=0,reserved_bytes=0))
            old['allocated_bytes'] = max(old['allocated_bytes'],torch.cuda.max_memory_allocated())
            old['reserved_bytes'] = max(old['reserved_bytes'],torch.cuda.max_memory_reserved())
            atomic_json(output/'cuda_memory.json',memory)
    try:
        status(output,'running',phase)
        torch.cuda.set_per_process_memory_fraction(38*2**30/torch.cuda.get_device_properties(0).total_memory)
        model, codec = load_model(manifest['source_configuration'],torch.device('cuda:0'))
        adapter = torch.load(output/'adapter.pt',map_location='cpu',weights_only=False)
        model.load_trainable(adapter)
        model.parameter_version = 3500
        if not model.training or model.native.training or not model.adapter.training \
                or any(p.requires_grad for p in model.native.parameters()) \
                or not all(p.requires_grad for p in model.adapter.parameters()) \
                or codec.alphabet_sha != manifest['alphabet']['alphabet_sha256']:
            raise ValueError('Accepted source execution flags/alphabet changed')
        assert_equal(adapter,model.trainable_state())
        atomic_json(output/'restore_audit.json',dict(passed=True,optimizer_updates=0,
            adapter_sha256=sha(output/'adapter.pt'),source_execution_flags_preserved=True,
            grad_enabled=torch.is_grad_enabled()))
        phase = 'source_replay_mechanisms'
        status(output,'running',phase)
        acceptance = read(output/'mechanisms.json')
        if acceptance is None:
            acceptance = mechanisms(model,codec,manifest,check,
                audit=lambda value:atomic_json(output/'mechanism_diagnostics.json',value))
            acceptance['binding'] = binding
            atomic_json(output/'mechanisms.json',acceptance)
        if not acceptance['passed'] or acceptance['binding'] != binding or acceptance['gradients_created']:
            raise ValueError('Source numerical replay acceptance differs')
        save_memory()
        root = Path(manifest['source_configuration']['data']['directory'])
        results = {}
        for name in TARGETS:
            series = {name: Series(root/'targets'/(name+'.npz'))}
            for native in (True,False):
                mode = 'native' if native else 'gofa'
                phase = f'validation-{name}-{mode}'
                check()
                torch.cuda.reset_peak_memory_stats()
                status(output,'running',phase,dataset=name,model=mode)
                progress.tick(phase,force=True,dataset=name,model=mode)
                result = evaluate(model,codec,series,manifest['queries'][name],output,phase,
                    result_binding(manifest,binding,name,mode),progress,check,native)
                progress.completed[phase] = result
                progress.evaluation(128,128,result['numeric_slots_in_eight_paths'],
                    result['numeric_slots_in_eight_paths'],result['active_seconds'],
                    evaluation_kind=phase,evaluation_sample=8)
                file = output/(phase+'-result.json')
                immutable(file,result)
                results[phase] = sha(file)
                assert_equal(adapter,model.trainable_state())
                if any(p.grad is not None for p in model.parameters()):
                    raise ValueError('Validation inference created gradients')
                save_memory()
            verify_inference(cfg,output)
        immutable(output/'validation_complete.json',dict(binding=binding,complete=True,
            results=results,queries_per_model=1152,trajectories=18432,optimizer_updates=0))
        progress.tick('validation_complete',force=True,completed_model_datasets=18)
        status(output,'validation_complete','validation_complete',requires_cpu_finalization=True)
    except Paused as exc:
        save_memory()
        status(output,'resource_paused',phase,reason=str(exc))
    except torch.cuda.OutOfMemoryError as exc:
        save_memory()
        attempts = read(output/'oom_attempts.json',{})
        key = phase+':'+str(progress.state.get('evaluation_done'))+':'+str(progress.state.get('evaluation_sample'))
        attempts[key] = attempts.get(key,0)+1
        atomic_json(output/'oom_attempts.json',attempts)
        if attempts[key] < 3:
            status(output,'resource_paused',phase,reason='cuda_oom; saved paths retained',attempts=attempts[key])
        else:
            atomic_json(output/'failures.json',dict(type=type(exc).__name__,phase=phase,error=str(exc)))
            status(output,'failed',phase,error='Three identical OOMs; no blind restart')
            raise
    except Exception as exc:
        save_memory()
        atomic_json(output/'failures.json',dict(type=type(exc).__name__,phase=phase,error=str(exc)))
        status(output,'failed',phase,error=str(exc))
        raise


def finish_cpu(cfg, output):
    cpu_only()
    started = time.monotonic()
    if read(output/'activity.json',{}).get('worker') is not None:
        raise ValueError('GPU worker must exit before CPU finalization')
    try:
        calibrate(cfg,output)
        replay_test(cfg,output)
        atomic_json(output/'cpu_finalization.json',dict(passed=True,cuda_initialized=torch.cuda.is_initialized(),
            seconds=time.monotonic()-started))
        report(cfg,output)
    except Exception as exc:
        atomic_json(output/'failures.json',dict(type=type(exc).__name__,phase='cpu_finalization',error=str(exc)))
        status(output,'failed','cpu_finalization',error=str(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--resume',action='store_true')
    modes = parser.add_mutually_exclusive_group()
    for name in ('prepare','predict-validation','calibrate','replay-test','report-only','finish-cpu'):
        modes.add_argument('--'+name,action='store_true')
    args = parser.parse_args()
    cfg, output = yaml.safe_load(Path(args.config).read_text()), Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    if args.prepare:
        cpu_only()
        prepare(cfg,output)
    elif args.calibrate:
        calibrate(cfg,output)
    elif args.replay_test:
        replay_test(cfg,output)
    elif args.report_only:
        report(cfg,output)
    elif args.finish_cpu:
        finish_cpu(cfg,output)
    else:
        predict_validation(cfg,output)


if __name__ == '__main__':
    main()
