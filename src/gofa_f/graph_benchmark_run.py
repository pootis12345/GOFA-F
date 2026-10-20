"""Read-only Stage21 six-target sampling; accepted Stage19 numerical core unchanged."""
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from gofa_f.allnode_graph import build_graph
from gofa_f.allnode_stage import Paused, read, supervisor_alive
from gofa_f.generative_data import Query, Series, sha
from gofa_f.graph_benchmark_data import TARGETS
from gofa_f.graph_benchmark_progress import Progress
from gofa_f.graph_benchmark_state import verify
from gofa_f.prefix_cache import atomic_json
from gofa_f.short_generative_stage import load_model
from gofa_f.short_migration import assert_equal
from gofa_f.short_sampled_forecast import evaluate
from gofa_f.short_sampled_targets import mechanisms
from gofa_f.short_sampling_diagnostic import SETTINGS, generate, sample_seed
from gofa_f.short_source_diagnostic import compare_scores


@torch.no_grad()
def target_path_acceptance(model,codec,data,query,native,output,phase,binding,check):
    path=output/'target_mechanisms'/f'{phase}.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    old=read(path)
    if old is not None:
        if old['binding']!=binding or old['query']!=query or not old['passed']:
            raise ValueError('Target path acceptance binding changed')
        return
    q=Query(**query)
    history=np.where(data.mask[q.origin-q.length:q.origin,q.variable],
        data.values[q.origin-q.length:q.origin,q.variable],np.nan)
    if not np.isfinite(history).any():
        raise ValueError('First fixed query needs observed history for path acceptance; do not replace it')
    context=None if native else model.encode_origins([build_graph(data,q)],codec,check=check)
    a,b=[],[]
    kwargs=dict(contexts=context,seed=sample_seed(query,0),check=check)
    cached=generate(model,codec,history,q.horizon,trace=a,**kwargs)
    full=generate(model,codec,history,q.horizon,cached=False,trace=b,**kwargs)
    comparison=compare_scores(a,b,2e-3)
    passed=cached['generated_ids']==full['generated_ids'] and not comparison['violations']
    atomic_json(path,dict(binding=binding,query=query,passed=passed,comparison=comparison,
        cached=cached,full=full,rtol=2e-3,atol=2e-3,
        scope='first frozen query sample0, same model; extra mechanism work counted in activity'))
    if not passed:
        raise ValueError('New target full/incremental path mismatch: '+phase)


def report(output):
    manifest=read(output/'manifest.json')
    binding=sha(output/'manifest.json')
    result={}
    for name in TARGETS:
        pair={mode:read(output/f'target-{name}-{mode}-result.json') for mode in ('native','gofa')}
        for mode,value in pair.items():
            if value is None:
                continue
            digest=manifest['weights_sha256'] if mode=='native' else manifest['configuration']['adapter_sha256']
            expected=f'{binding}:sample8:target-{name}-{mode}:{digest}:'+manifest['alphabet']['alphabet_sha256']
            if value['binding']!=expected or value['sampling']!=SETTINGS or value['queries']!=128 \
                    or value['numeric_slots_in_eight_paths']!=8*sum(q['horizon'] for q in manifest['queries']['targets'][name]):
                raise ValueError('Mismatched target result identity/coverage')
        complete=all(v is not None and v.get('complete') and v['queries']==128 for v in pair.values())
        result[name]=dict(complete=complete,results=pair if complete else None,expected_queries=128)
    value=dict(complete=all(v['complete'] for v in result.values()),targets=result,
        selected_update=3500,adapter_sha256=manifest['configuration']['adapter_sha256'],
        manifest_sha256=binding,optimizer_updates=0,calibration_enabled=False,
        completed_datasets=sum(v['complete'] for v in result.values()),activity=read(output/'activity.json'),
        memory=read(output/'cuda_memory.json'),coverage=manifest['queries']['coverage'],
        limitations=['fixed small sample; high-dimensional targets only128 selected variables, one origin each',
            'no self-loop/shuffled-graph ablation: migration performance cannot isolate graph contribution',
            'BF16 constrained sample8 single-block protocol differs from official FP16 rolling protocol',
            'single seed; ChatTime pretraining exposure cannot be independently verified',
            'targets selected after earlier nine-target results were viewed; new suite is exploratory',
            'road support binarized; EngRAD is model-derived; publisher preprocessing disclosed',
            'no pooled raw-unit score across datasets; macro MAE averages sampled variables'])
    atomic_json(output/'report.json',value)
    return value


@torch.no_grad()
def run(cfg,output):
    if os.environ.get('GOFA_STAGE19_ADMITTED')!='1':
        raise RuntimeError('Target evaluation requires fully-idle supervised admission')
    manifest=verify(cfg,output)
    binding=sha(output/'manifest.json')
    proof=read(output/'cpu_preflight.json',{})
    if not proof.get('passed') or proof['manifest_sha256']!=binding:
        raise ValueError('Code/config/subset-bound CPU preflight required')
    stopped=[False]
    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    def check():
        if stopped[0] or not supervisor_alive(os.environ['GOFA_STAGE19_SUPERVISOR_PID'],os.environ['GOFA_STAGE19_SUPERVISOR_TICKS']):
            raise Paused('cooperative_resource_or_user_pause')
    progress=Progress(output,manifest['total_numeric_work'])
    phase='restore_frozen_3500'
    def status(state,**fields):
        atomic_json(output/'status.json',dict(state=state,phase=phase,updated_at=time.time(),
            selected_update=3500,optimizer_updates=0,**fields))
    memory=read(output/'cuda_memory.json',{})
    def save_memory():
        old=memory.setdefault(phase,dict(allocated_bytes=0,reserved_bytes=0))
        old['allocated_bytes']=max(old['allocated_bytes'],torch.cuda.max_memory_allocated())
        old['reserved_bytes']=max(old['reserved_bytes'],torch.cuda.max_memory_reserved())
        atomic_json(output/'cuda_memory.json',memory)
    try:
        status('running')
        torch.cuda.set_per_process_memory_fraction(38*2**30/torch.cuda.get_device_properties(0).total_memory)
        model,codec=load_model(manifest['source_configuration'],torch.device('cuda:0'))
        if codec.alphabet_sha!=manifest['alphabet']['alphabet_sha256']:
            raise ValueError('Evaluation numeric alphabet differs')
        adapter=torch.load(output/'adapter.pt',map_location='cpu',weights_only=False)
        model.load_trainable(adapter)
        model.parameter_version=3500
        # Preserve source-validation module/parameter flags. Changing them can
        # change FP32 graph kernel rounding and sampled BF16 trajectories. The
        # entire entry and both evaluators are no_grad; no optimizer exists.
        if not model.training or model.native.training or not model.adapter.training \
                or any(p.requires_grad for p in model.native.parameters()) \
                or not all(p.requires_grad for p in model.adapter.parameters()):
            raise ValueError('Source validation execution flags must be retained')
        assert_equal(adapter,model.trainable_state())
        atomic_json(output/'restore_audit.json',dict(passed=True,adapter_sha256=sha(output/'adapter.pt'),
            selected_update=3500,backbone_frozen=True,adapter_requires_grad_preserved=True,
            evaluation_grad_enabled=torch.is_grad_enabled(),optimizer_updates=0))
        phase='source_replay_mechanisms'
        status('running')
        old=read(output/'mechanisms.json')
        if old is None:
            old=mechanisms(model,codec,manifest,check,
                audit=lambda value:atomic_json(output/'mechanism_diagnostics.json',value))
            old['binding']=binding
            atomic_json(output/'mechanisms.json',old)
        if not old['passed'] or old['binding']!=binding or old['gradients_created']:
            raise ValueError('Source replay acceptance missing or changed')
        save_memory()
        root=Path(manifest['data_directory'])
        for name in TARGETS:
            data={name:Series(root/'targets'/(name+'.npz'))}
            for native in (True,False):
                mode='native' if native else 'gofa'
                phase=f'target-{name}-{mode}'
                torch.cuda.reset_peak_memory_stats()
                check()
                status('running',dataset=name,model=mode)
                progress.tick(phase,force=True,dataset=name,model=mode)
                weight_hash=manifest['weights_sha256'] if native else cfg['adapter_sha256']
                target_path_acceptance(model,codec,data[name],manifest['queries']['targets'][name][0],
                    native,output,phase,binding,check)
                result=evaluate(model,codec,data,manifest['queries']['targets'][name],output,phase,
                    binding+':sample8:'+phase+':'+weight_hash+':'+codec.alphabet_sha,progress,check,native)
                progress.completed[phase]=result
                progress.evaluation(result['queries'],result['queries'],result['numeric_slots_in_eight_paths'],
                    result['numeric_slots_in_eight_paths'],result['active_seconds'],evaluation_kind=phase,evaluation_sample=8)
                atomic_json(output/(phase+'-result.json'),result)
                assert_equal(adapter,model.trainable_state())
                if any(p.grad is not None for p in model.parameters()):
                    raise ValueError('Read-only target phase created gradients')
                save_memory()
                report(output)
            verify(cfg,output)
        assert_equal(adapter,model.trainable_state())
        if any(p.grad is not None for p in model.parameters()):
            raise ValueError('Read-only target evaluation created gradients')
        progress.tick('complete',force=True,completed_model_datasets=12)
        if not report(output)['complete']:
            raise ValueError('Paired target coverage is incomplete')
        status('complete',target_datasets_complete=6)
    except Paused as exc:
        status('resource_paused',reason=str(exc))
    except torch.cuda.OutOfMemoryError as exc:
        attempts=read(output/'oom_attempts.json',{})
        key=phase+':'+str(progress.state.get('evaluation_done'))+':'+str(progress.state.get('evaluation_sample'))
        attempts[key]=attempts.get(key,0)+1
        atomic_json(output/'oom_attempts.json',attempts)
        if attempts[key]<3:
            status('resource_paused',reason='cuda_oom; atomic paths retained',attempts=attempts[key])
        else:
            atomic_json(output/'failures.json',dict(type=type(exc).__name__,phase=phase,error=str(exc)))
            status('failed',error='Repeated identical OOM; no blind restart')
            raise
    except Exception as exc:
        save_memory()
        atomic_json(output/'failures.json',dict(type=type(exc).__name__,phase=phase,error=str(exc)))
        status('failed',error=str(exc))
        raise

