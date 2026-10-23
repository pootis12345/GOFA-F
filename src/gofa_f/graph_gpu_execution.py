"""State-preserving GPU execution; scientific CPU manifests remain immutable provenance."""
import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import torch

from gofa_f.dual_memory_fitting import fit,evaluate,read
from gofa_f.dual_memory_engines import model_digest
from gofa_f.dual_memory_policy import digest,SourceSchedule,summarize
from gofa_f.dual_memory_store import SourceStore
from gofa_f.generative_data import sha
from gofa_f.graph_gpu_engine import DeviceEngine
from gofa_f.idle_launcher import CapOnlyGate,process_identity,query_gpu
from gofa_f.short_launcher import own_usage
from gofa_f.prefix_cache import atomic_json
from gofa_f.ufg_shared import SharedUFGEngine
from gofa_f.ufg_shared_source import runtime as primary_runtime
from gofa_f.variable_graph_data import VariableStore,VariableSchedule,length_balanced_summary
from gofa_f.variable_graph_source import ExpertEngine,runtime as variable_runtime

CASES=('expert','primary_ufg','variable_ufg')
GPU_UUID='GPU-1c2c50c3-5cda-1ef1-f1f7-45682de07fd4'


def runtime():
    names=set(variable_runtime())|set(primary_runtime())|{'graph_gpu_engine.py','graph_gpu_execution.py','graph_gpu_qa.py','short_launcher.py','idle_launcher.py'}
    return {n:sha(Path(__file__).parent/n) for n in sorted(names)}


def paths(root,case):
    source=root/('primary' if case=='primary_ufg' else 'variable')
    work=source if case=='primary_ufg' else source/('expert' if case=='expert' else 'ufg')
    return source,work


def source_settings(root,case):
    source,work=paths(root,case);m=read(source/'manifest.json')
    cfg=m['shared_ufg_configuration'] if case=='primary_ufg' else m['variable_configuration']
    return source,work,m,cfg


def construct(root,case,device,base_only=False):
    source,work,m,cfg=source_settings(root,case)
    store=SourceStore(m,source/'source_inputs') if case=='primary_ufg' else VariableStore(m,source/'source_inputs')
    base=SharedUFGEngine(cfg['ufg_root'],m,store) if case=='primary_ufg' else ExpertEngine(cfg,m,store,'expert' if case=='expert' else 'ufg')
    return base if base_only else DeviceEngine(base,'primary' if case=='primary_ufg' else 'expert' if case=='expert' else 'ufg',device)


def prepare(root,variable,primary):
    if root.exists():raise ValueError('Preserve previous execution; do not overwrite')
    for origin,kind in ((variable,'expert'),(primary,'')):
        folder=origin/kind
        if read(folder/'activity.json')['worker'] is not None or read(folder/'status.json')['state']!='paused':raise ValueError('Cooperative CPU save and exit required')
        launch=Path(str(origin)+'.bootstrap.launch.json');identity=read(launch)
        if process_identity(identity['pid'])==identity['start_ticks']:raise ValueError('Old controller is still live')
    root.mkdir(parents=True);origins={};copies={}
    for label,origin in (('variable',variable),('primary',primary)):
        dest=root/label;dest.mkdir();shutil.copyfile(origin/'manifest.json',dest/'manifest.json')
        m=read(dest/'manifest.json')
        if m.get('target_labels_used'):raise ValueError('Source-only migration required')
        cfg=m['variable_configuration'] if label=='variable' else m['shared_ufg_configuration']
        if cfg['target_training_enabled']:raise ValueError('No target training')
        cache=origin/'source_inputs' if label=='variable' else Path(cfg['source_output'])/'source_inputs'
        if not read(cache/'index.json')['complete']:raise ValueError('Sealed shared source cache required')
        (dest/'source_inputs').symlink_to(cache.resolve(),target_is_directory=True)
        for name in ('target_plan.json','cpu_preflight.json','target_history_acceptance.json'):
            if (origin/name).exists():shutil.copyfile(origin/name,dest/name)
        old_fit=origin/'expert/all' if label=='variable' else origin/'all'
        new_fit=dest/'expert/all' if label=='variable' else dest/'all'
        new_fit.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(old_fit,new_fit)
        inherited=read(origin/'expert/activity.json' if label=='variable' else origin/'activity.json')
        atomic_json(new_fit.parent/'cpu_inheritance.json',dict(origin=str(origin),activity=inherited,origin_manifest_sha256=sha(origin/'manifest.json')))
        inherited_files={str(p.relative_to(old_fit)):sha(p) for p in old_fit.rglob('*') if p.is_file()}
        for name,value in inherited_files.items():
            if sha(new_fit/name)!=value:raise ValueError('State or evaluation bytes changed during migration')
        copies[label]=dict(files=inherited_files,all_bytes_identical=True,cache_index_sha256=sha(cache/'index.json'))
        origins[label]=str(origin)
    atomic_json(root/'execution_manifest.json',dict(protocol='stage66-gpu-state-migration-v1',origins=origins,
        runtime=runtime(),copied_state=copies,scientific_manifests_unchanged=True,original_cpu_resources_are_historical=True,
        active_resources=dict(device='cuda_fp32_complex64',admission='memory_cap_only',max_workers=2,gpu_uuid=GPU_UUID,
            project_cap_gib=40,aggregate_torch_cap_gib=38,torch_per_worker_cap_gib=18,audit_seconds=10,deadline=None),
        graph_builder='accepted CPU historical FFT/knn/Chebyshev; learned computation CUDA',
        state_transfer='weights/Adam/partial gradients/CPU RNG/cursor/selection preserved; CUDA RNG initialized seed42 on first device transfer',
        CPU_GPU_bitwise_future_trajectory_claimed=False,target_labels_used=False))
    for case in CASES:
        _,work=paths(root,case);work.mkdir(parents=True,exist_ok=True)
        atomic_json(work/'status.json',dict(state='prepared',case=case,goal_achieved=False,target_predictions=0))
    print(json.dumps(dict(prepared=str(root),execution_manifest_sha256=sha(root/'execution_manifest.json'))),flush=True)


def gpu_environment():
    if os.environ.get('GOFA_GPU_ADMITTED')!='1':raise ValueError('A verified memory-cap supervisor is required')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends,'mha'):torch.backends.mha.set_fastpath_enabled(False)
    torch.cuda.set_per_process_memory_fraction(18*1024**3/torch.cuda.get_device_properties(0).total_memory)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)


def worker(root,case):
    execution=read(root/'execution_manifest.json')
    if execution['runtime']!=runtime():raise ValueError('GPU execution bytes changed')
    accepted=read(root/'gpu_acceptance'/f'{case}.json',{})
    if not accepted.get('passed') or accepted['execution_manifest_sha256']!=sha(root/'execution_manifest.json'):
        raise ValueError('Case-specific GPU acceptance required')
    source,work,m,cfg=source_settings(root,case);lock=(work/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if read(work/'status.json',{}).get('state')=='complete':return
    if read(work/'status.json',{}).get('state')=='failed':raise ValueError('Audit failure; no blind restart')
    stopped=[False]
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    gpu_environment();started=time.monotonic();identity=dict(pid=os.getpid(),start_ticks=process_identity(os.getpid()))
    prior=read(work/'activity.json',{});old_gpu=prior.get('spent_gpu_residency_seconds',0.);visible={};engine=None
    inherited=read(work/'cpu_inheritance.json',{}).get('activity',{}).get('spent_cpu_residency_seconds',0.)
    def check():
        if stopped[0]:raise InterruptedError('Cooperative GPU state save')
    def activity(worker_identity):
        atomic_json(work/'activity.json',dict(worker=worker_identity,inherited_cpu_residency_seconds=inherited,
            spent_gpu_residency_seconds=old_gpu+time.monotonic()-started,cuda_initialized=True,threads=2,
            peak_torch_reserved_mib=torch.cuda.max_memory_reserved()/1024**2))
    def progress(**fields):
        visible.update(fields);atomic_json(work/'status.json',dict(state='running',case=case,execution_device='cuda_fp32',
            target_predictions=0,goal_achieved=False,**visible));activity(identity)
    try:
        engine=construct(root,case,'cuda');binding=sha(source/'manifest.json')
        fit_binding=digest([binding,'one_shared_ufg' if case=='primary_ufg' else 'expert' if case=='expert' else 'ufg','all'])
        options={} if case=='primary_ufg' else dict(schedule_factory=VariableSchedule,summarizer=length_balanced_summary)
        fit(engine,m,cfg,work/'all','all',None,fit_binding,check,progress,**options)
        selection=read(work/'all/selection.json');adequate=selection['stop_reason']=='source_early_stop'
        selected=dict(binding=binding,selection=selection,source_only=True,one_parameter_set=True,shape_specific_weights=False,
            checkpoint_path=str(work/'all/best.pt'),source_adequacy=adequate,target_labels_used=False,
            gpu_execution_manifest_sha256=sha(root/'execution_manifest.json'))
        if case=='primary_ufg':
            selected.update(training_shapes=[[12,12],[48,24]],adaptations=engine.model.provenance,
                global_optimum_not_claimed=True,upstream_effective_graph_core_preserved=True)
            atomic_json(source/'ufg_shared_lock.json',selected)
        else:
            atomic_json(work/'source_lock.json',selected);engine.store.confirmation=True
            rows=[q for d in m['queries'].values() for v in d.values() for q in v['confirmation']]
            engine.model.eval();tensor_sha=model_digest(engine.model)
            records=evaluate(engine,rows,'confirmation',work/'confirmation',binding,tensor_sha,check,progress)
            if model_digest(engine.model)!=tensor_sha:raise ValueError('Confirmation modified source weights')
            atomic_json(work/'confirmation.json',dict(binding=binding,model_sha256=tensor_sha,
                metrics=length_balanced_summary(records,m['training_unit_normalizers']),used_for_selection=False))
        atomic_json(work/'status.json',dict(state='complete',phase='source_locked',case=case,best_update=selection['best_update'],
            stopped_update=selection['stopped_update'],source_adequacy=adequate,target_predictions=0,goal_achieved=False,eta_seconds=None))
    except InterruptedError:
        atomic_json(work/'status.json',dict(state='paused',case=case,goal_achieved=False));raise
    except BaseException as exc:
        atomic_json(work/f'failure-{time.time_ns()}.json',dict(error=repr(exc)))
        atomic_json(work/'status.json',dict(state='failed',case=case,error=repr(exc),goal_achieved=False));raise
    finally:
        if engine is not None:engine.close()
        activity(None)


def supervise(root,snapshot,qa_case=None):
    lock=(root/'gpu_supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    project_lock=(root.parent/'stage19-gpu.lock').open('a');fcntl.flock(project_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if read(root/'execution_manifest.json')['runtime']!=runtime():raise ValueError('Immutable GPU runtime required')
    if not qa_case:
        for case in CASES:
            for folder in ('cpu_acceptance','gpu_acceptance'):
                proof=read(root/folder/f'{case}.json',{})
                if not proof.get('passed') or proof.get('execution_manifest_sha256')!=sha(root/'execution_manifest.json'):
                    raise ValueError('All case-specific CPU/GPU acceptances required before deployment')
    stopped=[False]
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    gate=CapOnlyGate();gate.uuid=GPU_UUID;active={};queue=[qa_case] if qa_case else list(CASES);done=[];last=time.monotonic();last_probe=0.
    ledger=read(root/('qa_activity.json' if qa_case else 'gpu_activity.json'),{});spent=ledger.get('project_gpu_union_seconds',0.);peak=ledger.get('peak_project_mib',0)
    def status(state,**fields):
        atomic_json(root/('qa_status.json' if qa_case else 'gpu_status.json'),dict(state=state,completed=done,queued=queue,
            running=list(active),goal_achieved=False,**fields))
        atomic_json(root/('qa_activity.json' if qa_case else 'gpu_activity.json'),dict(worker=dict(pid=os.getpid(),start_ticks=process_identity(os.getpid())),
            project_gpu_union_seconds=spent,peak_project_mib=peak,resource_audit_seconds=10,parallel_worker_limit=2,
            resource_policy='memory_cap_only',project_cap_gib=40,aggregate_torch_cap_gib=38))
    def stop_children():
        nonlocal spent,last
        for v in active.values():
            if process_identity(v['pid'])==v['ticks']:os.killpg(v['pid'],signal.SIGTERM)
        deadline=time.monotonic()+60
        while any(v['process'].poll() is None for v in active.values()) and time.monotonic()<deadline:
            now=time.monotonic();spent+=now-last;last=now;time.sleep(1)
        for v in active.values():
            if v['process'].poll() is None:raise RuntimeError('Cooperative save timeout; identity audit required')
    try:
        while queue or active:
            now=time.monotonic();spent+=now-last if active else 0.;last=now
            if stopped[0]:stop_children();status('paused_by_user');return
            gpu=None
            if now-last_probe>=10:
                try:gpu=query_gpu()
                except Exception:gpu=None
                last_probe=now;measurements=[own_usage(gpu,v['pid'],gate.uuid) for v in active.values()]
                total=None if any(v is None for v in measurements) else sum(measurements)
                if active and (total is None or total>40*1024):
                    stop_children();queue=list(active)+queue;active.clear();status('resource_paused');time.sleep(10);continue
                if total is not None:peak=max(peak,total)
            for case,v in list(active.items()):
                code=v['process'].poll()
                if code is None:continue
                del active[case]
                if code:raise RuntimeError(f'{case} exited {code}; preserve evidence and audit')
                done.append(case)
            if queue and len(active)<(1 if qa_case else 2):
                if gpu is None:
                    try:gpu=query_gpu()
                    except Exception:gpu=None
                if gate.observe(gpu):
                    case=queue.pop(0);source,work=paths(root,case)
                    if not qa_case and read(work/'status.json',{}).get('state')=='complete':done.append(case);continue
                    mode='--qa-worker' if qa_case else '--worker'
                    command=[sys.executable,'-u','-m','gofa_f.graph_gpu_execution','--output',str(root),mode,case]
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gate.uuid,GOFA_GPU_ADMITTED='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
                        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPATH=str(snapshot/'src'))
                    with (root/(case+('-qa.log' if qa_case else '.log'))).open('ab') as log:
                        p=subprocess.Popen(command,cwd=snapshot,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    v=dict(process=p,pid=p.pid,ticks=process_identity(p.pid));active[case]=v
                    atomic_json(root/(case+('-qa.launch.json' if qa_case else '.launch.json')),dict(pid=p.pid,start_ticks=v['ticks'],command=command))
            status('running');time.sleep(1)
        if not qa_case:
            status('gpu_source_complete_waiting_cpu_evaluation')
            finalize(root,stopped,status)
        status('complete')
    except BaseException as exc:
        stop_children();status('failed',error=repr(exc));atomic_json(root/f'gpu-supervisor-failure-{time.time_ns()}.json',dict(error=repr(exc)));raise
    finally:
        path=root/('qa_activity.json' if qa_case else 'gpu_activity.json');value=read(path,{})
        value['worker']=None;atomic_json(path,value)


def finalize(root,stopped,status):
    """Use the original immutable CPU inference paths and exact data manifests."""
    execution=read(root/'execution_manifest.json')
    for label in ('primary','variable'):
        origin=Path(execution['origins'][label]);launch=read(Path(str(origin)+'.bootstrap.launch.json'))
        snapshot=Path(launch['snapshot']);source=root/label;m=read(source/'manifest.json')
        cfg=m['shared_ufg_configuration'] if label=='primary' else m['variable_configuration']
        if label=='primary':
            code="from pathlib import Path;import json;from gofa_f.ufg_shared_evaluation import evaluate;s=Path(__import__('sys').argv[1]);m=json.loads((s/'manifest.json').read_text());c=m['shared_ufg_configuration'];evaluate(c,s,Path(c['evaluation']['output']))"
        else:
            code="from pathlib import Path;import json;from gofa_f.variable_graph_evaluation import evaluate_targets;s=Path(__import__('sys').argv[1]);m=json.loads((s/'manifest.json').read_text());c=m['variable_configuration'];evaluate_targets(c,s,lambda:None,lambda **fields:print(fields,flush=True))"
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(snapshot/'src'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        with (root/(label+'-cpu-evaluation.log')).open('ab') as log:
            child=subprocess.Popen([sys.executable,'-u','-c',code,str(source)],cwd=snapshot,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while child.poll() is None:
            status('cpu_evaluation',model=label)
            if stopped[0]:
                child.send_signal(signal.SIGTERM);child.wait(timeout=60);raise InterruptedError('CPU evaluation saved')
            time.sleep(2)
        if child.returncode:raise RuntimeError('CPU evaluation failed; preserve source states and audit')
    atomic_json(root/'evaluation_completion.json',dict(complete=True,target_labels_used_for_fitting=False,
        gpu_execution_manifest_sha256=sha(root/'execution_manifest.json'),goal_achieved=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--snapshot')
    p.add_argument('--prepare',action='store_true');p.add_argument('--variable-origin');p.add_argument('--primary-origin')
    p.add_argument('--worker',choices=CASES);p.add_argument('--qa-worker',choices=CASES);p.add_argument('--accept',choices=CASES)
    p.add_argument('--cpu-accept',choices=CASES)
    p.add_argument('--supervise',action='store_true');p.add_argument('--progress',action='store_true');a=p.parse_args();root=Path(a.output)
    if a.prepare:prepare(root,Path(a.variable_origin),Path(a.primary_origin))
    elif a.worker:
        try:worker(root,a.worker)
        except InterruptedError:raise SystemExit(75)
    elif a.qa_worker:
        from gofa_f.graph_gpu_qa import acceptance
        acceptance(root,a.qa_worker)
    elif a.cpu_accept:
        from gofa_f.graph_gpu_qa import acceptance
        acceptance(root,a.cpu_accept,cpu_only=True)
    elif a.accept or a.supervise:supervise(root,Path(a.snapshot),a.accept)
    elif a.progress:
        print(json.dumps(read(root/'gpu_status.json',{}),indent=2))
        for c in CASES:
            _,work=paths(root,c);print(c,json.dumps(read(work/'status.json',{})))
    else:p.error('Choose prepare, accept, supervise, worker or progress')


if __name__=='__main__':main()
