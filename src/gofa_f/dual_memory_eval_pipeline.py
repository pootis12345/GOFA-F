"""One local pipeline: wait for source locks, confirm, then evaluate six targets.

This is a process tied to the authorized run, not a recurring app automation.
Existing healthy source processes are observed, never restarted or modified.
"""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from gofa_f.dual_memory_eval_data import read,seal,source_ready,deployment
from gofa_f.dual_memory_eval_prepare import prepare_scope
from gofa_f.dual_memory_evaluation import configuration
from gofa_f.generative_data import sha
from gofa_f.idle_launcher import process_identity
from gofa_f.prefix_cache import atomic_json


def alive(handle):
    if not handle or process_identity(handle['pid'])!=handle['start_ticks']:return False
    stat=Path('/proc')/str(handle['pid'])/'stat'
    return stat.read_text().split(') ',1)[1].split()[0]!='Z'


def source_handles(source):
    result=[]
    for suffix in ('.launch.json','.ufg.launch.json'):
        path=Path(str(source)+suffix)
        if path.exists():result.append(read(path))
    return result


def launch(snapshot,cfg,output,ufg=False):
    path=Path(str(output)+('.ufg.launch.json' if ufg else '.launch.json'))
    if path.exists():
        handle=read(path)
        if alive(handle):return handle
    worker_status=output/('ufg_status.json' if ufg else 'status.json')
    if worker_status.exists() and read(worker_status).get('state')=='failed':
        raise ValueError('A failed evaluation must be audited, not blindly restarted')
    complete_files=('ufg_complete.json','ufg_frozen_weights.json') if ufg else \
        ('native_complete.json','enhanced_complete.json','fusion_complete.json','frozen_gpu_weights.json')
    if worker_status.exists() and read(worker_status).get('state')=='complete' and all((output/n).exists() for n in complete_files):
        return None
    python=cfg['ufg_python'] if ufg else sys.executable
    command=[python,'-u','-m','gofa_f.dual_memory_evaluation','--config',str(snapshot/'configs/stage59_dual_memory_evaluation.yaml'),
        '--output',str(output),'--ufg'] if ufg else [python,'-u','-m','gofa_f.short_launcher',
        '--config',str(snapshot/'configs/stage59_dual_memory_evaluation.yaml'),'--output',str(output),
        '--snapshot',str(snapshot),'--worker-mode','dual_memory_evaluation']
    environment=dict(os.environ,PYTHONPATH=str(snapshot/'src'),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    with Path(str(output)+('.ufg.log' if ufg else '.log')).open('ab') as log:
        child=subprocess.Popen(command,env=environment,cwd=snapshot,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    handle=dict(pid=child.pid,start_ticks=process_identity(child.pid),command=command,snapshot=str(snapshot),at=time.time())
    atomic_json(path,handle);return handle


def run(cfg,root,snapshot,proof):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('The connected controller is CPU-only')
    root.mkdir(parents=True,exist_ok=True);lock=(root/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cpu=read(proof)
    if not cpu.get('passed') or not cpu.get('cuda_initialized') is False:
        raise ValueError('Successful related CPU tests with CUDA disabled are required')
    files=cpu['code_sha256']
    for name,expected in files.items():
        if sha(snapshot/name)!=expected:raise ValueError('CPU-tested evaluation file changed: '+name)
    stopped=[False];children=[]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    def check():
        if stopped[0]:raise InterruptedError('Evaluation pipeline stopped cooperatively')
    def state(phase,**fields):atomic_json(root/'status.json',dict(state='running',phase=phase,goal_achieved=False,**fields))
    source=Path(cfg['source_output']);handles=source_handles(source)
    try:
        while not source_ready(source):
            check();statuses={n:read(source/n) for n in ('status.json','ufg_status.json') if (source/n).exists()}
            if any(v['state']=='failed' for v in statuses.values()):raise ValueError('Source job failed; preserve evidence and audit before recovery')
            live=[h for h in handles if alive(h)]
            if not live:raise ValueError('Source locks are incomplete and the verified source job handles have exited')
            state('waiting_for_verified_source_jobs',live_handles=[dict(pid=h['pid'],start_ticks=h['start_ticks']) for h in live],
                source_states=statuses,target_inputs_prepared=False)
            time.sleep(10)
        while any(alive(h) for h in handles):
            check();state('waiting_for_source_process_exit');time.sleep(1)
        seal(root/'source_exit_audit.json',dict(handles=[dict(pid=h['pid'],start_ticks=h['start_ticks'],exited=True) for h in handles],
            source_activity_worker=read(source/'activity.json')['worker'],ufg_activity_worker=read(source/'ufg_activity.json')['worker']))
        locked=deployment(source,Path(cfg['accepted_snapshot']));seal(root/'deployment_before_confirmation.json',locked)
        for scope,name in (('source_confirmation','source_confirmation_output'),('targets','target_output')):
            check();output=Path(cfg[name])
            if (output/'final_status.json').exists():
                final=read(output/'final_status.json')
                if final['state']=='complete':continue
            state('preparing_'+scope);manifest=prepare_scope(cfg,output,scope,locked)
            seal(output/'cpu_preflight.json',dict(passed=True,binding=sha(output/'manifest.json'),
                proof_sha256=sha(proof),code_sha256=files,cuda_initialized=False))
            # Scope-specific ledger starts at zero; final accounting adds the
            # prior source and mechanism ledger exactly once, never copies it.
            if not (output/'ufg_activity.json').exists():atomic_json(output/'ufg_activity.json',dict(spent_seconds=0.,worker=None))
            for ufg in (True,False):
                handle=launch(snapshot,cfg,output,ufg)
                if handle is not None:children.append(handle)
            while True:
                check();live=[h for h in children if alive(h)]
                statuses={n:read(output/n) for n in ('status.json','ufg_status.json') if (output/n).exists()}
                if any(v['state']=='failed' for v in statuses.values()):raise ValueError('Frozen evaluation failed; evidence preserved for implementation audit')
                if not live:break
                state('predicting_'+scope,scope=scope,prediction_states=statuses)
                time.sleep(10)
            children=[]
            if not all((output/f'{n}_complete.json').exists() for n in ('native','enhanced','fusion','ufg')):
                raise ValueError('Evaluation workers exited before complete bound prediction seals')
            if read(output/'activity.json').get('worker') is not None:raise ValueError('GPU ledger still has a worker after controller exit')
            state('scoring_'+scope)
            command=[sys.executable,'-m','gofa_f.dual_memory_evaluation','--config',str(snapshot/'configs/stage59_dual_memory_evaluation.yaml'),
                '--output',str(output),'--summarize']
            subprocess.run(command,cwd=snapshot,env=dict(os.environ,PYTHONPATH=str(snapshot/'src'),CUDA_VISIBLE_DEVICES='',
                OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),check=True)
        source_activity=read(source/'activity.json');confirm=Path(cfg['source_confirmation_output']);target=Path(cfg['target_output'])
        target_report=read(target/'report.json');activities={scope:read(path/'activity.json') for scope,path in (('source_confirmation',confirm),('targets',target))}
        report=dict(state='complete',goal_achieved=target_report['goal_achieved'],deployment=locked,
            source_confirmation_report_sha256=sha(confirm/'report.json'),target_report_sha256=sha(target/'report.json'),
            inherited_source_gpu_seconds=source_activity['spent_seconds'],
            new_evaluation_gpu_seconds=sum(v['spent_seconds'] for v in activities.values()),
            cumulative_gpu_seconds=source_activity['spent_seconds']+sum(v['spent_seconds'] for v in activities.values()),
            peak_group_mib=max(source_activity.get('peak_group_mib',0),*(v.get('peak_group_mib',0) for v in activities.values())),
            source_ufg_cpu=read(source/'ufg_activity.json'),evaluation_ufg_cpu={name:read(path/'ufg_activity.json') for name,path in (('source_confirmation',confirm),('targets',target))},
            target_results_not_used_to_refit=True)
        seal(root/'report.json',report);atomic_json(root/'status.json',report)
    finally:
        for handle in children:
            if alive(handle):os.kill(handle['pid'],signal.SIGTERM)
        until=time.monotonic()+60
        while any(alive(h) for h in children) and time.monotonic()<until:time.sleep(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--snapshot',required=True);parser.add_argument('--cpu-proof',required=True);args=parser.parse_args()
    cfg=configuration(args.config);root=Path(args.output)
    try:run(cfg,root,Path(args.snapshot),Path(args.cpu_proof))
    except InterruptedError:
        atomic_json(root/'status.json',dict(state='paused',phase='evaluation_pipeline',goal_achieved=False));raise SystemExit(75)
    except BaseException as exc:
        atomic_json(root/f'failure-{time.time_ns()}.json',dict(error=repr(exc)))
        atomic_json(root/'status.json',dict(state='failed',phase='evaluation_pipeline',error=repr(exc),goal_achieved=False));raise


if __name__=='__main__':main()
