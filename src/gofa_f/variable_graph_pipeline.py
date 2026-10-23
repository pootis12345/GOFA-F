"""Run fresh graph expert and equal-data shared UFG, then sealed target evaluation."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from gofa_f.dual_memory_fitting import read
from gofa_f.dual_memory_warmup import cpu_only
from gofa_f.generative_data import sha
from gofa_f.idle_launcher import process_identity
from gofa_f.prefix_cache import atomic_json
from gofa_f.variable_graph_source import configuration,runtime
from gofa_f.variable_graph_evaluation import evaluate_targets


def pipeline(cfg,root,snapshot,config):
    cpu_only();proof=read(root/'cpu_preflight.json',{})
    if not proof.get('passed') or proof['runtime_code']!=runtime() or proof['manifest_sha256']!=sha(root/'manifest.json') or \
        proof['target_plan_sha256']!=sha(root/'target_plan.json'):
        raise ValueError('Exact source-bound acceptance required')
    lock=(root/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=read(root/'pipeline_status.json',{})
    if state.get('state')=='complete':return
    if state.get('state')=='failed':raise ValueError('Audit before repairing a failed implementation')
    stopped=[False];worker=dict(pid=os.getpid(),start_ticks=process_identity(os.getpid()))
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    started=time.monotonic();elapsed=read(root/'pipeline_activity.json',{}).get('spent_cpu_residency_seconds',0.)
    def check():
        if stopped[0]:raise InterruptedError('Cooperative variable-length pipeline save')
    def progress(**fields):
        atomic_json(root/'pipeline_status.json',dict(state='running',goal_achieved=False,**fields))
        atomic_json(root/'pipeline_activity.json',dict(worker=worker,spent_cpu_residency_seconds=elapsed+time.monotonic()-started,
            cpu_seconds=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_utime,
            new_gpu_seconds=0,cuda_initialized=False))
    try:
        if not (root/'target_plan.json').exists():raise ValueError('Target queries must be sealed before training')
        for kind in ('expert','ufg'):
            check();status=read(root/kind/'status.json',{})
            if status.get('state')=='complete':continue
            prior=read(root/(kind+'.launch.json'),{})
            if prior and Path('/proc',str(prior['pid'])).exists():
                if process_identity(prior['pid'])==prior['start_ticks']:raise ValueError('Existing live source process; do not duplicate')
            cmd=[sys.executable,'-u','-m','gofa_f.variable_graph_source','--config',str(config),'--output',str(root),'--model',kind]
            with (root/(kind+'.log')).open('ab') as log:
                child=subprocess.Popen(cmd,cwd=snapshot,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            child_id=dict(pid=child.pid,start_ticks=process_identity(child.pid))
            atomic_json(root/(kind+'.launch.json'),dict(**child_id,command=cmd))
            while child.poll() is None:
                progress(phase='source_fit',model=kind,source_worker=child_id)
                if stopped[0]:
                    child.send_signal(signal.SIGTERM)
                    try:child.wait(timeout=60)
                    except subprocess.TimeoutExpired:raise RuntimeError('Source cooperative stop timed out; identity audit required')
                    check()
                time.sleep(10)
            if child.returncode:raise RuntimeError('Source worker stopped without completion; inspect failure evidence')
        check();evaluate_targets(cfg,root,check,progress)
        atomic_json(root/'pipeline_status.json',dict(state='complete',phase='expert_shared_ufg_and_zero_shot_retest_complete',goal_achieved=False,eta_seconds=None))
    except InterruptedError:
        atomic_json(root/'pipeline_status.json',dict(state='paused',phase='cooperative_save',goal_achieved=False));raise
    except BaseException as exc:
        atomic_json(root/f'pipeline-failure-{time.time_ns()}.json',dict(error=repr(exc)))
        atomic_json(root/'pipeline_status.json',dict(state='failed',error=repr(exc),goal_achieved=False));raise
    finally:
        atomic_json(root/'pipeline_activity.json',dict(worker=None,spent_cpu_residency_seconds=elapsed+time.monotonic()-started,
            new_gpu_seconds=0,cuda_initialized=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--snapshot',required=True);a=p.parse_args();cfg=configuration(a.config)
    pipeline(cfg,Path(a.output),Path(a.snapshot),Path(a.config))


if __name__=='__main__':main()
