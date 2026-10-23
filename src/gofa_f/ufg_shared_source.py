"""Fresh source training and one checkpoint lock for shared-shape UFG."""
import argparse
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
import yaml

from gofa_f.dual_memory_fitting import fit,read
from gofa_f.dual_memory_policy import digest
from gofa_f.dual_memory_store import SourceStore
from gofa_f.dual_memory_warmup import cpu_only
from gofa_f.generative_data import sha
from gofa_f.prefix_cache import atomic_json
from gofa_f.ufg_shared import SharedUFGEngine

PROTOCOL='stage65-shared-ufg-source-v1'
FILES=('ufg_shared.py','ufg_shared_source.py','ufg_shared_evaluation.py',
    'dual_memory_fitting.py','dual_memory_policy.py','dual_memory_store.py','dual_memory_data.py',
    'dual_memory_engines.py','dual_memory_warmup.py','ufg.py','ufg_memory_roster.py','ufg_subgraph.py',
    'generative_data.py','prefix_cache.py','tsgfm_target_evaluation.py')
RECIPE=dict(max_updates=40000,effective_batch=64,microbatch=64,learning_rate=.001,warmup_updates=1,
    interface_only_updates=0,weight_decay=0.,gradient_clip=5.,validation_every=500,
    early_min_updates=4000,early_patience=8,significant_improvement=.001)


def runtime():return {n:sha(Path(__file__).parent/n) for n in FILES}


def configuration(path):
    cfg=yaml.safe_load(Path(path).read_text().replace('${GOFA_WORKSPACE}',os.environ['GOFA_WORKSPACE']))
    if cfg['protocol']!=PROTOCOL or cfg['training']!=RECIPE or \
        cfg['model']!=dict(width=128,canonical_history=48,maximum_horizon=24,single_parameter_set=True,shape_specific_weights=False):
        raise ValueError('Exactly one preregistered shared UFG source model required')
    if cfg['target_training_enabled'] or cfg['resources']!=dict(cuda_enabled=False,cpu_threads=2,deadline=None):
        raise ValueError('No target update; CPU two-thread resource policy')
    return cfg


def prepare(cfg,root):
    cpu_only();parent=Path(cfg['source_output'])
    if sha(parent/'manifest.json')!=cfg['source_manifest_sha256']:raise ValueError('Original source registry changed')
    old=read(parent/'manifest.json');m=deepcopy(old)
    if read(parent/'status.json')['state']!='complete' or read(parent/'activity.json')['worker'] is not None:
        raise ValueError('Read-only exited original source parent required')
    m['configuration']['training']=cfg['training'];m['configuration']['queries']['initial_training_per_view']=8192
    m['shared_ufg_configuration']=cfg;m['shared_ufg_runtime']=runtime()
    m['source_cache_index_sha256']=sha(parent/'source_inputs/index.json')
    m['shape_specialized_models_allowed']=False;m['fresh_seed42_no_shape_checkpoint_initialization']=True
    m['source_confirmation_opened']=False;m['target_labels_used']=False
    root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists() and read(root/'manifest.json')!=m:raise ValueError('Shared model preparation binding changed')
    atomic_json(root/'manifest.json',m)
    if not (root/'status.json').exists():atomic_json(root/'status.json',dict(state='prepared',goal_achieved=False))
    return m


def run(cfg,root):
    cpu_only();lock=(root/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    m=read(root/'manifest.json');binding=sha(root/'manifest.json');proof=read(root/'cpu_preflight.json',{})
    if m['shared_ufg_configuration']!=cfg or m['shared_ufg_runtime']!=runtime() or not proof.get('passed') or \
        proof.get('manifest_sha256')!=binding or proof.get('runtime_code')!=runtime():raise ValueError('Fixed tested source runtime required')
    prior=read(root/'status.json',{})
    if prior.get('state')=='complete':return
    if prior.get('state')=='failed':raise ValueError('Audit implementation failure before resume')
    parent=Path(cfg['source_output'])
    if sha(parent/'source_inputs/index.json')!=m['source_cache_index_sha256']:raise ValueError('Shared source cache changed')
    from gofa_f.idle_launcher import process_identity
    worker=dict(pid=os.getpid(),start_ticks=process_identity(os.getpid()));stopped=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    start=time.monotonic();elapsed=read(root/'activity.json',{}).get('spent_cpu_residency_seconds',0.);visible={};engine=None
    def check():
        if stopped[0]:raise InterruptedError('Cooperative shared-UFG save')
    def progress(**fields):
        visible.update(fields)
        atomic_json(root/'status.json',dict(state='running',goal_achieved=False,target_predictions=0,
            one_checkpoint_for_both_shapes=True,**visible))
        atomic_json(root/'activity.json',dict(worker=worker,spent_cpu_residency_seconds=elapsed+time.monotonic()-start,
            cuda_initialized=False,new_gpu_seconds=0,threads=2))
    try:
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        store=SourceStore(m,parent/'source_inputs');engine=SharedUFGEngine(cfg['ufg_root'],m,store)
        # ufg=False deliberately selects the ALL-SHAPE source schedule. With
        # weight_decay0,warmup1 andclip5 this is the same fixed Adam update rule,
        # not the legacy shape-filtered UFG branch of the generic fitter.
        fit(engine,m,cfg,root/'all','all',None,digest([binding,'one_shared_ufg','all']),check,progress,ufg=False)
        s=read(root/'all/selection.json');adequate=s['stop_reason']=='source_early_stop'
        atomic_json(root/'ufg_shared_lock.json',dict(binding=binding,selection=s,source_only=True,
            checkpoint_path=str(root/'all/best.pt'),one_parameter_set=True,shape_specific_weights=False,
            source_adequacy=adequate,global_optimum_not_claimed=True,target_labels_used=False,
            training_shapes=[[12,12],[48,24]],upstream_effective_graph_core_preserved=True,
            adaptations=engine.model.provenance))
        atomic_json(root/'status.json',dict(state='complete',phase='single_shared_ufg_source_locked',goal_achieved=False,
            source_adequacy=adequate,best_update=s['best_update'],stopped_update=s['stopped_update'],
            one_checkpoint_for_both_shapes=True,target_predictions=0,eta_seconds=None))
    except InterruptedError:
        atomic_json(root/'status.json',dict(state='paused',phase='shared_ufg_source',goal_achieved=False));raise
    except BaseException as exc:
        atomic_json(root/f'failure-{time.time_ns()}.json',dict(error=repr(exc),binding=binding))
        atomic_json(root/'status.json',dict(state='failed',error=repr(exc),goal_achieved=False));raise
    finally:
        if engine is not None:engine.close()
        atomic_json(root/'activity.json',dict(worker=None,spent_cpu_residency_seconds=elapsed+time.monotonic()-start,
            cuda_initialized=False,new_gpu_seconds=0,threads=2,
            peak_rss_kib=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--progress',action='store_true');p.add_argument('--watch',action='store_true')
    a=p.parse_args();cfg=configuration(a.config);root=Path(a.output)
    if a.progress or a.watch:
        while True:
            s=read(root/'status.json',{});target=read(Path(cfg['evaluation']['output'])/'status.json',{})
            state=target if target else s
            if s.get('state')=='complete' and not target:state=dict(state='running',phase='source_locked_waiting_cpu_retest')
            pipeline=read(root/'pipeline_status.json',{})
            if pipeline.get('state') in ('failed','paused'):state=pipeline
            end=state.get('state') in ('failed','paused','complete')
            eta=None if end or s.get('state')!='running' else s.get('eta_current_fit_seconds')
            selection=s.get('selection',{})
            print('SharedUFG',state.get('state','not_started'),state.get('phase','source'),
                'source_updates',s.get('updates',s.get('stopped_update',0)),
                'source_best',s.get('best_update',selection.get('best_update')),
                'target',target.get('model'),target.get('queries',0),
                'ETA','n/a' if end else 'unknown' if eta is None else f'{eta/3600:.2f}h plus final CPU retest',flush=True)
            if not a.watch or end:break
            time.sleep(5)
    elif a.prepare:prepare(cfg,root)
    else:run(cfg,root)


if __name__=='__main__':main()
