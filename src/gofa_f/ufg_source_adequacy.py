"""Source-only training-adequacy continuation of the unchanged official UFG.

Only the original short-shape update cap changes. All optimizer, gradient,
RNG, query-order, selection and numeric state are transferred exactly; the
stopped parent remains immutable. No target prediction entry point exists.
"""
import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import time

import numpy as np
import torch
import yaml

from gofa_f.dual_memory_engines import UFGEngine, model_digest
from gofa_f.dual_memory_fitting import fit, read
from gofa_f.dual_memory_policy import digest
from gofa_f.dual_memory_store import SourceStore, atomic_torch
from gofa_f.dual_memory_warmup import cpu_only
from gofa_f.generative_data import sha
from gofa_f.prefix_cache import atomic_json

PROTOCOL='stage62-official-ufg-source-adequacy-v1'
FILES=('ufg_source_adequacy.py','dual_memory_fitting.py','dual_memory_policy.py',
    'dual_memory_store.py','dual_memory_engines.py','dual_memory_data.py',
    'ufg_memory_roster.py','ufg_subgraph.py','ufg.py','generative_data.py','prefix_cache.py')


def configuration(path):
    cfg=yaml.safe_load(Path(path).read_text().replace('${GOFA_WORKSPACE}',os.environ['GOFA_WORKSPACE']))
    if cfg.get('protocol')!=PROTOCOL or cfg.get('target_training_enabled') is not False or cfg.get('target_evaluation_enabled') is not False:
        raise ValueError('Source-only official UFG adequacy configuration required')
    if cfg.get('maximum_updates')!=20000 or cfg.get('shape')!='12-12':
        raise ValueError('Only the preregistered short-shape cap extension is permitted')
    if cfg.get('resources')!=dict(cuda_enabled=False,cpu_threads=2,deadline=None):
        raise ValueError('CUDA-disabled two-thread execution required')
    return cfg


def runtime():return {n:sha(Path(__file__).parent/n) for n in FILES}


def tree_digest(value):
    """Canonical numerical-state digest, independent of archive/pickle metadata."""
    h=hashlib.sha256()
    def visit(v):
        if isinstance(v,torch.Tensor):
            x=v.detach().cpu().contiguous();h.update(str(('torch',x.dtype,tuple(x.shape))).encode())
            h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v,np.ndarray):
            h.update(str(('numpy',v.dtype,v.shape)).encode());h.update(v.tobytes())
        elif isinstance(v,dict):
            h.update(b'dict')
            for key in sorted(v,key=lambda x:(type(x).__name__,str(x))):visit(key);visit(v[key])
        elif isinstance(v,(list,tuple)):
            h.update(type(v).__name__.encode());h.update(str(len(v)).encode())
            for x in v:visit(x)
        elif v is None or isinstance(v,(str,bool,int,float,np.generic)):
            h.update(str((type(v).__name__,v)).encode())
        else:raise TypeError('Unexpected checkpoint state: '+str(type(v)))
    visit(value);return h.hexdigest()


def transferable(value):
    copied=deepcopy(value);copied.pop('binding')
    copied['state'].pop('completed',None);copied['state'].pop('stop_reason',None)
    return copied


def continue_state(value,old_binding,new_binding):
    if value['binding']!=old_binding:raise ValueError('Original transaction binding changed')
    state=value['state']
    if state['updates']!=10000 or not state.get('completed') or state.get('stop_reason')!='maximum_updates' \
        or state['selection'].get('stop') or state['selection'].get('last_validated')!=10000:
        raise ValueError('Only the fully saved capped UFG transaction can be continued')
    result=deepcopy(value);result['binding']=new_binding
    result['state']['completed']=False;result['state'].pop('stop_reason',None)
    if tree_digest(transferable(value))!=tree_digest(transferable(result)):
        raise ValueError('Optimizer/partial gradients/RNG/cursor/selection changed')
    return result


def prepare(cfg,output):
    cpu_only();parent=Path(cfg['parent_source_output'])
    if sha(parent/'manifest.json')!=cfg['parent_source_manifest_sha256']:raise ValueError('Original source manifest changed')
    old=read(parent/'manifest.json')
    if read(parent/'ufg_status.json')['state']!='complete' or read(parent/'ufg_activity.json')['worker'] is not None:
        raise ValueError('Original UFG workers must be finished and exited')
    for name,value in old['code'].items():
        if sha(Path(__file__).parent/name)!=value:raise ValueError('Audited input core changed')
    selected=read(parent/'ufg-12-12/selection.json')
    if selected['stopped_update']!=10000 or selected['stop_reason']!='maximum_updates' or selected['target_labels_used']:
        raise ValueError('Original short-shape cap evidence required')
    if sha(parent/'ufg-12-12/best.pt')!=selected['checkpoint_sha256']:raise ValueError('Original UFG best changed')
    long=read(parent/'ufg-48-24/selection.json')
    if long['stop_reason']!='source_early_stop' or long['target_labels_used'] or sha(parent/'ufg-48-24/best.pt')!=long['checkpoint_sha256']:
        raise ValueError('Long shape must retain its already early-stopped source best')
    m=deepcopy(old);m['configuration']['ufg']['maximum_updates']=20000
    m['ufg_adequacy_configuration']=cfg;m['ufg_adequacy_runtime_code']=runtime()
    m['parent_source_manifest_sha256']=cfg['parent_source_manifest_sha256']
    m['source_cache_index_sha256']=sha(parent/'source_inputs/index.json')
    m['original_short_selection_sha256']=sha(parent/'ufg-12-12/selection.json')
    m['original_last_checkpoint_sha256']=sha(parent/'ufg-12-12/last.pt')
    m['original_best_checkpoint_sha256']=sha(parent/'ufg-12-12/best.pt')
    m['reused_long_selection_sha256']=sha(parent/'ufg-48-24/selection.json')
    m['inherited_ufg_cpu_activity']=read(parent/'ufg_activity.json')
    costs=Path(cfg['inherited_costs_file'])
    if sha(costs)!=cfg['inherited_costs_sha256']:raise ValueError('Original cumulative cost evidence changed')
    m['inherited_prior_gpu_seconds']=read(costs)['cumulative_gpu_seconds']
    m['inherited_costs_sha256']=sha(costs)
    m['target_labels_opened']=False;m['source_confirmation_labels_opened']=False
    output.mkdir(parents=True,exist_ok=True)
    if (output/'manifest.json').exists() and read(output/'manifest.json')!=m:raise ValueError('Adequacy manifest is immutable')
    atomic_json(output/'manifest.json',m)
    atomic_json(output/'status.json',dict(state='prepared',phase='original_transaction_ready',goal_achieved=False,target_predictions=0))
    return m


def seed_transaction(parent,folder,binding,manifest):
    """Run in the original UFG environment to preserve NumPy/pickle versions."""
    folder.mkdir(parents=True,exist_ok=True);old=Path(parent)/'ufg-12-12'
    if (folder/'last.pt').exists():
        proof=read(folder/'continuation_origin.json')
        if proof is None or proof['new_binding']!=binding:raise ValueError('Unproved continuation checkpoint')
        return
    if sha(old/'last.pt')!=manifest['original_last_checkpoint_sha256'] or sha(old/'best.pt')!=manifest['original_best_checkpoint_sha256']:
        raise ValueError('Original saved transaction changed')
    selected=read(old/'selection.json');saved=torch.load(old/'last.pt',map_location='cpu',weights_only=False)
    continued=continue_state(saved,selected['binding'],binding)
    best=torch.load(old/'best.pt',map_location='cpu',weights_only=False)
    if best['binding']!=selected['binding'] or best['model_sha256']!=selected['model_sha256']:
        raise ValueError('Original best checkpoint binding changed')
    before_best=tree_digest(best['model']);best['binding']=binding
    for name in ('curve.jsonl',*[p.name for p in sorted(old.glob('validation-*.json'))]):
        if (folder/name).exists() and sha(folder/name)!=sha(old/name):raise ValueError('Original curve evidence changed')
        if not (folder/name).exists():shutil.copyfile(old/name,folder/name)
    atomic_torch(folder/'best.pt',best)
    atomic_json(folder/'continuation_origin.json',dict(new_binding=binding,old_binding=selected['binding'],
        old_last_checkpoint_sha256=sha(old/'last.pt'),old_best_checkpoint_sha256=sha(old/'best.pt'),
        old_training_state_sha256=tree_digest(transferable(saved)),
        continued_training_state_sha256=tree_digest(transferable(continued)),
        same_best_tensor_sha256=before_best,maximum_updates_changed=[10000,20000],
        data_model_optimizer_rng_cursor_selection_unchanged=True,target_labels_used=False))
    atomic_torch(folder/'last.pt',continued)


def run(cfg,output):
    cpu_only();lock=(output/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    m=read(output/'manifest.json');binding=sha(output/'manifest.json')
    if m['ufg_adequacy_configuration']!=cfg or m['ufg_adequacy_runtime_code']!=runtime():raise ValueError('Fixed runtime changed')
    proof=read(output/'cpu_preflight.json',{})
    if not proof.get('passed') or proof.get('manifest_sha256')!=binding or proof.get('runtime_code')!=runtime():
        raise ValueError('Tested continuation proof required')
    previous=read(output/'status.json',{})
    if previous.get('state')=='complete':return
    if previous.get('state')=='failed':raise ValueError('Audit any implementation failure before restarting')
    parent=Path(cfg['parent_source_output'])
    if sha(parent/'source_inputs/index.json')!=m['source_cache_index_sha256']:raise ValueError('Original source cache changed')
    stopped=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    from gofa_f.idle_launcher import process_identity
    worker=dict(pid=os.getpid(),start_ticks=process_identity(os.getpid()))
    elapsed=read(output/'activity.json',{}).get('new_cpu_residency_seconds',0.);start=time.monotonic();visible={}
    engine=None
    def check():
        if stopped[0]:raise InterruptedError('Cooperative UFG adequacy pause')
    def progress(**fields):
        visible.update(fields)
        atomic_json(output/'status.json',dict(state='running',goal_achieved=False,target_predictions=0,**visible))
        atomic_json(output/'activity.json',dict(worker=worker,new_cpu_residency_seconds=elapsed+time.monotonic()-start,
            inherited_ufg_cpu_activity=m['inherited_ufg_cpu_activity'],inherited_prior_gpu_seconds=m['inherited_prior_gpu_seconds'],
            new_gpu_seconds=0,cuda_initialized=False,threads=2))
    try:
        progress(phase='checking_original_saved_transaction')
        folder=output/'ufg-12-12';transaction=digest([binding,'official-ufg-cap-continuation','12-12'])
        seed_transaction(parent,folder,transaction,m)
        store=SourceStore(m,parent/'source_inputs')
        engine=UFGEngine(cfg['ufg_root'],12,12,m,store,microbatch=64)
        fit(engine,m,cfg,folder,'all',None,transaction,check,progress,ufg=True)
        selected=read(folder/'selection.json');long=read(parent/'ufg-48-24/selection.json')
        adequately_stopped=selected['stop_reason']=='source_early_stop'
        atomic_json(output/'ufg_source_lock.json',dict(binding=binding,source_only=True,
            adequacy_criterion='same_source_0.1pct_reference_eight_validations_without_improvement',
            both_shapes_stopped_by_source_criterion=adequately_stopped,
            global_optimum_not_claimed=True,target_labels_used=False,
            models={'12-12':dict(directory=str(folder),selection=selected),
                '48-24':dict(directory=str(parent/'ufg-48-24'),selection=long,reused_unchanged=True)}))
        atomic_json(output/'status.json',dict(state='complete',phase='source_early_stop_met' if adequately_stopped else 'extended_cap_not_convergence',
            goal_achieved=False,target_predictions=0,source_adequacy=adequately_stopped,
            best_update=selected['best_update'],stopped_update=selected['stopped_update'],eta_seconds=None))
    except InterruptedError:
        atomic_json(output/'status.json',dict(state='paused',phase='official_ufg_adequacy',goal_achieved=False));raise SystemExit(75)
    except BaseException as exc:
        atomic_json(output/f'failure-{time.time_ns()}.json',dict(error=repr(exc),binding=binding))
        atomic_json(output/'status.json',dict(state='failed',error=repr(exc),goal_achieved=False));raise
    finally:
        if engine is not None:engine.close()
        atomic_json(output/'activity.json',dict(worker=None,new_cpu_residency_seconds=elapsed+time.monotonic()-start,
            inherited_ufg_cpu_activity=m['inherited_ufg_cpu_activity'],inherited_prior_gpu_seconds=m['inherited_prior_gpu_seconds'],
            new_gpu_seconds=0,cuda_initialized=False,threads=2,
            peak_rss_kib=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    cfg=configuration(args.config);output=Path(args.output)
    if args.prepare:prepare(cfg,output)
    else:run(cfg,output)


if __name__=='__main__':main()
