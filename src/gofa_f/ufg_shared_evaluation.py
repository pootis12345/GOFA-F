"""Run one shared source UFG to completion, then retest all six frozen targets.

No shape lookup selects a model. The two task shapes use the SAME object,
checkpoint and tensor SHA. Native and TSGFM predictions are immutable reuse.
"""
import argparse
import fcntl
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
import torch

from gofa_f.dual_memory_eval_data import InputStore,read,seal,TARGETS
from gofa_f.dual_memory_eval_records import PredictionStore
from gofa_f.dual_memory_eval_metrics import macro_metrics,strict_improvement,paired_block_intervals,spatial_blocks
from gofa_f.dual_memory_engines import model_digest
from gofa_f.dual_memory_policy import digest
from gofa_f.dual_memory_warmup import cpu_only
from gofa_f.generative_data import Query,Series,sha
from gofa_f.prefix_cache import atomic_json
from gofa_f.tsgfm_target_evaluation import (validate_native,query_contract,identity,checked_prediction,require_complete)
from gofa_f.ufg_shared import SharedRosterUFG,pack_roster
from gofa_f.ufg_shared_source import configuration,runtime,prepare as prepare_source


def prepare(cfg,source,root):
    cpu_only();root=Path(root);source=Path(source);old=Path(cfg['evaluation']['reference'])
    m=read(source/'manifest.json');locked=read(source/'ufg_shared_lock.json');selected=read(source/'all/selection.json')
    if read(source/'status.json')['state']!='complete' or read(source/'activity.json')['worker'] is not None or \
        locked['binding']!=sha(source/'manifest.json') or locked['selection']!=selected or \
        not locked['one_parameter_set'] or locked['shape_specific_weights'] or selected['target_labels_used'] or \
        sha(source/'all/best.pt')!=selected['checkpoint_sha256']:
        raise ValueError('One locked source checkpoint with no shape-specialized branch required')
    if m['shared_ufg_runtime']!=runtime() or cfg!=m['shared_ufg_configuration']:raise ValueError('Shared training/inference code changed')
    if sha(old/'manifest.json')!=cfg['evaluation']['reference_manifest_sha256']:
        raise ValueError('Frozen comparison reference changed')
    previous=read(old/'manifest.json');query_contract(previous['queries'])
    if read(old/'status.json')['state']!='complete' or read(old/'activity.json')['worker'] is not None:
        raise ValueError('Frozen previous predictions required')
    root.mkdir(parents=True,exist_ok=True)
    value=dict(protocol='stage65-shared-ufg-targets-v1',source_manifest_sha256=sha(source/'manifest.json'),
        source_lock_sha256=sha(source/'ufg_shared_lock.json'),shared_checkpoint=str(source/'all/best.pt'),
        selection=selected,source_adequacy=locked['source_adequacy'],shape_specific_weights=False,
        reference_manifest_sha256=sha(old/'manifest.json'),reference_output=str(old),
        original_target_reference=previous['configuration']['target_reference'],
        original_target_manifest_sha256=previous['target_manifest_sha256'],
        input_index_sha256=previous['input_index_sha256'],queries=previous['queries'],datasets=previous['datasets'],
        runtime_code=runtime(),target_labels_used_for_fitting=False,old_target_development_exposure=True)
    seal(root/'manifest.json',value);return value


@torch.no_grad()
def evaluate(cfg,source,root):
    from gofa_f.idle_launcher import process_identity
    cpu_only();root=Path(root);m=prepare(cfg,source,root);binding=sha(root/'manifest.json')
    lock=(root/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'status.json').exists() and read(root/'status.json')['state']=='complete':return
    if (root/'status.json').exists() and read(root/'status.json')['state']=='failed':raise ValueError('Audit before evaluation repair')
    original=Path(m['original_target_reference']);reference=Path(m['reference_output'])
    if sha(original/'manifest.json')!=m['original_target_manifest_sha256'] or sha(original/'inputs/index.json')!=m['input_index_sha256']:
        raise ValueError('Original history/roster binding changed')
    inputs=InputStore(original);reuse_manifest=read(reference/'manifest.json')
    require_complete(reference,reuse_manifest,inputs)
    model=SharedRosterUFG(cfg['ufg_root']).eval()
    model.load_state_dict(torch.load(m['shared_checkpoint'],map_location='cpu',weights_only=False)['model'])
    for p in model.parameters():p.requires_grad_(False)
    before=model_digest(model)
    if before!=m['selection']['model_sha256']:raise ValueError('Shared UFG tensor SHA changed')
    stopped=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    elapsed=read(root/'activity.json')['spent_cpu_residency_seconds'] if (root/'activity.json').exists() else 0.
    start=time.monotonic();worker=dict(pid=os.getpid(),start_ticks=process_identity(os.getpid()))
    def progress(**fields):
        atomic_json(root/'status.json',dict(state='running',goal_achieved=False,shared_ufg_checkpoint=True,**fields))
        atomic_json(root/'activity.json',dict(worker=worker,spent_cpu_residency_seconds=elapsed+time.monotonic()-start,
            new_gpu_seconds=0,cuda_initialized=False,threads=2))
    try:
        progress(phase='read_only_native_eight_path_audit')
        native_hash=validate_native(original,PredictionStore(original),inputs)
        for name in ('native','tsgfm','ufg'):
            shards={}
            for i,row in enumerate(m['queries']):
                if stopped[0]:raise InterruptedError('Cooperative shared baseline evaluation stop')
                item,_=inputs.get(row);key=digest(Query(**row).key)
                old_path=reference/'predictions'/name/(key+'.json')
                old_record=read(old_path) if name!='ufg' else None
                own_sha=before if name=='ufg' else old_record['identity']['model_sha256']
                expected=identity(binding,row,item,name,own_sha);path=root/'predictions'/name/(key+'.json')
                value=checked_prediction(path,expected)
                if value is None:
                    tick=time.monotonic()
                    if name=='ufg':pred=model(pack_roster(item),row['horizon'])[0].numpy();reuse=None
                    else:
                        if old_record['identity']['query']!=row or old_record['identity']['input_sha256']!=item.identity or \
                            old_record['identity']['binding']!=m['reference_manifest_sha256']:
                            raise ValueError('Previous Native/TSGFM input binding differs')
                        pred=np.asarray(old_record['prediction']);reuse=dict(path=str(old_path),sha256=sha(old_path))
                    if pred.shape!=(row['horizon'],) or not np.isfinite(pred).all():raise ValueError('Invalid frozen prediction')
                    path.parent.mkdir(parents=True,exist_ok=True)
                    atomic_json(path,dict(identity=expected,prediction=pred.tolist(),seconds=time.monotonic()-tick,
                        real_nodes=len(item.node_ids),node_ids=list(item.node_ids),sampling_audit=item.audit,reuse=reuse))
                shards[str(path.relative_to(root))]=sha(path)
                if i%32==0 or i==767:progress(phase='six_target_prediction',model=name,queries=i+1,total=768)
            seal(root/(name+'_complete.json'),dict(complete=True,binding=binding,model=name,queries=768,
                input_index_sha256=m['input_index_sha256'],shards=shards))
        if model_digest(model)!=before or any(p.grad is not None for p in model.parameters()):raise ValueError('Source weights changed')
        seal(root/'frozen_weights.json',dict(model_sha256=before,same_tensor_sha_for_both_shapes=True,
            checkpoint_sha256=sha(m['shared_checkpoint']),parameters_modified=False,gradients_created=False,
            native_completion_sha256=native_hash))
        score(cfg,root,m,inputs)
        atomic_json(root/'status.json',dict(state='complete',phase='single_checkpoint_six_target_retest',
            goal_achieved=False,report_sha256=sha(root/'report.json'),eta_seconds=None))
    except InterruptedError:
        atomic_json(root/'status.json',dict(state='paused',phase='shared_baseline_evaluation',goal_achieved=False));raise
    except BaseException as exc:
        atomic_json(root/f'failure-{time.time_ns()}.json',dict(error=repr(exc)))
        atomic_json(root/'status.json',dict(state='failed',error=repr(exc),goal_achieved=False));raise
    finally:
        atomic_json(root/'activity.json',dict(worker=None,spent_cpu_residency_seconds=elapsed+time.monotonic()-start,
            cuda_initialized=False,new_gpu_seconds=0,threads=2,
            peak_rss_kib=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss))


def score(cfg,root,m,inputs):
    completion=require_complete(root,m,inputs);results={};binding=sha(root/'manifest.json')
    for name in TARGETS:
        info=m['datasets'][name];p=Path(info['path'])
        if sha(p)!=info['sha256'] or sha(p.with_suffix('.json'))!=info['metadata_sha256']:raise ValueError('Target data changed')
        data=Series(p);rows=[q for q in m['queries'] if q['source']==name]
        ys=[];masks=[];variables=[];clocks=[];preds={k:[] for k in ('native','tsgfm','ufg')}
        for row in rows:
            q=Query(**row);item,_=inputs.get(row)
            y=np.asarray(data.values[q.origin:q.origin+q.horizon,q.variable],float)
            mask=np.asarray(data.mask[q.origin:q.origin+q.horizon,q.variable],bool)&np.isfinite(y)
            ys.append(np.where(mask,y,0));masks.append(mask);variables.append(q.variable);clocks.append(int(data.times[q.origin]))
            for model in preds:preds[model].append(read(root/'predictions'/model/(digest(q.key)+'.json'))['prediction'])
        metrics={k:macro_metrics(v,ys,masks,variables) for k,v in preds.items()}
        independent={}
        for k,v in preds.items():
            err=defaultdict(list)
            for variable,pred,y,mask in zip(variables,v,ys,masks):
                err[variable].extend(abs(float(a)-float(b)) for a,b,valid in zip(pred,y,mask) if valid)
            independent[k]=math.fsum(math.fsum(x)/len(x) for x in err.values())/len(err)
            if not math.isclose(independent[k],metrics[k]['macro']['mae'],rel_tol=1e-6,abs_tol=1e-6):raise ValueError('MAE audit differs')
        spatial=spatial_blocks(data.external,data.ids)
        ci=paired_block_intervals(dict(native=preds['native'],ufg=preds['ufg'],fusion=preds['tsgfm']),
            ys,masks,variables,[spatial[v] for v in variables],np.asarray(clocks)//(7*24*3600*10**9))
        results[name]=dict(metrics=metrics,independent_scalar_mae=independent,tsgfm_comparison={b:dict(
            strictly_improved=strict_improvement(independent['tsgfm'],independent[b]),
            mae_ratio=independent['tsgfm']/independent[b] if independent[b]>0 else None) for b in ('native','ufg')},
            shared_ufg_better_than_native=strict_improvement(independent['ufg'],independent['native']),
            uncertainty_tsgfm_minus_baselines=ci,queries=128,variables=len(set(variables)))
    seal(root/'report.json',dict(binding=binding,results=results,completion_sha256=completion,
        source_selection=m['selection'],source_adequacy=m['source_adequacy'],shape_specific_weights=False,
        same_checkpoint_and_tensor_sha_all_768=True,target_labels_used_for_fitting=False,
        source_parameters_unchanged=True,old_target_development_exposure=True,single_training_seed=True,
        independent_blind_test_claimed=False,goal_achieved=False,new_gpu_seconds=0,
        not_a_final_Memory_system_result=True))


def pipeline(cfg,source,snapshot,config_path,proof_path):
    from gofa_f.idle_launcher import process_identity
    from gofa_f.dual_memory_eval_pipeline import alive
    cpu_only();source=Path(source);proof=read(proof_path)
    if not proof['passed'] or proof['runtime_code']!=runtime() or proof['manifest_sha256']!=sha(source/'manifest.json'):
        raise ValueError('Tested exact shared baseline snapshot required')
    prior=Path(str(source)+'.launch.json')
    if prior.exists() and alive(read(prior)):raise ValueError('Shared UFG source is already live; do not duplicate')
    stopped=[False];child=None
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    command=[sys.executable,'-u','-m','gofa_f.ufg_shared_source','--config',str(config_path),'--output',str(source)]
    with Path(str(source)+'.log').open('ab') as log:
        child=subprocess.Popen(command,cwd=snapshot,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(prior,dict(pid=child.pid,start_ticks=process_identity(child.pid),command=command,snapshot=str(snapshot)))
    while child.poll() is None:
        if stopped[0]:
            child.send_signal(signal.SIGTERM)
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:raise RuntimeError('Source save timeout; audit identity before further action')
            raise InterruptedError('Shared baseline pipeline stopped')
        time.sleep(2)
    if child.returncode:raise RuntimeError('Source process did not complete; no automatic restart')
    if stopped[0]:raise InterruptedError('Stopped before target prediction')
    target=Path(cfg['evaluation']['output']);prepare(cfg,source,target);evaluate(cfg,source,target)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--source',required=True)
    p.add_argument('--snapshot',required=True);p.add_argument('--cpu-proof',required=True);a=p.parse_args()
    cfg=configuration(a.config);source=Path(a.source)
    atomic_json(source/'pipeline_status.json',dict(state='running',phase='source_then_six_target_retest',goal_achieved=False))
    try:
        pipeline(cfg,source,Path(a.snapshot),Path(a.config),Path(a.cpu_proof))
        atomic_json(source/'pipeline_status.json',dict(state='complete',phase='source_and_six_target_retest_complete',goal_achieved=False))
    except InterruptedError:
        atomic_json(source/'pipeline_status.json',dict(state='paused',phase='cooperative_stop',goal_achieved=False));raise
    except BaseException as exc:
        atomic_json(source/'pipeline_status.json',dict(state='failed',phase='shared_ufg_pipeline',error=repr(exc),goal_achieved=False));raise


if __name__=='__main__':main()
