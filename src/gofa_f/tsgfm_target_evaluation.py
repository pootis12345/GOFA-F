"""One frozen six-target comparison; no training, screening or model search.

Source-screen failure is preserved as evidence, never interpreted as a target
score. The source-selected expert and adequately stopped official UFG are fixed
before target inputs open. Reuse the old immutable target queries/Native paths.
"""
import argparse
from collections import Counter, defaultdict
import csv
import math
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
import yaml

from gofa_f.dual_memory_data import SAMPLER, memory_input
from gofa_f.dual_memory_engines import model_digest
from gofa_f.dual_memory_eval_data import InputStore, TARGETS, read, seal
from gofa_f.dual_memory_eval_metrics import (macro_metrics, strict_improvement,
    spatial_blocks, paired_block_intervals)
from gofa_f.dual_memory_eval_records import PredictionStore
from gofa_f.dual_memory_policy import digest
from gofa_f.dual_memory_store import atomic_torch
from gofa_f.dual_memory_warmup import cpu_only
from gofa_f.generative_data import Query, Series, sha
from gofa_f.prefix_cache import atomic_json
from gofa_f.tsgfm_expert import TSGFMExpert

PROTOCOL='stage64-independent-tsgfm-targets-v1'
FILES=('tsgfm_target_evaluation.py','tsgfm_expert.py','dual_memory_data.py',
    'dual_memory_engines.py','dual_memory_eval_data.py','dual_memory_eval_records.py',
    'dual_memory_eval_metrics.py','dual_memory_policy.py','generative_data.py',
    'ufg_memory_roster.py','ufg_subgraph.py','ufg.py','prefix_cache.py')


def runtime():return {n:sha(Path(__file__).parent/n) for n in FILES}


def configuration(path):
    cfg=yaml.safe_load(Path(path).read_text().replace('${GOFA_WORKSPACE}',os.environ['GOFA_WORKSPACE']))
    if cfg['protocol']!=PROTOCOL or cfg['training']!={'max_updates':0} or \
        cfg['data']!={'target_training_enabled':False,'target_evaluation_enabled':True} or \
        cfg['resources']!=dict(cuda_enabled=False,cpu_threads=2,deadline=None):
        raise ValueError('Frozen CPU evaluation only, no training or target selection')
    return cfg


def exited(root,manifest_sha):
    from gofa_f.dual_memory_eval_pipeline import alive
    root=Path(root)
    if sha(root/'manifest.json')!=manifest_sha or read(root/'status.json')['state']!='complete' or \
        read(root/'activity.json').get('worker') is not None:
        raise ValueError('Completed immutable source fit required')
    launch=Path(str(root)+'.launch.json')
    if launch.exists() and alive(read(launch)):raise ValueError('Source process remains live')


def query_contract(rows):
    if Counter(q['source'] for q in rows)!=Counter({n:128 for n in TARGETS}) or \
        len({Query(**q).key for q in rows})!=768:
        raise ValueError('All six fixed128-query targets required')
    for q in rows:
        if (q['length'],q['horizon'])!=((12,12) if q['source'] in TARGETS[:2] else (48,24)):
            raise ValueError('Target forecasting task changed')


def source_locks(cfg):
    expert=Path(cfg['expert_output']);adequacy=Path(cfg['ufg_output'])
    exited(expert,cfg['expert_manifest_sha256']);exited(adequacy,cfg['ufg_manifest_sha256'])
    selected=read(expert/'all/selection.json');m=read(expert/'manifest.json')
    if selected['target_labels_used'] or sha(expert/'all/best.pt')!=selected['checkpoint_sha256']:
        raise ValueError('Expert source checkpoint changed or used target labels')
    for name,value in m['tsgfm_runtime_code'].items():
        if sha(Path(__file__).parent/name)!=value:raise ValueError('Trained expert execution changed: '+name)
    locked=read(adequacy/'ufg_source_lock.json')
    if locked['binding']!=cfg['ufg_manifest_sha256'] or locked['target_labels_used'] or \
        not locked['both_shapes_stopped_by_source_criterion']:
        raise ValueError('Adequately stopped source UFG required')
    models={}
    for shape,entry in locked['models'].items():
        folder=Path(entry['directory']);s=read(folder/'selection.json')
        if s!=entry['selection'] or s['target_labels_used'] or s['stop_reason']!='source_early_stop' or \
            sha(folder/'best.pt')!=s['checkpoint_sha256']:
            raise ValueError('UFG source lock changed')
        models[shape]=dict(path=str(folder/'best.pt'),selection=s)
    if set(models)!={'12-12','48-24'}:raise ValueError('Both source UFG shapes required')
    return dict(expert=dict(path=str(expert/'all/best.pt'),selection=selected),ufg=models,
        source_screen=read(expert/'report.json')['screen'],
        source_screen_is_not_a_target_gate=True,target_labels_used_for_selection=False)


def prepare(cfg,root):
    cpu_only();root=Path(root);root.mkdir(parents=True,exist_ok=True)
    locks=source_locks(cfg);reference=Path(cfg['target_reference'])
    if sha(reference/'manifest.json')!=cfg['target_manifest_sha256']:
        raise ValueError('Previously frozen target query manifest changed')
    old=read(reference/'manifest.json');query_contract(old['queries'])
    if old['scope']!='targets' or old['sampler']!=SAMPLER or old['query_choice_uses_prediction_errors'] or \
        old['target_labels_used_for_fitting']:
        raise ValueError('Lawful source-only, error-independent target reference required')
    model=TSGFMExpert();checkpoint=torch.load(locks['expert']['path'],map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model'])
    if model_digest(model)!=locks['expert']['selection']['model_sha256']:
        raise ValueError('Loaded expert is not source-selected best')
    # Export model tensors only: old optimizer/RNG pickle is not transported
    # across the two independently pinned Python/NumPy environments.
    portable=root/'expert-model-only.pt'
    if not portable.exists():atomic_torch(portable,dict(model=model.state_dict()))
    check=TSGFMExpert();check.load_state_dict(torch.load(portable,map_location='cpu',weights_only=False)['model'])
    if model_digest(check)!=model_digest(model):raise ValueError('Portable expert export changed weights')
    value=dict(protocol=PROTOCOL,configuration=cfg,locks=locks,queries=old['queries'],
        target_manifest_sha256=sha(reference/'manifest.json'),input_index_sha256=sha(reference/'inputs/index.json'),
        datasets=old['datasets'],runtime_sha256=runtime(),portable_expert_sha256=sha(portable),
        trainable_updates=0,old_target_development_exposure=True,independent_blind_test_claimed=False,
        original_source_screen_preserved=True,source_selected_weights_not_reselected=True,
        exact_paper_tsgfm_replication=False,shared_source_training_rosters_and_all_node_labels=True,
        ufg_shape_specific_weights_and_padding_disclosed=True)
    seal(root/'manifest.json',value)
    if not (root/'status.json').exists():atomic_json(root/'status.json',dict(state='prepared',goal_achieved=False))
    return value


def identity(binding,row,item,model,model_sha):
    return dict(binding=binding,query=row,input_sha256=item.identity,model=model,model_sha256=model_sha)


def checked_prediction(path,expected):
    if not Path(path).exists():return None
    value=read(path);p=np.asarray(value['prediction'],np.float64)
    if value['identity']!=expected or p.shape!=(expected['query']['horizon'],) or not np.isfinite(p).all():
        raise ValueError('Frozen prediction binding/coverage changed')
    return value


def require_complete(root,manifest,inputs):
    binding=sha(Path(root)/'manifest.json');result={}
    for name in ('native','tsgfm','ufg'):
        path=Path(root)/(name+'_complete.json');s=read(path)
        if not s['complete'] or s['binding']!=binding or s['model']!=name or \
            s['queries']!=768 or s['input_index_sha256']!=manifest['input_index_sha256']:
            raise ValueError('All three groups must finish before opening target truth')
        if len(s['shards'])!=768:raise ValueError('Incomplete group shards')
        for filename,value in s['shards'].items():
            if sha(Path(root)/filename)!=value:raise ValueError('Completed prediction was modified')
        result[name]=sha(path)
    return result


@torch.no_grad()
def predict(cfg,root):
    import fcntl
    from gofa_f.idle_launcher import process_identity
    from gofa_f.ufg_memory_roster import MemoryRosterUFG,pack_roster
    cpu_only();root=Path(root);handle=(root/'worker.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    m=read(root/'manifest.json');binding=sha(root/'manifest.json')
    if m['configuration']!=cfg or m['runtime_sha256']!=runtime() or source_locks(cfg)!=m['locks']:
        raise ValueError('Frozen evaluation runtime or source lock changed')
    if read(root/'status.json')['state']=='complete':return
    if read(root/'status.json')['state']=='failed':raise ValueError('Audit failure before resuming')
    proof=read(root/'cpu_preflight.json')
    if not proof['passed'] or proof['manifest_sha256']!=binding or proof['runtime_sha256']!=runtime():
        raise ValueError('CPU proof does not bind the deployed evaluator')
    ref=Path(cfg['target_reference'])
    if sha(ref/'manifest.json')!=m['target_manifest_sha256'] or sha(ref/'inputs/index.json')!=m['input_index_sha256'] or \
        sha(root/'expert-model-only.pt')!=m['portable_expert_sha256']:
        raise ValueError('Target inputs or portable weights changed')
    inputs=InputStore(ref);old=PredictionStore(ref);models={};digests={};stopped=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    activity=root/'activity.json';prior=read(activity)['spent_cpu_residency_seconds'] if activity.exists() else 0.
    started=time.monotonic();worker=dict(pid=os.getpid(),start_ticks=process_identity(os.getpid()))
    def progress(**fields):
        atomic_json(root/'status.json',dict(state='running',goal_achieved=False,**fields))
        atomic_json(activity,dict(worker=worker,spent_cpu_residency_seconds=prior+time.monotonic()-started,
            cuda_initialized=False,new_gpu_seconds=0,threads=2))
    try:
        progress(phase='verify_native_eight_saved_paths')
        # Verifies every pre-existing independent path and its pointwise median;
        # seal already exists, so this cannot cause any Native generation.
        native_seal=ref/'native_complete.json'
        if not native_seal.exists():raise ValueError('Frozen Native trajectories required; generation is forbidden')
        original_native_sha=sha(native_seal);old.complete('native',inputs)
        if sha(native_seal)!=original_native_sha:raise ValueError('Native completion evidence changed')
        expert=TSGFMExpert().eval();expert.load_state_dict(torch.load(root/'expert-model-only.pt',map_location='cpu',weights_only=False)['model'])
        for p in expert.parameters():p.requires_grad_(False)
        models['tsgfm']=expert;digests['tsgfm']=model_digest(expert)
        if digests['tsgfm']!=m['locks']['expert']['selection']['model_sha256']:raise ValueError('Expert tensor SHA changed')
        for name in ('native','tsgfm','ufg'):
            shards={}
            for index,row in enumerate(m['queries']):
                if stopped[0]:raise InterruptedError('Cooperative prediction pause')
                item,_=inputs.get(row);shape=f"{row['length']}-{row['horizon']}"
                key=digest(Query(**row).key);path=root/'predictions'/name/(key+'.json');path.parent.mkdir(parents=True,exist_ok=True)
                entry=m['locks']['ufg'][shape] if name=='ufg' else None
                model_sha=old.identity(row,item,'native')['model_sha256'] if name=='native' else \
                    digests['tsgfm'] if name=='tsgfm' else entry['selection']['model_sha256']
                expected=identity(binding,row,item,name,model_sha);value=checked_prediction(path,expected)
                if value is None:
                    tick=time.monotonic();origin=None
                    if name=='native':
                        saved=old.get(row,item,'native')
                        if saved is None:raise ValueError('Missing frozen Native result')
                        prediction=np.asarray(saved['prediction']);origin=dict(path=str(old.path(row,'native')),
                            sha256=sha(old.path(row,'native')),completion_sha256=original_native_sha,reused_not_new_generation=True)
                    elif name=='tsgfm':prediction=expert.forecast(item)[0].numpy()
                    else:
                        if shape not in models:
                            model=MemoryRosterUFG(cfg['ufg_root'],row['length'],row['horizon'],'cpu').eval()
                            model.load_state_dict(torch.load(entry['path'],map_location='cpu',weights_only=False)['model'])
                            for p in model.parameters():p.requires_grad_(False)
                            models[shape]=model;digests[shape]=model_digest(model)
                            if digests[shape]!=model_sha:raise ValueError('Loaded adequate UFG tensor SHA changed')
                        prediction=models[shape](pack_roster(item))[0].numpy()
                    p=np.asarray(prediction,np.float64)
                    if p.shape!=(row['horizon'],) or not np.isfinite(p).all():raise ValueError('Nonfinite or incomplete forecast')
                    atomic_json(path,dict(identity=expected,prediction=p.tolist(),seconds=time.monotonic()-tick,
                        real_nodes=len(item.node_ids),node_ids=list(item.node_ids),sampling_audit=item.audit,
                        reuse=origin,training_updates=0))
                shards[str(path.relative_to(root))]=sha(path)
                if index%16==0 or index==767:progress(phase='frozen_target_prediction',model=name,queries=index+1,total=768)
            seal(root/(name+'_complete.json'),dict(complete=True,binding=binding,model=name,queries=768,
                input_index_sha256=m['input_index_sha256'],shards=shards))
        if any(model_digest(v)!=digests[k] or any(p.grad is not None for p in v.parameters()) for k,v in models.items()):
            raise ValueError('Inference changed source weights or created gradients')
        source_locks(cfg)
        seal(root/'frozen_weights.json',dict(binding=binding,model_sha256=digests,parameters_modified=False,gradients_created=False))
        progress(phase='all_predictions_sealed_before_scoring')
        report=score(cfg,root,m,inputs)
        atomic_json(root/'status.json',dict(state='complete',phase='frozen_six_target_comparison',
            standalone_passes_both_all_six=report['standalone_passes_both_all_six'],goal_achieved=False,
            report_sha256=sha(root/'report.json'),eta_seconds=None))
    except InterruptedError:
        atomic_json(root/'status.json',dict(state='paused',phase='frozen_prediction',goal_achieved=False));raise
    except BaseException as exc:
        atomic_json(root/f'failure-{time.time_ns()}.json',dict(error=repr(exc),binding=binding))
        atomic_json(root/'status.json',dict(state='failed',error=repr(exc),goal_achieved=False));raise
    finally:
        atomic_json(activity,dict(worker=None,spent_cpu_residency_seconds=prior+time.monotonic()-started,
            cuda_initialized=False,new_gpu_seconds=0,threads=2,
            peak_rss_kib=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss))


def score(cfg,root,m,inputs):
    completion=require_complete(root,m,inputs);root=Path(root);results={};audit={};binding=sha(root/'manifest.json')
    with (root/'metrics.csv').open('w',newline='') as out:
        writer=csv.writer(out);writer.writerow(['dataset','model','variable_equal_mae','rmse','mape','wape'])
        for name in TARGETS:
            info=m['datasets'][name];data_path=Path(info['path'])
            if sha(data_path)!=info['sha256'] or sha(data_path.with_suffix('.json'))!=info['metadata_sha256']:
                raise ValueError('Frozen evaluation observations changed')
            data=Series(data_path);rows=[q for q in m['queries'] if q['source']==name]
            ys=[];masks=[];variables=[];clocks=[];predictions={k:[] for k in ('native','tsgfm','ufg')};node_counts=[]
            for row in rows:
                q=Query(**row);item,_=inputs.get(row)
                if memory_input(data,q).identity!=item.identity:raise ValueError('History/roster differs from frozen shared input')
                y=np.asarray(data.values[q.origin:q.origin+q.horizon,q.variable],np.float64)
                mask=np.asarray(data.mask[q.origin:q.origin+q.horizon,q.variable],bool)&np.isfinite(y)
                if not mask.any():raise ValueError('Missing all evaluation labels')
                ys.append(np.where(mask,y,0));masks.append(mask);variables.append(q.variable);clocks.append(int(data.times[q.origin]));node_counts.append(len(item.node_ids))
                for model in predictions:
                    path=root/'predictions'/model/(digest(q.key)+'.json');r=read(path)
                    predictions[model].append(r['prediction'])
            metrics={k:macro_metrics(p,ys,masks,variables) for k,p in predictions.items()}
            independent={}
            for k,p in predictions.items():
                errors=defaultdict(list)
                for v,pred,y,mask in zip(variables,p,ys,masks):
                    errors[v].extend(abs(float(a)-float(b)) for a,b,valid in zip(pred,y,mask) if valid)
                independent[k]=math.fsum(math.fsum(e)/len(e) for e in errors.values())/len(errors)
                if not math.isclose(independent[k],metrics[k]['macro']['mae'],rel_tol=1e-6,abs_tol=1e-6):
                    raise ValueError('Independent scalar MAE aggregation differs')
                macro=metrics[k]['macro'];writer.writerow([name,k,*[macro[x] for x in ('mae','rmse','mape','wape')]])
            comparisons={b:dict(strictly_improved=strict_improvement(independent['tsgfm'],independent[b]),
                mae_ratio=independent['tsgfm']/independent[b] if independent[b]>0 else None) for b in ('native','ufg')}
            space=spatial_blocks(data.external,data.ids)
            # Existing interval function calls the candidate 'fusion'. Here it
            # is strictly an internal alias for TSGFM, no output blending.
            intervals=paired_block_intervals(dict(native=predictions['native'],ufg=predictions['ufg'],fusion=predictions['tsgfm']),
                ys,masks,variables,[space[v] for v in variables],np.asarray(clocks)//(7*24*3600*10**9))
            results[name]=dict(metrics=metrics,comparison=comparisons,passes_both=all(v['strictly_improved'] for v in comparisons.values()),
                uncertainty=intervals,coverage=dict(queries=128,variables=len(set(variables)),real_node_counts=node_counts))
            audit[name]=independent
    report=dict(binding=binding,results=results,standalone_passes_both_all_six=all(v['passes_both'] for v in results.values()),
        goal_achieved=False,not_a_Memory_integration_result=True,source_screen=m['locks']['source_screen'],
        source_screen_did_not_block_target_evaluation=True,checkpoint_selection_used_only_source=True,
        target_labels_used_for_fitting=False,no_target_feedback_to_this_run=True,old_target_development_exposure=True,
        exact_paper_tsgfm_replication=False,single_seed=True,point_estimates_not_general_stability=True,
        completion_sha256=completion,frozen_weights_sha256=sha(root/'frozen_weights.json'),independent_scalar_mae=audit,
        new_gpu_seconds=0,native_predictions_reused_not_regenerated=True)
    seal(root/'report.json',report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--prepare',action='store_true');a=parser.parse_args();cfg=configuration(a.config)
    if a.prepare:prepare(cfg,Path(a.output))
    else:predict(cfg,Path(a.output))


if __name__=='__main__':main()
