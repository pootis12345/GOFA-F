"""Stage31 finite simpler blend families, reusing frozen Stage30 CPU models."""
import argparse
import copy
from pathlib import Path
import shutil
import time

import joblib
import numpy as np

from gofa_f.allnode_stage import read
from gofa_f.autonomous_iteration import (DEADLINE, check_time, exited_parent, inference_binding)
from gofa_f.autonomous_iteration_models import RECIPES, predict
from gofa_f.generative_data import Series, sha
from gofa_f.graph_benchmark_data import TARGETS
from gofa_f.history_calibration_cpu import exact_native_identity, frames_for
from gofa_f.history_calibration_math import TAUS, comparison, derived, fit
from gofa_f.history_calibration_state import verify_data
from gofa_f.prefix_cache import atomic_json
from gofa_f.short_forecast import summarize
from gofa_f.six_calibration_cpu import no_worker, read_shard, test_records
from gofa_f.six_calibration_state import immutable
from gofa_f.source_retraining import configuration
from gofa_f.supervised_ensemble_math import fit as simple_fit
from gofa_f.supervised_ensemble_state import original_development
from gofa_f.target_calibration import cpu_only
from gofa_f.target_calibration_math import improves
from gofa_f.target_calibration_state import code_hashes


def status(output,state,phase,**fields):
    atomic_json(output/'status.json',dict(state=state,phase=phase,stage=31,target_supervised=True,
        strict_zero_shot=False,deadline_unix=DEADLINE,updated_at=time.time(),**fields))


def done(output,reason,phase,goal=False):
    status(output,'complete',phase,stop_reason=reason,goal_achieved=goal)
    p=read(output/'progress.json',{})
    p.update(state='complete',overall_eta_seconds=None,stop_reason=reason)
    atomic_json(output/'progress.json',p)
    report(output)


def config_validate(cfg):
    if cfg['protocol']!='stage31-bounded-simplification-v1' or cfg['deadline_unix']!=DEADLINE \
            or cfg['resources']['admission_policy']!='memory_cap_only' \
            or cfg['resources']['memory_cap_gib']!=40 or cfg['resources']['torch_cap_gib']!=38 \
            or not cfg['data']['target_training_enabled'] or cfg['training']['max_updates']!=0:
        raise ValueError('Fixed Stage31 protocol changed')


def prepare(cfg,output):
    cpu_only()
    config_validate(cfg)
    check_time()
    root=Path(cfg['stage30_output'])
    exited_parent(root)
    if sha(root/'manifest.json')!=cfg['stage30_manifest_sha256'] \
            or read(root/'status.json')['stop_reason']!='bounded_cpu_candidates_exhausted':
        raise ValueError('Sealed failed Stage30 required')
    old=read(root/'manifest.json')
    verify_data(old)
    if any(code_hashes().get(n)!=h for n,h in old['unchanged_tensor_files'].items()):
        raise ValueError('Frozen numerical core changed')
    m=copy.deepcopy(old)
    models={}
    for name in TARGETS:
        models[name]={}
        for recipe in RECIPES:
            path=root/'candidates'/name/(recipe+'.joblib')
            meta=read(path.with_suffix('.json'))
            if not meta or sha(path)!=meta['model_sha256']:
                raise ValueError('All eight immutable parent models required')
            models[name][recipe]=meta['model_sha256']
    m.update(configuration=cfg,code=code_hashes(),stage30_development_sha256=sha(root/'development.json'),
        cpu_models_sha256=models,stage31=True)
    output.mkdir(parents=True,exist_ok=True)
    immutable(output/'manifest.json',m)
    for n in ['adapter.pt','gpu_identity.json']:
        if not (output/n).exists():
            shutil.copyfile(root/n,output/n)
    if not (output/'status.json').exists():
        status(output,'prepared','cpu_preparation')
    if not (output/'activity.json').exists():
        atomic_json(output/'activity.json',dict(spent_seconds=0.,worker=None,waiting_seconds=0.,
            peak_group_mib=0,resource_pause_count=0))
    return m


def manifest(cfg,output):
    config_validate(cfg)
    m=read(output/'manifest.json')
    if not m or m['configuration']!=cfg or m['code']!=code_hashes():
        raise ValueError('Stage31 manifest code/config changed')
    verify_data(m)
    root=Path(cfg['stage30_output'])
    if sha(root/'development.json')!=m['stage30_development_sha256']:
        raise ValueError('Inherited candidate evidence changed')
    return m


def model_predictions(cfg,m,name,data,queries,recipe):
    if recipe is None:
        return None
    path=Path(cfg['stage30_output'])/'candidates'/name/(recipe+'.joblib')
    if sha(path)!=m['cpu_models_sha256'][name][recipe]:
        raise ValueError('Selected CPU model/input binding changed')
    return predict(joblib.load(path),data,queries,recipe)


def frames(native,graph,data,cpu,control=False):
    own=frames_for(native,graph,data,control)
    if cpu is not None:
        for f,c in zip(own,cpu):
            f['names']=f['names']+['cpu']
            f['experts']=np.column_stack((f['experts'],c))
    return own


def convert_simple(parameters,names):
    w=dict(zip(['native','gofa','cpu'],parameters['weights']))
    return dict(experts=names,weights={n:w.get(n,0.) for n in names},bias=[0.,0.],tau=0.,
        source_solver=parameters,stored_parameters=sum(w.get(n,0.)!=0 for n in names))


def key(candidate):
    return (candidate['worst_fold_ratio'],candidate['pooled_mae'],
        ['simple','history_only','history_augmented'].index(candidate['family']),
        RECIPES.index(candidate['recipe']) if candidate['recipe'] else len(RECIPES),-candidate['tau'])


def eligible(folds,score,base):
    return all(f['mae']<=f['native_mae'] for f in folds) and score<=.995*base and improves(score,base)


def develop(cfg,output):
    cpu_only()
    no_worker(output)
    m=manifest(cfg,output)
    proof=read(output/'cpu_preflight.json',{})
    if not proof.get('passed') or proof.get('manifest_sha256')!=sha(output/'manifest.json'):
        raise ValueError('Bound full CPU proof required')
    if (output/'coefficient_lock.json').exists():
        raise ValueError('Cannot return to development after seal')
    if (output/'development.json').exists():
        return read(output/'development.json')
    start=time.monotonic()
    pairs,inputs=original_development(m['parent_configuration'])
    if inputs!=m['development_inputs_sha256']:
        raise ValueError('Development inputs changed')
    prior=read(Path(cfg['stage30_output'])/'development.json')
    datasets,selected={},{}
    for name in TARGETS:
        check_time()
        n,g=pairs[name]['native'],pairs[name]['gofa']
        data=Series(Path(m['data_directory'])/'targets'/(name+'.npz'))
        fold_pairs=((64,96),(96,128)) if name=='sdwpf' else ((32,48),(48,64))
        base=summarize(n[fold_pairs[0][0]:])['macro']['mae']
        routes=[('simple',recipe,0.) for recipe in RECIPES]+[('history_only',None,tau) for tau in TAUS]
        candidates=[]
        for family,recipe,tau in routes:
            check_time()
            cpu=model_predictions(cfg,m,name,data,[r['query'] for r in n],recipe)
            fs=frames(n,g,data,cpu)
            predictions,details=[],[]
            for a,b in fold_pairs:
                if family=='simple':
                    p=convert_simple(simple_fit(n[:a],g[:a],cpu[:a]),fs[0]['names'])
                else:
                    p=fit(n[:a],g[:a],fs[:a],tau)
                rows,metric=derived(n[a:b],g[a:b],fs[a:b],p)
                predictions.extend(rows)
                details.append(dict(begin=a,end=b,mae=metric['macro']['mae'],
                    native_mae=summarize(n[a:b])['macro']['mae'],parameters=p))
            score=summarize(predictions)['macro']['mae']
            row=dict(family=family,recipe=recipe,tau=tau,pooled_mae=score,native_mae=base,folds=details,
                worst_fold_ratio=max(f['mae']/f['native_mae'] for f in details),passed=eligible(details,score,base))
            candidates.append(row)
        previous=prior['datasets'][name]['selected']
        if previous:
            candidates.append(dict(previous,family='history_augmented',reused_stage30=True))
        choices=[x for x in candidates if x['passed']]
        choice=min(choices,key=key) if choices else None
        datasets[name]=dict(candidates=candidates,selected=choice,passed=choice is not None)
        if choice:
            selected[name]=choice
        atomic_json(output/'progress.json',dict(state='cpu_development',stage=31,dataset=name,
            datasets_done=len(datasets),datasets_total=6,overall_eta_seconds=None))
        print('Stage31',name,'eligible',len(choices),'selected',None if choice is None else
            (choice['family'],choice['recipe'],choice['tau'],choice['pooled_mae']),flush=True)
    value=dict(binding=sha(output/'manifest.json'),datasets=datasets,selected=selected,
        all_six_passed=len(selected)==6,confirmation_or_test_read=False,maximum_candidates_per_dataset=12)
    immutable(output/'development.json',value)
    atomic_json(output/'cpu_activity.json',dict(spent_seconds=time.monotonic()-start))
    if not value['all_six_passed']:
        done(output,'bounded_simplification_exhausted','cpu_development')
    else:
        status(output,'cpu_passed','cpu_development')
    return value


def seal(cfg,output):
    cpu_only()
    no_worker(output)
    m=manifest(cfg,output)
    if (output/'coefficient_lock.json').exists():
        return load_coefficients(cfg,output)[1]
    dev=read(output/'development.json')
    if not dev or not dev['all_six_passed']:
        raise ValueError('All six source-free development gates required')
    pairs,_=original_development(m['parent_configuration'])
    params={}
    for name,choice in dev['selected'].items():
        data=Series(Path(m['data_directory'])/'targets'/(name+'.npz'))
        n,g=pairs[name]['native'],pairs[name]['gofa']
        cpu=model_predictions(cfg,m,name,data,[r['query'] for r in n],choice['recipe'])
        own={}
        for control in [False,True]:
            fs=frames(n,g,data,cpu,control)
            p=convert_simple(simple_fit(n,g,cpu,control),fs[0]['names']) if choice['family']=='simple' \
                else fit(n,g,fs,choice['tau'])
            own['native_history' if control else 'full']=p
        params[name]=dict(own,recipe=choice['recipe'],family=choice['family'],
            model_sha256=m['cpu_models_sha256'][name][choice['recipe']] if choice['recipe'] else None)
    value=dict(binding=sha(output/'manifest.json'),coefficients=params,
        development_sha256=sha(output/'development.json'),confirmation_or_test_read=False)
    immutable(output/'coefficients.json',value)
    immutable(output/'coefficient_lock.json',dict(binding=value['binding'],coefficients_sha256=sha(output/'coefficients.json')))
    if any(exact_native_identity(x['full']) for x in params.values()):
        done(output,'native_identity_after_refit','postfit_feasibility')
    else:
        status(output,'coefficients_sealed','cpu_fit')
    return value


def load_coefficients(cfg,output):
    m=manifest(cfg,output)
    p,lock=read(output/'coefficients.json'),read(output/'coefficient_lock.json')
    if not p or not lock or lock['coefficients_sha256']!=sha(output/'coefficients.json') \
            or p['binding']!=sha(output/'manifest.json') \
            or p['development_sha256']!=sha(output/'development.json') \
            or not read(output/'development.json')['all_six_passed'] \
            or any(exact_native_identity(x['full']) for x in p['coefficients'].values()):
        raise ValueError('Sealed nonidentity all-six coefficients required')
    for name,x in p['coefficients'].items():
        if x['recipe'] and sha(Path(cfg['stage30_output'])/'candidates'/name/(x['recipe']+'.joblib'))!=x['model_sha256']:
            raise ValueError('CPU model changed after seal')
    return m,p,lock


def verify_inference(cfg,output):
    m,_,_=load_coefficients(cfg,output)
    if sha(output/'adapter.pt')!=cfg['adapter_sha256']:
        raise ValueError('Adapter changed')
    weights=Path(m['source_configuration']['weights']['directory'])
    if sha(weights/'snapshot.json')!=m['weights_sha256'] or any((weights/n).stat().st_size!=v['size'] \
            or (weights/n).stat().st_mtime_ns!=v['mtime_ns'] for n,v in m['weight_stats'].items()):
        raise ValueError('Frozen backbone files changed')
    return m


def compare(cfg,output,name,n,g):
    m,p,_=load_coefficients(cfg,output)
    params=p['coefficients'][name]
    data=Series(Path(m['data_directory'])/'targets'/(name+'.npz'))
    cpu=model_predictions(cfg,m,name,data,[r['query'] for r in n],params['recipe'])
    graph=g if g is not None else n
    result=comparison(n,graph,frames(n,graph,data,cpu),frames(n,graph,data,cpu,True),
        params['full'],params['native_history'],bootstrap=True)
    if g is None:
        result['metrics']['gofa']=None
        result['extra_graph_gain_demonstrated']=False
        for changes in result['changes'].values():
            changes.pop('gofa',None)
        result['paired_query_counts'].pop('gofa',None)
        result['paired_variable_cluster_bootstrap']['mae_difference_95pct'].pop('gofa',None)
    if cpu is None:
        result['metrics']['cpu']=None
    else:
        result['metrics']['cpu']=derived(n,graph,frames(n,graph,data,cpu),dict(params['full'],
            weights={k:float(k=='cpu') for k in params['full']['experts']},bias=[0.,0.]))[1]
    return result


def confirm_dataset(cfg,output,name,inference=False):
    if not inference:
        cpu_only()
        no_worker(output)
    m,p,lock=load_coefficients(cfg,output)
    for old in list(TARGETS)[:list(TARGETS).index(name)]:
        if not read(output/'confirmation'/(old+'.json'),{}).get('comparison',{}).get('strictly_improved'):
            raise ValueError('Earlier confirmation incomplete/failed')
    pair={}
    for mode in ['native']+(['gofa'] if p['coefficients'][name]['full']['weights'].get('gofa',0)!=0 else []):
        folder=output/'evaluations'/f'confirm-{name}-{mode}'
        if not read(folder/'complete.json',{}).get('complete'):
            raise ValueError('Confirmation forecasts incomplete')
        pair[mode]=[read_shard(folder,i,q,inference_binding(m,sha(output/'manifest.json'),name,mode),{})
            for i,q in enumerate(m['queries'][name][64:])]
    result=compare(cfg,output,name,pair['native'],pair.get('gofa'))
    value=dict(binding=p['binding'],coefficients_sha256=lock['coefficients_sha256'],comparison=result,refitted=False)
    (output/'confirmation').mkdir(exist_ok=True)
    immutable(output/'confirmation'/(name+'.json'),value)
    if not result['strictly_improved']:
        done(output,'failed_confirmation_'+name,'independent_confirmation')
    return value


def advance(cfg,output):
    cpu_only()
    no_worker(output)
    m,p,lock=load_coefficients(cfg,output)
    if (output/'test_replay.json').exists():
        return read(output/'test_replay.json')
    for name in TARGETS:
        if not confirm_dataset(cfg,output,name)['comparison']['strictly_improved']:
            return report(output)
    immutable(output/'confirm_gate.json',dict(binding=p['binding'],all_six_strictly_improved=True,
        coefficients_sha256=lock['coefficients_sha256']))
    results={}
    for name in TARGETS:
        n,g=(test_records(m['parent_configuration'],m,name,mode) for mode in ['native','gofa'])
        results[name]=compare(cfg,output,name,n,g)
    value=dict(binding=p['binding'],datasets=results,complete=True,new_test_trajectories=0,
        all_six_strictly_improved=all(x['strictly_improved'] for x in results.values()))
    immutable(output/'test_replay.json',value)
    done(output,'completed_frozen_test_replay','test_replay',value['all_six_strictly_improved'])
    return value


def report(output):
    value={n:read(output/(n+'.json')) for n in ['status','development','coefficients','activity',
        'cpu_activity','confirm_gate','test_replay','cuda_memory']}
    atomic_json(output/'report.json',value)
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=['prepare','run-cpu','report','progress'])
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--advance-cpu',action='store_true')
    args=parser.parse_args()
    cfg,output=configuration(args.config),args.output
    if args.advance_cpu:
        return advance(cfg,output)
    if args.phase=='prepare':
        return prepare(cfg,output)
    if args.phase=='run-cpu':
        if develop(cfg,output)['all_six_passed']:
            return seal(cfg,output)
        return report(output)
    if args.phase=='report':
        return report(output)
    if args.phase=='progress':
        print(read(output/'status.json'),read(output/'progress.json'))
        return
    from gofa_f.bounded_simplification_gpu import predict as gpu_predict
    return gpu_predict(cfg,output)


if __name__=='__main__':
    main()
