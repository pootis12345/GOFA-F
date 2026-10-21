"""Stage33 finite simpler blend families, reusing frozen Stage30 CPU models."""
import argparse
import copy
from pathlib import Path
import shutil
import time

import joblib
import numpy as np

from gofa_f.allnode_stage import read
from gofa_f.autonomous_iteration import (DEADLINE, check_time, exited_parent, inference_binding)
from gofa_f.autonomous_iteration_models import predict
from gofa_f.scada_wind_models import (RECIPES, arrays as wind_arrays, estimator, fresh_confirmation,
    predict as wind_predict, load_scada, folds)
from gofa_f.scada_wind_data import convert
from gofa_f.autonomous_iteration_models import save_model
from gofa_f.target_calibration_state import digest
from gofa_f.generative_data import Series, sha
from gofa_f.graph_benchmark_data import TARGETS
from gofa_f.history_calibration_cpu import exact_native_identity, frames_for
from gofa_f.history_calibration_math import comparison, derived
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
    atomic_json(output/'status.json',dict(state=state,phase=phase,stage=33,target_supervised=True,
        strict_zero_shot=False,deadline_unix=DEADLINE,updated_at=time.time(),**fields))


def done(output,reason,phase,goal=False):
    status(output,'complete',phase,stop_reason=reason,goal_achieved=goal)
    p=read(output/'progress.json',{})
    p.update(state='complete',overall_eta_seconds=None,stop_reason=reason)
    atomic_json(output/'progress.json',p)
    report(output)


def config_validate(cfg):
    if cfg['protocol']!='stage33-causal-scada-v1' or cfg['deadline_unix']!=DEADLINE \
            or cfg['seed']!=42 or cfg['scada_recipes']!=list(RECIPES) \
            or cfg['resources']['admission_policy']!='memory_cap_only' \
            or cfg['resources']['memory_cap_gib']!=40 or cfg['resources']['torch_cap_gib']!=38 \
            or not cfg['data']['target_training_enabled'] or cfg['training']['max_updates']!=0:
        raise ValueError('Fixed Stage33 protocol changed')


def exposed_development(cfg, m):
    pairs, inputs = original_development(m['parent_configuration'])
    if inputs != m['development_inputs_sha256']:
        raise ValueError('Old development inputs changed')
    rows = list(pairs['sdwpf']['native'])
    for stage in (31, 32):
        root = Path(cfg[f'stage{stage}_output'])
        old = read(root/'manifest.json')
        binding = inference_binding(old, sha(root/'manifest.json'), 'sdwpf', 'native')
        folder = root/'evaluations/confirm-sdwpf-native'
        if not read(folder/'complete.json', {}).get('complete'):
            raise ValueError('Sealed exposed SDWPF predictions incomplete')
        rows.extend(read_shard(folder, i, q, binding, {})
                    for i, q in enumerate(old['queries']['sdwpf'][64:]))
    if len(rows) != 256 or len({digest(r['query']) for r in rows}) != 256:
        raise ValueError('Exactly256 distinct exposed development records required')
    return rows


def prepare(cfg, output):
    cpu_only()
    config_validate(cfg)
    check_time()
    root = Path(cfg['stage32_output'])
    exited_parent(root)
    if sha(root/'manifest.json') != cfg['stage32_manifest_sha256'] \
            or sha(root/'coefficients.json') != cfg['stage32_coefficients_sha256'] \
            or read(root/'status.json')['stop_reason'] != 'failed_confirmation_sdwpf' \
            or (root/'test_replay.json').exists():
        raise ValueError('Sealed stopped Stage32 with no test replay required')
    old = read(root/'manifest.json')
    verify_data(old)
    if any(code_hashes().get(n) != h for n, h in old['unchanged_tensor_files'].items()):
        raise ValueError('Frozen numerical core changed')
    data = Series(Path(old['data_directory'])/'targets/sdwpf.npz')
    development = exposed_development(cfg, old)
    folds(data, development)
    confirmation = fresh_confirmation(data, old, development)
    # Seal input identities before conversion, fitting, or generation.
    output.mkdir(parents=True, exist_ok=True)
    immutable(output/'preparation_queries.json', dict(development_queries=[r['query'] for r in development],
        confirmation_queries=confirmation, old_confirmation_exposed=True,
        old_confirmation_used_for_development=True, frozen_test_read=False))
    scada = convert(old['data_directory'], output, cfg['data_reader_directory'])
    m = copy.deepcopy(old)
    m['queries']['sdwpf'] = m['queries']['sdwpf'][:64]+confirmation
    m.update(configuration=cfg, code=code_hashes(), wind_recipes=list(RECIPES),
        scada=scada, exposed_development_sha256=digest(development),
        preparation_queries_sha256=sha(output/'preparation_queries.json'),
        frozen_stage31_coefficients=read(Path(cfg['stage31_output'])/'coefficients.json')['coefficients'],
        previous_confirmation_exposed=True, previous_confirmation_used_for_development=True)
    immutable(output/'manifest.json', m)
    for n in ['adapter.pt', 'gpu_identity.json']:
        if not (output/n).exists():
            shutil.copyfile(root/n, output/n)
    if not (output/'status.json').exists():
        status(output, 'prepared', 'cpu_preparation')
    if not (output/'activity.json').exists():
        activity = copy.deepcopy(read(root/'activity.json'))
        if activity.get('worker') is not None:
            raise ValueError('Old GPU worker not cleared')
        activity['inherited_prior_seconds'] = activity['spent_seconds']
        activity.pop('this_stage_seconds', None)
        atomic_json(output/'activity.json', activity)
    return m


def manifest(cfg,output):
    config_validate(cfg)
    m=read(output/'manifest.json')
    if not m or m['configuration']!=cfg or m['code']!=code_hashes():
        raise ValueError('Stage33 manifest code/config changed')
    verify_data(m)
    if sha(output/'scada.npz') != m['scada']['sha256'] or sha(output/'preparation_queries.json') != m['preparation_queries_sha256']:
        raise ValueError('SCADA/query binding changed')
    root=Path(cfg['stage31_output'])
    if sha(root/'coefficients.json')!=m['stage31_coefficients_sha256']:
        raise ValueError('Frozen other-five coefficients changed')
    return m


def model_predictions(cfg,m,name,data,queries,recipe):
    if recipe is None:
        return None
    if name=='sdwpf':
        output=Path(cfg['output_directory'])
        path=output/'candidates/sdwpf'/(recipe+'.joblib')
        meta=read(path.with_suffix('.json'))
        if not meta or meta['model_sha256']!=sha(path) \
                or meta['binding']!=digest(dict(manifest=sha(output/'manifest.json'),recipe=recipe)):
            raise ValueError('Wind CPU model/input binding changed')
        return wind_predict(joblib.load(path),data,load_scada(str(output),m['scada']['sha256']),queries,recipe)
    path=Path(cfg['stage30_output'])/'candidates'/name/(recipe+'.joblib')
    if sha(path)!=m['cpu_models_sha256'][name][recipe]:
        raise ValueError('Frozen other-five CPU model changed')
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
    n=exposed_development(cfg,m)
    if digest(n)!=m['exposed_development_sha256']:
        raise ValueError('Exposed development records changed')
    g=n
    data=Series(Path(m['data_directory'])/'targets/sdwpf.npz')
    folder=output/'candidates/sdwpf'
    folder.mkdir(parents=True,exist_ok=True)
    candidates=[]
    oldcost=read(output/'cpu_activity.json',dict(spent_seconds=0.))
    previous=time.monotonic()
    split=folds(data,n)
    valid=[int(i) for _,indices in split for i in indices]
    base=summarize([n[i] for i in valid])['macro']['mae']
    scada=load_scada(str(output),m['scada']['sha256'])
    for recipe in RECIPES:
        check_time()
        status(output,'running','wind_cpu_training',dataset='sdwpf',recipe=recipe)
        path=folder/(recipe+'.joblib')
        binding=digest(dict(manifest=sha(output/'manifest.json'),recipe=recipe))
        meta=read(path.with_suffix('.json'))
        if meta:
            if meta['binding']!=binding or meta['model_sha256']!=sha(path):
                raise ValueError('Wind candidate cache rebound')
            model=joblib.load(path)
        else:
            x,y,weights=wind_arrays(data,scada,m['wind_training_queries'],recipe)
            if 'extra' in recipe:
                x=np.nan_to_num(x,nan=0.)
            model=estimator()
            model.fit(x,y,sample_weight=weights)
            save_model(model,path)
            if not np.allclose(model.predict(x[:64]),joblib.load(path).predict(x[:64]),rtol=1e-6,atol=1e-6):
                raise ValueError('Wind saved model differs')
            immutable(path.with_suffix('.json'),dict(binding=binding,model_sha256=sha(path),
                training_queries=16384,recipe=recipe,training_loss='raw-unit absolute_error; equal-variable point weights'))
        cpu=wind_predict(model,data,scada,[r['query'] for r in n],recipe)
        fs=frames(n,g,data,cpu)
        predictions,details=[],[]
        for fit_indices,valid_indices in split:
            fitting=[n[int(i)] for i in fit_indices]
            validating=[n[int(i)] for i in valid_indices]
            own_cpu=[cpu[int(i)] for i in fit_indices]
            p=convert_simple(simple_fit(fitting,fitting,own_cpu,True),fs[0]['names'])
            rows,metric=derived(validating,validating,[fs[int(i)] for i in valid_indices],p)
            predictions.extend(rows)
            details.append(dict(fit_indices=fit_indices.tolist(),validation_indices=valid_indices.tolist(),
                mae=metric['macro']['mae'],native_mae=summarize(validating)['macro']['mae'],parameters=p))
        score=summarize(predictions)['macro']['mae']
        row=dict(recipe=recipe,family='simple',tau=0.,pooled_mae=score,native_mae=base,folds=details,
            worst_fold_ratio=max(f['mae']/f['native_mae'] for f in details),
            passed=eligible(details,score,base),model_sha256=sha(path))
        immutable(folder/(recipe+'-fit.json'),row)
        candidates.append(row)
        now=time.monotonic()
        oldcost['spent_seconds']+=now-previous
        previous=now
        atomic_json(output/'cpu_activity.json',oldcost)
        atomic_json(output/'progress.json',dict(state='wind_cpu_candidates',dataset='sdwpf',
            recipe=recipe,candidates_done=len(candidates),candidates_total=4,overall_eta_seconds=None))
        print('Stage33',recipe,'MAE',score,'passed',row['passed'],flush=True)
    choices=[x for x in candidates if x['passed']]
    selected=min(choices,key=lambda x:(x['worst_fold_ratio'],x['pooled_mae'],RECIPES.index(x['recipe']))) if choices else None
    value=dict(binding=sha(output/'manifest.json'),candidates=candidates,selected=selected,
        all_six_passed=selected is not None,frozen_other_five=True,confirmation_or_test_read=False,
        previously_exposed_confirmation_used_for_development=True, frozen_test_used_for_selection=False)
    immutable(output/'development.json',value)
    if not selected:
        done(output,'bounded_wind_candidates_exhausted','cpu_development')
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
        raise ValueError('Wind development gate required; other five frozen')
    choice=dev['selected']
    n=exposed_development(cfg,m)
    g=n
    data=Series(Path(m['data_directory'])/'targets/sdwpf.npz')
    cpu=model_predictions(cfg,m,'sdwpf',data,[r['query'] for r in n],choice['recipe'])
    params=copy.deepcopy(m['frozen_stage31_coefficients'])
    own={}
    for control in [False,True]:
        fs=frames(n,g,data,cpu,control)
        own['native_history' if control else 'full']=convert_simple(simple_fit(n,n,cpu,True),fs[0]['names'])
    params['sdwpf']=dict(own,recipe=choice['recipe'],family='simple',model_sha256=choice['model_sha256'])
    value=dict(binding=sha(output/'manifest.json'),coefficients=params,
        development_sha256=sha(output/'development.json'),confirmation_or_test_read=False,
        five_frozen_exactly=True,fresh_wind_confirmation_sha256=digest(m['queries']['sdwpf'][64:]))
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
        raise ValueError('Sealed nonidentity coefficients required')
    for name,x in p['coefficients'].items():
        if name=='sdwpf':
            path=output/'candidates/sdwpf'/(x['recipe']+'.joblib')
        else:
            if x!=m['frozen_stage31_coefficients'][name]:
                raise ValueError('Other-five frozen parameters changed')
            path=Path(cfg['stage30_output'])/'candidates'/name/(x['recipe']+'.joblib') if x['recipe'] else None
        if path and sha(path)!=x['model_sha256']:
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
    from gofa_f.scada_wind_iteration_gpu import predict as gpu_predict
    return gpu_predict(cfg,output)


if __name__=='__main__':
    main()
