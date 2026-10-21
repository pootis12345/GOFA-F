"""Stage30 bounded CPU candidates and one sealed confirmation/test, never target retries."""
import argparse
import copy
from pathlib import Path
import shutil
import time

import joblib
import numpy as np

from gofa_f.allnode_stage import read
from gofa_f.autonomous_iteration_models import RECIPES, arrays, estimator, predict, save_model
from gofa_f.generative_data import Series, sha
from gofa_f.graph_benchmark_data import TARGETS
from gofa_f.history_calibration_cpu import frames_for
from gofa_f.history_calibration_math import TAUS, derived, fit
from gofa_f.history_calibration_state import verify_data
from gofa_f.prefix_cache import atomic_json
from gofa_f.short_forecast import summarize
from gofa_f.short_sampled_state import exited
from gofa_f.six_calibration_cpu import no_worker, read_shard, test_records
from gofa_f.six_calibration_state import immutable
from gofa_f.source_retraining import configuration
from gofa_f.supervised_ensemble_state import original_development
from gofa_f.target_calibration import cpu_only
from gofa_f.target_calibration_math import improves
from gofa_f.target_calibration_state import code_hashes, digest

DEADLINE = 1792638912.  # 2026-10-22 03:15:12 UTC, original authorization +11h.


def check_time():
    if time.time() >= DEADLINE-60:
        raise TimeoutError('Fixed eleven-hour cutoff; no new active work')


def status(output, state, phase, **fields):
    atomic_json(output/'status.json', dict(state=state, phase=phase, stage=30,
        target_supervised=True, strict_zero_shot=False, neural_optimizer_updates=0,
        deadline_utc='2026-10-22T03:15:12Z', updated_at=time.time(), **fields))


def config_validate(cfg):
    if cfg['protocol'] != 'stage30-autonomous-iteration-v1' or cfg['recipes'] != list(RECIPES) \
            or cfg['deadline_unix'] != DEADLINE or cfg['taus'] != list(TAUS) \
            or cfg['resources']['admission_policy'] != 'memory_cap_only' \
            or cfg['resources']['memory_cap_gib'] != 40 or cfg['resources']['torch_cap_gib'] != 38 \
            or not cfg['data']['target_training_enabled'] or cfg['training']['max_updates'] != 0:
        raise ValueError('Stage30 preregistered candidate/resource/deadline policy changed')


def exited_parent(root):
    """Stage29 legitimately stopped before creating any GPU supervisor."""
    if (root/'launcher.json').exists():
        exited(root)
        return
    from gofa_f.idle_launcher import process_identity
    for path in [root.with_suffix('.launch.json'),root.with_suffix('.bootstrap.launch.json')]:
        identity=read(path,{})
        if identity.get('pid') and process_identity(identity['pid'])==str(identity.get('start_ticks')):
            raise ValueError('Parent CPU controller still alive')
    if read(root/'activity.json',{}).get('worker') is not None \
            or read(root/'cpu_activity.json',{}).get('active') is not None \
            or read(root/'status.json',{}).get('state')!='complete':
        raise ValueError('Parent activity must be terminal and empty')


def prepare(cfg, output):
    cpu_only()
    config_validate(cfg)
    check_time()
    root = Path(cfg['stage29_output'])
    exited_parent(root)
    if read(root/'status.json')['stop_reason'] != 'failed_cpu_development_gate' \
            or sha(root/'manifest.json') != cfg['stage29_manifest_sha256']:
        raise ValueError('Sealed failed Stage29 and immutable inputs required')
    old = read(root/'manifest.json')
    verify_data(old)
    if any(code_hashes().get(n) != h for n, h in old['unchanged_tensor_files'].items()):
        raise ValueError('Frozen numerical core changed')
    m = copy.deepcopy(old)
    m.update(configuration=cfg, code=code_hashes(), parent_manifest_sha256=sha(root/'manifest.json'),
        parent_configuration=old['configuration'], deadline_unix=DEADLINE)
    output.mkdir(parents=True, exist_ok=True)
    immutable(output/'manifest.json', m)
    for f in ['adapter.pt','gpu_identity.json']:
        if not (output/f).exists():
            shutil.copyfile(root/f, output/f)
    if sha(output/'adapter.pt') != cfg['adapter_sha256']:
        raise ValueError('Parent adapter SHA changed')
    if not (output/'activity.json').exists():
        atomic_json(output/'activity.json', dict(spent_seconds=0., worker=None, waiting_seconds=0.,
            peak_group_mib=0, resource_pause_count=0))
    if not (output/'status.json').exists():
        status(output, 'prepared', 'cpu_preparation')
    return m


def manifest(cfg, output):
    config_validate(cfg)
    m = read(output/'manifest.json')
    if not m or m['configuration'] != cfg or m['code'] != code_hashes():
        raise ValueError('Stage30 manifest/code/config differs')
    verify_data(m)
    return m


def done(output, reason, phase, goal=False):
    status(output, 'complete', phase, stop_reason=reason, goal_achieved=goal)
    progress = read(output/'progress.json', {})
    progress.update(state='complete', overall_eta_seconds=None, stop_reason=reason)
    atomic_json(output/'progress.json', progress)
    report(output)


def augmented(native, graph, data, cpu, no_graph=False):
    frames = frames_for(native, graph, data, no_graph)
    for f, own in zip(frames, cpu):
        f['names'] = f['names']+['cpu']
        f['experts'] = np.column_stack((f['experts'], own))
    return frames


def candidate_key(row):
    return (row['worst_fold_ratio'], row['pooled_mae'], RECIPES.index(row['recipe']), -row['tau'])


def eligibility(folds, pooled, baseline):
    return all(x['mae'] <= x['native_mae'] for x in folds) \
        and pooled <= .995*baseline and improves(pooled, baseline)


def develop(cfg, output):
    cpu_only()
    no_worker(output)
    m = manifest(cfg, output)
    proof = read(output/'cpu_preflight.json', {})
    if not proof.get('passed') or proof.get('manifest_sha256') != sha(output/'manifest.json'):
        raise ValueError('Complete bound CPU proof required')
    if (output/'coefficient_lock.json').exists():
        raise ValueError('Cannot return to development after seal')
    if (output/'development.json').exists():
        return read(output/'development.json')
    pairs, inputs = original_development(m['parent_configuration'])
    if inputs != m['development_inputs_sha256']:
        raise ValueError('Development prediction provenance changed')
    cost = read(output/'cpu_activity.json', dict(spent_seconds=0.))
    started, previous = time.monotonic(), time.monotonic()
    results, selected = {}, {}
    for name in TARGETS:
        check_time()
        n, g = pairs[name]['native'], pairs[name]['gofa']
        data = Series(Path(m['data_directory'])/'targets'/(name+'.npz'))
        folder = output/'candidates'/name
        folder.mkdir(parents=True, exist_ok=True)
        candidates = []
        folds = ((64,96),(96,128)) if name == 'sdwpf' else ((32,48),(48,64))
        native_mae = summarize(n[folds[0][0]:])['macro']['mae']
        for recipe in RECIPES:
            check_time()
            status(output, 'running', 'cpu_candidates', dataset=name, recipe=recipe)
            path = folder/(recipe+'.joblib')
            marker_path = folder/(recipe+'.json')
            binding = digest(dict(manifest=sha(output/'manifest.json'), dataset=name, recipe=recipe))
            meta = read(marker_path)
            if meta:
                if meta['binding'] != binding or meta['model_sha256'] != sha(path):
                    raise ValueError('Candidate model cache rebound')
                model = joblib.load(path)
            elif recipe == 'raw150':
                old = Path(cfg['stage29_output'])/'models'/(name+'.joblib')
                oldmeta = read(old.with_suffix('.json'))
                if sha(old) != oldmeta['model_sha256']:
                    raise ValueError('Reused Stage29 model SHA changed')
                shutil.copyfile(old, path)
                model = joblib.load(path)
                immutable(marker_path, dict(binding=binding, model_sha256=sha(path), reused=True))
            else:
                x, y, weights = arrays(data, m['training_queries'][name], recipe)
                if 'extra' in recipe:
                    x = np.nan_to_num(x, nan=0.)
                model = estimator(recipe)
                model.fit(x,y,sample_weight=weights)
                save_model(model,path)
                before = model.predict(x[:32])
                after = joblib.load(path).predict(x[:32])
                if not np.allclose(before,after,rtol=1e-6,atol=1e-6):
                    raise ValueError('Candidate serialization changed prediction')
                immutable(marker_path, dict(binding=binding, model_sha256=sha(path), reused=False))
            cpu = predict(model,data,[r['query'] for r in n],recipe)
            for tau in TAUS:
                fitpath = folder/(recipe+f'-tau{tau:g}.json')
                row = read(fitpath)
                if row is None:
                    fs = augmented(n,g,data,cpu)
                    predictions, details = [], []
                    for a,b in folds:
                        p = fit(n[:a],g[:a],fs[:a],tau)
                        own,metric = derived(n[a:b],g[a:b],fs[a:b],p)
                        predictions.extend(own)
                        details.append(dict(begin=a,end=b,mae=metric['macro']['mae'],
                            native_mae=summarize(n[a:b])['macro']['mae'],parameters=p))
                    score = summarize(predictions)['macro']['mae']
                    row = dict(binding=binding, recipe=recipe, tau=tau, model_sha256=sha(path),
                        pooled_mae=score, native_mae=native_mae, folds=details,
                        worst_fold_ratio=max(f['mae']/f['native_mae'] for f in details),
                        passed=eligibility(details,score,native_mae))
                    immutable(fitpath,row)
                if row['binding'] != binding:
                    raise ValueError('Candidate fit rebound')
                candidates.append(row)
            now=time.monotonic()
            cost['spent_seconds']+=now-previous
            previous=now
            atomic_json(output/'cpu_activity.json',cost)
            atomic_json(output/'progress.json',dict(state='cpu_candidates',dataset=name,recipe=recipe,
                datasets_done=len(results),datasets_total=6,candidate_tasks_done=len(candidates),
                candidate_tasks_total=24,elapsed_cpu_seconds=time.monotonic()-started,overall_eta_seconds=None))
        eligible = [x for x in candidates if x['passed']]
        choice = min(eligible,key=candidate_key) if eligible else None
        results[name]=dict(candidates=candidates,selected=choice,passed=choice is not None)
        print('Stage30',name,'eligible',len(eligible),'selected',
            None if choice is None else (choice['recipe'],choice['tau'],choice['pooled_mae']),flush=True)
        if choice:
            selected[name]=choice
    value=dict(binding=sha(output/'manifest.json'),datasets=results,all_six_passed=len(selected)==6,
        selected=selected,confirmation_or_test_read=False,finite_candidates_per_dataset=24)
    immutable(output/'development.json',value)
    if not value['all_six_passed']:
        done(output,'bounded_cpu_candidates_exhausted','cpu_development')
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
    if not dev or not dev['all_six_passed'] or dev['binding']!=sha(output/'manifest.json'):
        raise ValueError('All six development gates required')
    pairs,_=original_development(m['parent_configuration'])
    params={}
    for name,choice in dev['selected'].items():
        check_time()
        data=Series(Path(m['data_directory'])/'targets'/(name+'.npz'))
        path=output/'candidates'/name/(choice['recipe']+'.joblib')
        if sha(path)!=choice['model_sha256']:
            raise ValueError('Selected CPU model changed')
        n,g=pairs[name]['native'],pairs[name]['gofa']
        cpu=predict(joblib.load(path),data,[r['query'] for r in n],choice['recipe'])
        full=fit(n,g,augmented(n,g,data,cpu),choice['tau'])
        control=fit(n,g,augmented(n,g,data,cpu,True),choice['tau'])
        params[name]=dict(recipe=choice['recipe'],model_sha256=sha(path),full=full,native_history=control)
    value=dict(binding=sha(output/'manifest.json'),coefficients=params,
        development_sha256=sha(output/'development.json'),confirmation_or_test_read=False)
    immutable(output/'coefficients.json',value)
    immutable(output/'coefficient_lock.json',dict(binding=value['binding'],coefficients_sha256=sha(output/'coefficients.json')))
    from gofa_f.history_calibration_cpu import exact_native_identity
    if any(exact_native_identity(x['full']) for x in params.values()):
        done(output,'native_identity_after_refit','postfit_feasibility')
        return value
    status(output,'coefficients_sealed','cpu_fit')
    return value


def load_coefficients(cfg,output):
    m=manifest(cfg,output)
    p,lock=read(output/'coefficients.json'),read(output/'coefficient_lock.json')
    if not p or not lock or p['binding']!=sha(output/'manifest.json') \
            or lock['coefficients_sha256']!=sha(output/'coefficients.json') \
            or p['development_sha256']!=sha(output/'development.json') \
            or not read(output/'development.json')['all_six_passed']:
        raise ValueError('Sealed coefficients required')
    from gofa_f.history_calibration_cpu import exact_native_identity
    if any(exact_native_identity(x['full']) for x in p['coefficients'].values()):
        raise ValueError('Native identity cannot satisfy strict improvement')
    for name,x in p['coefficients'].items():
        if sha(output/'candidates'/name/(x['recipe']+'.joblib'))!=x['model_sha256']:
            raise ValueError('Sealed CPU model changed')
    return m,p,lock


def inference_binding(m,binding,name,mode):
    weight=m['weights_sha256'] if mode=='native' else m['configuration']['adapter_sha256']
    return f'{binding}:sample8:confirm-{name}-{mode}:{weight}:'+m['alphabet']['alphabet_sha256']


def verify_inference(cfg,output):
    m,_,_=load_coefficients(cfg,output)
    verify_data(m)
    if sha(output/'adapter.pt')!=cfg['adapter_sha256']:
        raise ValueError('Frozen parent adapter changed')
    return m


def compare(cfg,output,name,n,g):
    from gofa_f.history_calibration_math import comparison
    m,p,_=load_coefficients(cfg,output)
    params=p['coefficients'][name]
    data=Series(Path(m['data_directory'])/'targets'/(name+'.npz'))
    cpu=predict(joblib.load(output/'candidates'/name/(params['recipe']+'.joblib')),
        data,[r['query'] for r in n],params['recipe'])
    # A skipped graph path is mathematically unused; don't manufacture graph metrics.
    graph=g if g is not None else n
    result=comparison(n,graph,augmented(n,graph,data,cpu),augmented(n,graph,data,cpu,True),
        params['full'],params['native_history'],bootstrap=True)
    result['metrics']['cpu']=derived(n,graph,augmented(n,graph,data,cpu),dict(params['full'],
        weights={key:float(key=='cpu') for key in params['full']['experts']},bias=[0.,0.]))[1]
    if g is None:
        result['metrics']['gofa']=None
        result['graph_metrics']='skipped_exact_zero_weight'
        result['extra_graph_gain_demonstrated']=False
        for changes in result['changes'].values():
            changes.pop('gofa',None)
        result.get('paired_variable_cluster_bootstrap',{}).get('mae_difference_95pct',{}).pop('gofa',None)
        result['paired_query_counts'].pop('gofa',None)
    return result


def confirm_dataset(cfg,output,name,inference=False):
    if not inference:
        cpu_only()
        no_worker(output)
    m,p,lock=load_coefficients(cfg,output)
    for old in list(TARGETS)[:list(TARGETS).index(name)]:
        if not read(output/'confirmation'/(old+'.json'),{}).get('comparison',{}).get('strictly_improved'):
            raise ValueError('Earlier dataset failed or incomplete')
    modes=['native']+(['gofa'] if p['coefficients'][name]['full']['weights'].get('gofa',0)!=0 else [])
    pairs={}
    for mode in modes:
        label=f'confirm-{name}-{mode}'
        marker=read(output/'evaluations'/label/'complete.json',{})
        if not marker.get('complete'):
            raise ValueError('Incomplete required confirmation predictions')
        pairs[mode]=[read_shard(output/'evaluations'/label,i,q,
            inference_binding(m,sha(output/'manifest.json'),name,mode),{}) for i,q in enumerate(m['queries'][name][64:])]
    result=compare(cfg,output,name,pairs['native'],pairs.get('gofa'))
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
    value=dict(binding=p['binding'],datasets=results,complete=True,
        all_six_strictly_improved=all(x['strictly_improved'] for x in results.values()),new_test_trajectories=0)
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
    parser.add_argument('--phase',choices=['prepare','develop','seal','run-cpu','report','progress'])
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--advance-cpu',action='store_true')
    args=parser.parse_args()
    cfg,output=configuration(args.config),args.output
    try:
        if args.advance_cpu:
            return advance(cfg,output)
        if args.phase=='prepare':
            return prepare(cfg,output)
        if args.phase in ('develop','run-cpu'):
            if not develop(cfg,output)['all_six_passed']:
                return report(output)
        if args.phase in ('seal','run-cpu'):
            return seal(cfg,output)
        if args.phase=='report':
            return report(output)
        if args.phase=='progress':
            print(read(output/'status.json'),read(output/'progress.json'))
            return
        from gofa_f.autonomous_iteration_gpu import predict as gpu_predict
        return gpu_predict(cfg,output)
    except TimeoutError:
        done(output,'eleven_hour_deadline','cutoff')
    except Exception as exc:
        atomic_json(output/'failures.json',dict(type=type(exc).__name__,error=str(exc)))
        status(output,'failed',args.phase or 'predict',error=str(exc))
        raise


if __name__=='__main__':
    main()
