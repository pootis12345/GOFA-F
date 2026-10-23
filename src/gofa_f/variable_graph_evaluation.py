"""Source-locked zero-update expert/shared-UFG evaluation, including new lengths."""
from collections import defaultdict
import math
from pathlib import Path
import time

import numpy as np
import torch

from gofa_f.dual_memory_eval_data import TARGETS,InputStore,seal,read
from gofa_f.dual_memory_eval_metrics import macro_metrics,strict_improvement,paired_block_intervals,spatial_blocks
from gofa_f.dual_memory_engines import model_digest
from gofa_f.dual_memory_policy import digest
from gofa_f.generative_data import Query,Series,sha
from gofa_f.graph_memory_data import fresh_queries
from gofa_f.prefix_cache import atomic_json
from gofa_f.tsgfm_target_evaluation import validate_native,query_contract
from gofa_f.dual_memory_eval_records import PredictionStore
from gofa_f.variable_graph_data import variable_input,interval_seconds,HELD_SHAPES
from gofa_f.variable_graph_expert import VariableGraphExpert,pack_items
from gofa_f.ufg_memory_roster import pack_roster


def seal_target_plan(cfg,source):
    """Select only IDs, lawful timestamps and missing eligibility, before any fit."""
    reference=Path(cfg['evaluation']['reference']);path=reference/'manifest.json'
    if sha(path)!=cfg['evaluation']['reference_manifest_sha256']:raise ValueError('Frozen primary registry changed')
    old=read(path);query_contract(old['queries']);supplement=[]
    for name in TARGETS:
        info=old['datasets'][name];p=Path(info['path']);data=Series(p)
        if sha(p)!=info['sha256'] or sha(p.with_suffix('.json'))!=info['metadata_sha256']:raise ValueError('Target data changed')
        variables=sorted({q['variable'] for q in old['queries'] if q['source']==name})
        exclusions=[q for q in old['queries'] if q['source']==name]
        for l,h in HELD_SHAPES:
            own=fresh_queries(data,name,l,h,'test',8,variables,exclusions+supplement,66)
            for row in own:
                q=Query(**row);variable_input(data,q);interval_seconds(data,q)
            supplement.extend(own)
    plan=dict(primary=old['queries'],supplement=supplement,datasets=old['datasets'],
        reference_output=str(reference),reference_manifest_sha256=sha(path),
        original_target_reference=old['configuration']['target_reference'],
        original_target_manifest_sha256=old['target_manifest_sha256'],
        original_input_index_sha256=old['input_index_sha256'],target_values_used_for_selection=False,
        selection_rule='ID/time/eligibility only; fixed seed42/stage66;8 per target per held length pair',
        old_target_development_exposure=True,independent_blind_test_claimed=False)
    seal(Path(source)/'target_plan.json',plan);return plan


def identity(binding,row,item,model,model_sha):
    return dict(binding=binding,query=row,input_sha256=item.identity,model=model,model_sha256=model_sha)


@torch.no_grad()
def evaluate_targets(cfg,source,check,progress):
    from gofa_f.variable_ufg import VariableRosterUFG
    source=Path(source);plan=read(source/'target_plan.json');out=Path(cfg['evaluation']['output'])
    locks={kind:read(source/kind/'source_lock.json') for kind in ('expert','ufg')}
    for kind,lock in locks.items():
        if read(source/kind/'status.json')['state']!='complete' or read(source/kind/'activity.json')['worker'] is not None or \
            lock['binding']!=sha(source/'manifest.json') or lock['target_labels_used'] or not lock['one_parameter_set'] or \
            sha(lock['checkpoint_path'])!=lock['selection']['checkpoint_sha256']:
            raise ValueError('Exited complete source training and immutable single checkpoint required')
    m=dict(protocol='stage66-variable-graph-targets-v1',source_manifest_sha256=sha(source/'manifest.json'),
        target_plan_sha256=sha(source/'target_plan.json'),locks=locks,plan=plan,target_labels_used_for_fitting=False)
    out.mkdir(parents=True,exist_ok=True);seal(out/'manifest.json',m);binding=sha(out/'manifest.json')
    if (out/'status.json').exists() and read(out/'status.json')['state']=='complete':return read(out/'report.json')
    original=Path(plan['original_target_reference'])
    if sha(original/'manifest.json')!=plan['original_target_manifest_sha256'] or \
        sha(original/'inputs/index.json')!=plan['original_input_index_sha256']:raise ValueError('Original Native input binding changed')
    original_inputs=InputStore(original);old_native=PredictionStore(original)
    native_complete=validate_native(original,old_native,original_inputs)
    datasets={};all_rows=plan['primary']+plan['supplement'];primary_keys={Query(**q).key for q in plan['primary']}
    def data_for(row):
        name=row['source']
        if name not in datasets:
            info=plan['datasets'][name];p=Path(info['path'])
            if sha(p)!=info['sha256'] or sha(p.with_suffix('.json'))!=info['metadata_sha256']:raise ValueError('Target observations changed')
            datasets[name]=Series(p)
        return datasets[name]
    for kind in ('native','expert','ufg'):
        if kind=='native':model=None;rows=plan['primary'];before=None
        else:
            model=VariableGraphExpert(**cfg['expert']) if kind=='expert' else VariableRosterUFG(cfg['ufg_root'])
            model.load_state_dict(torch.load(locks[kind]['checkpoint_path'],map_location='cpu',weights_only=False)['model'])
            model.eval()
            for p in model.parameters():p.requires_grad_(False)
            before=model_digest(model);rows=all_rows
            if before!=locks[kind]['selection']['model_sha256']:raise ValueError('Source-best tensors changed')
        hashes={}
        for i,row in enumerate(rows):
            check();q=Query(**row);data=data_for(row);item=variable_input(data,q)
            if q.key in primary_keys:
                old_item,_=original_inputs.get(row)
                if item.identity!=old_item.identity:raise ValueError('Primary graph/history differs from original frozen input')
            model_sha=old_native.identity(row,item,'native')['model_sha256'] if kind=='native' else before
            expected=identity(binding,row,item,kind,model_sha);path=out/'predictions'/kind/(digest(q.key)+'.json')
            value=read(path) if path.exists() else None
            if value is not None:
                if value['identity']!=expected:raise ValueError('Saved forecast binding changed')
                pred=np.asarray(value['prediction'],float)
            else:
                started=time.monotonic();reuse=None
                if kind=='native':
                    previous=old_native.get(row,item,'native')
                    if previous is None:raise ValueError('Missing completed Native prediction')
                    pred=np.asarray(previous['prediction']);reuse=dict(path=str(old_native.path(row,'native')),
                        sha256=sha(old_native.path(row,'native')),reused_not_new_generation=True)
                elif kind=='expert':
                    pred=model(**pack_items([item]),interval=torch.tensor([interval_seconds(data,q)]),horizon=q.horizon)[0,0].numpy()
                else:pred=model(pack_roster(item),q.horizon)[0].numpy()
                if pred.shape!=(q.horizon,) or not np.isfinite(pred).all():raise ValueError('Complete finite prediction required')
                path.parent.mkdir(parents=True,exist_ok=True)
                atomic_json(path,dict(identity=expected,prediction=pred.tolist(),seconds=time.monotonic()-started,
                    reuse=reuse,real_nodes=len(item.node_ids),training_updates=0))
            if pred.shape!=(q.horizon,) or not np.isfinite(pred).all():raise ValueError('Saved prediction invalid')
            hashes[str(path.relative_to(out))]=sha(path)
            if i%16==0 or i+1==len(rows):progress(phase='zero_shot_target_prediction',model=kind,queries=i+1,total=len(rows))
        if model is not None and (model_digest(model)!=before or any(p.grad is not None for p in model.parameters())):
            raise ValueError('Inference changed source weights or created gradients')
        seal(out/(kind+'_complete.json'),dict(binding=binding,complete=True,queries=len(rows),shards=hashes))
        del model
    # No target truth is accessed for scoring until ALL registered predictions exist.
    results={}
    for panel in ('primary','supplement'):
        own_rows=plan[panel];by_task=defaultdict(list)
        for q in own_rows:by_task[(q['source'],q['length'],q['horizon'])].append(q)
        panel_results={}
        for (name,l,h),rows in by_task.items():
            data=data_for(rows[0]);ys=[];masks=[];variables=[];clocks=[]
            kinds=('native','expert','ufg') if panel=='primary' else ('expert','ufg');preds={k:[] for k in kinds}
            for row in rows:
                q=Query(**row);y=np.asarray(data.values[q.origin:q.origin+h,q.variable],np.float64)
                mask=np.asarray(data.mask[q.origin:q.origin+h,q.variable],bool)&np.isfinite(y)
                if not mask.any():raise ValueError('No valid evaluation target')
                ys.append(np.where(mask,y,0));masks.append(mask);variables.append(q.variable);clocks.append(int(data.times[q.origin]))
                for k in kinds:preds[k].append(read(out/'predictions'/k/(digest(q.key)+'.json'))['prediction'])
            metrics={k:macro_metrics(p,ys,masks,variables) for k,p in preds.items()};independent={}
            for k,p in preds.items():
                errors=defaultdict(list)
                for v,a,y,mask in zip(variables,p,ys,masks):errors[v].extend(abs(float(x)-float(t)) for x,t,z in zip(a,y,mask) if z)
                independent[k]=math.fsum(math.fsum(x)/len(x) for x in errors.values())/len(errors)
                if not math.isclose(independent[k],metrics[k]['macro']['mae'],rel_tol=1e-6,abs_tol=1e-6):raise ValueError('Independent MAE replay failed')
            ci=None
            if panel=='primary':
                space=spatial_blocks(data.external,data.ids)
                ci=paired_block_intervals(dict(native=preds['native'],ufg=preds['ufg'],fusion=preds['expert']),
                    ys,masks,variables,[space[v] for v in variables],np.asarray(clocks)//(7*24*3600*10**9))
            panel_results[name+f'/{l}-{h}']=dict(metrics=metrics,independent_scalar_mae=independent,
                expert_strictly_better_ufg=strict_improvement(independent['expert'],independent['ufg']),
                expert_strictly_better_native=strict_improvement(independent['expert'],independent['native']) if 'native' in kinds else None,
                uncertainty=ci,queries=len(rows),variables=len(set(variables)))
        results[panel]=panel_results
    report=dict(binding=binding,results=results,
        primary_all_six_better_than_ufg=all(v['expert_strictly_better_ufg'] for v in results['primary'].values()),
        shared_ufg_source_adequacy=locks['ufg']['source_adequacy'],
        supplementary_all_cells_better_than_ufg=all(v['expert_strictly_better_ufg'] for v in results['supplement'].values()),
        source_selection_only=True,target_labels_used_for_fitting=False,single_parameter_set_per_model=True,
        native_completion_sha256=native_complete,new_gpu_seconds=0,single_training_seed=True,
        old_target_development_exposure=True,independent_blind_test_claimed=False,
        general_population_stability_proved=False,GOFA_Memory_integration_completed=False,goal_achieved=False)
    seal(out/'report.json',report)
    atomic_json(out/'status.json',dict(state='complete',goal_achieved=False,report_sha256=sha(out/'report.json'),eta_seconds=None))
    return report
