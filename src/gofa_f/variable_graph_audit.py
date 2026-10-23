"""Read-only final evidence audit; no training, model search or prediction calls."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from gofa_f.dual_memory_data import GROUPS
from gofa_f.dual_memory_engines import model_digest
from gofa_f.dual_memory_policy import digest
from gofa_f.dual_memory_warmup import cpu_only
from gofa_f.generative_data import Query,Series,sha
from gofa_f.prefix_cache import atomic_json


def read(path):return json.loads(Path(path).read_text())


def close(a,b):
    if a is None or b is None:
        if a is not None or b is not None:raise ValueError('Missing metric is not zero')
    elif not math.isfinite(a) or not math.isfinite(b) or not math.isclose(a,b,rel_tol=1e-6,abs_tol=1e-6):
        raise ValueError(f'Independent metric replay differs: {a} versus {b}')


def scalar_metrics(records):
    """Independent scalar sums: valid points/variable, then equal variables."""
    variables=defaultdict(lambda:dict(error=[],squared=[],truth=[],percentage=[],steps=defaultdict(list),queries=0))
    for variable,prediction,target,mask in records:
        if not len(prediction)==len(target)==len(mask) or not any(mask):raise ValueError('Invalid forecast shape or empty mask')
        own=variables[str(variable)];own['queries']+=1
        for h,(p,y,valid) in enumerate(zip(prediction,target,mask),1):
            if not math.isfinite(float(p)):raise ValueError('Nonfinite prediction')
            if not valid:continue
            if not math.isfinite(float(y)):raise ValueError('Observed target must be finite')
            e=abs(float(p)-float(y));own['error'].append(e);own['squared'].append(e*e)
            own['truth'].append(abs(float(y)));own['steps'][h].append(e)
            if abs(float(y))>1e-5:own['percentage'].append(e/abs(float(y)))
    if not variables:raise ValueError('Empty evaluation')
    per={};steps=defaultdict(list)
    for v,s in variables.items():
        n=len(s['error']);absolute=math.fsum(s['error']);denominator=math.fsum(s['truth'])
        per[v]=dict(mae=absolute/n,rmse=math.sqrt(math.fsum(s['squared'])/n),
            mape=math.fsum(s['percentage'])/len(s['percentage']) if s['percentage'] else None,
            wape=absolute/denominator if denominator else None,observed_points=n,queries=s['queries'])
        for h,errors in s['steps'].items():steps[str(h)].append(math.fsum(errors)/len(errors))
    macro={}
    for name in ('mae','rmse','mape','wape'):
        values=[v[name] for v in per.values() if v[name] is not None]
        macro[name]=math.fsum(values)/len(values) if values else None
    return dict(macro=macro,per_variable=per,per_step_macro_mae={h:math.fsum(x)/len(x) for h,x in steps.items()},
        variables=len(per),queries=sum(v['queries'] for v in per.values()),observed_points=sum(v['observed_points'] for v in per.values()))


def compare_metrics(actual,saved):
    for key in ('variables','queries','observed_points'):
        if actual[key]!=saved[key]:raise ValueError('Evaluation coverage changed: '+key)
    for key in actual['macro']:close(actual['macro'][key],saved['macro'][key])
    for panel in ('per_variable','per_step_macro_mae'):
        if set(actual[panel])!=set(saved[panel]):raise ValueError('Variable or horizon coverage changed')
    for v,values in actual['per_variable'].items():
        for key,value in values.items():close(value,saved['per_variable'][v][key])
    for h,value in actual['per_step_macro_mae'].items():close(value,saved['per_step_macro_mae'][h])


def check_shards(root,completion,expected):
    root=Path(root)
    if not completion['complete'] or set(completion['shards'])!=set(expected):raise ValueError('Incomplete forecast seal')
    for name,hash_value in completion['shards'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path)!=hash_value:raise ValueError('Forecast shard changed: '+name)


def source_audit(root,manifest,kind):
    own=root/kind;lock=read(own/'source_lock.json');selection=read(own/'all/selection.json')
    if read(own/'status.json')['state']!='complete' or read(own/'activity.json')['worker'] is not None:
        raise ValueError('Completed source process required')
    if lock['binding']!=sha(root/'manifest.json') or lock['selection']!=selection or lock['target_labels_used'] or \
        not lock['one_parameter_set'] or lock['shape_specific_weights'] or selection['target_labels_used']:
        raise ValueError('Invalid shared source lock')
    checkpoint=own/'all/best.pt'
    if checkpoint.resolve()!=Path(lock['checkpoint_path']).resolve() or sha(checkpoint)!=selection['checkpoint_sha256']:
        raise ValueError('Locked checkpoint changed')
    best=torch.load(checkpoint,map_location='cpu',weights_only=False)
    tensors=model_digest(SimpleNamespace(state_dict=lambda:best['model']))
    if tensors!=selection['model_sha256'] or tensors!=best['model_sha256'] or best['update']!=selection['best_update']:
        raise ValueError('Source-best tensors or update changed')
    last=torch.load(own/'all/last.pt',map_location='cpu',weights_only=False);state=last['state']
    steps={int(v['step']) for v in last['optimizer']['state'].values() if 'step' in v}
    if not state['completed'] or state['pending'] or state['updates']!=selection['stopped_update'] or \
        steps!={state['updates']} or state['cursor']['total']!=state['updates']*manifest['variable_configuration']['training']['effective_batch']:
        raise ValueError('Optimizer, accumulation or query cursor is inconsistent')
    directory=own/'all/evaluations'/f"update-{best['update']:06d}"
    records=[read(p) for p in sorted(directory.glob('*.json'))]
    expected={Query(**q).key for d in manifest['queries'].values() for v in d.values() for q in v['development']}
    if len(records)!=len(expected) or {Query(**r['query']).key for r in records}!=expected:raise ValueError('Source validation coverage changed')
    shapes=defaultdict(lambda:defaultdict(list))
    for r in records:
        if r['identity']['checkpoint_sha']!=tensors or r['identity']['binding']!=last['binding'] or r['identity']['split']!='development':
            raise ValueError('Source prediction checkpoint binding changed')
        q=r['query'];shapes[(q['length'],q['horizon'])][q['source']].append((q['variable'],r['prediction'],r['target'],r['observed']))
    shape_scores=[]
    for (length,horizon),views in shapes.items():
        ratios=defaultdict(list);saved=selection['source_metrics']['per_shape'][f'{length}-{horizon}']
        for name,rows in views.items():
            metric=scalar_metrics(rows)['macro']['mae'];close(metric,saved['views'][name]['macro_mae'])
            ratios[GROUPS[name]].append(max(metric/manifest['training_unit_normalizers'][name],1e-12))
        domains=[math.exp(math.fsum(math.log(v) for v in x)/len(x)) for x in ratios.values()]
        score=math.exp(math.fsum(math.log(v) for v in domains)/len(domains));close(score,saved['selection_score']);shape_scores.append(score)
    score=math.exp(math.fsum(math.log(v) for v in shape_scores)/len(shape_scores));close(score,selection['source_metrics']['selection_score'])
    validations=[read(p) for p in sorted((own/'all').glob('validation-*.json'))]
    selected=min(validations,key=lambda r:(r['metrics']['selection_score'],r['update']))
    if selected['update']!=best['update']:raise ValueError('Checkpoint is not the strict source best')
    adequate=selection['stop_reason']=='source_early_stop'
    if adequate!=lock['source_adequacy']:raise ValueError('Source adequacy claim changed')
    return dict(best_update=best['update'],stopped_update=state['updates'],source_adequacy=adequate,
        checkpoint_sha256=sha(checkpoint),model_sha256=tensors,independent_source_score=score,
        optimizer_steps=sorted(steps),source_records=len(records),target_labels_used=False)


def audit(source,target):
    source=Path(source);target=Path(target);m=read(source/'manifest.json');plan=read(source/'target_plan.json')
    statuses={k:read(source/k/'status.json') if (source/k/'status.json').exists() else dict(state='not_started') for k in ('expert','ufg')}
    if any(v['state']!='complete' for v in statuses.values()) or not (target/'status.json').exists() or read(target/'status.json')['state']!='complete':
        return dict(state='incomplete',source_statuses=statuses,target_results_complete=False,goal_achieved=False)
    source_evidence={k:source_audit(source,m,k) for k in ('expert','ufg')}
    target_manifest=read(target/'manifest.json');binding=sha(target/'manifest.json');report=read(target/'report.json')
    if target_manifest['source_manifest_sha256']!=sha(source/'manifest.json') or target_manifest['target_plan_sha256']!=sha(source/'target_plan.json') or \
        target_manifest['plan']!=plan or report['binding']!=binding or read(target/'status.json')['report_sha256']!=sha(target/'report.json'):
        raise ValueError('Final report or data plan binding changed')
    for kind in ('expert','ufg'):
        if target_manifest['locks'][kind]!=read(source/kind/'source_lock.json'):raise ValueError('Target deployment differs from source lock')
    all_rows=plan['primary']+plan['supplement']
    for kind in ('native','expert','ufg'):
        rows=plan['primary'] if kind=='native' else all_rows
        expected={f'predictions/{kind}/{digest(Query(**q).key)}.json' for q in rows}
        complete=read(target/(kind+'_complete.json'))
        if complete['binding']!=binding or complete['queries']!=len(rows):raise ValueError('Model forecast coverage changed')
        check_shards(target,complete,expected)
    result={};data={}
    for panel in ('primary','supplement'):
        groups=defaultdict(list)
        for q in plan[panel]:groups[(q['source'],q['length'],q['horizon'])].append(q)
        for (name,length,horizon),rows in groups.items():
            if name not in data:
                info=plan['datasets'][name];p=Path(info['path'])
                if sha(p)!=info['sha256'] or sha(p.with_suffix('.json'))!=info['metadata_sha256']:raise ValueError('Target data changed')
                data[name]=Series(p)
            metrics={};kinds=('native','expert','ufg') if panel=='primary' else ('expert','ufg')
            for kind in kinds:
                actual=[]
                for row in rows:
                    q=Query(**row);record=read(target/f'predictions/{kind}/{digest(q.key)}.json')
                    if record['identity']['binding']!=binding or record['identity']['query']!=row:raise ValueError('Forecast query binding changed')
                    if kind!='native' and record['identity']['model_sha256']!=source_evidence[kind]['model_sha256']:raise ValueError('Forecast source tensors changed')
                    y=data[name].values[q.origin:q.origin+horizon,q.variable]
                    observed=data[name].mask[q.origin:q.origin+horizon,q.variable]&np.isfinite(y)
                    actual.append((q.variable,record['prediction'],y,observed))
                replay=scalar_metrics(actual);saved=report['results'][panel][f'{name}/{length}-{horizon}']['metrics'][kind]
                compare_metrics(replay,saved);metrics[kind]=replay['macro']
            result[f'{panel}/{name}/{length}-{horizon}']=metrics
    primary=[v for k,v in result.items() if k.startswith('primary/')]
    all_win=len(primary)==6 and all(x['expert']['mae']<x['ufg']['mae']-max(1e-12,1e-9*x['ufg']['mae']) for x in primary)
    if all_win!=report['primary_all_six_better_than_ufg']:raise ValueError('Six-target success flag changed')
    return dict(state='verified',source=source_evidence,independent_metrics=result,
        fixed_six_point_estimates_better_than_ufg=all_win,
        stage_criterion_met=all_win and source_evidence['ufg']['source_adequacy'],
        multiple_seed_stability_proved=False,GOFA_Memory_integration_completed=False,goal_achieved=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--target',required=True)
    p.add_argument('--output');a=p.parse_args();cpu_only();value=audit(a.source,a.target)
    if a.output:atomic_json(Path(a.output),value)
    print(json.dumps(value,indent=2))


if __name__=='__main__':main()
