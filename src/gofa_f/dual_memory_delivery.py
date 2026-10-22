"""CPU-only completion audit and portable deliverables from sealed predictions.

No model training, prediction, calibration, selection or results substitution.
Unfinished runs are rejected before source-confirmation/target artifacts open.
"""
import argparse
from collections import defaultdict
import csv
import io
import json
import math
import os
from pathlib import Path
import tempfile

from gofa_f.dual_memory_eval_data import MODELS,TARGETS,read,seal
from gofa_f.dual_memory_data import GROUPS,SHAPES
from gofa_f.dual_memory_eval_metrics import strict_improvement
from gofa_f.generative_data import sha


def checked_reports(pipeline,confirmation,targets):
    pipeline,confirmation,targets=map(Path,(pipeline,confirmation,targets))
    final=read(pipeline/'status.json')
    if final.get('state')!='complete' or not (pipeline/'report.json').exists():
        raise ValueError('The connected experiment is incomplete; no final result may be exported')
    result=read(pipeline/'report.json')
    if final!=result or result.get('target_results_not_used_to_refit') is not True:
        raise ValueError('Pipeline completion/report binding differs')
    reports={};manifests={}
    for scope,root,expected in (('source_confirmation',confirmation,result['source_confirmation_report_sha256']),
            ('targets',targets,result['target_report_sha256'])):
        if sha(root/'report.json')!=expected:raise ValueError('A final report changed after pipeline completion')
        report=read(root/'report.json');manifest=read(root/'manifest.json')
        if report['binding']!=sha(root/'manifest.json') or report['scope']!=scope or manifest['scope']!=scope or \
                report['deployment']!=result['deployment'] or manifest['deployment']!=result['deployment'] or \
                not report['all_model_groups_complete'] or report['parameters_modified'] or report['target_labels_used_for_fitting']:
            raise ValueError('Unbound/incomplete source-only deployment report')
        if report['gpu_activity'].get('worker') is not None or report['cpu_ufg_activity'].get('worker') is not None:
            raise ValueError('Prediction workers have not exited')
        reports[scope]=report;manifests[scope]=manifest
    source_rows=manifests['source_confirmation']['queries']
    if len(source_rows)!=256 or any(sum(q['source']==name and (q['length'],q['horizon'])==(length,horizon) for q in source_rows)!=32
            for name in GROUPS for length,horizon in SHAPES):raise ValueError('Fixed source confirmation coverage changed')
    target=reports['targets']
    if set(target['results'])!=set(TARGETS):raise ValueError('Missing target dataset; selective export forbidden')
    rows=manifests['targets']['queries']
    if len(rows)!=768 or any(sum(q['source']==name for q in rows)!=128 for name in TARGETS):
        raise ValueError('Original fixed six-target query budget changed')
    for scope,manifest in manifests.items():
        own=manifest['queries'];keys=[tuple(q.get(k,0) for k in ('source','variable','origin','length','horizon','chunk')) for q in own]
        if len(set(keys))!=len(own):raise ValueError('Duplicate queries cannot establish complete coverage')
    for row in rows:
        shape=(12,12) if row['source'] in ('pems03','pems_bay') else (48,24)
        if (row['length'],row['horizon'])!=shape:raise ValueError('Target forecast shape changed')
    for name,value in target['results'].items():
        if set(value['metrics'])!=set(MODELS):raise ValueError('A model group is missing')
        actual=value['metrics']['fusion']['macro']['mae']
        passes=all(strict_improvement(actual,value['metrics'][base]['macro']['mae']) for base in ('native','ufg'))
        if passes!=value['passes_both']:raise ValueError('Reported target gate does not match actual MAEs')
    achieved=all(value['passes_both'] for value in target['results'].values())
    if achieved!=target['goal_achieved'] or achieved!=result['goal_achieved']:
        raise ValueError('Goal claim differs from the six actual dual-baseline gates')
    return result,reports,manifests


def independent_mae(records,forecasts,scope):
    """Independent scalar summation; variable-equal valid-point raw-unit MAE."""
    errors={name:defaultdict(lambda:defaultdict(list)) for name in MODELS}
    for index,row in enumerate(records):
        q=row['query'];name=q['source'] if scope=='targets' else f"{q['source']}/{q['length']}-{q['horizon']}"
        y=row['target'];mask=row['observed']
        if len(y)!=q['horizon'] or len(mask)!=len(y) or not any(mask):raise ValueError('Invalid saved truth mask')
        for model in MODELS:
            values=forecasts[model][index]
            if len(values)!=len(y) or not all(math.isfinite(v) for v in values):raise ValueError('Incomplete/nonfinite model group')
            errors[model][name][q['variable']].extend(abs(float(p)-float(t)) for p,t,m in zip(values,y,mask) if m)
    output={}
    for model,groups in errors.items():
        output[model]={name:math.fsum(math.fsum(values)/len(values) for values in variables.values())/len(variables)
            for name,variables in groups.items()}
    return output


def audit_predictions(root,report,manifest):
    from gofa_f.dual_memory_eval_data import InputStore,truth_records,verify_deployment
    from gofa_f.dual_memory_eval_records import PredictionStore
    from gofa_f.dual_memory_eval_prepare import verify_runtime
    verify_runtime(manifest)
    inputs=InputStore(root);store=PredictionStore(root);verify_deployment(manifest['deployment'])
    seals={name:store.complete(name,inputs) for name in MODELS}
    if report['prediction_completion_sha256']!={name:sha(path) for name,path in seals.items()}:
        raise ValueError('Prediction completion evidence changed after scoring')
    truths=truth_records(root,seals);predictions={name:[] for name in MODELS};sampling=[];texts={}
    for row in manifest['queries']:
        item,text=inputs.get(row)
        from gofa_f.dual_memory_policy import digest
        text_hash=digest(text);texts[text_hash]=text
        for name in MODELS:predictions[name].append(store.get(row,item,name)['prediction'])
        sampling.append(dict(query=row,input_sha256=item.identity,real_nodes=len(item.node_ids),
            node_ids=list(item.node_ids),sampling_audit=item.audit,text_sha256=text_hash))
    independent=independent_mae(truths,predictions,manifest['scope']);differences={}
    for model,groups in independent.items():
        differences[model]={}
        for name,value in groups.items():
            reported=report['results'][name]['metrics'][model]['macro']['mae']
            if not math.isclose(value,reported,rel_tol=1e-6,abs_tol=1e-6):raise ValueError('Independent raw prediction MAE disagrees with report')
            differences[model][name]=abs(value-reported)
    if manifest['scope']=='targets':
        gates={name:all(strict_improvement(independent['fusion'][name],independent[base][name]) for base in ('native','ufg')) for name in TARGETS}
        if any(gates[n]!=report['results'][n]['passes_both'] for n in TARGETS):raise ValueError('Independent target gate changed')
    return dict(independent_macro_mae=independent,absolute_report_difference=differences,
        tolerance=1e-6,all_raw_prediction_bindings_valid=True,eight_native_trajectories_revalidated=True),sampling,texts


def csv_artifacts(reports):
    tables={name:[] for name in ('metrics','per_variable','per_step','paired_comparisons')}
    for scope,report in reports.items():
        for dataset,result in report['results'].items():
            for model,value in result['metrics'].items():
                base=dict(scope=scope,dataset=dataset,model=model)
                tables['metrics'].append(dict(base,**value['macro'],queries=value['queries'],variables=value['variables'],
                    observed_points=value['observed_points'],mape_valid_fraction=value['mape_valid_fraction'],
                    mape_fraction_not_percent=True,wape_fraction_not_percent=True))
                for variable,item in value['per_variable'].items():tables['per_variable'].append(dict(base,variable=variable,**item))
                for step,item in value['per_step'].items():tables['per_step'].append(dict(base,step=step,**item))
            for baseline,value in result['comparison'].items():
                tables['paired_comparisons'].append(dict(scope=scope,dataset=dataset,baseline=baseline,
                    strictly_improved=value['strictly_improved'],mae_ratio=value['mae_ratio'],
                    **{'queries_'+k:v for k,v in value['paired_queries'].items()},
                    **{'variables_'+k:v for k,v in value['paired_variables'].items()}))
    output={}
    for name,rows in tables.items():
        keys=list(dict.fromkeys(k for row in rows for k in row));stream=io.StringIO(newline='')
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
        output[name+'.csv']=stream.getvalue()
    return output


def write_artifact(path,value):
    """Byte-exact idempotence includes CSV CRLF, with atomic initial writes."""
    path=Path(path);payload=value.encode('utf-8')
    if path.exists():
        if path.read_bytes()!=payload:raise ValueError('A final exported artifact changed')
        return
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix=path.name+'.',suffix='.partial',delete=False) as stream:
        stream.write(payload);stream.flush();os.fsync(stream.fileno());temporary=Path(stream.name)
    os.replace(temporary,path)


def configuration_hashes(pipeline,source,manifests):
    from gofa_f.dual_memory_policy import digest
    launch=read(Path(str(source)+'.launch.json'))
    command=launch['command'];source_config=Path(command[command.index('--config')+1])
    runtime=read(Path(pipeline)/'deployment_source.json');evaluation_config=Path(runtime['snapshot'])/'configs/stage59_dual_memory_evaluation.yaml'
    return dict(source_configuration_file_sha256=sha(source_config),evaluation_configuration_file_sha256=sha(evaluation_config),
        evaluation_canonical_configuration_sha256=digest(manifests['targets']['configuration']),
        source_configuration_canonical_sha256=digest(read(Path(source)/'manifest.json')['formal_configuration']))


def render_summary(report,source,training,costs):
    achieved=report['goal_achieved'];deployment=report['deployment']
    lines=['# GOFA-F双流Memory零样本预测：封存结果','',
        '**固定六集均超过双基线，精度目标达成。**' if achieved else '**固定六集未全部超过双基线，精度目标未达成。**','',
        '下表均为变量等权原始单位MAE，不跨数据集混合物理量。增强流和融合为连续预测，不能另计为语言生成覆盖率。','',
        '| 数据集 | 原生ChatTime | 增强流 | 最终融合 | UFGTime | 融合/原生 | 融合/UFG | 双基线通过 |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for name in TARGETS:
        value=report['results'][name];mae={m:value['metrics'][m]['macro']['mae'] for m in MODELS}
        ratio=lambda base:'不适用' if mae[base]==0 else f"{mae['fusion']/mae[base]:.6f}"
        lines.append(f"| {name} | {mae['native']:.9g} | {mae['enhanced']:.9g} | {mae['fusion']:.9g} | {mae['ufg']:.9g} | {ratio('native')} | {ratio('ufg')} | {'是' if value['passes_both'] else '否'} |")
    lines+=['','每集完整MAE、RMSE、MAPE、WAPE以及逐变量/步指标见CSV；MAPE和WAPE以比例存储，空项表示无有效分母。',
        '配对空间/时间分块区间、覆盖、端点、回退和可表示范围审计见final_metrics.json；单个空间簇不提供虚假的空间置信区间。','',
        f"唯一增强专家SHA：`{deployment['expert']['checkpoint_sha256']}`。",
        f"共享融合SHA：`{deployment['gate']['sha256']}`；全局交织深度：{deployment['depth']}。",
        '原生保护流没有图回写；增强流仅更新专用Memory。所有真实节点经过同一冻结ChatTime，三跳整层各限10个候选，合并去重后最多31个真实节点。',
        'UFG保留官方有效结构和width128，使用相同历史节点/源监督资格；12→12与48→24的形状专用权重、尾部零填充已披露。','',
        f"累计GPU驻留活动：{costs['cumulative_gpu_seconds']/3600:.4f}小时；项目组观测峰：{costs['peak_group_mib']}MiB。",
        '累计值包含已继承机制和源训练成本一次；新增评测与各CPU worker驻留分别列出。驻留墙钟不是CPU核秒，不补造未观测峰值。','',
        '源确认使用从未参与本轮选模的预先封存时间窗；确认不返回挑选另一个模型。源和目标成绩、全部排名均完整保留。',
        '单训练种子、旧六集目标开发暴露、骨干预训练暴露不可核均须披露；本轮新查询不能消除历史开发暴露。',
        '气象/风电坐标四近邻为构造代理图，EngRAD是模式生成的气象场；本轮没有额外SCADA协变量。',
        '固定样本严格全胜也不等于总体稳定优势，不自动证明收益来自真实图或双向交织。',
        '本次仅源标签训练，没有目标标签训练、校准、提示或候选选择。未通过时不删除集合、返选模型、放宽门槛或根据该测试调参。','',
        f"独立源确认：{len(source['results'])}个视图/形状组合，四组结果在source_confirmation_metrics.json中完整保留。",
        '源选模分数按训练段单位归一化后聚合，不应误读为相对原生MAE比值。',
        f"训练选模记录共{len(training['fits'])}个拟合；若最佳为更新0或触及训练上限，会如实保留，不声称充分收敛。",'']
    return '\n'.join(lines)


def source_training(source):
    fits=[];curve=[];validation=[]
    for path in sorted(Path(source).rglob('selection.json')):
        # Only direct fit selections, not cache manifests/target diagnostics.
        if path.parent.name not in ('hold-traffic','hold-weather','hold-wind','final-source-model','ufg-12-12','ufg-48-24'):continue
        value=read(path);fit=str(path.parent.relative_to(source))
        fits.append(dict(fit=fit,**{k:value[k] for k in ('depth','best_update','stopped_update','stop_reason','active_seconds','source_metrics') if k in value},
            checkpoint_sha256=value['checkpoint_sha256'],model_sha256=value['model_sha256'],selection_sha256=sha(path),
            initial_reference_selected=value['best_update']==0,cap_is_not_convergence_claim=value['stop_reason']=='maximum_updates'))
        training=path.parent/'curve.jsonl'
        if training.exists():
            for line in training.read_text().splitlines():
                row=json.loads(line);curve.append(dict(fit=fit,**{k:row[k] for k in ('update','loss','active_seconds','training_queries') if k in row}))
        for file in sorted(path.parent.glob('validation-*.json')):
            row=read(file);validation.append(dict(fit=fit,update=row['update'],selection_score=row['metrics']['selection_score'],
                best_update=row['selection']['best_update'],view_metrics=row['metrics']['views']))
    return dict(fits=fits,training_curves=curve,source_validation_curves=validation,
        global_depth_selection=read(Path(source)/'depth_selection.json'),
        source_labels_only=True,traffic_channels_held_out_together=True)


def export(pipeline,output,confirmation=None,targets=None):
    import torch
    if torch.cuda.is_initialized():raise ValueError('Final delivery is CPU-only')
    pipeline=Path(pipeline);output=Path(output)
    confirmation=Path(confirmation) if confirmation else pipeline.parent/'stage59-dual-memory-source-confirm-v1'
    targets=Path(targets) if targets else pipeline.parent/'stage59-dual-memory-targets-v1'
    final,reports,manifests=checked_reports(pipeline,confirmation,targets)
    audits={};sampling={};texts={}
    for scope,path in (('source_confirmation',confirmation),('targets',targets)):
        audits[scope],sampling[scope],own=audit_predictions(path,reports[scope],manifests[scope]);texts.update(own)
    training=source_training(Path(final['deployment']['source_directory']))
    costs={k:final[k] for k in ('inherited_source_gpu_seconds','new_evaluation_gpu_seconds','cumulative_gpu_seconds','peak_group_mib')}
    costs['source_ufg_cpu_residency_seconds']=final['source_ufg_cpu']['spent_seconds']
    costs['evaluation_ufg_cpu_residency_seconds']={k:v['spent_seconds'] for k,v in final['evaluation_ufg_cpu'].items()}
    costs['inherited_cost_counted_once']=math.isclose(costs['cumulative_gpu_seconds'],costs['inherited_source_gpu_seconds']+costs['new_evaluation_gpu_seconds'],rel_tol=1e-12,abs_tol=1e-9)
    if not costs['inherited_cost_counted_once']:raise ValueError('Cumulative GPU cost double-counted or reset')
    deployment=final['deployment'];portable=dict(depth=deployment['depth'],weights_sha256=deployment['weights_sha256'],
        source_manifest_sha256=deployment['source_manifest_sha256'],source_model_lock_sha256=deployment['source_model_lock_sha256'],
        expert={k:v for k,v in deployment['expert'].items() if k!='path'},gate={k:v for k,v in deployment['gate'].items() if k!='path'},
        ufg={shape:{k:v for k,v in item.items() if k!='path'} for shape,item in deployment['ufg'].items()},
        core_sha256=deployment['core_sha256'],shared_fusion_code_sha256=deployment['shared_fusion_code_sha256'],
        evaluation_runtime_sha256=manifests['targets']['evaluation_runtime_sha256'])
    portable_reports={}
    for scope,report in reports.items():
        portable_reports[scope]={k:v for k,v in report.items() if k not in ('deployment','gpu_activity','cpu_ufg_activity')}
        portable_reports[scope]['deployment']=portable
    portable['configuration_hashes']=configuration_hashes(pipeline,Path(final['deployment']['source_directory']),manifests)
    bindings=dict(pipeline_report_sha256=sha(pipeline/'report.json'),source_confirmation_report_sha256=sha(confirmation/'report.json'),
        target_report_sha256=sha(targets/'report.json'),manifests={scope:report['binding'] for scope,report in reports.items()},
        goal_achieved=reports['targets']['goal_achieved'],portable_deployment=portable,
        target_labels_used_for_training_or_selection=False,independent_predictions_revalidated=True)
    output.mkdir(parents=True,exist_ok=True)
    json_files={'completion_audit.json':dict(bindings=bindings,audits=audits),'final_metrics.json':portable_reports['targets'],
        'source_confirmation_metrics.json':portable_reports['source_confirmation'],'training_summary.json':training,
        'costs.json':costs,'sampling_manifest.json':sampling,'text_manifest.json':texts,'deployment_sha.json':portable}
    for name,value in json_files.items():seal(output/name,value)
    documents=csv_artifacts(reports)
    documents['RESULTS.md']=render_summary(reports['targets'],reports['source_confirmation'],training,costs)
    for name,value in documents.items():write_artifact(output/name,value)
    checksums={name:sha(output/name) for name in [*json_files,*documents]}
    seal(output/'checksums.json',dict(complete=True,goal_achieved=bindings['goal_achieved'],sha256=checksums,
        source_reports_revalidated=True,no_new_predictions=True))
    return bindings


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--evaluation-output',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=export(args.evaluation_output,args.output)
    print(json.dumps(dict(goal_achieved=result['goal_achieved'],complete_delivery=True)))


if __name__=='__main__':main()
