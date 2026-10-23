"""Stage60 CPU completion audit, including warmup and adequate UFG provenance."""
import argparse
import json
import math
from pathlib import Path
import shlex

from gofa_f.dual_memory_delivery import (checked_reports,audit_predictions,source_training,
    csv_artifacts,render_summary,write_artifact)
from gofa_f.dual_memory_eval_data import read,seal
from gofa_f.dual_memory_policy import digest
from gofa_f.generative_data import sha


def warmup_training(source):
    m=read(Path(source)/'manifest.json');root=Path(m['formal_configuration']['warmup_output']);fits={}
    if sha(root/'selection_seal.json')!=m['warmup_selection_seal_sha256']:
        raise ValueError('Temporal initialization source seal changed')
    locked=read(root/'selection_seal.json')
    for name,entry in locked['selections'].items():
        selected=read(root/name/'selection.json')
        if sha(root/name/'selection.json')!=entry['selection_sha256'] or sha(root/name/'best.pt')!=entry['checkpoint_sha256']:
            raise ValueError('A source temporal initialization changed')
        fits[name]=dict(selection=selected,training_curve=[json.loads(v) for v in (root/name/'curve.jsonl').read_text().splitlines()],
            validation_curve=[read(p) for p in sorted((root/name).glob('validation-*.json'))])
    return dict(fits=fits,source_only=True,source_confirmation_used=False,target_labels_used=False,
        activity=read(root/'activity.json'),selection_seal_sha256=sha(root/'selection_seal.json'))


def costs_from(final):
    values={k:final[k] for k in ('inherited_source_gpu_seconds','new_evaluation_gpu_seconds','cumulative_gpu_seconds','peak_group_mib')}
    if not math.isclose(values['cumulative_gpu_seconds'],values['inherited_source_gpu_seconds']+values['new_evaluation_gpu_seconds'],rel_tol=1e-12,abs_tol=1e-9):
        raise ValueError('Cumulative GPU cost reset or counted twice')
    baseline=final['baseline_adequacy_cpu']
    old=final['source_ufg_cpu']['spent_seconds']
    if baseline['inherited_ufg_cpu_activity']['spent_seconds']!=old:raise ValueError('UFG ancestor CPU cost differs')
    values.update(inherited_cost_counted_once=True,inherited_prior_stage_gpu_seconds=final['inherited_prior_stage_gpu_seconds'],
        new_joint_source_gpu_seconds=final['new_joint_source_gpu_seconds'],
        source_ufg_cpu_residency_seconds=old+baseline['new_cpu_residency_seconds'],
        source_ufg_inherited_cpu_seconds=old,source_ufg_adequacy_new_cpu_seconds=baseline['new_cpu_residency_seconds'],
        temporal_initialization_cpu=final['source_temporal_initialization_cpu'],
        standalone_expert_source_cpu=final['standalone_expert_source_cpu'],
        source_alternative_review_activity=final['source_alternative_review_activity'],
        inherited_prior_cpu_costs=final['inherited_prior_cpu_costs'],
        evaluation_ufg_cpu_residency_seconds={k:v['spent_seconds'] for k,v in final['evaluation_ufg_cpu'].items()},
        coordinator_wait_is_not_GPU_or_CPU_compute=True,
        inherited_cpu_values_are_lineage_references_not_additional_training=True)
    return values


def readout_diagnostic(model):
    return dict(memory_read_columns_nonzero=bool(model['head.0.weight'][:,:512].any()),
        exact_persistence=not bool(model['head.2.weight'].any()) and not bool(model['head.2.bias'].any()),
        nonzero_columns_do_not_prove_graph_contribution=True)


def export(cfg,pipeline,output):
    import torch
    if torch.cuda.is_initialized():raise ValueError('Completion delivery must be CPU-only')
    pipeline,output=Path(pipeline),Path(output)
    confirmation,targets=Path(cfg['source_confirmation_output']),Path(cfg['target_output'])
    final,reports,manifests=checked_reports(pipeline,confirmation,targets)
    audits={};sampling={};texts={}
    for scope,path in (('source_confirmation',confirmation),('targets',targets)):
        audits[scope],sampling[scope],own=audit_predictions(path,reports[scope],manifests[scope]);texts.update(own)
    source=Path(final['deployment']['source_directory']);training=source_training(source)
    training['temporal_initializations']=warmup_training(source)
    for item in training['fits']:
        item['initial_reference_kind']='fresh_official_UFG' if item['fit'].startswith('ufg-') else 'pretrained_source_temporal_expert'
    checkpoint=torch.load(final['deployment']['expert']['path'],map_location='cpu',weights_only=False)
    model=checkpoint['model']
    training['final_expert_readout_diagnostic']=readout_diagnostic(model)
    training['adequate_ufg_selections']={shape:{k:v for k,v in value.items() if k!='path'}
        for shape,value in final['deployment']['ufg'].items()}
    short=Path(final['deployment']['ufg']['12-12']['path']).parent
    selected=read(short/'selection.json')
    training['legacy_UFG_fit_references']=[v for v in training['fits'] if v['fit'].startswith('ufg-')]
    training['fits']=[v for v in training['fits'] if v['fit']!='ufg-12-12']
    training['fits'].append(dict(fit='ufg-12-12',**{k:selected[k] for k in
        ('best_update','stopped_update','stop_reason','active_seconds','source_metrics','checkpoint_sha256','model_sha256')},
        selection_sha256=sha(short/'selection.json'),cap_is_not_convergence_claim=False,
        initial_reference_kind='fresh_official_UFG_before_exact_cap_continuation'))
    training['training_curves']=[v for v in training['training_curves'] if v['fit']!='ufg-12-12']
    for line in (short/'curve.jsonl').read_text().splitlines():
        row=json.loads(line);training['training_curves'].append(dict(fit='ufg-12-12',**{k:row[k] for k in
            ('update','loss','active_seconds','training_queries') if k in row}))
    training['source_validation_curves']=[v for v in training['source_validation_curves'] if v['fit']!='ufg-12-12']
    for path in sorted(short.glob('validation-*.json')):
        row=read(path);training['source_validation_curves'].append(dict(fit='ufg-12-12',update=row['update'],
            selection_score=row['metrics']['selection_score'],best_update=row['selection']['best_update'],
            view_metrics=row['metrics']['views']))
    costs=costs_from(final);deployment=final['deployment']
    portable={k:deployment[k] for k in ('depth','weights_sha256','source_manifest_sha256','source_model_lock_sha256',
        'core_sha256','shared_fusion_code_sha256','ufg_source_adequacy')}
    for key in ('expert','gate'):portable[key]={k:v for k,v in deployment[key].items() if k!='path'}
    portable['ufg']={shape:{k:v for k,v in value.items() if k!='path'} for shape,value in deployment['ufg'].items()}
    launch=read(Path(str(source)+'.launch.json'));command=launch['command']
    command=shlex.split(command) if isinstance(command,str) else command
    source_config=Path(command[command.index('--config')+1]);runtime=read(pipeline/'deployment_source.json')
    if sha(runtime['evaluation_config'])!=runtime['evaluation_config_sha256']:raise ValueError('Evaluation configuration file changed')
    portable['configuration_hashes']=dict(source_file_sha256=sha(source_config),evaluation_file_sha256=runtime['evaluation_config_sha256'],
        source_canonical_sha256=digest(read(source/'manifest.json')['formal_configuration']),evaluation_canonical_sha256=digest(cfg))
    portable['evaluation_runtime_sha256']=manifests['targets']['evaluation_runtime_sha256']
    portable_reports={scope:dict({k:v for k,v in report.items() if k not in ('deployment','gpu_activity','cpu_ufg_activity')},
        deployment=portable) for scope,report in reports.items()}
    bindings=dict(pipeline_report_sha256=sha(pipeline/'report.json'),source_confirmation_report_sha256=sha(confirmation/'report.json'),
        target_report_sha256=sha(targets/'report.json'),manifests={scope:r['binding'] for scope,r in reports.items()},
        goal_achieved=reports['targets']['goal_achieved'],portable_deployment=portable,
        target_labels_used_for_training_or_selection=False,independent_predictions_revalidated=True)
    output.mkdir(parents=True,exist_ok=True)
    json_files={'completion_audit.json':dict(bindings=bindings,audits=audits),'final_metrics.json':portable_reports['targets'],
        'source_confirmation_metrics.json':portable_reports['source_confirmation'],'training_summary.json':training,
        'costs.json':costs,'sampling_manifest.json':sampling,'text_manifest.json':texts,'deployment_sha.json':portable}
    for name,value in json_files.items():seal(output/name,value)
    documents=csv_artifacts(reports)
    documents['RESULTS.md']=render_summary(reports['targets'],reports['source_confirmation'],training,costs)+ \
        '\n联合更新0若被保留，指已经源训练的时间专家初始化，不等于未训练或持久性。四组CPU初始化及独立专家尝试的真实成本均保留。\n'+ \
        'UFG两形状都按原源早停标准锁定；短形状充分性续训及新增CPU成本已计入，原旧上限结果仍保留。非零Memory读出不自动证明图贡献。\n'
    for name,value in documents.items():write_artifact(output/name,value)
    seal(output/'checksums.json',dict(complete=True,goal_achieved=bindings['goal_achieved'],
        sha256={name:sha(output/name) for name in (*json_files,*documents)},no_new_predictions=True,source_reports_revalidated=True))
    return bindings


def main():
    from gofa_f.dual_memory_revalidation import configuration
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    parser.add_argument('--evaluation-output',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    result=export(configuration(args.config),args.evaluation_output,args.output)
    print(json.dumps(dict(complete_delivery=True,goal_achieved=result['goal_achieved'])))


if __name__=='__main__':main()
