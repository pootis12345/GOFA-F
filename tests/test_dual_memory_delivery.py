from copy import deepcopy
import csv
import io
import json

import pytest

from gofa_f.dual_memory_delivery import checked_reports,independent_mae,csv_artifacts,write_artifact,render_summary
from gofa_f.dual_memory_eval_data import MODELS,TARGETS
from gofa_f.dual_memory_data import GROUPS,SHAPES
from gofa_f.generative_data import sha
from gofa_f.prefix_cache import atomic_json


def test_unfinished_run_cannot_read_confirmation_or_targets(tmp_path,monkeypatch):
    import gofa_f.dual_memory_delivery as module
    pipeline=tmp_path/'pipeline';pipeline.mkdir();atomic_json(pipeline/'status.json',dict(state='running'))
    original=module.read;calls=[]
    def read(path):
        calls.append(path)
        if path!=pipeline/'status.json':raise AssertionError('Premature result access')
        return original(path)
    monkeypatch.setattr(module,'read',read)
    with pytest.raises(ValueError,match='incomplete'):checked_reports(pipeline,tmp_path/'confirmation',tmp_path/'targets')
    assert len(calls)==1


def test_independent_mae_is_valid_point_then_variable_equal():
    rows=[]
    for variable,origin,mask in ((0,10,[True,False]),(0,12,[True,True]),(1,10,[True,True])):
        rows.append(dict(query=dict(source='pems03',variable=variable,origin=origin,length=2,horizon=2),target=[0.,99.] if not mask[-1] else [0.,0.],observed=mask))
    base=[[2.,0.],[2.,2.],[10.,10.]]
    forecasts={model:[[v*(.5 if model=='fusion' else 1.) for v in row] for row in base] for model in MODELS}
    result=independent_mae(rows,forecasts,'targets')
    assert result['native']['pems03']==6. and result['fusion']['pems03']==3.
    forecasts['native'][0]=[2.]
    with pytest.raises(ValueError,match='Incomplete'):independent_mae(rows,forecasts,'targets')


def test_csv_retains_all_groups_null_metrics_and_step_values():
    result=dict(metrics={},comparison={})
    for model in MODELS:
        item=dict(mae=2.,rmse=3.,mape=None,wape=None)
        result['metrics'][model]=dict(macro=item,queries=128,variables=6,observed_points=2048,mape_valid_fraction=0.,
            per_variable={'v':dict(item,observed_points=3)},per_step={'1':item})
    for baseline in ('native','ufg'):
        result['comparison'][baseline]=dict(strictly_improved=False,mae_ratio=1.,
            paired_queries=dict(improved=1,tied=2,degraded=125),paired_variables=dict(improved=0,tied=0,degraded=6))
    tables=csv_artifacts(dict(targets=dict(results={'kelmarsh2019':result})))
    rows=list(csv.DictReader(io.StringIO(tables['metrics.csv'])))
    assert {r['model'] for r in rows}==set(MODELS) and all(r['mape']=='' and r['wape']=='' for r in rows)
    assert len(list(csv.DictReader(io.StringIO(tables['per_step.csv']))))==4


def test_export_text_is_atomic_and_byte_exact_across_csv_newlines(tmp_path):
    path=tmp_path/'metrics.csv';value='model,mae\r\nnative,1.25\r\n'
    write_artifact(path,value);first=path.read_bytes();write_artifact(path,value)
    assert first==value.encode() and not list(tmp_path.glob('*.partial'))
    with pytest.raises(ValueError,match='changed'):write_artifact(path,value.replace('1.25','1.24'))


def bound_reports(tmp_path):
    pipeline=tmp_path/'pipeline';confirmation=tmp_path/'confirmation';targets=tmp_path/'targets'
    for root in (pipeline,confirmation,targets):root.mkdir()
    deployment=dict(expert='unchanged');reports={}
    for scope,root in (('source_confirmation',confirmation),('targets',targets)):
        rows=[dict(source=name,variable=0,origin=100+turn*horizon,length=length,horizon=horizon) for name in GROUPS for length,horizon in SHAPES for turn in range(32)] if scope=='source_confirmation' else \
            [dict(source=name,variable=0,origin=100+i*24,length=12 if name in ('pems03','pems_bay') else 48,horizon=12 if name in ('pems03','pems_bay') else 24) for name in TARGETS for i in range(128)]
        manifest=dict(scope=scope,deployment=deployment,queries=rows);atomic_json(root/'manifest.json',manifest)
        values={name:dict(metrics={model:dict(macro=dict(mae=1. if model=='fusion' else 2.)) for model in MODELS},passes_both=True) for name in TARGETS}
        report=dict(binding=sha(root/'manifest.json'),scope=scope,deployment=deployment,all_model_groups_complete=True,
            parameters_modified=False,target_labels_used_for_fitting=False,gpu_activity=dict(worker=None),cpu_ufg_activity=dict(worker=None),
            results=values,goal_achieved=scope=='targets')
        reports[scope]=report;atomic_json(root/'report.json',report)
    result=dict(state='complete',deployment=deployment,goal_achieved=True,target_results_not_used_to_refit=True,
        source_confirmation_report_sha256=sha(confirmation/'report.json'),target_report_sha256=sha(targets/'report.json'))
    atomic_json(pipeline/'report.json',result);atomic_json(pipeline/'status.json',result)
    return pipeline,confirmation,targets,reports


def test_completion_requires_all_six_actual_dual_baseline_gates(tmp_path):
    pipeline,confirmation,targets,reports=bound_reports(tmp_path)
    assert checked_reports(pipeline,confirmation,targets)[0]['goal_achieved']
    report=reports['targets'];report['results']['sdwpf']['metrics']['ufg']['macro']['mae']=.5
    atomic_json(targets/'report.json',report)
    result=json.loads((pipeline/'report.json').read_text());result['target_report_sha256']=sha(targets/'report.json')
    atomic_json(pipeline/'report.json',result);atomic_json(pipeline/'status.json',result)
    with pytest.raises(ValueError,match='actual MAEs'):checked_reports(pipeline,confirmation,targets)


def test_changed_report_or_missing_dataset_never_becomes_a_success(tmp_path):
    pipeline,confirmation,targets,reports=bound_reports(tmp_path)
    report=reports['targets'];del report['results']['kelmarsh2019'];atomic_json(targets/'report.json',report)
    with pytest.raises(ValueError,match='changed'):checked_reports(pipeline,confirmation,targets)
    result=json.loads((pipeline/'report.json').read_text());result['target_report_sha256']=sha(targets/'report.json')
    atomic_json(pipeline/'report.json',result);atomic_json(pipeline/'status.json',result)
    with pytest.raises(ValueError,match='selective'):checked_reports(pipeline,confirmation,targets)
