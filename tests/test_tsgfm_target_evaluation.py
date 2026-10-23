from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from gofa_f.generative_data import sha
from gofa_f.dual_memory_engines import model_digest
from gofa_f.tsgfm_expert import TSGFMExpert
from gofa_f.tsgfm_target_evaluation import (TARGETS, PROTOCOL, query_contract, configuration,
    identity, checked_prediction, require_complete, source_locks)


def rows():
    return [dict(source=n,variable=i,origin=1000+i*30,length=12 if n in TARGETS[:2] else 48,
        horizon=12 if n in TARGETS[:2] else 24,chunk=0) for n in TARGETS for i in range(128)]


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path


@pytest.mark.parametrize('mutation',[
    lambda q:q.pop(),lambda q:q.append(deepcopy(q[0])),
    lambda q:q[0].update(source='other'),lambda q:q[0].update(horizon=24)])
def test_query_contract_rejects_partial_duplicate_substituted_or_different_tasks(mutation):
    values=rows();query_contract(values);mutation(values)
    with pytest.raises(ValueError):query_contract(values)


def test_frozen_configuration_rejects_training_and_target_selection(tmp_path,monkeypatch):
    monkeypatch.setenv('GOFA_WORKSPACE','/workspace');path=tmp_path/'config.yaml'
    cfg=dict(protocol=PROTOCOL,training={'max_updates':0},data=dict(target_training_enabled=False,target_evaluation_enabled=True),
        resources=dict(cuda_enabled=False,cpu_threads=2,deadline=None))
    path.write_text(yaml.safe_dump(cfg));assert configuration(path)==cfg
    cfg['data']['target_training_enabled']=True;path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):configuration(path)


@pytest.mark.parametrize('change',[
    lambda v:v['identity'].update(model_sha256='changed-weights'),
    lambda v:v['identity'].update(input_sha256='other-roster'),
    lambda v:v['identity']['query'].update(origin=999),
    lambda v:v['prediction'].pop(),lambda v:v['prediction'].__setitem__(0,float('nan'))])
def test_resume_rejects_wrong_model_input_query_or_number_slots(tmp_path,change):
    q=rows()[0];expected=identity('manifest',q,SimpleNamespace(identity='input'),'tsgfm','model')
    value=dict(identity=deepcopy(expected),prediction=[1.]*12);path=put(tmp_path/'record.json',value)
    assert checked_prediction(path,expected)==value
    change(value);put(path,value)
    with pytest.raises(ValueError):checked_prediction(path,expected)


def test_truth_remains_closed_for_missing_or_partial_prediction_group(tmp_path):
    put(tmp_path/'manifest.json',{})
    with pytest.raises(FileNotFoundError):require_complete(tmp_path,dict(input_index_sha256='input'),None)
    put(tmp_path/'native_complete.json',dict(complete=True,binding=sha(tmp_path/'manifest.json'),model='native',
        queries=768,input_index_sha256='input',shards={}))
    with pytest.raises(ValueError,match='Incomplete group'):require_complete(tmp_path,dict(input_index_sha256='input'),None)


def test_source_screen_failure_does_not_block_fixed_target_comparison(tmp_path,monkeypatch):
    import gofa_f.tsgfm_target_evaluation as mod
    expert=tmp_path/'expert';ufg=tmp_path/'ufg'
    selected=dict(target_labels_used=False,checkpoint_sha256='checkpoint',model_sha256='tensor')
    put(expert/'all/selection.json',selected);put(expert/'manifest.json',dict(tsgfm_runtime_code={}))
    put(expert/'report.json',dict(screen=dict(passed=False,geometric_domain_ratio=1.15)))
    entries={}
    for shape in ('12-12','48-24'):
        folder=ufg/shape;s=dict(selected,stop_reason='source_early_stop')
        put(folder/'selection.json',s);entries[shape]=dict(directory=str(folder),selection=s)
    put(ufg/'ufg_source_lock.json',dict(binding='ufg-manifest',target_labels_used=False,
        both_shapes_stopped_by_source_criterion=True,models=entries))
    monkeypatch.setattr(mod,'exited',lambda *_:None);monkeypatch.setattr(mod,'sha',lambda _:'checkpoint')
    result=source_locks(dict(expert_output=str(expert),expert_manifest_sha256='expert-manifest',
        ufg_output=str(ufg),ufg_manifest_sha256='ufg-manifest'))
    assert not result['source_screen']['passed']
    assert result['source_screen_is_not_a_target_gate']


def test_portable_model_tensor_export_has_identical_prediction_and_no_gradients(tmp_path):
    from test_tsgfm_expert import item
    torch.manual_seed(42);a=TSGFMExpert(width=8).eval();path=tmp_path/'model.pt'
    torch.save(dict(model=a.state_dict()),path);b=TSGFMExpert(width=8).eval()
    b.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'])
    for p in b.parameters():p.requires_grad_(False)
    before=model_digest(b)
    with torch.no_grad():
        for length,horizon in ((12,12),(48,24)):
            data=item(length,horizon,nodes=4,missing=True)
            torch.testing.assert_close(a.forecast(data),b.forecast(data),rtol=0,atol=0)
    assert model_digest(b)==before and all(p.grad is None for p in b.parameters())
