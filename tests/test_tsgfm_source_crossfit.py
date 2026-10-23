from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gofa_f.dual_memory_data import GROUPS
from gofa_f.generative_data import Query,sha
from gofa_f.idle_launcher import process_identity
from gofa_f.tsgfm_source_crossfit import rows_for,paired_metrics,parent_complete,recheck_full_source,controller,configuration,PROTOCOL


def manifest():
    return dict(queries={name:{s:dict(development=[dict(source=name,variable=0,origin=100,
        length=int(s.split('-')[0]),horizon=int(s.split('-')[1]),chunk=0)],train=[],confirmation=[])
        for s in ('12-12','48-24')} for name in GROUPS},training_unit_normalizers={name:1. for name in GROUPS})


def records(m,error):
    return [dict(query=q,input_sha256=Query(**q).key,target=[2.]*q['horizon'],observed=[True]*q['horizon'],
        prediction=[2.+error]*q['horizon']) for d in m['queries'].values() for own in d.values() for q in own['development']]


def test_traffic_channels_are_held_together_and_shape_filter_is_explicit():
    m=manifest();rows=rows_for(m,'traffic')
    assert len(rows)==4 and {r['source'] for r in rows}=={'pems04_flow','pems04_speed'}
    assert len(rows_for(m,'weather','48-24'))==1
    with pytest.raises(ValueError):rows_for(m,'pems04_flow')
    with pytest.raises(ValueError):rows_for(m,'traffic',split='confirmation')


def test_paired_metrics_use_same_queries_rosters_and_masks_and_do_not_claim_targets():
    m=manifest();a,b=records(m,.8),records(m,1.)
    result=paired_metrics(a,b,m)
    assert result['gate']['passed'] and result['queries']==8
    assert result['gate']['scope']=='matched_whole_source_group_out_of_fit'
    assert result['gate']['fixed_six_target_superiority_proven'] is False
    assert result['same_real_history_rosters_and_valid_slots']


@pytest.mark.parametrize('change',[
    lambda r:r.pop(),lambda r:r.append(deepcopy(r[0])),
    lambda r:r[0].update(input_sha256='other-roster'),
    lambda r:r[0]['target'].__setitem__(0,1e10),
    lambda r:r[0]['observed'].__setitem__(0,False)])
def test_unmatched_source_evidence_is_rejected(change):
    m=manifest();a,b=records(m,.8),records(m,1.);change(b)
    with pytest.raises(ValueError):paired_metrics(a,b,m)


def test_pairing_checks_real_source_cache_not_only_mutually_agreeing_records():
    m=manifest();a,b=records(m,.8),records(m,1.)
    class Store:
        def get(self,q,split):
            assert split=='development'
            return SimpleNamespace(identity=Query(**q).key),torch.ones(1,q['horizon'])*2,torch.ones(1,q['horizon'],dtype=torch.bool)
    assert paired_metrics(a,b,m,Store())['gate']['passed']
    a[0]['target'][0]=b[0]['target'][0]=100
    with pytest.raises(ValueError,match='audited source'):paired_metrics(a,b,m,Store())


def test_parent_requires_verified_actual_process_and_exit_before_completion(tmp_path):
    root=tmp_path/'parent';root.mkdir();(root/'manifest.json').write_text('{}')
    (root/'status.json').write_text(json.dumps(dict(state='running')))
    (root/'activity.json').write_text(json.dumps(dict(worker=None)))
    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)',str(root),'gofa_f.tsgfm_source'])
    handle=dict(pid=child.pid,start_ticks=process_identity(child.pid))
    launch=Path(str(root)+'.launch.json');launch.write_text(json.dumps(handle))
    try:
        assert parent_complete(root,sha(root/'manifest.json'),'gofa_f.tsgfm_source') is False
        (root/'status.json').write_text(json.dumps(dict(state='complete')))
        assert parent_complete(root,sha(root/'manifest.json'),'gofa_f.tsgfm_source') is False
        child.terminate();child.wait()
        assert parent_complete(root,sha(root/'manifest.json'),'gofa_f.tsgfm_source')
        (root/'status.json').write_text(json.dumps(dict(state='running')))
        with pytest.raises(ValueError,match='no verified live'):parent_complete(root,sha(root/'manifest.json'),'gofa_f.tsgfm_source')
    finally:
        if child.poll() is None:child.terminate();child.wait()


def test_failed_first_screen_stops_before_UFG_or_new_training(tmp_path,monkeypatch):
    import gofa_f.tsgfm_source_crossfit as mod
    output=tmp_path/'output';output.mkdir();expert=tmp_path/'expert';expert.mkdir()
    (expert/'report.json').write_text(json.dumps(dict(screen=dict(passed=False))))
    cfg=dict(expert_output=str(expert),ufg_adequacy_output=str(tmp_path/'unopened'),
        expert_manifest_sha256='expert',ufg_adequacy_manifest_sha256='ufg',source_cache_manifest_sha256='cache')
    monkeypatch.setattr(mod,'require_proof',lambda *_:({},'binding'))
    monkeypatch.setattr(mod,'cpu_only',lambda:None)
    monkeypatch.setattr(mod,'parent_complete',lambda *_:True)
    monkeypatch.setattr(mod,'recheck_full_source',lambda *_:(_ for _ in ()).throw(AssertionError('must not reach UFG')))
    controller(cfg,output,tmp_path,tmp_path/'config')
    report=json.loads((output/'report.json').read_text())
    assert report['new_training_updates']==0 and report['target_predictions']==0
    assert not (output/'fits').exists()
    assert json.loads((output/'activity.json').read_text())['worker'] is None


def test_config_rejects_target_or_unmatched_group_changes(tmp_path,monkeypatch):
    import yaml
    monkeypatch.setenv('GOFA_WORKSPACE','/workspace')
    cfg=dict(protocol=PROTOCOL,held_groups=['traffic','weather','wind'],target_training_enabled=False,
        target_evaluation_enabled=False,ufg_maximum_updates=20000,resources=dict(cuda_enabled=False,cpu_threads=2,deadline=None))
    p=tmp_path/'config.yaml';p.write_text(yaml.safe_dump(cfg));assert configuration(p)==cfg
    cfg['held_groups']=['pems04_flow','weather','wind'];p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='whole-source'):configuration(p)


def test_new_UFG_recheck_must_actually_beat_the_improved_baseline(tmp_path,monkeypatch):
    import gofa_f.tsgfm_source_crossfit as mod
    m=manifest();expert=tmp_path/'expert';adequacy=tmp_path/'adequacy';cache=tmp_path/'cache'
    expert.mkdir();adequacy.mkdir();cache.mkdir()
    def selection(folder,error,shape=None,tensor='tensor'):
        folder.mkdir(parents=True,exist_ok=True);(folder/'best.pt').write_bytes(b'locked-weights')
        selected=dict(checkpoint_sha256=sha(folder/'best.pt'),target_labels_used=False,model_sha256=tensor,
            best_update=250,stop_reason='source_early_stop')
        (folder/'selection.json').write_text(json.dumps(selected));directory=folder/'evaluations/update-000250';directory.mkdir(parents=True)
        rows=[r for r in records(m,error) if shape is None or f"{r['query']['length']}-{r['query']['horizon']}"==shape]
        for i,r in enumerate(rows):
            r['identity']=dict(checkpoint_sha=tensor);(directory/f'{i:05d}.json').write_text(json.dumps(r))
        return selected
    selection(expert/'all',.8,tensor='expert')
    models={}
    for shape in ('12-12','48-24'):
        folder=adequacy/shape;s=selection(folder,.5,shape)
        models[shape]=dict(directory=str(folder),selection=s)
    lock=dict(binding='adequacy-manifest',both_shapes_stopped_by_source_criterion=True,target_labels_used=False,models=models)
    (adequacy/'ufg_source_lock.json').write_text(json.dumps(lock))
    class Store:
        def __init__(self,*_):pass
        def get(self,q,split):
            assert split=='development'
            return SimpleNamespace(identity=Query(**q).key),torch.ones(1,q['horizon'])*2,torch.ones(1,q['horizon'],dtype=torch.bool)
    monkeypatch.setattr(mod,'SourceStore',Store)
    cfg=dict(expert_output=str(expert),ufg_adequacy_output=str(adequacy),ufg_adequacy_manifest_sha256='adequacy-manifest',source_cache_output=str(cache))
    output=tmp_path/'out';output.mkdir()
    result=recheck_full_source(cfg,output,m,'manifest',lambda:None,lambda *_:None)
    assert result['passed'] is False and result['gate']['geometric_domain_ratio']>1
    assert result['source_prediction_provenance']
