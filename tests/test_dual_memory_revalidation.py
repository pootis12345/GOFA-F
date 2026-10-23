from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gofa_f.dual_memory_data import GROUPS
from gofa_f.dual_memory_revalidation import fresh_targets,prepare_scope,adequate_deployment
from gofa_f.dual_memory_revalidation_pipeline import route_review,launch
from gofa_f.dual_memory_revalidation_delivery import costs_from,readout_diagnostic
from gofa_f.generative_data import sha
from gofa_f.prefix_cache import atomic_json


def test_fresh_target_windows_exclude_original_and_stage59_without_values():
    class Data:
        ids=['a','b']
        @property
        def values(self):raise AssertionError('Query choice read target values')
        def candidates(self,*_):return np.arange(16000,24000)
    original=[dict(source='pems03',variable=i%2,origin=16000+i//2*24,length=12,horizon=12,chunk=0) for i in range(128)]
    previous=[dict(q,origin=q['origin']+2200) for q in original]
    rows=fresh_targets(Data(),'pems03',original,previous)
    assert rows==fresh_targets(Data(),'pems03',original,previous)
    assert Counter(q['variable'] for q in rows)==Counter(q['variable'] for q in original)
    for i,q in enumerate(rows):
        for other in original+previous+rows[:i]:
            if q['variable']==other['variable']:
                assert q['origin']+12<=other['origin'] or q['origin']>=other['origin']+12


def test_fresh_capacity_failure_does_not_change_variables_or_budget():
    data=SimpleNamespace(ids=['a'],candidates=lambda *_:np.array([100]))
    old=[dict(source='pems03',variable=0,origin=100,length=12,horizon=12,chunk=0)]*128
    with pytest.raises(ValueError,match='capacity'):fresh_targets(data,'pems03',old,[])


def test_target_preparation_checks_confirmation_before_old_registry_access(tmp_path,monkeypatch):
    import gofa_f.dual_memory_revalidation as mod
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');source=tmp_path/'source';source.mkdir()
    atomic_json(source/'manifest.json',{})
    cfg=dict(source_training_manifest_sha256=sha(source/'manifest.json'),source_confirmation_output=str(tmp_path/'unopened'))
    monkeypatch.setattr(mod,'verify_deployment',lambda *_:None)
    monkeypatch.setattr(mod,'require_confirmation',lambda *_:(_ for _ in ()).throw(ValueError('confirmation required')))
    with pytest.raises(ValueError,match='confirmation'):prepare_scope(cfg,tmp_path/'out','targets',dict(source_directory=str(source)))
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('phase,decision',[
    ('not_started_standalone_source_screen_failed','continue_existing_memory_candidate'),
    ('not_started_source_recheck_or_adequacy_failed','continue_existing_memory_candidate'),
    ('matched_source_gate_failed','continue_existing_memory_candidate'),
    ('matched_source_gate_passed','prefer_new_expert_integration')])
def test_route_choice_is_fixed_by_source_review_not_targets(tmp_path,monkeypatch,phase,decision):
    import gofa_f.dual_memory_revalidation_pipeline as mod
    root=tmp_path/'review';root.mkdir();atomic_json(root/'manifest.json',{})
    atomic_json(root/'status.json',dict(state='complete'));atomic_json(Path(str(root)+'.launch.json'),dict(pid=123,start_ticks='x'))
    atomic_json(root/'activity.json',dict(worker=None))
    binding=sha(root/'manifest.json');atomic_json(root/'report.json',dict(binding=binding,target_predictions=0,phase=phase))
    monkeypatch.setattr(mod,'alive',lambda _:False)
    assert route_review(dict(source_alternative_review_output=str(root),source_alternative_review_manifest_sha256=binding))==decision


def test_unfinished_source_review_does_not_open_its_report(tmp_path,monkeypatch):
    import gofa_f.dual_memory_revalidation_pipeline as mod
    root=tmp_path/'review';root.mkdir();atomic_json(root/'manifest.json',{})
    atomic_json(root/'status.json',dict(state='running'));atomic_json(Path(str(root)+'.launch.json'),dict(pid=1,start_ticks='x'))
    monkeypatch.setattr(mod,'alive',lambda _:True)
    assert route_review(dict(source_alternative_review_output=str(root),source_alternative_review_manifest_sha256=sha(root/'manifest.json')))=='waiting'
    assert not (root/'report.json').exists()


def test_no_duplicate_gpu_spawn_or_reload_after_completion(tmp_path,monkeypatch):
    import gofa_f.dual_memory_revalidation_pipeline as mod
    out=tmp_path/'out';out.mkdir();handle=dict(pid=1,start_ticks='x')
    atomic_json(Path(str(out)+'.launch.json'),handle)
    monkeypatch.setattr(mod,'alive',lambda h:h==handle)
    monkeypatch.setattr(mod.subprocess,'Popen',lambda *a,**k:(_ for _ in ()).throw(AssertionError('duplicate GPU')))
    assert launch(tmp_path,tmp_path/'config',{},out)==handle
    monkeypatch.setattr(mod,'alive',lambda _:False);atomic_json(out/'status.json',dict(state='complete'))
    for name in ('native_complete.json','enhanced_complete.json','fusion_complete.json','frozen_gpu_weights.json'):atomic_json(out/name,{})
    assert launch(tmp_path,tmp_path/'config',{},out) is None
    atomic_json(out/'status.json',dict(state='failed'))
    with pytest.raises(ValueError,match='audit'):launch(tmp_path,tmp_path/'config',{},out)


def test_memory_off_pretrained_temporal_expert_is_not_persistence():
    model={'head.0.weight':torch.zeros(8,644),'head.2.weight':torch.ones(1,8),'head.2.bias':torch.zeros(1)}
    result=readout_diagnostic(model)
    assert not result['memory_read_columns_nonzero'] and not result['exact_persistence']
    model['head.2.weight'].zero_();assert readout_diagnostic(model)['exact_persistence']


def test_costs_count_UFG_extension_once_and_preserve_all_source_attempts():
    final=dict(inherited_source_gpu_seconds=30.,new_evaluation_gpu_seconds=9.,cumulative_gpu_seconds=39.,peak_group_mib=20000,
        inherited_prior_stage_gpu_seconds=20.,new_joint_source_gpu_seconds=10.,source_ufg_cpu=dict(spent_seconds=100.),
        baseline_adequacy_cpu=dict(inherited_ufg_cpu_activity=dict(spent_seconds=100.),new_cpu_residency_seconds=20.),
        source_temporal_initialization_cpu=dict(spent_cpu_residency_seconds=10.),standalone_expert_source_cpu=dict(spent_cpu_residency_seconds=50.),
        source_alternative_review_activity=dict(coordinator_wait_seconds=60.,model_cpu_residency_seconds={}),
        inherited_prior_cpu_costs={},evaluation_ufg_cpu={'targets':dict(spent_seconds=5.)})
    value=costs_from(final)
    assert value['source_ufg_cpu_residency_seconds']==120 and value['cumulative_gpu_seconds']==39
    assert value['standalone_expert_source_cpu']['spent_cpu_residency_seconds']==50
    with pytest.raises(ValueError,match='reset'):costs_from(dict(final,cumulative_gpu_seconds=40.))


def test_adequate_UFG_override_requires_same_sources_and_source_early_stop(tmp_path,monkeypatch):
    import gofa_f.dual_memory_revalidation as mod
    source=tmp_path/'source';adequacy=tmp_path/'adequacy';source.mkdir();adequacy.mkdir()
    m={key:{} for key in ('source_hashes','metadata_hashes','sampler','partitions','training_unit_normalizers')}
    m['queries']={name:{shape:dict(train=[],development=[]) for shape in ('12-12','48-24')} for name in GROUPS}
    atomic_json(source/'manifest.json',m);atomic_json(adequacy/'manifest.json',m)
    atomic_json(adequacy/'status.json',dict(state='complete',source_adequacy=True));atomic_json(adequacy/'activity.json',dict(worker=None))
    atomic_json(Path(str(adequacy)+'.launch.json'),dict(pid=1,start_ticks='x'))
    binding=sha(adequacy/'manifest.json');models={}
    for shape in ('12-12','48-24'):
        path=adequacy/shape;path.mkdir();(path/'best.pt').write_bytes(b'stronger-official-source-UFG')
        selected=dict(checkpoint_sha256=sha(path/'best.pt'),model_sha256='tensor',target_labels_used=False,
            stop_reason='source_early_stop',best_update=10500,stopped_update=12500)
        atomic_json(path/'selection.json',selected);models[shape]=dict(directory=str(path),selection=selected)
    atomic_json(adequacy/'ufg_source_lock.json',dict(binding=binding,both_shapes_stopped_by_source_criterion=True,target_labels_used=False,models=models))
    monkeypatch.setattr(mod,'deployment',lambda *_:dict(ufg={'old':{}},source_directory=str(source)))
    monkeypatch.setattr(mod,'verify_deployment',lambda *_:None)
    import gofa_f.dual_memory_eval_pipeline as pipeline
    monkeypatch.setattr(pipeline,'alive',lambda _:False)
    cfg=dict(source_output=str(source),accepted_snapshot=str(tmp_path/'accepted'),ufg_adequacy_output=str(adequacy),ufg_adequacy_manifest_sha256=binding)
    result=adequate_deployment(cfg)
    assert result['ufg']['12-12']['best_update']==10500 and 'old' not in result['ufg']
    assert result['ufg_source_adequacy']['source_early_stop_both_shapes']
    atomic_json(adequacy/'status.json',dict(state='complete',source_adequacy=False))
    with pytest.raises(ValueError,match='adequately stopped'):adequate_deployment(cfg)


def test_new_launcher_mode_has_same_shared_caps_and_original_modes_still_work():
    from gofa_f.short_launcher import admission_gate
    from gofa_f.idle_launcher import CapOnlyGate
    r=dict(admission_policy='memory_cap_only',resource_check_seconds=10,admission_samples=1,memory_cap_gib=40,
        torch_cap_gib=38,yield_on_foreign=False,leave_free_gib=0)
    for mode in ('dual_memory_revalidation','dual_memory_evaluation','dual_memory_retraining'):
        assert isinstance(admission_gate(r,mode),CapOnlyGate)
    with pytest.raises(ValueError):admission_gate(dict(r,memory_cap_gib=41),'dual_memory_revalidation')
