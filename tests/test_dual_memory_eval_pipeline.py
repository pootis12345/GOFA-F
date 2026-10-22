from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gofa_f.dual_memory_eval_data import read,seal
from gofa_f.dual_memory_eval_pipeline import alive,source_handles,launch
from gofa_f.dual_memory_eval_records import PredictionStore
from gofa_f.generative_data import Query,sha
from gofa_f.prefix_cache import atomic_json


def test_pipeline_does_not_duplicate_a_verified_existing_worker(tmp_path,monkeypatch):
    import gofa_f.dual_memory_eval_pipeline as module
    output=tmp_path/'evaluation';output.mkdir();handle=dict(pid=123,start_ticks='tick')
    atomic_json(Path(str(output)+'.launch.json'),handle)
    monkeypatch.setattr(module,'alive',lambda h:h==handle)
    monkeypatch.setattr(module.subprocess,'Popen',lambda *a,**k:(_ for _ in ()).throw(AssertionError('duplicate spawn')))
    assert launch(tmp_path,{},output)==handle


def test_pipeline_refuses_blind_restart_of_failed_evaluation(tmp_path,monkeypatch):
    import gofa_f.dual_memory_eval_pipeline as module
    output=tmp_path/'evaluation';output.mkdir();atomic_json(output/'status.json',dict(state='failed'))
    monkeypatch.setattr(module.subprocess,'Popen',lambda *a,**k:(_ for _ in ()).throw(AssertionError('blind restart')))
    with pytest.raises(ValueError,match='audited'):launch(tmp_path,{},output)


def test_completed_gpu_prediction_does_not_reload_chattime_on_cpu_scoring_resume(tmp_path,monkeypatch):
    import gofa_f.dual_memory_eval_pipeline as module
    output=tmp_path/'evaluation';output.mkdir();atomic_json(output/'status.json',dict(state='complete'))
    for name in ('native_complete.json','enhanced_complete.json','fusion_complete.json','frozen_gpu_weights.json'):
        atomic_json(output/name,dict(complete=True))
    monkeypatch.setattr(module.subprocess,'Popen',lambda *a,**k:(_ for _ in ()).throw(AssertionError('unnecessary GPU reload')))
    assert launch(tmp_path,{},output) is None


def test_completion_seals_are_exactly_immutable(tmp_path):
    path=tmp_path/'seal.json';seal(path,dict(weights='frozen',queries=768))
    seal(path,dict(weights='frozen',queries=768))
    with pytest.raises(ValueError,match='immutable'):seal(path,dict(weights='other',queries=768))


def test_evaluation_launcher_shares_gpu_lock_and_caps(monkeypatch):
    from gofa_f.short_launcher import admission_gate
    from gofa_f.idle_launcher import CapOnlyGate
    resources=dict(admission_policy='memory_cap_only',resource_check_seconds=10,admission_samples=1,
        memory_cap_gib=40,torch_cap_gib=38,yield_on_foreign=False,leave_free_gib=0)
    assert isinstance(admission_gate(resources,'dual_memory_evaluation'),CapOnlyGate)
    with pytest.raises(ValueError):admission_gate(dict(resources,torch_cap_gib=39),'dual_memory_evaluation')
