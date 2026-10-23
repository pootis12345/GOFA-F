from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from gofa_f.dual_memory_engines import model_digest
from gofa_f.dual_memory_policy import digest
from gofa_f.dual_memory_retraining import configuration, initialization, initialization_specs
from gofa_f.dual_memory_warmup import TemporalWarmup, transfer
from gofa_f.generative_data import sha


def resources():
    return dict(admission_policy='memory_cap_only',admission_samples=1,resource_check_seconds=10,
        memory_cap_gib=40,torch_cap_gib=38,activity_limit_seconds=None,wall_deadline=None,
        yield_on_foreign=False,leave_free_gib=0)


def test_new_mode_uses_shared_caps_and_rejects_target_access(tmp_path,monkeypatch):
    from gofa_f.short_launcher import admission_gate
    from gofa_f.idle_launcher import CapOnlyGate
    monkeypatch.setenv('GOFA_WORKSPACE','/workspace')
    cfg=dict(protocol='stage60-dual-memory-source-v1',training={'max_updates':3000},
        data=dict(target_training_enabled=False,target_evaluation_enabled=False),resources=resources())
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(cfg))
    assert configuration(path)==cfg
    assert isinstance(admission_gate(cfg['resources'],'dual_memory_retraining'),CapOnlyGate)
    cfg['data']['target_evaluation_enabled']=True;path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='source-only'):configuration(path)
    cfg['data']['target_evaluation_enabled']=False;cfg['resources']['torch_cap_gib']=39
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='Resource'):configuration(path)


def test_unfinished_cpu_fit_cannot_start_joint_preparation(tmp_path):
    (tmp_path/'status.json').write_text(json.dumps({'state':'running'}))
    (tmp_path/'activity.json').write_text(json.dumps({'worker':{'pid':123,'start_ticks':'x'}}))
    with pytest.raises(ValueError,match='complete'):initialization_specs(tmp_path,{})


def test_initialization_preserves_source_specific_binding_and_rejects_corruption(tmp_path):
    model=TemporalWarmup(width=8)
    saved=dict(binding='held-weather',model=model.state_dict(),model_sha256=model_digest(model))
    path=tmp_path/'best.pt';torch.save(saved,path)
    entry=dict(path=str(path),binding='held-weather',model_sha256=saved['model_sha256'],checkpoint_sha256=sha(path))
    manifest={'temporal_initializations':{'hold-weather':entry}}
    value,binding=initialization(manifest,'hold-weather')
    assert binding=='held-weather' and value['model_sha256']==saved['model_sha256']
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='changed'):initialization(manifest,'hold-weather')


def test_transfer_then_joint_update_opens_memory_and_preserves_cached_gradients():
    # This uses the accepted real production Memory/tiny frozen-LLM path rather
    # than a numerical mock of its head or gradient recurrence.
    from test_dual_memory_native_training import fixture
    model,item,tokens=fixture();warm=TemporalWarmup(width=8)
    with torch.no_grad():warm.head[-1].weight.normal_(std=.02)
    selected=dict(binding='source',model=deepcopy(warm.state_dict()),model_sha256=model_digest(warm))
    transfer(model.adapter,selected,'source')
    prefix=model.frozen_history(tokens);meta=torch.ones(16)
    optimizer=torch.optim.AdamW(model.adapter.parameters(),lr=.001,weight_decay=0.)
    value,_=model.enhanced(tokens,item,meta,prefix=prefix,optimized=True)
    (value/item.extent[:,None]).square().mean().backward();optimizer.step()
    assert bool(model.adapter.head[0].weight[:,:32].abs().sum())
    model.zero_grad(set_to_none=True)
    full,memory=model.enhanced(tokens,item,meta,prefix=prefix,optimized=False)
    (full/item.extent[:,None]).square().mean().backward()
    expected={n:p.grad.detach().clone() for n,p in model.adapter.named_parameters()}
    model.zero_grad(set_to_none=True)
    cached,cached_memory=model.enhanced(tokens,item,meta,prefix=prefix,optimized=True,checkpoint_tail=True)
    torch.testing.assert_close(cached,full,rtol=1e-6,atol=1e-6)
    torch.testing.assert_close(cached_memory,memory,rtol=1e-6,atol=1e-6)
    (cached/item.extent[:,None]).square().mean().backward()
    for name,p in model.adapter.named_parameters():
        torch.testing.assert_close(p.grad,expected[name],rtol=1e-6,atol=1e-6)
    assert all(p.grad is None for p in model.native.parameters())


def test_progress_uses_actual_maximum_and_distinguishes_initialization(tmp_path,capsys):
    from gofa_f.dual_memory_source_progress import report
    (tmp_path/'manifest.json').write_text(json.dumps({'configuration':{'training':{'max_updates':3000,'effective_batch':8}}}))
    (tmp_path/'status.json').write_text(json.dumps(dict(state='running',phase='training',fit='hold-wind',depth=6,
        updates=1100,maximum=3000,initialization_best_update=900,selection={'best_update':1000},
        fit_scope='source_group_crossfit')))
    report(tmp_path);text=capsys.readouterr().out
    assert 'updates=1100/3000' in text and 'CPU initialization best=900; joint best=1000' in text
    assert 'within this fit only' in text


def test_terminal_source_progress_does_not_keep_old_eta(tmp_path,capsys):
    from gofa_f.dual_memory_source_progress import report
    (tmp_path/'status.json').write_text(json.dumps(dict(state='complete',eta_current_fit_seconds=1e6,estimated_seconds_per_update=100.)))
    report(tmp_path);text=capsys.readouterr().out
    assert 'conditionalETA=not_applicable' in text
