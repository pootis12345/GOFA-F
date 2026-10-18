import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import LlamaTokenizer

from gofa_f.allnode_stage import Paused,read,rng_state
from gofa_f.generative_data import Query
from gofa_f.short_codec import ShortCodec
from gofa_f.short_migration import assert_equal
from gofa_f.short_sampled_continuation import commit_validation,run
from gofa_f.short_sampled_forecast import evaluate
from gofa_f.short_sampled_progress import Progress
from gofa_f.short_sampled_state import (fresh_ledger,migrated_payload,save,temporary_adapter)
from gofa_f.prefix_cache import atomic_json
from gofa_f.generative_data import sha
import gofa_f.short_sampled_continuation as continuation
import gofa_f.short_sampled_forecast as forecasting
from test_allnode import tiny,source


@pytest.fixture(scope='module')
def codec():
    root=Path('/home/dcg/dcg_data/GOFA-F/data/weights/chattime-7b-20baa88')
    if not root.exists():
        pytest.skip('Official tokenizer unavailable')
    return ShortCodec(LlamaTokenizer.from_pretrained(root,local_files_only=True))


class Display:
    def evaluation(self,*args,**kwargs):
        self.last=args


def test_resumable_eight_paths_exact_seeds_median_masks_and_binding(codec,tmp_path,monkeypatch):
    model,data=tiny(),source(3)
    with torch.no_grad():
        for m in model.adapter.exchanges:
            m.up.weight.normal_(std=.003)
    data.mask[501,0]=False
    q=Query('pems04',0,500,12,3)
    queries=[vars(q)]
    original=forecasting.generate
    count=[0]
    def interrupt(*args,**kwargs):
        count[0]+=1
        if count[0]==4:
            raise Paused('test resource pause')
        return original(*args,**kwargs)
    monkeypatch.setattr(forecasting,'generate',interrupt)
    with pytest.raises(Paused):
        evaluate(model,codec,{'pems04':data},queries,tmp_path,'partial','binding',Display(),lambda:None)
    assert len(list((tmp_path/'evaluations/partial/trajectories').glob('*.json')))==3
    assert not (tmp_path/'evaluations/partial/complete.json').exists()
    called=[]
    def resumed(*args,**kwargs):
        called.append(kwargs['seed'])
        return original(*args,**kwargs)
    monkeypatch.setattr(forecasting,'generate',resumed)
    actual=evaluate(model,codec,{'pems04':data},queries,tmp_path,'partial','binding',Display(),lambda:None)
    assert len(called)==5
    monkeypatch.setattr(forecasting,'generate',original)
    expected=evaluate(model,codec,{'pems04':data},queries,tmp_path,'whole','binding2',Display(),lambda:None)
    assert actual['errors']==expected['errors']
    a,b=(read(tmp_path/f'evaluations/{name}/000000.json') for name in ('partial','whole'))
    assert a['prediction']==b['prediction'] and a['observed']==[True,False,True]
    paths=[read(tmp_path/f'evaluations/partial/trajectories/000000-{i:02d}.json') for i in range(8)]
    assert len(set(p['seed'] for p in paths))==8
    assert a['prediction']==np.median([p['prediction'] for p in paths],axis=0).tolist()
    assert actual['numeric_slots_in_eight_paths']==24
    with pytest.raises(ValueError,match='protocol'):
        evaluate(model,codec,{'pems04':data},queries,tmp_path,'partial','wrong',Display(),lambda:None)


def test_empty_history_fallback_not_dropped_or_called_generation(codec,tmp_path,monkeypatch):
    data=source(3)
    data.mask[488:500,0]=False
    def forbidden(*args,**kwargs):
        raise AssertionError('Empty history must not call LLM generation')
    monkeypatch.setattr(forecasting,'generate',forbidden)
    result=evaluate(None,codec,{'pems04':data},[vars(Query('pems04',0,500,12,3))],
                    tmp_path,'empty','binding',Display(),lambda:None,True)
    item=read(tmp_path/'evaluations/empty/000000.json')
    assert item['prediction']==[0.]*3 and not item['format_complete']
    assert result['metrics']['formatting']['fallback_slots']==3


def test_checkpoint_ledger_and_early_reference_not_greedy_or_double_counted(tmp_path):
    candidate=tmp_path/'reference.pt'
    torch.save({'weight':torch.ones(1)},candidate)
    ledger=fresh_ledger()
    commit_validation(tmp_path,ledger,{'score':1.},1000,candidate)
    commit_validation(tmp_path,ledger,{'score':.9995},1500,candidate)
    assert ledger['best_score']==.9995 and ledger['early_reference']==1.
    assert ledger['bad_validations']==1 and ledger['best_update']==1500
    commit_validation(tmp_path,ledger,{'score':.9995},1500,candidate)
    assert ledger['bad_validations']==1 and ledger['validated_updates']==[1000,1500]
    with pytest.raises(ValueError,match='score changed'):
        commit_validation(tmp_path,ledger,{'score':.5},1500,candidate)


def test_exact_partial_training_payload_separate_selection():
    original=dict(binding='old',adapter={'x':torch.ones(2)},optimizer={'moment':torch.rand(2)},
        gradients={'x':torch.tensor([3.,4.])},rng=rng_state(),parameter_version=1177,
        state=dict(updates=1177,pending=30,pending_tokens=390,pending_loss=123.,
                   best_score=1.0285,early_reference=1.0285,bad_validations=0,last_validated_update=1000),
        progress=dict(measured_updates=1177,train_seconds=500.,validation_runs=2,validation_seconds=2500.))
    before=copy.deepcopy(original)
    after=migrated_payload(original,'new')
    assert_equal(original,before)
    for key in ('adapter','optimizer','gradients','rng','parameter_version','state'):
        assert_equal(original[key],after[key])
    assert after['sampling_ledger']['best_score'] is None
    assert after['state']['best_score']==1.0285
    assert after['progress']['throughput_anchor_seconds']==500.


def test_evaluation_adapter_scope_preserves_pending_gradient_rng_version_on_failure():
    model=tiny()
    for p in model.adapter.parameters():
        p.grad=torch.ones_like(p)
    original=model.trainable_state()
    grads={n:p.grad.clone() for n,p in model.adapter.named_parameters()}
    rng,version=rng_state(),model.parameter_version
    changed={k:v+1 for k,v in original.items()}
    with pytest.raises(RuntimeError,match='test'):
        with temporary_adapter(model,changed):
            torch.rand(4)
            raise RuntimeError('test')
    assert_equal(model.trainable_state(),original)
    assert_equal(rng_state(),rng)
    assert model.parameter_version==version
    assert_equal({n:p.grad for n,p in model.adapter.named_parameters()},grads)


def test_save_sampling_ledger_preserves_partial_gradients(tmp_path):
    model=tiny()
    opt=torch.optim.AdamW(model.adapter.parameters())
    for p in model.adapter.parameters():
        p.grad=torch.ones_like(p)
    class P:
        state={'updates':1177}
        def tick(self,**kwargs):
            pass
    ledger=fresh_ledger()
    save(tmp_path/'last.pt',model,opt,{'pending':30,'pending_tokens':390},P(),ledger,'binding')
    saved=torch.load(tmp_path/'last.pt',weights_only=False)
    assert saved['state']=={'pending':30,'pending_tokens':390}
    assert all(g is not None for g in saved['gradients'].values())
    assert saved['sampling_ledger']==ledger


def test_progress_review_cap_numeric_eight_work_and_no_wait_accounting(tmp_path):
    clock=[0.]
    state=dict(updates=1177,pending_queries=30,measured_updates=1177,train_seconds=500.,
        throughput_anchor_seconds=500.,throughput_anchor_updates=1177,sampled_evaluation_durations={})
    p=Progress(tmp_path,state,lambda:clock[0])
    p.tick('sampled-native')
    clock[0]=100.
    p.evaluation(1,10,8,80,2.,evaluation_kind='sampled-native')
    assert state['evaluation_eta_seconds']==18 and state['train_seconds']==500.
    assert state['max_updates']==1500 and state['original_update_cap']==4000
    p.tick('train')
    clock[0]=110.
    p.tick(updates=1178,measured_updates=1178)
    assert state['train_seconds']==510. and state['training_eta_seconds']==3220.
    state['sampled_evaluation_durations']={'sampled-native':10.,'sampled-update-001000':20.}
    p.tick('sampled-update-001500',updates=1500,pending_queries=0)
    p.evaluation(2,10,20,80,4.,evaluation_kind='sampled-update-001500')
    assert state['overall_eta_seconds']==12.


def test_unadmitted_training_is_impossible(tmp_path,monkeypatch):
    monkeypatch.delenv('GOFA_STAGE19_ADMITTED',raising=False)
    with pytest.raises(RuntimeError,match='fully-idle'):
        run({},tmp_path,True)


def test_end_to_end_evaluation_before_training_and_hard_review_stop(codec,tmp_path,monkeypatch):
    monkeypatch.setattr(continuation.signal,'signal',lambda *_:None)
    model=tiny()
    opt=torch.optim.AdamW(model.adapter.parameters(),lr=2e-4,weight_decay=1e-4)
    for p in model.adapter.parameters():
        p.grad=torch.ones_like(p)
    state=dict(updates=1499,pending=64,pending_tokens=64,pending_loss=64.,microbatch=2,
        peak_memory={},best_score=1.0285,early_reference=1.0285,bad_validations=0,last_validated_update=1000)
    progress=dict(updates=1499,pending_queries=64,measured_updates=1499,train_seconds=100.,
        throughput_anchor_updates=1499,throughput_anchor_seconds=100.,sampled_evaluation_durations={})
    model.parameter_version=1499
    class P:
        def __init__(self):
            self.state=progress
        def tick(self,**kwargs):
            pass
    atomic_json(tmp_path/'manifest.json',{'test':'CPU orchestration'})
    binding=sha(tmp_path/'manifest.json')
    save(tmp_path/'last.pt',model,opt,state,P(),fresh_ledger(),binding)
    torch.save(model.trainable_state(),tmp_path/'reference-001000.pt')
    (tmp_path/'pools').mkdir()
    atomic_json(tmp_path/'pools/queries.json',dict(train={},validation=[]))
    atomic_json(tmp_path/'pools/metric_epsilons.json',{})
    atomic_json(tmp_path/'cpu_preflight.json',dict(passed=True,manifest_sha256=binding,
        alphabet={'alphabet_sha256':codec.alphabet_sha}))
    manifest=dict(source_configuration={'data':{'directory':str(tmp_path)}},weights_sha256='test')
    monkeypatch.setattr(continuation,'verify',lambda *_:manifest)
    monkeypatch.setattr(continuation,'load_model',lambda *_:(model,codec))
    monkeypatch.setattr(continuation,'Series',lambda *_:source(3))
    monkeypatch.setattr(continuation,'supervisor_alive',lambda *_:True)
    monkeypatch.setattr(torch.cuda,'set_per_process_memory_fraction',lambda *_:None)
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda *_:type('GPU',(),{'total_memory':80*2**30})())
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda:0)
    events=[]
    def evaluation(*args,**kwargs):
        label=args[5]
        events.append(label)
        return dict(active_seconds=10.,errors=[])
    monkeypatch.setattr(continuation,'evaluate',evaluation)
    monkeypatch.setattr(continuation,'relative_score',lambda *_:{'score':.9})
    update=continuation.apply_update
    def apply(*args):
        assert events==['sampled-native','sampled-update-001000']
        events.append('one_update')
        return update(*args)
    monkeypatch.setattr(continuation,'apply_update',apply)
    for key in ('GOFA_STAGE19_ADMITTED','GOFA_STAGE19_SUPERVISOR_PID','GOFA_STAGE19_SUPERVISOR_TICKS'):
        monkeypatch.setenv(key,'1')
    run({},tmp_path,True)
    assert events==['sampled-native','sampled-update-001000','one_update','sampled-update-001500']
    status=read(tmp_path/'status.json')
    saved=torch.load(tmp_path/'last.pt',weights_only=False)
    assert status['state']=='complete' and status['phase']=='review_paused'
    assert saved['state']['updates']==1500 and saved['state']['pending']==0
    assert saved['state']['best_score']==1.0285
    assert saved['sampling_ledger']['best_update']==1000  # Exact tie keeps earlier.
    assert saved['sampling_ledger']['bad_validations']==1
    assert not (tmp_path/'final_selection.json').exists()
