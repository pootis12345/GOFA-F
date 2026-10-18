import copy

import numpy as np
import pytest
import torch

from gofa_f.allnode_stage import read,rng_state
from gofa_f.generative_data import Query,sha
from gofa_f.prefix_cache import atomic_json
from gofa_f.short_migration import assert_equal
from gofa_f.short_sampled_continuation import commit_validation,report
from gofa_f.short_sampled_handoff import after_review_payload,copy_evaluations,migrate,wait_ready
from gofa_f.short_sampled_progress import Progress
from gofa_f.short_sampled_state import fresh_ledger
from gofa_f.short_sampling_diagnostic import SETTINGS,sample_seed
import gofa_f.short_sampled_handoff as handoff


def test_review_migration_preserves_sampling_selection_optimizer_rng_and_cost_progress():
    ledger=fresh_ledger()
    ledger.update(best_score=.9,best_update=1000,early_reference=.9,bad_validations=1,
        last_validated_update=1500,validated_updates=[1000,1500])
    original=dict(binding='old',adapter={'x':torch.ones(2)},optimizer={'moment':torch.rand(2)},
        gradients={'x':None},rng=rng_state(),parameter_version=1500,
        state=dict(updates=1500,pending=0,pending_tokens=0,pending_loss=0.,stop_reason='review_at_1500',
            microbatch=2,cache_totals={'hits':30},peak_memory={'train':{'allocated_bytes':100}}),
        progress=dict(train_seconds=500.,sampled_evaluation_durations={'sampled-native':200.}),sampling_ledger=ledger)
    before=copy.deepcopy(original)
    migrated=after_review_payload(original,'new')
    assert_equal(original,before)
    expected=copy.deepcopy(original)
    expected['binding']='new'
    expected['state']['stop_reason']=None
    assert_equal(migrated,expected)
    for key,value in [('updates',1499),('pending',1),('stop_reason','failed')]:
        broken=copy.deepcopy(original)
        broken['state'][key]=value
        with pytest.raises(ValueError,match='completed'):
            after_review_payload(broken,'new')


def sample_shards(source,label,queries,digest='weights'):
    root=source/'evaluations'/label
    (root/'trajectories').mkdir(parents=True)
    errors=[]
    for i,q in enumerate(queries):
        paths=[]
        for s in range(8):
            p=[float(s)]*q['horizon']
            paths.append(p)
            atomic_json(root/'trajectories'/f'{i:06d}-{s:02d}.json',dict(
                binding=f'old:sample8:{label}:{digest}:alphabet',query=q,
                seed=sample_seed(q,s),sample=s,prediction=p,numeric_ids=[s]*q['horizon'],
                generated_ids=[s]*q['horizon'],active_seconds=float(s+1)))
        errors.append(3.5)
        atomic_json(root/f'{i:06d}.json',dict(binding=f'old:sample8:{label}:{digest}:alphabet',
            query=q,prediction=np.median(paths,axis=0).tolist(),mae=3.5,active_seconds=36.))
    atomic_json(root/'complete.json',dict(binding=f'old:sample8:{label}:{digest}:alphabet',
        complete=True,queries=len(queries),sampling=SETTINGS,errors=errors,active_seconds=36.*len(queries)))


def test_shard_handoff_keeps_tokens_seeds_predictions_and_cost_and_is_resumable(tmp_path):
    source,output=tmp_path/'source',tmp_path/'dest'
    queries=[vars(Query('pems04',0,500,12,12)),vars(Query('nn5',1,600,14,7))]
    sample_shards(source,'native',queries)
    audit=copy_evaluations(source,output,queries,'old','new',{'native':'weights'},'alphabet')
    assert audit['files']==19
    for p in (source/'evaluations').rglob('*.json'):
        a,b=read(p),read(output/'evaluations'/p.relative_to(source/'evaluations'))
        a['binding']=a['binding'].replace('old:','new:',1)
        assert a==b
    again=copy_evaluations(source,output,queries,'old','new',{'native':'weights'},'alphabet')
    assert again==audit
    broken=source/'evaluations/native/trajectories/000000-00.json'
    item=read(broken)
    item['seed']+=1
    atomic_json(broken,item)
    with pytest.raises(ValueError,match='seed'):
        copy_evaluations(source,output,queries,'old','new',{'native':'weights'},'alphabet')


def test_waits_for_review_controller_exit_and_refuses_stopped_source(tmp_path,monkeypatch):
    atomic_json(tmp_path/'launcher.json',{'state':'running'})
    assert not wait_ready(tmp_path)
    atomic_json(tmp_path/'launcher.json',{'state':'complete'})
    def live(*_):
        raise ValueError('controller finishing ledger')
    monkeypatch.setattr(handoff,'exited',live)
    assert not wait_ready(tmp_path)
    monkeypatch.setattr(handoff,'exited',lambda *_:{'worker':None})
    assert wait_ready(tmp_path)
    atomic_json(tmp_path/'launcher.json',{'state':'stopped_by_user'})
    with pytest.raises(ValueError,match='failed/stopped'):
        wait_ready(tmp_path)


def test_continued_progress_cap_and_eta_and_source_early_stop_ledger(tmp_path):
    clock=[0.]
    state=dict(updates=1500,pending_queries=0,measured_updates=1500,train_seconds=500.,
        throughput_anchor_updates=1177,throughput_anchor_seconds=177.,
        sampled_evaluation_durations={'sampled-native':10.,'sampled-update-001000':20.,'sampled-update-001500':30.})
    p=Progress(tmp_path,state,lambda:clock[0],cap=4000)
    p.tick('train')
    assert state['max_updates']==4000 and state['sampled_validation_rounds_done']==2
    assert state['sampled_validation_rounds_total']==7 and state['overall_eta_seconds']==2650.
    candidate=tmp_path/'candidate.pt'
    torch.save({'w':torch.ones(1)},candidate)
    ledger=fresh_ledger()
    for u in range(1000,3501,500):
        commit_validation(tmp_path,ledger,{'score':1.},u,candidate)
    assert ledger['bad_validations']==5 and ledger['last_validated_update']==3500
    commit_validation(tmp_path,ledger,{'score':1.},3500,candidate)
    assert ledger['bad_validations']==5  # Never duplicate patience after migration/restart.


def test_final_source_report_locks_best_only_after_full_validation(tmp_path):
    candidate=tmp_path/'reference.pt'
    torch.save({'w':torch.ones(1)},candidate)
    ledger=fresh_ledger()
    commit_validation(tmp_path,ledger,{'score':1.},1000,candidate)
    commit_validation(tmp_path,ledger,{'score':.9},4000,candidate)
    torch.save(dict(state=dict(updates=4000,pending=0,stop_reason='max_updates'),
        sampling_ledger=ledger),tmp_path/'last.pt')
    atomic_json(tmp_path/'manifest.json',{'scope':'source-only'})
    result=report(tmp_path)
    assert result['complete'] and not result['training_requires_new_authorization_after_review']
    selection=read(tmp_path/'source_final_selection.json')
    assert selection['adapter_sha256']==sha(tmp_path/'sampled_best.pt')
    assert not selection['target_evaluation_authorized']
    assert not (tmp_path/'final_selection.json').exists()


def test_handoff_serializes_exact_trained_state_and_never_readds_diagnostic_costs(tmp_path,monkeypatch):
    source,output=tmp_path/'source',tmp_path/'destination'
    source.mkdir()
    output.mkdir()
    (source/'pools').mkdir()
    adapter={'weight':torch.tensor([1.,2.])}
    torch.save(adapter,source/'candidate-001500.pt')
    torch.save(adapter,source/'reference-001000.pt')
    ledger=fresh_ledger()
    commit_validation(source,ledger,{'score':1.},1000,source/'reference-001000.pt')
    commit_validation(source,ledger,{'score':.9},1500,source/'candidate-001500.pt')
    atomic_json(source/'manifest.json',{'lineage':'old'})
    original=dict(binding=sha(source/'manifest.json'),adapter=adapter,
        optimizer={'state':{0:{'exp_avg':torch.tensor([.01,.02])}}},gradients={'weight':None},
        rng=rng_state(),parameter_version=1500,sampling_ledger=ledger,
        state=dict(updates=1500,pending=0,pending_tokens=0,pending_loss=0.,stop_reason='review_at_1500'),
        progress=dict(train_seconds=100.,sampled_evaluation_durations={'sampled-native':20.}))
    torch.save(original,source/'last.pt')
    atomic_json(source/'pools/queries.json',dict(train={},validation=[vars(Query('pems04',0,500,12,12))]*2264))
    atomic_json(source/'pools/metric_epsilons.json',{'epsilon':1e-6})
    atomic_json(source/'cpu_preflight.json',{'alphabet':{'alphabet_sha256':'alphabet'}})
    activity=dict(spent_seconds=10000.,waiting_seconds=300.,resource_pause_count=2,worker=None,peak_group_mib=16000)
    atomic_json(source/'activity.json',activity)
    manifest={'weights_sha256':'backbone'}
    atomic_json(output/'manifest.json',manifest)
    monkeypatch.setattr(handoff,'audit',lambda *_:manifest)
    monkeypatch.setattr(handoff,'exited',lambda *_:copy.deepcopy(activity))
    observed=[]
    monkeypatch.setattr(handoff,'copy_evaluations',lambda *args:observed.append(args) or {'files':61131})
    cfg=dict(source_output=str(source),source_manifest_sha256=sha(source/'manifest.json'),
        reference_1000_sha256=sha(source/'reference-001000.pt'))
    source_sha=sha(source/'last.pt')
    migrate(cfg,output)
    restored=torch.load(output/'last.pt',map_location='cpu',weights_only=False)
    assert_equal(restored,after_review_payload(original,sha(output/'manifest.json')))
    assert read(output/'activity.json')==activity
    assert sha(source/'last.pt')==source_sha
    assert read(output/'migration.json')['costs_added_again'] is False
    assert len(observed)==1 and observed[0][3]==cfg['source_manifest_sha256']
    assert read(output/'sampled_best_validation.json')==read(source/'sampled_best_validation.json')
