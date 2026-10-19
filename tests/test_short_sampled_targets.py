import copy
from pathlib import Path

import pytest
import torch
from transformers import LlamaTokenizer

from gofa_f.allnode_stage import read
from gofa_f.generative_data import Query,sha
from gofa_f.prefix_cache import atomic_json
from gofa_f.short_codec import ShortCodec
from gofa_f.short_data import TARGETS
from gofa_f.short_sampling_diagnostic import SETTINGS,generate,sample_seed
from gofa_f.short_target_state import failed_preflight,subset
from gofa_f.short_target_progress import Progress
import gofa_f.short_sampled_targets as target
from test_allnode import tiny,source


def original_pool():
    result={}
    for name,(length,horizon,n) in TARGETS.items():
        variables=[104] if name=='nasdaq100' else list(range(n))
        result[name]=[vars(Query(name,variables[i%len(variables)],10000+i*24,length,horizon)) for i in range(2048)]
    return result


def test_balanced_subset_is_deterministic_original_only_all_variables_and_no_value_input():
    original=original_pool()
    before=copy.deepcopy(original)
    a,b=subset(original),subset(original)
    assert a==b and original==before
    assert sum(len(q) for q in a['targets'].values())==2121
    assert sum(q['horizon'] for rows in a['targets'].values() for q in rows)*16==699904
    for name,rows in a['targets'].items():
        assert rows==[original[name][i] for i in a['original_indices'][name]]
        assert set(q['variable'] for q in rows)==set(q['variable'] for q in original[name])
        assert len(rows)==max(128,len(set(q['variable'] for q in original[name])))
        for v in set(q['variable'] for q in rows):
            times=[q['origin'] for q in rows if q['variable']==v]
            assert len(times)==len(set(times))
    original['traffic'][0]['variable']=9999
    with pytest.raises(ValueError,match='coverage'):
        subset(original)


def test_target_progress_resume_work_and_finish(tmp_path):
    p=Progress(tmp_path,100)
    p.completed={'prior':dict(numeric_slots_in_eight_paths=40,active_seconds=20.)}
    p.tick('target-current-native',dataset='current',model='native')
    p.evaluation(1,2,20,40,10.,evaluation_kind='target-current-native',evaluation_sample=4)
    assert p.state['overall_work_done']==60 and p.state['overall_eta_seconds']==20.
    assert p.state['evaluation_eta_seconds']==10.
    p.tick(force=True)
    other=Progress(tmp_path,100)
    assert other.state['overall_work_done']==60
    assert not any(k.startswith('epoch') for k in other.state)


@pytest.fixture(scope='module')
def codec():
    root=Path('/home/dcg/dcg_data/GOFA-F/data/weights/chattime-7b-20baa88')
    if not root.exists():
        pytest.skip('Official tokenizer unavailable')
    return ShortCodec(LlamaTokenizer.from_pretrained(root,local_files_only=True))


@pytest.mark.parametrize('nodes',[7,8,9])
def test_trained_nonzero_source_replay_full_and_incremental_fp32(codec,tmp_path,monkeypatch,nodes):
    model=tiny().eval().requires_grad_(False)
    with torch.no_grad():
        for layer in model.adapter.exchanges:
            layer.up.weight.normal_(std=.003)
    data=source(nodes)
    q=Query('pems04',0,500,12,3)
    (tmp_path/'sources').mkdir()
    datafile=tmp_path/'sources/pems04.npz'
    datafile.write_bytes(b'fixture only; Series mocked')
    monkeypatch.setattr(target,'Series',lambda *_:data)
    references={}
    history=data.values[488:500,0]
    for mode in ('native','gofa'):
        context=model.encode_origins([target.build_graph(data,q)],codec) if mode=='gofa' else None
        prediction=generate(model,codec,history,3,context,seed=sample_seed(vars(q),0))
        path=tmp_path/(mode+'.json')
        atomic_json(path,prediction)
        references[mode]=dict(path=str(path),sha256=sha(path))
    manifest=dict(source_configuration={'data':{'directory':str(tmp_path)}},
        source_raw_hashes={'pems04':sha(datafile)},probes=[dict(query=vars(q),references=references)])
    proof=target.mechanisms(model,codec,manifest,lambda:None,1e-6,1e-6)
    assert proof['passed'] and not proof['gradients_created']
    assert all(r['raw_tokens_equal'] and not r['comparison']['violations'] for r in proof['records'])
    path=Path(references['gofa']['path'])
    item=read(path)
    item['generated_ids'][-1]+=1
    atomic_json(path,item)
    references['gofa']['sha256']=sha(path)
    with pytest.raises(ValueError,match='sampling failed'):
        target.mechanisms(model,codec,manifest,lambda:None,1e-6,1e-6)


def test_no_gpu_admission_no_evaluation(tmp_path,monkeypatch):
    monkeypatch.delenv('GOFA_STAGE19_ADMITTED',raising=False)
    with pytest.raises(RuntimeError,match='fully-idle'):
        target.run({},tmp_path)


def test_failed_preflight_carries_costs_without_predictions_or_live_worker(tmp_path,monkeypatch):
    old,new=tmp_path/'failed',tmp_path/'new'
    old.mkdir()
    new.mkdir()
    manifest=dict(queries={'fixed':[1]},source_selection={'update':3500})
    atomic_json(old/'manifest.json',manifest)
    (old/'adapter.pt').write_bytes(b'locked')
    atomic_json(Path(str(old)+'.launch.json'),dict(pid=123,start_ticks='123'))
    activity=dict(worker=None,spent_seconds=10.1,waiting_seconds=20.,peak_group_mib=12)
    atomic_json(old/'activity.json',activity)
    atomic_json(old/'launcher.json',dict(state='failed'))
    atomic_json(old/'failures.json',dict(phase='source_replay_mechanisms'))
    monkeypatch.setattr('gofa_f.short_target_state.process_identity',lambda _:None)
    cfg=dict(repair_source_output=str(old),repair_source_manifest_sha256=sha(old/'manifest.json'),
        adapter_sha256=sha(old/'adapter.pt'))
    failed_preflight(cfg,new,copy.deepcopy(manifest))
    assert read(new/'activity.json')==activity
    assert read(new/'migration.json')['target_predictions']==0
    (old/'evaluations').mkdir()
    atomic_json(old/'evaluations/prediction.json',{})
    with pytest.raises(ValueError,match='Only exited'):
        failed_preflight(cfg,new,copy.deepcopy(manifest))


def test_target_only_end_to_end_same_adapter_two_models_nine_datasets(codec,tmp_path,monkeypatch):
    monkeypatch.setattr(target.signal,'signal',lambda *_:None)
    for name in ('GOFA_STAGE19_ADMITTED','GOFA_STAGE19_SUPERVISOR_PID','GOFA_STAGE19_SUPERVISOR_TICKS'):
        monkeypatch.setenv(name,'1')
    model=tiny()
    torch.save(model.trainable_state(),tmp_path/'adapter.pt')
    manifest=dict(source_configuration={'data':{'directory':str(tmp_path)}},
        weights_sha256='backbone',alphabet={'alphabet_sha256':codec.alphabet_sha},total_numeric_work=144,
        queries=dict(targets={n:[vars(Query(n,0,500,10,1))] for n in TARGETS},coverage={}))
    atomic_json(tmp_path/'manifest.json',manifest)
    atomic_json(tmp_path/'cpu_preflight.json',dict(passed=True,manifest_sha256=sha(tmp_path/'manifest.json')))
    monkeypatch.setattr(target,'verify',lambda *_:manifest)
    monkeypatch.setattr(target,'load_model',lambda *_:(model,codec))
    monkeypatch.setattr(target,'Series',lambda *_:source(7))
    monkeypatch.setattr(target,'supervisor_alive',lambda *_:True)
    monkeypatch.setattr(torch.cuda,'set_per_process_memory_fraction',lambda *_:None)
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda *_:type('GPU',(),{'total_memory':80*2**30})())
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda:0)
    events=[]
    def checks(*_,**kwargs):
        assert not list(tmp_path.glob('target-*-result.json'))
        events.append('acceptance')
        return dict(passed=True,gradients_created=False)
    monkeypatch.setattr(target,'mechanisms',checks)
    def evaluate(*args,**kwargs):
        assert events[0]=='acceptance' and not any(p.requires_grad for p in model.parameters())
        events.append((args[5],args[9]))
        return dict(complete=True,queries=1,numeric_slots_in_eight_paths=8,active_seconds=1.,metrics={},
            binding=args[6],sampling=SETTINGS,errors=[.1])
    monkeypatch.setattr(target,'evaluate',evaluate)
    cfg={'adapter_sha256':sha(tmp_path/'adapter.pt')}
    target.run(cfg,tmp_path)
    assert len(events)==19 and sum(native for _,native in events[1:])==9
    assert read(tmp_path/'report.json')['complete'] and read(tmp_path/'status.json')['state']=='complete'
    assert not (tmp_path/'last.pt').exists() and not (tmp_path/'optimizer.pt').exists()
    before=list(events)
    # Accepted source mechanism is reused after a resource resume.
    target.run(cfg,tmp_path)
    assert events.count('acceptance')==1 and len(events)==len(before)+18
    path=tmp_path/'target-etth1-gofa-result.json'
    result=read(path)
    result['binding']='wrong-checkpoint-or-decoding'
    atomic_json(path,result)
    with pytest.raises(ValueError,match='different bound'):
        target.report(tmp_path)
