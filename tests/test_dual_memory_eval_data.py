from collections import Counter
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from gofa_f.dual_memory_eval_data import (InputStore,fresh_target_queries,source_ready,
    target_text,truth_records,write_inputs)
from gofa_f.generative_data import Query


def test_source_not_ready_without_reading_any_target_data(tmp_path):
    assert source_ready(tmp_path/'source') is False


def test_fresh_queries_preserve_variables_quotas_and_exclude_old_and_new_windows():
    class Data:
        ids=['a','b']
        @property
        def values(self):raise AssertionError('Selection must not inspect target values')
        def candidates(self,v,l,h,split):return np.arange(16000,20000-h+1)
    old=[dict(source='pems03',variable=i%2,origin=16000+(i//2)*24,length=12,horizon=12,chunk=0) for i in range(128)]
    actual=fresh_target_queries(Data(),'pems03',old)
    assert Counter(q['variable'] for q in actual)==Counter(q['variable'] for q in old)
    assert actual==fresh_target_queries(Data(),'pems03',old)
    for i,q in enumerate(actual):
        for other in old+actual[:i]:
            if q['variable']==other['variable']:
                assert q['origin']+q['horizon']<=other['origin'] or q['origin']>=other['origin']+other['horizon']


def test_capacity_failure_does_not_substitute_target_variables():
    data=SimpleNamespace(ids=['a'],candidates=lambda *_:np.array([100]))
    old=[dict(source='pems03',variable=0,origin=100,length=12,horizon=12,chunk=0) for _ in range(128)]
    with pytest.raises(ValueError,match='capacity'):fresh_target_queries(data,'pems03',old)


def test_metadata_frequency_uses_past_segment_not_future_or_global_frequency():
    first=np.arange(100)*900*10**9;second=first[-1]+np.arange(1,101)*600*10**9
    data=SimpleNamespace(times=np.r_[first,second],meta={'provenance':{'unit':'kW','channel':'Patv'}})
    a=target_text(data,'sdwpf',Query('sdwpf',0,80,48,24))
    b=target_text(data,'sdwpf',Query('sdwpf',0,180,48,24))
    assert '900 seconds' in a and '600 seconds' in b
    data.times[180:]=1
    assert target_text(data,'sdwpf',Query('sdwpf',0,180,48,24))==b
    data.meta['provenance']['unit']='MW'
    with pytest.raises(ValueError,match='unit'):target_text(data,'sdwpf',Query('sdwpf',0,180,48,24))


def test_prediction_shards_contain_histories_only_and_truth_requires_all_seals(tmp_path):
    t=np.arange(400)[:,None];v=np.arange(4)[None]
    data=SimpleNamespace(values=(t*.1+v).astype('float32'),mask=np.ones((400,4),bool),ids=list('abcd'),
        external=np.ones((4,4))-np.eye(4),segments=np.zeros(400,int),times=np.arange(400,dtype=np.int64)*300*10**9,
        meta={'provenance':{'unit':'vehicles per 5-minute bin','channel':'flow'}})
    query=dict(source='pems03',variable=0,origin=200,length=12,horizon=12,chunk=0)
    manifest=dict(scope='targets',queries=[query])
    write_inputs(tmp_path,manifest,{'pems03':data})
    store=InputStore(tmp_path);first,text=store.get(query)
    data.values[200:]=1e9;data.mask[200:]=False
    again,_=store.get(query)
    assert torch.equal(first.history,again.history) and 'traffic flow' in text
    shard=next((tmp_path/'inputs').glob('history-*.pt'))
    value=torch.load(shard,weights_only=False)
    assert set(value['records'][0])=={'query','item','text','text_sha256'}
    with pytest.raises(ValueError,match='all model'):truth_records(tmp_path,{})
    shard.write_bytes(shard.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='changed'):InputStore(tmp_path).get(query)


def test_truth_access_rejects_wrong_model_group_or_changed_prediction(tmp_path):
    from gofa_f.generative_data import sha
    from gofa_f.prefix_cache import atomic_json
    query=dict(source='pems03',variable=0,origin=200,length=12,horizon=12,chunk=0)
    atomic_json(tmp_path/'manifest.json',dict(queries=[query]));(tmp_path/'inputs').mkdir();atomic_json(tmp_path/'inputs/index.json',{})
    binding=sha(tmp_path/'manifest.json');shard=tmp_path/'prediction.json';atomic_json(shard,dict(p=[1.]*12))
    seals={}
    for model in ('native','enhanced','fusion','ufg'):
        path=tmp_path/(model+'_complete.json');seals[model]=path
        atomic_json(path,dict(binding=binding,model='native',queries=1,complete=True,
            input_index_sha256=sha(tmp_path/'inputs/index.json'),shards={'prediction.json':sha(shard)}))
    with pytest.raises(ValueError,match='bound elsewhere'):truth_records(tmp_path,seals)
    for model,path in seals.items():
        value=json.loads(path.read_text());value['model']=model;atomic_json(path,value)
    atomic_json(shard,dict(p=[0.]*12))
    with pytest.raises(ValueError,match='changed'):truth_records(tmp_path,seals)


def test_deployment_binds_old_tensor_acceptance_and_later_formal_fusion_separately(tmp_path):
    from pathlib import Path
    import gofa_f.dual_memory_eval_data as module
    from gofa_f.generative_data import sha
    from gofa_f.prefix_cache import atomic_json
    source=tmp_path/'source';source.mkdir();accepted=tmp_path/'accepted';code=accepted/'src/gofa_f';code.mkdir(parents=True)
    runtime=Path(module.__file__).parent
    for name in module.CORE:(code/name).write_bytes((runtime/name).read_bytes())
    assert not (code/'dual_memory_fusion.py').exists()
    atomic_json(source/'manifest.json',dict(weights_sha256='frozen-weights',formal_runtime_code={
        'dual_memory_fusion.py':sha(runtime/'dual_memory_fusion.py')}))
    for name in ('status.json','ufg_status.json'):atomic_json(source/name,dict(state='complete'))
    for name in ('activity.json','ufg_activity.json'):atomic_json(source/name,dict(worker=None))
    expert=source/'final-source-model';expert.mkdir();(expert/'best.pt').write_bytes(b'locked expert')
    atomic_json(source/'shared_gate.json',dict(depth=6,source_expert_sha256='expert-model',target_labels_used=False))
    atomic_json(source/'source_model_lock.json',dict(binding=sha(source/'manifest.json'),target_labels_used=False,
        source_confirmation_labels_used=False,depth=6,expert_model_sha256='expert-model',
        expert_checkpoint_sha256=sha(expert/'best.pt'),shared_gate_sha256=sha(source/'shared_gate.json')))
    for shape in ('12-12','48-24'):
        own=source/('ufg-'+shape);own.mkdir();(own/'best.pt').write_bytes(b'locked UFG')
        atomic_json(own/'selection.json',dict(checkpoint_sha256=sha(own/'best.pt'),target_labels_used=False,
            model_sha256='ufg-model',best_update=3000,stopped_update=5000,stop_reason='source_early_stop'))
    assert module.source_ready(source)
    locked=module.deployment(source,accepted);module.verify_deployment(locked)
    assert locked['shared_fusion_code_sha256']==sha(runtime/'dual_memory_fusion.py')
    assert 'dual_memory_fusion.py' not in locked['core_sha256']
    (expert/'best.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):module.verify_deployment(locked)
