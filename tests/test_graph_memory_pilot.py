from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import LlamaTokenizer

from gofa_f.graph_memory_pilot import (CONTROLS,VIEWS,batch_queries,checkpoint,configuration,gate_report,
    history_target,restore_checkpoint,tensor_digest)
from gofa_f.graph_memory_data import memory_graph
from gofa_f.short_codec import ShortCodec
from gofa_f.short_training import compact_forward,compact_loss,teacher_prefix
from test_graph_memory import data,tiny,nonzero,q


@pytest.fixture(scope='module')
def codec():
    path=Path('/home/dcg/dcg_data/GOFA-F/data/weights/chattime-7b-20baa88')
    if not path.exists():pytest.skip('Pinned official tokenizer on C only')
    return ShortCodec(LlamaTokenizer.from_pretrained(path,local_files_only=True))


def manifest():
    return dict(queries={name:dict(train=[dict(source=name,variable=i,origin=100+4*i,length=12,horizon=12,chunk=0)
        for i in range(64)]) for name in VIEWS})


def test_same_query_order_and_three_domain_loss_weights():
    m=manifest();rows=batch_queries(m,0)
    assert len(rows)==8 and sum(v['weight'] for v in rows)==pytest.approx(1.)
    assert sum(v['weight'] for v in rows if v['query']['source'].startswith('pems'))==pytest.approx(1/3)
    assert rows==batch_queries(deepcopy(m),0)
    for name in VIEWS:
        first=[r['query']['variable'] for update in range(32) for r in batch_queries(m,update) if r['query']['source']==name]
        second=[r['query']['variable'] for update in range(32,64) for r in batch_queries(m,update) if r['query']['source']==name]
        assert sorted(first)==sorted(second)==list(range(64)) and first!=second


def test_locked_config_rejects_target_training_and_budget_changes(tmp_path,monkeypatch):
    import yaml
    monkeypatch.setenv('GOFA_WORKSPACE','/source')
    path=Path(__file__).parents[1]/'configs/stage57_graph_memory_pilot.yaml'
    cfg=yaml.safe_load(path.read_text());target=tmp_path/'config.yaml';target.write_text(yaml.safe_dump(cfg))
    assert configuration(target)['training']['max_updates']==100
    for key,value in [('max_updates',101),('effective_batch',64)]:
        altered=deepcopy(cfg);altered['training'][key]=value;target.write_text(yaml.safe_dump(altered))
        with pytest.raises(ValueError,match='protocol'):configuration(target)
    altered=deepcopy(cfg);altered['policy']['target_training_enabled']=True;target.write_text(yaml.safe_dump(altered))
    with pytest.raises(ValueError,match='protocol'):configuration(target)


def test_history_inputs_do_not_use_future_truth_or_future_missing_masks():
    d=data(9);query=dict(source='pems04_flow',variable=0,origin=200,length=12,horizon=12,chunk=0)
    before,_=history_target(d,query,False);graph=memory_graph(d,q())
    d.values[200:]=1e8;d.mask[200:]=False
    after,target=history_target(d,query,False)
    np.testing.assert_array_equal(before,after);assert target is None and graph.identity==memory_graph(d,q()).identity


def test_partial_gradients_optimizer_rng_and_cursor_restore_exactly(codec,tmp_path):
    model=tiny(codec);nonzero(model);d=data();query=q();graph=memory_graph(d,query)
    tokens=codec.batch([codec.record(d.values[188:200,0],d.values[200:212,0])])
    optimizer=torch.optim.AdamW(model.adapter.parameters(),lr=1e-4)
    def micro():
        ctx=model.encode_origins([graph],codec)
        loss=compact_loss(compact_forward(model,codec,tokens,ctx,teacher_prefix(model,tokens)))[0]/2
        loss.backward();return float(loss.detach())
    micro();optimizer.step();model.mark_updated();optimizer.zero_grad(set_to_none=True)
    micro();state=dict(binding='b',condition='physical',updates=1,pending=1,partial_losses=[.5],losses=[],training_seconds=2.)
    path=tmp_path/'last.pt';checkpoint(model,optimizer,state,path)
    expected_loss=micro();optimizer.step();model.mark_updated();expected=model.trainable_state();random_after=torch.rand(4)
    restored=restore_checkpoint(model,optimizer,path,'b','physical')
    assert restored['updates']==1 and restored['pending']==1 and restored['partial_losses']==[.5]
    assert micro()==expected_loss;optimizer.step();model.mark_updated()
    for name,value in model.trainable_state().items():assert torch.equal(value,expected[name])
    assert torch.equal(torch.rand(4),random_after)
    with pytest.raises(ValueError,match='binding'):restore_checkpoint(model,optimizer,path,'different','physical')
    with pytest.raises(ValueError,match='binding'):restore_checkpoint(model,optimizer,path,'b','self')


def test_tensor_digest_binds_values_dtype_and_shape():
    first={'a':torch.ones(4)}
    assert tensor_digest(first)==tensor_digest(deepcopy(first))
    assert tensor_digest(first)!=tensor_digest({'a':torch.ones(2,2)})
    assert tensor_digest(first)!=tensor_digest({'a':torch.ones(4,dtype=torch.float64)})


def records(physical=1.,time_failure=False):
    result={label:[] for label in ('native',*CONTROLS)}
    for label in result:
        for name in VIEWS:
            for i in range(16):
                error=physical if label=='physical' else 2.
                if time_failure and label=='physical' and i<4:error=2.1
                result[label].append(dict(query=dict(source=name,variable=i,origin=100+i,length=12,horizon=2,chunk=0),
                    prediction=[error,error],target=[0.,1000.],observed=[True,False]))
    return result


def test_gate_requires_real_graph_gain_against_native_self_and_permutation():
    good=gate_report(records());assert good['passed'] and not good['goal_achieved'] and good['target_predictions']==0
    equal=gate_report(records(physical=2.));assert not equal['passed']
    worse=records();worse['rewired']=deepcopy(worse['physical'])
    assert not gate_report(worse)['passed']


def test_gate_time_block_regression_cannot_hide_in_average_gain():
    value=gate_report(records(physical=.5,time_failure=True))
    assert not value['passed'] and any(not c['passed'] and c['scope']=='time_block_0' for c in value['checks'])
    assert value['macro_mae_by_view']['physical']['pems04_flow']<value['macro_mae_by_view']['native']['pems04_flow']


def test_pilot_shared_admission_keeps_legacy_rules():
    import yaml
    import gofa_f.short_launcher as launcher
    cfg=yaml.safe_load((Path(__file__).parents[1]/'configs/stage57_graph_memory_pilot.yaml').read_text())
    assert type(launcher.admission_gate(cfg['resources'],'graph_memory_pilot')).__name__=='CapOnlyGate'
    with pytest.raises(ValueError):launcher.admission_gate(cfg['resources'],'source_retraining')
