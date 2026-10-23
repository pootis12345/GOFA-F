from copy import deepcopy
import json
import numpy as np
import pytest
import torch

from gofa_f.dual_memory_policy import capture,restore,accumulate
from gofa_f.ufg_source_adequacy import tree_digest,transferable,continue_state,configuration


def value():
    torch.manual_seed(42);model=torch.nn.Linear(3,2);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    model(torch.ones(1,3)).square().mean().backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
    state=dict(updates=10000,pending=0,completed=True,stop_reason='maximum_updates',
        cursor=dict(total=640000,pools={'flow':{'epoch':30,'offset':71}}),
        selection=dict(last_validated=10000,best_update=9000,best_score=1.,reference_score=1.,stale=4,stop=False),
        active_seconds=2980.)
    return model,optimizer,capture(model,optimizer,state,'original')


def test_exact_transfer_changes_only_binding_and_completed_bookkeeping():
    model,optimizer,old=value();before=deepcopy(old)
    new=continue_state(old,'original','extended')
    assert tree_digest(old)==tree_digest(before)
    assert tree_digest(transferable(old))==tree_digest(transferable(new))
    assert new['binding']=='extended' and new['state']['completed'] is False
    assert 'stop_reason' not in new['state']
    assert new['state']['cursor']==old['state']['cursor'] and new['state']['selection']==old['state']['selection']


def test_next_optimizer_update_is_identical_after_transaction_transfer():
    model,optimizer,old=value();new=continue_state(old,'original','extended')
    def step(checkpoint,binding):
        state=restore(model,optimizer,checkpoint,binding)
        accumulate(model(torch.randn(4,3)).square().mean(),list(model.parameters()))
        optimizer.step()
        return capture(model,optimizer,state,'same')
    a=step(old,'original');b=step(new,'extended')
    for key in ('model','optimizer','gradients','rng'):
        assert tree_digest(a[key])==tree_digest(b[key])


@pytest.mark.parametrize('mutate',[
    lambda v:v.update(binding='wrong'),
    lambda v:v['state'].update(updates=9999),
    lambda v:v['state'].update(completed=False),
    lambda v:v['state'].update(stop_reason='source_early_stop'),
    lambda v:v['state']['selection'].update(stop=True),
    lambda v:v['state']['selection'].update(last_validated=9750)])
def test_transfer_rejects_wrong_or_unvalidated_source_transaction(mutate):
    _,_,old=value();mutate(old)
    with pytest.raises(ValueError):continue_state(old,'original','new')


def test_canonical_digest_covers_numpy_rng_complex_optimizer_and_partial_gradients():
    v=dict(t=torch.tensor([1+2j],dtype=torch.complex64),a=np.arange(5,dtype=np.uint32),
        r=(None,{'state':torch.tensor(3.)}),gradients={'weight':torch.tensor([.1])})
    assert tree_digest(v)==tree_digest(deepcopy(v))
    for key in v:
        changed=deepcopy(v)
        if key=='t':changed[key][0]+=1j
        elif key=='a':changed[key][0]+=1
        elif key=='r':changed[key][1]['state']+=1
        else:changed[key]['weight']+=1
        assert tree_digest(changed)!=tree_digest(v)


def test_config_blocks_target_access_or_scientific_changes(tmp_path,monkeypatch):
    import yaml
    from gofa_f.ufg_source_adequacy import PROTOCOL
    monkeypatch.setenv('GOFA_WORKSPACE','/workspace')
    cfg=dict(protocol=PROTOCOL,target_training_enabled=False,target_evaluation_enabled=False,
        maximum_updates=20000,shape='12-12',resources=dict(cuda_enabled=False,cpu_threads=2,deadline=None))
    p=tmp_path/'config.yaml';p.write_text(yaml.safe_dump(cfg));assert configuration(p)==cfg
    cfg['target_evaluation_enabled']=True;p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='Source-only'):configuration(p)
    cfg['target_evaluation_enabled']=False;cfg['maximum_updates']=15000;p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='preregistered'):configuration(p)
