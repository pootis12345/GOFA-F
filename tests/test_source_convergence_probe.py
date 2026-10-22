import copy
from pathlib import Path

import pytest
import torch

from gofa_f import source_graph_probe as p
from gofa_f import source_neighborhood_probe as n
from gofa_f.source_convergence_probe import transplant_state,convergence_runtime
from gofa_f.lag_graph_probe import state_digest


def saved_state():
    model=torch.nn.Linear(2,1);optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    x=torch.tensor([[1.,2.]]);optimizer.zero_grad();model(x).square().sum().backward();optimizer.step()
    return dict(binding='old',update=1000,model=model.state_dict(),optimizer=optimizer.state_dict(),
        rng={'query_cursor':[1,2,3]},torch_rng=torch.get_rng_state(),cursor={'position':17},best_score=.9,best_update=900)


def test_transplant_preserves_every_numerical_state_and_original(tmp_path):
    torch.manual_seed(42);state=saved_state();old=tmp_path/'old.pt';new=tmp_path/'new.pt';torch.save(state,old)
    original=old.read_bytes();proof=transplant_state(old,new,'old','new');actual=torch.load(new,weights_only=False)
    assert old.read_bytes()==original and actual['binding']=='new' and proof['update']==1000
    def same(a,b):
        if torch.is_tensor(a):assert torch.equal(a,b)
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:same(a[k],b[k])
        elif isinstance(a,(list,tuple)):
            assert len(a)==len(b)
            for x,y in zip(a,b):same(x,y)
        else:assert a==b
    for key in state:
        if key!='binding':same(state[key],actual[key])


def test_wrong_prefix_and_binding_rejected(tmp_path):
    state=saved_state();source=tmp_path/'source.pt';torch.save(state,source)
    with pytest.raises(ValueError):transplant_state(source,tmp_path/'bad.pt','wrong','new')
    state['update']=999;torch.save(state,source)
    with pytest.raises(ValueError):transplant_state(source,tmp_path/'bad.pt','old','new')


def test_next_update_exact_after_rebinding(tmp_path):
    torch.manual_seed(42);state=saved_state();torch.save(state,tmp_path/'old.pt')
    transplant_state(tmp_path/'old.pt',tmp_path/'new.pt','old','new')
    def next_step(saved):
        model=torch.nn.Linear(2,1);model.load_state_dict(saved['model']);optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        optimizer.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['torch_rng'])
        x=torch.randn(3,2);optimizer.zero_grad();model(x).square().mean().backward();optimizer.step()
        return state_digest(model),optimizer.state_dict()['state'][0]['exp_avg']
    a=next_step(state);b=next_step(torch.load(tmp_path/'new.pt',weights_only=False))
    assert a[0]==b[0] and torch.equal(a[1],b[1])


def test_fixed_configuration_and_runtime_restore(monkeypatch):
    monkeypatch.setenv('GOFA_WORKSPACE','/tmp/source');old=(n.PROTOCOL,n.configuration,p.atomic_json)
    with convergence_runtime():
        cfg=n.configuration(Path(__file__).parents[1]/'configs/stage50_source_convergence_probe.yaml')
        assert cfg['training']['max_updates']==3000 and n.stage_number(cfg)==50
    assert (n.PROTOCOL,n.configuration,p.atomic_json)==old


def test_progress_budget_is_explicit(monkeypatch):
    captured=[];monkeypatch.setattr(p,'atomic_json',lambda path,value:captured.append(value))
    with convergence_runtime():
        p.atomic_json(Path('/tmp/progress.json'),dict(phase='matched_source_training',updates=1500,max_updates=1000))
    assert captured[0]['max_updates']==3000 and captured[0]['extra_updates']==500
