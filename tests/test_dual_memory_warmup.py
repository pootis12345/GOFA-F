from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gofa_f.dual_memory_model import MemoryInterface
from gofa_f.dual_memory_engines import model_digest
from gofa_f.dual_memory_policy import accumulate, capture, restore, query_loss
from gofa_f.dual_memory_warmup import TemporalWarmup, TemporalEngine, transfer, cpu_only, load_config, inherited_cost


def item(length=12, horizon=12, nodes=3, missing=False):
    x = torch.linspace(-.5, .5, length).expand(nodes, length).clone()
    mask = torch.ones_like(x, dtype=torch.bool)
    if missing:
        mask[-1] = False; x[-1] = 0
    return SimpleNamespace(history=x+10, normalized=x, observed=mask,
        extent=torch.arange(1, nodes+1).double(), last=torch.arange(nodes).double()+10,
        present=mask.any(1), node_ids=tuple(str(i) for i in range(nodes)), horizon=horizon,
        identity='history-only-binding')


def learned_model(width=8, slots=4):
    torch.manual_seed(42); m = TemporalWarmup(slots, width)
    with torch.no_grad():m.head[-1].weight.normal_(std=.03)
    return m


@pytest.mark.parametrize('shape', [(12, 12), (48, 24)])
def test_transfer_exact_production_prediction_for_arbitrary_memory(shape):
    m = learned_model(); selected = dict(binding='source', model=deepcopy(m.state_dict()), model_sha256=model_digest(m))
    adapter = MemoryInterface(16, depth=3, slots=4, width=8)
    untouched = {k:v.clone() for k,v in adapter.state_dict().items() if not k.startswith(('temporal.', 'head.'))}
    transfer(adapter, selected, 'source')
    x = item(*shape, missing=True)
    clock = torch.linspace(-1., 0., shape[0]).expand(3, shape[0])
    temporal, _ = adapter.temporal(torch.stack((x.normalized, x.observed.float(), clock), -1))
    actual = adapter.predict(torch.randn(3, 4, 16), temporal[:, -1], x)
    torch.testing.assert_close(actual, m.forecasts([x])[0], rtol=1e-6, atol=1e-6)
    assert all(torch.equal(v, adapter.state_dict()[k]) for k,v in untouched.items())


def test_zero_initial_head_returns_persistence_in_both_shapes_and_ragged_rosters():
    m = TemporalWarmup(width=4)
    samples = [item(12,12,1), item(48,24,5,True), item(12,12,4)]
    for x,p in zip(samples,m.forecasts(samples)):
        assert torch.equal(p, x.last[:,None].expand(-1,x.horizon))


def test_batched_mixed_shapes_match_individual_and_query_loss_reduction():
    m = learned_model(); samples = [item(12,12,2), item(48,24,4,True), item(12,12,5)]
    batched = m.forecasts(samples)
    for x,p in zip(samples,batched):
        torch.testing.assert_close(p,m.forecasts([x])[0],rtol=1e-6,atol=1e-6)
    rows = [dict(source=str(i)) for i in range(len(samples))]
    class Store:
        def get(self,row,split,role='fit'):
            assert split=='train'
            x=samples[int(row['source'])]
            y=x.last[:,None].expand(-1,x.horizon)+.5
            valid=torch.ones_like(y,dtype=torch.bool);valid[0,::2]=False
            return x,y,valid
    engine=TemporalEngine({'training_unit_normalizers':{'0':1.,'1':2.,'2':3.}},Store())
    engine.model=m
    actual=engine.loss(rows)
    expected=sum(query_loss(p,y,mask,x.present,i+1) for i,(x,p) in enumerate(zip(samples,batched))
        for _,y,mask in [engine.store.get(rows[i],'train')])
    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)


def test_future_target_and_mask_changes_do_not_change_predictions():
    m=learned_model();x=item(48,24,4,True)
    future=torch.ones((4,24)).double()
    future_mask=torch.zeros(4,24,dtype=torch.bool)
    class Store:
        def get(self,row,split,role='fit'):
            assert split=='development'
            return x,future,future_mask
    engine=TemporalEngine({},Store());engine.model=m
    before=engine.predict({},'development')[0].copy()
    future.fill_(1e12);future_mask.fill_(True)
    after,y,mask,_=engine.predict({},'development')
    assert np.array_equal(before,after)
    assert (y==1e12).all() and mask.all()


def test_real_learning_and_zero_memory_columns():
    torch.manual_seed(42);m=TemporalWarmup(width=4);x=item(nodes=2)
    target=x.last[:,None].expand(-1,12)+.2;mask=torch.ones_like(target,dtype=torch.bool)
    optimizer=torch.optim.AdamW(m.parameters(),lr=.003)
    def loss():return query_loss(m.forecasts([x])[0],target,mask,x.present,1.)
    initial=float(loss().detach())
    for _ in range(80):
        optimizer.zero_grad();loss().backward();optimizer.step()
    assert float(loss().detach())<initial*.5
    assert not bool(m.head[0].weight[:,:16].any())
    assert any(bool(p.grad.abs().sum()) for p in m.temporal.parameters())


def test_partial_gradient_and_optimizer_resume_exact():
    m=learned_model();optimizer=torch.optim.AdamW(m.parameters(),lr=.001)
    x=item(nodes=2);target=x.last[:,None].expand(-1,12)+.3
    def micro():
        accumulate((m.forecasts([x])[0]-target).abs().mean()/4,list(m.parameters()))
    micro();saved=capture(m,optimizer,dict(pending=1,cursor={'total':1}),'source')
    for _ in range(3):micro()
    optimizer.step();expected=deepcopy(m.state_dict())
    state=restore(m,optimizer,saved,'source');assert state['pending']==1
    for _ in range(3):micro()
    optimizer.step()
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in expected.items())


def test_transfer_rejects_wrong_binding_tensor_sha_and_memory_contribution():
    m=learned_model();adapter=MemoryInterface(16,3,width=8)
    selected=dict(binding='source',model=m.state_dict(),model_sha256=model_digest(m))
    with pytest.raises(ValueError,match='binding'):transfer(adapter,selected,'target')
    changed=deepcopy(selected);changed['model']['head.2.bias'].add_(1)
    with pytest.raises(ValueError,match='SHA'):transfer(adapter,changed,'source')
    with torch.no_grad():m.head[0].weight[0,0]=1
    selected.update(model=m.state_dict(),model_sha256=model_digest(m))
    with pytest.raises(ValueError,match='Memory'):transfer(adapter,selected,'source')


def test_cpu_guard_rejects_cuda_or_extra_threads(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    with pytest.raises(ValueError,match='CUDA'):cpu_only()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');monkeypatch.setenv('OMP_NUM_THREADS','8')
    monkeypatch.setenv('MKL_NUM_THREADS','2')
    with pytest.raises(ValueError,match='threads'):cpu_only()


def test_config_rejects_target_access_and_recipe_search(tmp_path,monkeypatch):
    import yaml
    monkeypatch.setenv('GOFA_WORKSPACE','/workspace')
    settings=dict(max_updates=2000,effective_batch=16,microbatch=16,learning_rate=.001,
        warmup_updates=50,interface_only_updates=0,weight_decay=.0001,gradient_clip=1.,
        validation_every=100,early_min_updates=1000,early_patience=8,significant_improvement=.001)
    value=dict(protocol='stage60-source-warmup-v1',target_training_enabled=True,
        target_evaluation_enabled=False,warmup=settings)
    p=tmp_path/'config.yaml';p.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError,match='Source-only'):load_config(p)
    value['target_training_enabled']=False;value['warmup']['learning_rate']=.1
    p.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError,match='recipe'):load_config(p)


def test_inherited_cost_preserves_source_and_evaluation_without_reset(tmp_path):
    import json
    from gofa_f.generative_data import sha
    parent=tmp_path/'parent';parent.mkdir();(parent/'activity.json').write_text(json.dumps({'spent_seconds':20.}))
    costs=tmp_path/'costs.json'
    costs.write_text(json.dumps(dict(inherited_source_gpu_seconds=20.,new_evaluation_gpu_seconds=5.,cumulative_gpu_seconds=25.)))
    cfg=dict(inherited_costs_file=str(costs),inherited_costs_sha256=sha(costs))
    assert inherited_cost(parent,cfg)==25.
    costs.write_text(json.dumps(dict(inherited_source_gpu_seconds=20.,new_evaluation_gpu_seconds=5.,cumulative_gpu_seconds=20.)))
    cfg['inherited_costs_sha256']=sha(costs)
    with pytest.raises(ValueError,match='reset'):inherited_cost(parent,cfg)
