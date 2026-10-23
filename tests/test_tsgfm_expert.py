from copy import deepcopy
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch
from torch import nn

from gofa_f.tsgfm_expert import TSGFMExpert, TSGFMEngine, edge_mean, pair_scores
from gofa_f.tsgfm_source import screen, configuration, ufg_baseline
from gofa_f.dual_memory_policy import capture, restore, accumulate


def item(length=12,horizon=12,nodes=3,missing=False):
    generator=torch.Generator().manual_seed(123)
    values=torch.rand(nodes,length,generator=generator)-.5
    observed=torch.ones_like(values,dtype=torch.bool)
    if missing:observed[-1]=False
    present=observed.any(1)
    adjacency=torch.ones(nodes,nodes,dtype=torch.bool)&present[:,None]&present[None]
    return SimpleNamespace(normalized=values,observed=observed,present=present,
        history=values*2+10,adjacency=adjacency,last=torch.arange(nodes).double()+10,
        extent=torch.arange(nodes).double()+1,horizon=horizon,node_ids=tuple(map(str,range(nodes))),identity='input')


def dense_reference(model,x):
    n,length=x.normalized.shape
    observed=x.observed&x.present[:,None]
    hidden=model.node_embedding(torch.stack((x.normalized.masked_fill(~observed,0),observed.float()),-1))
    messages=[]
    for node in range(n):
        sequence=[]
        for t in range(length):
            dest=hidden[node,t]
            neighbours=torch.where(x.adjacency[node]&observed[:,t])[0]
            if len(neighbours):
                source=hidden[neighbours,t]
                scores=model.spatial_attention(torch.cat((source,dest.expand_as(source)),-1)).squeeze(-1)
                spatial=(scores.softmax(0)[:,None]*source).sum(0)
            else:spatial=dest*0
            times=torch.where(observed[node,:t])[0]
            if len(times):
                source=hidden[node,times];edge=model.temporal_embedding(t-times)
                scores=model.temporal_attention(torch.cat((source,edge,dest.expand_as(source)),-1)).squeeze(-1)
                temporal=(scores.softmax(0)[:,None]*source).sum(0)
            else:temporal=dest*0
            sequence.append(model.combine_aggrs(torch.cat((temporal,spatial))))
        messages.append(torch.stack(sequence))
    updated=model.node_update(torch.stack(messages).reshape(n*length,-1),hidden.reshape(n*length,-1))
    return updated.reshape(n,length,-1)[:,-1].masked_fill(~x.present[:,None],0)


@pytest.mark.parametrize('edge_width',[0,5])
def test_factored_pair_mlp_matches_concat_values_and_gradients(edge_width):
    torch.manual_seed(42);width=8
    mlp=nn.Sequential(nn.Linear(2*width+edge_width,7),nn.LeakyReLU(),nn.Linear(7,1))
    a=torch.randn(13,width,requires_grad=True);b=torch.randn_like(a,requires_grad=True)
    edge=torch.randn(13,edge_width,requires_grad=True) if edge_width else None
    expected=mlp(torch.cat((a,b) if edge is None else (a,edge,b),-1)).squeeze(-1)
    actual=pair_scores(mlp,a,b,edge)
    parameters=[a,b,*mlp.parameters()]+([] if edge is None else [edge])
    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
    eg=torch.autograd.grad(expected.sum(),parameters,retain_graph=True)
    ag=torch.autograd.grad(actual.sum(),parameters)
    for x,y in zip(ag,eg):torch.testing.assert_close(x,y,rtol=1e-6,atol=1e-6)


def test_empty_neighbourhood_returns_zero_and_mixed_segment_attention():
    p=torch.randn(0,8)
    assert not edge_mean(p,torch.empty(0),torch.empty(0,dtype=torch.long),4).any()
    value=torch.tensor([[1.,2.],[3.,4.],[5.,6.]])
    actual=edge_mean(value,torch.tensor([1000.,1000.,-1000.]),torch.tensor([0,0,2]),4)
    torch.testing.assert_close(actual,torch.tensor([[2.,3.],[0.,0.],[5.,6.],[0.,0.]]))


@pytest.mark.parametrize('shape',[(12,12),(48,24)])
def test_parallel_space_time_matches_independent_dense_equations_and_gradients(shape):
    torch.manual_seed(42);model=TSGFMExpert(width=4);x=item(*shape,missing=True)
    expected=dense_reference(model,x);actual=model.encode(x)
    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
    parameters=list(model.parameters())
    eg=torch.autograd.grad(expected.square().sum(),parameters,allow_unused=True,retain_graph=True)
    ag=torch.autograd.grad(actual.square().sum(),parameters,allow_unused=True)
    for a,b in zip(ag,eg):
        assert (a is None)==(b is None)
        if a is not None:torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-6)


@pytest.mark.parametrize('shape,nodes', [((12,12),1),((12,12),31),((48,24),1),((48,24),31)])
def test_real_node_max_shape_all_parameter_gradients_finite(shape,nodes):
    torch.manual_seed(42);model=TSGFMExpert();x=item(*shape,nodes=nodes)
    p=model.forecast(x);assert p.shape==(nodes,shape[1]) and p.isfinite().all()
    (p-x.last[:,None]-.3).square().mean().backward()
    assert all(v.grad is not None and v.grad.isfinite().all() for v in model.parameters())
    assert model.spatial_attention[0].weight.grad.abs().sum()>0 or nodes==1
    assert model.temporal_attention[0].weight.grad.abs().sum()>0


def test_node_permutation_equivariance_without_node_id_parameters():
    torch.manual_seed(42);model=TSGFMExpert(width=8);x=item(nodes=5);expected=model.forecast(x)
    order=torch.tensor([0,4,1,3,2]);changed=deepcopy(x)
    for name in ('normalized','observed','present','history','last','extent'):
        setattr(changed,name,getattr(x,name)[order])
    changed.adjacency=x.adjacency[order][:,order]
    changed.node_ids=tuple(x.node_ids[int(i)] for i in order)
    torch.testing.assert_close(model.forecast(changed),expected[order],rtol=1e-6,atol=1e-6)


def test_all_missing_auxiliary_ignored_and_history_inputs_not_modified():
    model=TSGFMExpert(width=8);x=item(missing=True);before=deepcopy(x)
    expected=model.forecast(x)
    for key in ('normalized','observed','adjacency','present','last','extent'):
        assert torch.equal(getattr(x,key),getattr(before,key))
    x.normalized[-1]=1e10;x.history[-1]=-1e10
    assert torch.equal(expected[0],model.forecast(x)[0])


def test_cross_node_observations_can_change_target_and_receive_gradients():
    torch.manual_seed(42);model=TSGFMExpert(width=8);x=item()
    x.normalized.requires_grad_();model.forecast(x)[0].sum().backward()
    assert x.normalized.grad[1:,-1].abs().sum()>0
    x.adjacency=torch.eye(3,dtype=torch.bool);x.normalized.grad=None
    model.forecast(x)[0].sum().backward()
    assert not x.normalized.grad[1:].any()


def test_future_truth_and_mask_changes_never_change_forecast_input():
    x=item();future=torch.ones(3,12);mask=torch.ones(3,12,dtype=torch.bool)
    class Store:
        def get(self,row,split,role='fit'):return x,future,mask
    engine=TSGFMEngine({},Store());a=engine.predict({},'development')[0]
    future[:]=1e12;mask[:]=False
    assert np.array_equal(a,engine.predict({},'development')[0])


def test_model_save_load_and_partial_optimizer_accumulation_replay_exact(tmp_path):
    torch.manual_seed(42);m=TSGFMExpert(width=8);x=item();optimizer=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=0)
    def micro():accumulate((m.forecast(x)-11).abs().mean()/4,list(m.parameters()))
    micro();saved=capture(m,optimizer,dict(pending=1,cursor={'total':1}),'source')
    for _ in range(3):micro()
    optimizer.step();expected=deepcopy(m.state_dict())
    state=restore(m,optimizer,saved,'source');assert state['pending']==1
    for _ in range(3):micro()
    optimizer.step();assert all(torch.equal(v,m.state_dict()[k]) for k,v in expected.items())
    path=tmp_path/'model.pt';torch.save(m.state_dict(),path)
    other=TSGFMExpert(width=8);other.load_state_dict(torch.load(path,weights_only=True))
    assert torch.equal(m.forecast(x),other.forecast(x))


def test_screen_is_source_only_and_zero_error_ties_are_not_improvements():
    names=('pems04_flow','pems04_speed','boston2017','penmanshiel2017')
    def metrics(values):return dict(views={n:dict(macro_mae=v) for n,v in zip(names,values)})
    base=metrics([10.,1.,2.,100.])
    assert screen(metrics([9.,.9,1.8,90.]),base)['passed']
    assert not screen(metrics([9.,.9,2.1,90.]),base)['passed']
    assert not screen(base,base)['passed']
    result=screen(metrics([0,0,0,0]),metrics([0,0,0,0]));assert not result['passed']
    assert result['zero_shot_superiority_proven'] is False


def test_config_refuses_target_access_and_recipe_changes(tmp_path,monkeypatch):
    import yaml
    from gofa_f.tsgfm_source import RECIPE,PROTOCOL
    monkeypatch.setenv('GOFA_WORKSPACE','/workspace')
    cfg=dict(protocol=PROTOCOL,training=RECIPE,target_training_enabled=False,target_evaluation_enabled=False,
        model=dict(width=128,message_iterations=1,maximum_history=48,physical_subgraph=True,
            observed_origin_scaling=True,shared_multistep_head=True),
        resources=dict(cuda_enabled=False,cpu_threads=2,deadline=None,no_new_llm_training=True))
    p=tmp_path/'config.yaml';p.write_text(yaml.safe_dump(cfg));assert configuration(p)==cfg
    cfg['target_evaluation_enabled']=True;p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='target'):configuration(p)
    cfg['target_evaluation_enabled']=False;cfg['training']=dict(RECIPE,learning_rate=.01);p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='recipe'):configuration(p)


def test_ufg_reuse_checks_source_checkpoint_query_truth_mask_and_input_hash(tmp_path):
    from gofa_f.dual_memory_data import GROUPS,SHAPES
    from gofa_f.generative_data import sha
    parent=tmp_path/'parent';parent.mkdir();queries={n:{} for n in GROUPS}
    items={}
    for length,horizon in SHAPES:
        shape=f'{length}-{horizon}';root=parent/('ufg-'+shape);root.mkdir()
        (root/'best.pt').write_bytes(b'frozen-UFG-source-weight')
        (root/'selection.json').write_text(json.dumps(dict(target_labels_used=False,
            checkpoint_sha256=sha(root/'best.pt'),model_sha256='tensorSHA',best_update=250)))
        rows=root/'evaluations/update-000250';rows.mkdir(parents=True)
        for index,name in enumerate(GROUPS):
            row=dict(source=name,variable=0,origin=100,length=length,horizon=horizon)
            queries[name][shape]=dict(development=[row]);x=item(length,horizon)
            y=torch.ones(3,horizon).double()*2;mask=torch.ones_like(y,dtype=torch.bool)
            items[(name,length)]=(x,y,mask)
            r=dict(query=row,identity=dict(checkpoint_sha='tensorSHA'),input_sha256=x.identity,
                prediction=[2.1]*horizon,target=y[0].tolist(),observed=mask[0].tolist())
            (rows/f'{index:05d}.json').write_text(json.dumps(r))
    (parent/'ufg_activity.json').write_text(json.dumps(dict(spent_cpu_seconds=100)))
    class Store:
        def get(self,row,split):
            assert split=='development'
            return items[row['source'],row['length']]
    manifest=dict(queries=queries,training_unit_normalizers={n:1. for n in GROUPS})
    assert len(ufg_baseline(parent,manifest,Store())['records'])==8
    bad=parent/'ufg-12-12/evaluations/update-000250/00000.json'
    record=json.loads(bad.read_text());record['target'][0]=1e9;bad.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='truth'):ufg_baseline(parent,manifest,Store())
