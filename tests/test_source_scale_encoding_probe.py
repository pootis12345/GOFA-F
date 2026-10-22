import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gofa_f import source_graph_probe as p
from gofa_f import source_neighborhood_probe as n
from gofa_f.source_normalized_probe import normalized_runtime
from gofa_f.source_scale_encoding_probe import encode_pairs,history_edges,scale_encoding_runtime
from gofa_f.lag_graph_probe import state_digest
from test_source_graph_probe import model_input,random_model


def test_monotone_unclipped_encoding_direction_and_inverse():
    low=np.array([-30.,0.,500.]);scale=np.array([1.,20.,3.])
    edge=encode_pairs(low,scale).astype(np.float64)
    raw=np.stack((np.log(scale[None,:]/scale[:,None]),(low[None,:]-low[:,None])/scale[:,None]),-1)
    np.testing.assert_allclose(np.sinh(edge),raw,rtol=1e-6,atol=1e-6)
    assert edge[0,2,1]>edge[0,1,1]>0 and edge[2,0,1]<0
    assert np.all(np.diagonal(edge[:,:,0])==0)


def test_invalid_or_constant_scales():
    np.testing.assert_array_equal(encode_pairs([0,0],[1,1]),np.zeros((2,2,2)))
    for scales in ([0,1],[-1,1],[np.nan,1]):
        with pytest.raises(ValueError):encode_pairs([0,0],scales)


def test_history_causality_missing_nodes_and_unchanged_cached_arrays():
    values=np.arange(60.,dtype=float).reshape(20,3);values[:,1]=5.;values[:,2]=np.nan
    data=SimpleNamespace(values=values,mask=np.isfinite(values))
    raw=values[:12].T;mask=np.isfinite(raw);low=np.array([0.,5.,0.]);scale=np.array([33.,1.,1.])
    a=dict(nodes=np.array([[0,1,2,-1]]),edge_features=np.zeros((1,4,4,2),np.float32),
        low=np.array([0.]),scale=np.array([33.]),x=np.zeros((1,4,12,2),np.float32))
    a['x'][0,:3,:,0]=np.where(mask,(raw-low[:,None])/scale[:,None]-.5,0.)
    a['x'][0,:3,:,1]=mask
    a['edge_features'][0,:3,:3]=np.stack((np.clip(np.log(scale[None,:]/scale[:,None]),-5,5),
        np.clip((low[None,:]-low[:,None])/scale[:,None],-5,5)),-1)
    rows=[dict(query=dict(origin=12,length=12))];before=copy.deepcopy(a)
    first=history_edges(data,a,rows);data.values[12:]=1e9
    np.testing.assert_array_equal(first,history_edges(data,a,rows))
    assert np.isfinite(first).all() and not first[0,3].any()
    for k in a:np.testing.assert_array_equal(a[k],before[k])
    bad=copy.deepcopy(a);bad['scale'][0]=2
    with pytest.raises(ValueError):history_edges(data,bad,rows)


def test_runtime_exact_initial_parameters_configuration_and_restoration(monkeypatch):
    monkeypatch.setenv('GOFA_WORKSPACE','/tmp/source');root=Path(__file__).parents[1]
    factory=p.model_for;protocol=n.PROTOCOL
    with normalized_runtime():
        cfg=n.configuration(root/'configs/stage40_source_normalized_probe.yaml');expected=state_digest(p.model_for(cfg,42))
    with scale_encoding_runtime():
        cfg=n.configuration(root/'configs/stage45_source_scale_encoding_probe.yaml')
        assert state_digest(p.model_for(cfg,42))==expected and n.stage_number(cfg)==45
        assert sum(v.numel() for v in p.model_for(cfg,42).parameters())==10831
    assert p.model_for is factory and n.PROTOCOL==protocol


def test_self_path_ignores_encoding_and_nonzero_path_finite_backward():
    model=random_model();x,adj,edge=model_input();adj[:]=False
    changed=torch.asinh(edge*100)
    assert torch.equal(model(x,adj,edge),model(x,adj,changed))
    adj[:]=True;loss=model(x,adj,changed).square().mean();loss.backward()
    assert all(v.grad is None or v.grad.isfinite().all() for v in model.parameters())
