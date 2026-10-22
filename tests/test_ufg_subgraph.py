from pathlib import Path

import numpy as np
import pytest
import torch

from gofa_f.ufg import source_manifest
from gofa_f.ufg_subgraph import SubgraphUFGTime,pack_history

ROOT=Path('/home/dcg/dcg_data/GOFA-F/external/ufgtime')


def test_scaling_constant_missing_negative_and_padded_values():
    raw=np.array([[[-5.,2.,np.nan],[0.,2.,np.nan],[5.,2.,np.nan]]])
    value=pack_history(raw,np.isfinite(raw))
    torch.testing.assert_close(value.normalized[0,:,0],torch.tensor([-.5,0.,.5]),rtol=0,atol=0)
    assert value.extent[0,1]==1 and value.minimum[0,1]==2
    assert torch.equal(value.normalized[0,:,1],torch.full((3,),-.5))
    assert not value.observed[:,:,2:].any() and not value.present[:,2:].any()
    assert value.normalized[:,:,2:].eq(0).all()
    with pytest.raises(ValueError,match='observation'):pack_history(raw,np.zeros_like(raw,bool))
    with pytest.raises(ValueError,match='nine'):pack_history(np.zeros((1,12,10)),np.ones((1,12,10),bool))


def test_explicit_history_window_does_not_read_future_values_or_missingness():
    series=np.arange(100*4,dtype=float).reshape(100,4);mask=np.ones_like(series,bool)
    first=pack_history(series[20:68][None],mask[20:68][None])
    series[68:]=1e9;mask[68:]=False
    second=pack_history(series[20:68][None],mask[20:68][None])
    assert torch.equal(first.normalized,second.normalized) and torch.equal(first.minimum,second.minimum)


@pytest.mark.parametrize('length,horizon,nodes',[(12,12,2),(12,12,9),(48,24,3),(48,24,9)])
def test_exact_official_forward_raw_inverse_and_auxiliary_gradients(length,horizon,nodes):
    source_manifest(ROOT);torch.manual_seed(42)
    model=SubgraphUFGTime(ROOT,length,horizon)
    x=torch.randn(1,length,nodes,requires_grad=True);mask=torch.ones_like(x,dtype=torch.bool)
    if nodes>2:mask[:,2::3,2]=False
    value=pack_history(x,mask);prediction=model(value)
    direct=model.model(value.normalized,value.observed)
    expected=(direct[:,:,0].double()+.5)*value.extent[:,0:1]+value.minimum[:,0:1]
    torch.testing.assert_close(prediction,expected,rtol=0,atol=0)
    assert prediction.shape==(1,horizon) and torch.isfinite(prediction).all()
    prediction.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    assert x.grad[:,:,1:].abs().sum()>0 and torch.equal(x.grad[~mask],torch.zeros_like(x.grad[~mask]))
    with torch.no_grad():
        model2=SubgraphUFGTime(ROOT,length,horizon);model2.load_state_dict(model.state_dict())
        torch.testing.assert_close(model2(value),prediction,rtol=0,atol=0)
        garbage=x.detach().clone().masked_fill(~mask,1e10)
        torch.testing.assert_close(model(pack_history(garbage,mask)),prediction,rtol=0,atol=0)
    assert model.provenance['physical_roster_but_native_learned_frequency_graph']


def test_task_shapes_cannot_silently_share_incompatible_checkpoints():
    model=SubgraphUFGTime(ROOT,12,12)
    with pytest.raises(ValueError,match='shape'):model(pack_history(np.ones((1,48,9)),np.ones((1,48,9),bool)))
    with pytest.raises(ValueError,match='registered'):SubgraphUFGTime(ROOT,12,12,width=16)
