import copy
from types import SimpleNamespace

import numpy as np
import pytest

from gofa_f.robust_wind_models import RECIPES,arrays,features,fresh_confirmation


def fixture():
    v=np.arange(1000*3,dtype=np.float32).reshape(1000,3)/10-30
    external=np.ones((3,3))-np.eye(3)
    return SimpleNamespace(values=v,mask=np.ones_like(v,bool),times=np.arange(1000,dtype=np.int64)*600*10**9,
        segments=np.zeros(1000),external=external,ids=['A','B','C'],
        prefix_counts=np.vstack([np.zeros((1,3),np.int64),np.ones_like(v,dtype=np.int64).cumsum(0)]))


def test_all_wind_features_causal_and_graph_not_future():
    d=fixture()
    q=dict(source='sdwpf',variable=0,origin=100,length=48,horizon=24)
    changed=copy.deepcopy(d)
    changed.values[100:]=1e8
    changed.mask[100:]=False
    changed.times[100:]=0
    assert len(RECIPES)==8
    for recipe in RECIPES:
        a,b,s=features(d,q,recipe)
        x,y,t=features(changed,q,recipe)
        np.testing.assert_array_equal(a,x)
        np.testing.assert_array_equal(b,y)
        assert s==t
    for recipe in ['raw_hgb_neighbor','norm_hgb_neighbor']:
        d.mask[:100,1:]=False
        x,_,_=features(d,q,recipe)
        assert np.all(x[:,-11:]==0)


def test_normalized_weights_align_raw_unit_absolute_loss():
    d=fixture()
    q=dict(source='sdwpf',variable=0,origin=100,length=48,horizon=24)
    other=dict(q,variable=1,origin=200)
    d.values[:200,1]*=10
    _,_,w=arrays(d,[q,other],'norm_hgb_local')
    s1=features(d,q,'norm_hgb_local')[2]
    s2=features(d,other,'norm_hgb_local')[2]
    np.testing.assert_allclose(w[:24]/s1,w[24:]/s2,rtol=1e-6,atol=1e-6)
    assert np.isclose(w.mean(),1.)
    with pytest.raises(ValueError,match='First60'):
        arrays(d,[dict(q,origin=590)],'raw_hgb_local')


def test_fresh_confirmation_same_id_and_disjoint_or_fail():
    d=fixture()
    q=dict(source='sdwpf',variable=0,origin=770,length=48,horizon=24,chunk=0)
    m=dict(queries={'sdwpf':[dict(q,origin=620) for _ in range(64)]+[q]*64},
        development_queries={'sdwpf':[]})
    with pytest.raises(ValueError,match='unseen disjoint'):
        fresh_confirmation(d,m)
