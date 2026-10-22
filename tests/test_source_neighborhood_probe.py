import copy
from pathlib import Path

import numpy as np
import pytest

from gofa_f.graph_probe_data import controls,pack
from gofa_f.source_neighborhood_probe import configuration,neighborhood_pack,confirm
from test_source_graph_probe import fixture


def query(source='pems04_flow',origin=100):
    return dict(query=dict(source=source,variable=2,origin=origin,length=12,horizon=12),block=0)


def test_matched_rosters_target_scaling_labels_and_complete_support():
    d=fixture();allowed=list(range(8))
    a=neighborhood_pack(d,query(),allowed)
    b=neighborhood_pack(d,query(),allowed,random=True)
    for k in ('low','scale','persistence','y','raw_target','observed'):
        np.testing.assert_array_equal(a[k],b[k])
    np.testing.assert_array_equal(a['x'][0],b['x'][0])
    assert a['nodes'][0]==b['nodes'][0]==2
    assert (a['nodes']>=0).sum()==(b['nodes']>=0).sum()
    assert a['adjacency'].sum()==8*7 and not np.diag(a['adjacency']).any()
    # Legacy 'rewired' alias must leave dense support exactly invariant.
    np.testing.assert_array_equal(controls(b['adjacency'][None],b['nodes'][None],[query()],'rewired')[0],b['adjacency'])


def test_random_roster_deterministic_cross_view_date_and_partition_limited():
    d=fixture();d.external[:]=False;d.external[2,1]=True
    a=neighborhood_pack(d,query(),[0,1,2,3,4],random=True)
    b=neighborhood_pack(d,query('pems04_speed',120),[0,1,2,3,4],random=True)
    np.testing.assert_array_equal(a['nodes'],b['nodes'])
    assert (a['nodes']>=0).sum()==2 and set(a['nodes'])-{-1}<={0,1,2,3,4}
    d.values[:,5:]=1e9;d.mask[:,5:]=False
    c=neighborhood_pack(d,query(),[0,1,2,3,4],random=True)
    for k in a:np.testing.assert_array_equal(a[k],c[k])


@pytest.mark.parametrize('random',[False,True])
def test_future_and_auxiliary_future_independence_and_confirmation_no_labels(random):
    d=fixture();a=neighborhood_pack(d,query(),list(range(8)),random=random,labels=False)
    d.values[100:]=1e9;d.mask[100:]=False
    b=neighborhood_pack(d,query(),list(range(8)),random=random,labels=False)
    for k in a:np.testing.assert_array_equal(a[k],b[k])
    d=fixture()
    c=neighborhood_pack(d,query(origin=260),list(range(8)),random=random,labels=False,split='confirmation')
    assert not {'y','observed','raw_target'}&set(c)
    with pytest.raises(ValueError,match='Confirmation'):
        neighborhood_pack(d,query(origin=260),list(range(8)),random=random,split='confirmation')


def test_explicit_roster_refuses_leak_duplicate_or_wrong_target():
    d=fixture()
    for indices in ([1,2],[2,2],[2,7],[]):
        with pytest.raises(ValueError,match='roster'):pack(d,query(),[0,1,2,3],indices=indices)


def test_fixed_scope_and_confirmation_access_gate(tmp_path,monkeypatch):
    monkeypatch.setenv('GOFA_WORKSPACE','/tmp/source')
    cfg=configuration(Path(__file__).parents[1]/'configs/stage38_source_neighborhood_probe.yaml')
    assert cfg['seeds']==[42,43,44] and not cfg['policy']['gpu_enabled']
    with pytest.raises(ValueError,match='Repeated'):
        confirm(cfg,tmp_path,{},None,None,lambda:None)
