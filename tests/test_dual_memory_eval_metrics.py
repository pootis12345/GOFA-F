import numpy as np
import pytest

from gofa_f.dual_memory_eval_metrics import (macro_metrics,strict_improvement,
    spatial_blocks,paired_block_intervals,compare_dataset)


def test_variable_equal_mae_missing_zero_mape_and_zero_wape_denominator():
    p=np.array([[10.,0.],[0.,0.],[2.,2.]])
    y=np.zeros_like(p);mask=np.array([[True,True],[True,False],[True,True]])
    value=macro_metrics(p,y,mask,[0,0,1])
    assert value['macro']['mae']==pytest.approx((10/3+2)/2)
    assert value['macro']['mape'] is None and value['macro']['wape'] is None
    assert value['mape_valid_fraction']==0 and value['observed_points']==5
    with pytest.raises(ValueError,match='finite'):macro_metrics(p*np.nan,y,mask,[0,0,1])


def test_strict_goal_tolerance_and_missing_baseline_rejection():
    assert not strict_improvement(10-1e-9,10)
    assert strict_improvement(10-2e-8,10)
    assert not strict_improvement(0,0)
    records=[dict(query={'variable':0},target=[1.],observed=[True],origin_time_ns=1)]
    with pytest.raises(ValueError,match='four model'):compare_dataset(records,{'native':[[1.]]},{0:0})


def test_spatial_blocks_are_value_independent_and_leave_message_direction_unchanged():
    a=np.zeros((40,40));a[np.arange(39),np.arange(1,40)]=1;before=a.copy()
    groups=spatial_blocks(a,[str(i) for i in range(40)],maximum=8)
    assert len(groups)==40 and max(list(groups.values()).count(v) for v in set(groups.values()))<=8
    assert np.array_equal(a,before) and groups==spatial_blocks(a,[str(i) for i in range(40)],maximum=8)


def test_paired_space_time_intervals_reproducible_and_single_spatial_cluster_disclosed():
    y=np.ones((12,2));mask=np.ones_like(y,bool)
    predictions={'native':y+2,'ufg':y+1,'enhanced':y+.7,'fusion':y+.5}
    variables=np.repeat(np.arange(4),3);space=variables//2;clock=np.tile(np.arange(3),4)
    a=paired_block_intervals(predictions,y,mask,variables,space,clock,repetitions=200,seed=42)
    b=paired_block_intervals(predictions,y,mask,variables,space,clock,repetitions=200,seed=42)
    assert a==b and a['intervals']['space_time']['native']['mae_difference_95'][1]<0
    single=paired_block_intervals(predictions,y,mask,variables,np.zeros(12,int),clock,repetitions=200)
    assert single['insufficient_spatial_clusters'] and single['intervals']['space_time'] is None
    assert single['intervals']['time_only'] is not None
