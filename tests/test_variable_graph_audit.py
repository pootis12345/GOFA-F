import json
import math

import numpy as np
import pytest

from gofa_f.dual_memory_eval_metrics import macro_metrics
from gofa_f.generative_data import sha
from gofa_f.variable_graph_audit import scalar_metrics,compare_metrics,check_shards,close,audit


def test_scalar_replay_matches_variable_equal_metrics_and_missing_rules():
    prediction=[[2.,4.,8.],[0.,3.,9.],[-1.,5.,10.]]
    target=[[1.,float('nan'),0.],[0.,2.,8.],[-2.,4.,9.]]
    mask=[[True,False,True],[True,True,False],[True,True,True]];variables=[0,0,1]
    replay=scalar_metrics(list(zip(variables,prediction,target,mask)))
    compare_metrics(replay,macro_metrics(prediction,target,mask,variables))
    assert replay['macro']['mae']!=pytest.approx(np.mean([1,8,0,1,1,1,1]))
    zero=scalar_metrics([(0,[0.,0.],[0.,0.],[True,True])])
    assert zero['macro']['mape'] is None and zero['macro']['wape'] is None
    with pytest.raises(ValueError):close(None,0.)


@pytest.mark.parametrize('record',[(0,[float('nan')],[0.],[True]),(0,[1.],[0.],[False]),(0,[1.,2.],[0.],[True])])
def test_invalid_records_refused(record):
    with pytest.raises(ValueError):scalar_metrics([record])


def test_completed_shard_requires_exact_coverage_and_rejects_tampering(tmp_path):
    path=tmp_path/'a.json';path.write_text('{"prediction":[1]}')
    seal=dict(complete=True,shards={'a.json':sha(path)})
    check_shards(tmp_path,seal,{'a.json'})
    with pytest.raises(ValueError):check_shards(tmp_path,seal,{'a.json','missing.json'})
    path.write_text('{"prediction":[2]}')
    with pytest.raises(ValueError):check_shards(tmp_path,seal,{'a.json'})
    outside=tmp_path.parent/(tmp_path.name+'-outside.json');outside.write_text('{}')
    with pytest.raises(ValueError):check_shards(tmp_path,dict(complete=True,shards={'../'+outside.name:sha(outside)}),{'../'+outside.name})


def test_incomplete_training_cannot_be_marked_success_or_open_target_truth(tmp_path):
    source=tmp_path/'source';source.mkdir()
    for name in ['manifest.json','target_plan.json']:(source/name).write_text('{}')
    result=audit(source,tmp_path/'never_open_target')
    assert result['state']=='incomplete' and result['goal_achieved'] is False
    assert result['target_results_complete'] is False


def test_changed_metric_or_coverage_is_rejected():
    r=scalar_metrics([(0,[1.,2.],[0.,1.],[True,True])]);saved=json.loads(json.dumps(r))
    saved['macro']['mae']+=.01
    with pytest.raises(ValueError):compare_metrics(r,saved)
    saved=json.loads(json.dumps(r));saved['per_step_macro_mae'].pop('2')
    with pytest.raises(ValueError):compare_metrics(r,saved)
