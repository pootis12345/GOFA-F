import copy
from types import SimpleNamespace

import numpy as np
import pytest

from gofa_f.autonomous_iteration_models import RECIPES, arrays, design, estimator, predict
from gofa_f.autonomous_iteration import DEADLINE, candidate_key, check_time, eligibility


def fixture():
    values=np.arange(600*2,dtype=np.float32).reshape(600,2)/10-20
    data=SimpleNamespace(values=values,mask=np.ones_like(values,bool),
        times=np.arange(600,dtype=np.int64)*3600*10**9,segments=np.zeros(600))
    q=dict(source='engrad',variable=0,origin=100,length=48,horizon=24)
    return data,q


def test_causal_features_all_recipes_and_past_day_alignment():
    data,q=fixture()
    changed=copy.deepcopy(data)
    changed.values[100:]=999999
    changed.mask[100:]=False
    changed.times[100:]=0
    for recipe in RECIPES:
        a,b,s=design(data,q,recipe)
        c,d,t=design(changed,q,recipe)
        np.testing.assert_array_equal(a,c)
        np.testing.assert_array_equal(b,d)
        assert s==t
        if 'season' in recipe:
            np.testing.assert_array_equal(b,data.values[76:100,0])


def test_constant_missing_negative_and_normalization():
    data,q=fixture()
    data.values[:100,0]=-3
    data.mask[99,0]=False
    for recipe in RECIPES:
        x,b,s=design(data,q,recipe)
        assert s==1 and not np.isinf(x).any()
        assert np.all(b==-3)
        if recipe.startswith('norm'):
            assert np.nanmax(np.abs(x[:,:48]))==0
    data.mask[:100,0]=False
    with pytest.raises(ValueError,match='historical observation'):
        design(data,q,'norm150')


def test_training_boundary_and_variable_weights():
    data,q=fixture()
    qs=[q,dict(q,variable=1,origin=148)]
    x,y,w=arrays(data,qs,'norm150')
    assert len(x)==48 and w[:24].sum()==w[24:].sum()
    with pytest.raises(ValueError,match='boundary'):
        arrays(data,[dict(q,origin=350)],'norm150')


def test_fixed_candidates_predict_serialized_and_selection(tmp_path):
    import joblib
    data,q=fixture()
    x,y,w=arrays(data,[q,dict(q,origin=148)],'norm150')
    m=estimator('norm150')
    m.set_params(max_iter=3,min_samples_leaf=2)
    m.fit(x,y,sample_weight=w)
    a=predict(m,data,[q],'norm150')
    joblib.dump(m,tmp_path/'m.joblib')
    np.testing.assert_allclose(a,predict(joblib.load(tmp_path/'m.joblib'),data,[q],'norm150'),rtol=1e-6,atol=1e-6)
    assert len(RECIPES)==8 and estimator('normseason300').max_iter==300
    assert estimator('normextra64').n_jobs==2
    rows=[dict(mae=9.,native_mae=10.),dict(mae=10.,native_mae=10.)]
    assert eligibility(rows,9.5,10.)
    assert not eligibility([dict(mae=10.00001,native_mae=10.)],9.,10.)
    a=dict(recipe='raw150',tau=.01,worst_fold_ratio=.9,pooled_mae=8.)
    assert candidate_key(a)<candidate_key(dict(a,recipe='norm150'))


def test_deadline_is_fixed(monkeypatch):
    import gofa_f.autonomous_iteration as module
    monkeypatch.setattr(module.time,'time',lambda:DEADLINE-59)
    with pytest.raises(TimeoutError):
        check_time()
    assert DEADLINE==1792638912.


def test_cpu_only_parent_does_not_require_gpu_launcher(tmp_path,monkeypatch):
    from gofa_f.autonomous_iteration import exited_parent
    from gofa_f.prefix_cache import atomic_json
    import gofa_f.idle_launcher as launcher
    atomic_json(tmp_path/'status.json',dict(state='complete'))
    atomic_json(tmp_path/'activity.json',dict(worker=None))
    atomic_json(tmp_path/'cpu_activity.json',dict(active=None))
    atomic_json(tmp_path.with_suffix('.bootstrap.launch.json'),dict(pid=999,start_ticks='42'))
    monkeypatch.setattr(launcher,'process_identity',lambda _:None)
    exited_parent(tmp_path)
    monkeypatch.setattr(launcher,'process_identity',lambda _:'42')
    with pytest.raises(ValueError,match='still alive'):
        exited_parent(tmp_path)
