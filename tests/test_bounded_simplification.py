import copy

import numpy as np
import pytest

from gofa_f.bounded_simplification import (convert_simple, eligible, frames, model_predictions,
    key, advance, load_coefficients)
from gofa_f.history_calibration_math import predict
from gofa_f.supervised_ensemble_math import prediction


def test_simple_weights_exactly_embed_history_family_without_using_history():
    f=dict(names=['native','gofa','persistence','mean','trend','cpu'],
        experts=np.arange(24*6,dtype=np.float64).reshape(24,6),scale=999.,u=np.linspace(0,1,24))
    for weights in [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[.2,.3,.5]]:
        p=convert_simple(dict(weights=weights),f['names'])
        expected=prediction(f['experts'][:,0],f['experts'][:,1],f['experts'][:,-1],weights)
        np.testing.assert_allclose(predict(f,p),expected,rtol=1e-6,atol=1e-6)
        changed=copy.deepcopy(f)
        changed['experts'][:,2:5]=1e12
        np.testing.assert_array_equal(predict(f,p),predict(changed,p))


def test_history_only_has_no_fake_cpu_model():
    assert model_predictions({},None,'pems03',None,[],None) is None
    assert callable(frames)
    rows=[dict(mae=1.,native_mae=1.),dict(mae=.9,native_mae=1.)]
    assert eligible(rows,.95,1.)
    assert not eligible([dict(mae=1.00000000001,native_mae=1.)],.9,1.)
    simple=dict(family='simple',recipe='raw150',tau=0.,worst_fold_ratio=.9,pooled_mae=1.)
    history=dict(simple,family='history_only',recipe=None,tau=.1)
    assert key(simple)<key(history)


def test_unsealed_confirmation_and_test_refused(tmp_path,monkeypatch):
    import gofa_f.bounded_simplification as module
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(module,'manifest',lambda *_:{})
    with pytest.raises(ValueError,match='Sealed'):
        load_coefficients({},tmp_path)
    with pytest.raises(ValueError,match='Sealed'):
        advance({},tmp_path)
