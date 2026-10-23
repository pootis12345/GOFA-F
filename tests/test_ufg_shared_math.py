import torch
import pytest

from gofa_f.ufg_shared import history_weight,frequency_filter,spectral_linear


def test_history_basis_preserves_all_parameters_and_constant_response():
    w=torch.randn(8,48,requires_grad=True)
    assert history_weight(w,48) is w
    actual=history_weight(w,12)
    torch.testing.assert_close(actual.sum(-1),w.sum(-1),rtol=1e-6,atol=1e-6)
    actual.sum().backward();assert torch.equal(w.grad,torch.ones_like(w))
    with pytest.raises(ValueError):history_weight(w,24)


def test_frequency_gain_resampling_preserves_constant_filter_and_endpoints():
    w=torch.ones(31*25,1,requires_grad=True)
    assert frequency_filter(w,25) is w
    short=frequency_filter(w,9)
    assert short.shape==(31*9,1) and torch.equal(short,torch.ones_like(short))
    short.sum().backward();assert torch.isfinite(w.grad).all()


def test_complex_projection_resize_keeps_channels_separate_and_gradients_finite():
    width=3;w=torch.zeros(25,width,25,dtype=torch.complex64)
    for channel in range(width):w[:,channel,:]=complex(channel+1,-channel)
    w=w.reshape(25,width*25).requires_grad_();b=torch.ones(25,dtype=torch.complex64,requires_grad=True)
    cw,cb=spectral_linear(w,b,25,width)
    assert cw is w and cb is b
    sw,sb=spectral_linear(w,b,9,width)
    for channel in range(width):
        expected=torch.full((9,9),complex(channel+1,-channel)*25/9)
        torch.testing.assert_close(sw.reshape(9,width,9)[:,channel],expected,rtol=1e-6,atol=1e-6)
    (sw.abs().sum()+sb.abs().sum()).backward()
    assert w.grad.isfinite().all() and b.grad.isfinite().all()
    with pytest.raises(ValueError):spectral_linear(w,b,13,width)
