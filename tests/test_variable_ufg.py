import os
from pathlib import Path
import pytest
import torch

from gofa_f.variable_ufg import VariableRosterUFG,temporal_weight,resize_filter,resize_complex
from gofa_f.ufg_subgraph import pack_history
from gofa_f.ufg import active_name


def test_temporal_basis_preserves_constant_and_canonical_exact():
    torch.manual_seed(42);w=torch.randn(9,48)
    assert temporal_weight(w,48) is w
    for l in (10,12,14,49,96,192,288):
        torch.testing.assert_close(temporal_weight(w,l).sum(-1),w.sum(-1),atol=1e-5,rtol=1e-6)


def model():
    root=Path(os.environ['GOFA_WORKSPACE'])/'external/ufgtime'
    if not (root/'src/model.py').exists():pytest.skip('Pinned official source unavailable')
    torch.manual_seed(42);return VariableRosterUFG(root)


def test_canonical_actual_official_forward_and_gradients():
    m=model();x=torch.randn(1,48,31);mask=torch.ones_like(x,dtype=torch.bool)
    a=m.normalized_forecast(x,mask);b=m.core(x,mask)
    torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
    params=[p for p in m.parameters() if p.requires_grad]
    ga=torch.autograd.grad(a.sum(),params);gb=torch.autograd.grad(b.sum(),params)
    for l,r in zip(ga,gb):torch.testing.assert_close(l,r,atol=1e-6,rtol=1e-6)


@pytest.mark.parametrize('length,horizon',[(10,1),(97,25),(288,144)])
def test_variable_ufg_same_parameters_full_history_and_gradients(length,horizon):
    m=model();before=[(k,tuple(p.shape)) for k,p in m.named_parameters()]
    x=torch.randn(1,length,7);mask=torch.rand(x.shape)>.1;mask[:,-1]=True
    packed=pack_history(x,mask,maximum_nodes=31)
    y=m(packed,horizon,all_nodes=True);assert y.shape==(1,horizon,7)
    y.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
    assert before==[(k,tuple(p.shape)) for k,p in m.named_parameters()]
    torch.testing.assert_close(m(packed,1),m(packed,144)[:,:1],rtol=1e-6,atol=1e-6)


def test_variable_ufg_batch_query_isolation():
    m=model().eval();x=torch.randn(2,97,7);mask=torch.ones_like(x,dtype=torch.bool)
    a=m(pack_history(x[:1],mask[:1],maximum_nodes=31),25)
    b=m(pack_history(x,mask,maximum_nodes=31),25)
    x[1]*=100
    c=m(pack_history(x,mask,maximum_nodes=31),25)
    torch.testing.assert_close(a,b[:1],rtol=1e-6,atol=1e-6)
    torch.testing.assert_close(a,c[:1],rtol=1e-6,atol=1e-6)
