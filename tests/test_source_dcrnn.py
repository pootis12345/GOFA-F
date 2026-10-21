import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gofa_f.generative_graph import origin_graph
from gofa_f.source_dcrnn_data import sample
from gofa_f.source_dcrnn_model import DCRNN, Diffusion, masked_query_loss


def fixture():
    values = np.sin(np.arange(200)[:, None]/10+np.arange(3)[None]).astype(np.float32)
    d = SimpleNamespace(values=values, mask=np.ones_like(values, bool),
        segments=np.zeros(200), external=np.ones((3, 3))-np.eye(3), ids=['A', 'B', 'C'])
    d.graph = lambda q: origin_graph(d.values[q.origin-q.length:q.origin],
        d.mask[q.origin-q.length:q.origin], q.variable, d.external, d.ids)
    return d


def test_history_supports_and_scale_ignore_every_future_value():
    d = fixture()
    q = dict(source='pems04_flow', variable=0, origin=100, length=12, horizon=12)
    a = sample(d, q)
    d.values[100:] = 1e6
    d.mask[100:] = False
    b = sample(d, q)
    assert a.keys() == b.keys()
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    assert a['present'].sum() == 3
    assert np.all(a['supports'][:, 3:] == 0)
    assert np.all(a['supports'][:, :, 3:] == 0)


def test_source_split_labels_and_causal_missing_teacher_fill():
    d = fixture()
    q = dict(source='pems04_flow', variable=0, origin=100, length=12, horizon=12)
    d.mask[100:103, 0] = False
    row = sample(d, q, True, 'train')
    last = row['x'][-1, 0, 0]
    np.testing.assert_array_equal(row['teacher'][:3, 0], np.full(3, last))
    assert not row['observed'][:3, 0].any()
    with pytest.raises(ValueError, match='split'):
        sample(d, dict(q, origin=115), True, 'train')


def test_diffusion_direction_and_chebyshev_order():
    x = torch.tensor([[[1.], [2.], [4.]]])
    a = torch.tensor([[[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]])
    layer = Diffusion(1, 5)
    with torch.no_grad():
        layer.linear.weight.copy_(torch.eye(5))
        layer.linear.bias.zero_()
    result = layer(x, torch.stack((a, a.transpose(1, 2)), 1))
    np.testing.assert_allclose(result.detach().numpy()[:, :, 1:2], (a@x).numpy())
    np.testing.assert_allclose(result.detach().numpy()[:, :, 2:3], (2*a@(a@x)-x).numpy())


def test_variable_nodes_permutation_and_batch_isolation():
    torch.manual_seed(42)
    model = DCRNN(hidden=8)
    x = torch.randn(2, 4, 3, 2)
    supports = torch.eye(3).expand(2, 2, 3, 3).clone()
    present = torch.ones(2, 3)
    result = model(x, supports, present, 3)
    perm = torch.tensor([2, 0, 1])
    other = model(x[:, :, perm], supports[:, :, perm][:, :, :, perm], present[:, perm], 3)
    torch.testing.assert_close(result[:, :, perm], other, rtol=1e-6, atol=1e-6)
    single = model(x[:1], supports[:1], present[:1], 3)
    torch.testing.assert_close(result[:1], single, rtol=1e-6, atol=1e-6)
    zero = model(torch.zeros(1, 4, 9, 2), torch.zeros(1, 2, 9, 9), torch.zeros(1, 9), 3)
    assert torch.equal(zero, torch.zeros_like(zero))


def test_teacher_label_shift_and_no_inference_rng_consumption():
    torch.manual_seed(42)
    model = DCRNN(hidden=8)
    x = torch.randn(1, 4, 3, 2)
    supports = torch.eye(3).expand(1, 2, 3, 3)
    present = torch.ones(1, 3)
    teacher = torch.randn(1, 3, 3)
    a = model(x, supports, present, 3, teacher, 1., torch.Generator().manual_seed(42))
    b = model(x, supports, present, 3, teacher+1, 1., torch.Generator().manual_seed(42))
    torch.testing.assert_close(a[:, 0], b[:, 0], rtol=0, atol=0)
    assert not torch.equal(a[:, 1], b[:, 1])
    before = torch.get_rng_state().clone()
    model(x, supports, present, 3)
    assert torch.equal(before, torch.get_rng_state())


def test_optimizer_and_rng_replay_is_exact_and_all_gradients_finite():
    torch.manual_seed(42)
    model = DCRNN(hidden=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    rng = torch.Generator().manual_seed(42)
    x = torch.randn(2, 4, 3, 2)
    supports = torch.eye(3).expand(2, 2, 3, 3)
    present = torch.ones(2, 3)
    truth = torch.randn(2, 3, 3)
    mask = torch.ones_like(truth, dtype=torch.bool)
    saved = copy.deepcopy((model.state_dict(), optimizer.state_dict(), rng.get_state()))
    def step():
        optimizer.zero_grad(set_to_none=True)
        output = model(x, supports, present, 3, truth, .5, rng)
        loss = masked_query_loss(output, truth, mask)
        loss.backward()
        assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())
        optimizer.step()
        return {k:v.clone() for k,v in model.state_dict().items()}
    a = step()
    model.load_state_dict(saved[0])
    optimizer.load_state_dict(saved[1])
    rng.set_state(saved[2])
    b = step()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_target_access_requires_locked_source_checkpoint(tmp_path):
    from gofa_f.source_dcrnn import target_evaluation
    with pytest.raises(ValueError, match='source-selected checkpoint'):
        target_evaluation({}, tmp_path, {})
