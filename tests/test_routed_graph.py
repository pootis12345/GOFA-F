import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gofa_f.graph_context import ContextGraphExchange
from gofa_f.native import NativeBolt
from gofa_f.native_stage import Data, fit, seed, trained_state
from gofa_f.routed_graph import PriorRoutedGraphExchange
from gofa_f.stage_budget import Budget
from test_native import random_bolt


def sample():
    torch.manual_seed(9)
    return (torch.randn(2, 4, 3, 8), torch.ones(2, 4, 3, dtype=torch.bool),
            torch.tensor([[1., 2., 0., 0.], [2., 1., 1., 0.],
                          [0., 1., 1., 3.], [0., 0., 3., 1.]]), torch.randn(2, 4, 5))


def activated():
    module = PriorRoutedGraphExchange(8, 4, 5, attention_rank=2)
    with torch.no_grad():
        module.gate.fill_(.2)
        module.key.weight.normal_(0, .3)
        module.router.weight.normal_(0, .3)
        module.context_projection.weight.normal_(0, .1)
    return module


def projected(module, hidden, mask, features):
    return (module.down(module.norm(hidden)) + module.context_projection(features)[:, :, None]).masked_fill(
        ~mask[..., None], 0)


def test_paired_initialization_and_native_identity():
    hidden, mask, graph, features = sample()
    torch.manual_seed(42)
    old = ContextGraphExchange(8, 4, 5)
    rng = torch.get_rng_state()
    torch.manual_seed(42)
    new = PriorRoutedGraphExchange(8, 4, 5, 2)
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in old.state_dict().items():
        torch.testing.assert_close(new.state_dict()[name], value, rtol=0, atol=0)
    torch.testing.assert_close(new(hidden, graph, mask, features), hidden, rtol=0, atol=0)
    # Zero routing additions also recover a nonzero legacy graph to roundoff.
    old.gate.data.fill_(.2)
    new.gate.data.fill_(.2)
    torch.testing.assert_close(new(hidden, graph, mask, features), old(hidden, graph, mask, features))


def test_attention_support_normalization_and_prior_floor():
    hidden, mask, graph, features = sample()
    module = activated()
    mask[0, 1] = False
    attention, fraction, amplitude = module.routing(projected(module, hidden, mask, features), graph, mask)
    assert attention[:, graph == 0].eq(0).all()
    assert attention[0, :, 1].eq(0).all()
    torch.testing.assert_close(attention.sum(-1), torch.ones(2, 4))
    assert fraction.min() >= 0 and fraction.max() <= .5
    assert not torch.equal(fraction[:, :1], fraction[:, 1:2])
    assert not torch.equal(amplitude[:, :1], amplitude[:, 1:2])
    changed = hidden.clone()
    changed[0, 2] += torch.randn_like(changed[0, 2])
    after = module.routing(projected(module, changed, mask, features), graph, mask)[0]
    assert (after[0] - attention[0]).abs().max() > 1e-6
    torch.testing.assert_close(after[1], attention[1], rtol=0, atol=0)


def test_missing_sources_empty_neighborhoods_and_invalid_tokens():
    hidden, mask, graph, features = sample()
    module = activated()
    mask[0] = False
    mask[1, 1] = False
    graph[3] = 0
    output = module(hidden, graph, mask, features)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output[~mask], hidden[~mask], rtol=0, atol=0)
    torch.testing.assert_close(output[:, 3], hidden[:, 3], rtol=0, atol=0)
    altered = hidden.clone()
    altered[~mask] = float("nan")
    # Arbitrary missing-token payload cannot influence an observed target.
    changed = module(altered, graph, mask, features)
    torch.testing.assert_close(changed[mask], output[mask])
    changed[mask].square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None)


def test_permutation_equivariance_and_batch_isolation():
    hidden, mask, graph, features = sample()
    module = activated()
    expected = module(hidden, graph, mask, features)
    order = torch.tensor([2, 0, 3, 1])
    actual = module(hidden[:, order], graph[order][:, order], mask[:, order], features[:, order])
    torch.testing.assert_close(actual, expected[:, order])
    hidden[1] *= 10
    features[1] += 30
    torch.testing.assert_close(module(hidden, graph, mask, features)[0], expected[0], rtol=0, atol=0)


def test_self_graph_has_no_cross_node_path():
    hidden, mask, _, features = sample()
    graph, module = torch.eye(4), activated()
    expected = module(hidden, graph, mask, features)
    hidden[:, 1] += torch.randn_like(hidden[:, 1])
    features[:, 1] += 10
    actual = module(hidden, graph, mask, features)
    torch.testing.assert_close(actual[:, [0, 2, 3]], expected[:, [0, 2, 3]], rtol=0, atol=0)


def test_router_and_attention_receive_gradients_after_warm_start():
    hidden, mask, graph, features = sample()
    module = activated()
    module(hidden, graph, mask, features).square().mean().backward()
    for name in ("query.weight", "key.weight", "router.weight", "context_projection.weight"):
        gradient = dict(module.named_parameters())[name].grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().max() > 0


def test_cpu_bf16_autocast_keeps_routing_finite_and_output_dtype():
    hidden, mask, graph, features = sample()
    module = activated()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = module(hidden.bfloat16(), graph, mask, features)
    assert result.dtype == torch.bfloat16 and torch.isfinite(result).all()
    result.float().square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None)


@pytest.mark.parametrize("rank,floor", [(0, .5), (5, .5), (2, .49), (2, 1.), (2, float("nan"))])
def test_unsafe_routing_configuration_is_rejected(rank, floor):
    with pytest.raises(ValueError):
        PriorRoutedGraphExchange(8, 4, 5, rank, floor)


@pytest.mark.parametrize("length", [8, 12, 24])
def test_native_frozen_backbone_cache_and_common_initialization(length):
    native = random_bolt()
    torch.manual_seed(42)
    static = NativeBolt(native, np.ones((3, 3)), 3, (1, 2), rank=4,
                        graph_feature_dim=5, output_adapter_rank=4)
    torch.manual_seed(42)
    model = NativeBolt(native, np.ones((3, 3)), 3, (1, 2), rank=4, graph_feature_dim=5,
                       output_adapter_rank=4, graph_exchange="prior_routed", graph_attention_rank=2)
    for name, value in static.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    x, mask, features = torch.randn(2, length, 3), torch.ones(2, length, 3, dtype=torch.bool), torch.randn(2, 3, 5)
    torch.testing.assert_close(model(x, mask, features), static(x, mask, features), rtol=0, atol=0)
    for module in model.graph.values():
        module.gate.data.fill_(.1)
        module.key.weight.data.normal_(0, .1)
        module.router.weight.data.normal_(0, .1)
    outputs, gradients = [], []
    for cached in (False, True):
        model.zero_grad(set_to_none=True)
        output = model.from_prefix(model.prefix(x, mask, 2), 2, graph_features=features) if cached else model(
            x, mask, features)
        output.square().mean().backward()
        outputs.append(output.detach())
        gradients.append({n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad})
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    for name in gradients[0]:
        torch.testing.assert_close(gradients[0][name], gradients[1][name], rtol=0, atol=0)
    assert all(p.grad is None for p in native.parameters())
    # Node count is not encoded in trainable tensors.
    smaller = NativeBolt(copy.deepcopy(native), np.eye(2), 3, (1, 2), rank=4, graph_feature_dim=5,
                         output_adapter_rank=4, graph_exchange="prior_routed", graph_attention_rank=2)
    assert {n: p.shape for n, p in smaller.named_parameters()} == {
        n: p.shape for n, p in model.named_parameters()}


@pytest.mark.parametrize("layers,feature_dim,adjacency", [
    ((0, 2), 5, np.eye(3)), ((1, 2), 0, np.eye(3)), ((1, 2), 5, -np.eye(3)),
    ((1, 2), 5, np.full((3, 3), np.nan)),
])
def test_native_rejects_wrong_layout_context_or_prior(layers, feature_dim, adjacency):
    with pytest.raises(ValueError):
        NativeBolt(random_bolt(), adjacency, 3, layers, rank=4, graph_feature_dim=feature_dim,
                   graph_exchange="prior_routed", graph_attention_rank=2)


def test_optimizer_resume_preserves_router_and_rejects_changed_prior_floor(tmp_path):
    values = np.random.default_rng(2).normal(size=(160, 3)).astype(np.float32)
    path = tmp_path / "toy.npz"
    np.savez(path, values=values, mask=np.ones_like(values, bool))
    data = Data({"data": str(path), "context": 12, "horizons": [3],
                 "graph_context": {"enabled": True, "feature_mode": "values_only"}}, "cpu")
    native = random_bolt()

    def build(floor=.5):
        seed(42)
        return NativeBolt(copy.deepcopy(native), np.ones((3, 3)), 3, (1, 2), rank=4,
                          graph_feature_dim=5, output_adapter_rank=4, graph_exchange="prior_routed",
                          graph_attention_rank=2, graph_prior_floor=floor)

    budget = Budget(tmp_path / "budget.json", 100)
    budget.activate("test", 100)
    gpu = SimpleNamespace(micro=64, check=lambda b: b.check())
    model = build()
    fit(model, data, tmp_path / "resume", gpu, budget, .001, max_updates=2, adapter_lr=.0001)
    model = build()  # Reconstruct, then load optimizer/model/RNG/sampling cursor.
    fit(model, data, tmp_path / "resume", gpu, budget, .001, max_updates=3, adapter_lr=.0001)
    continuous = build()
    fit(continuous, data, tmp_path / "continuous", gpu, budget, .001, max_updates=3, adapter_lr=.0001)
    for name, value in trained_state(continuous).items():
        torch.testing.assert_close(value, trained_state(model)[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="protocol"):
        fit(build(.75), data, tmp_path / "resume", gpu, budget, .001, max_updates=4, adapter_lr=.0001)
