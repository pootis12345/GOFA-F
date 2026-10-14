import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gofa_f.data import file_sha256
from gofa_f.graph_context import FEATURE_DIM, ContextGraphExchange, clock_features, load_calendar, node_features
from gofa_f.native import HSDGNNAdapter, NativeBolt
from gofa_f.models import GraphExchange
from gofa_f.native_stage import Data, context_inputs, dataset_settings, stage_config
from test_native import random_bolt


def features(x, mask, clock=None):
    if clock is None:
        clock = torch.zeros(len(x), 6)
    return node_features(x, mask, torch.zeros(x.shape[-1]), torch.ones(x.shape[-1]), 0., 10., clock)


def test_statistics_align_physical_units_across_different_node_scalers():
    raw = torch.tensor([[[30., 30.], [40., 40.], [50., 50.]]])
    mean, std = torch.tensor([10., 40.]), torch.tensor([2., 20.])
    result = node_features((raw-mean)/std, torch.ones_like(raw, dtype=torch.bool), mean, std,
                           30., 10., torch.zeros(1, 6))
    torch.testing.assert_close(result[:, 0], result[:, 1])
    assert result[0, 0, 0] == 1  # Mean speed 40, centered at pooled training mean 30.
    assert result[0, 0, 2] == 2  # Last speed 50, in pooled training standard deviations.


def test_missing_values_are_excluded_and_unobserved_nodes_are_finite():
    x = torch.tensor([[[2., 999.], [999., 999.], [4., 999.]]])
    mask = torch.tensor([[[True, False], [False, False], [True, False]]])
    value = features(x, mask)
    assert value.shape == (1, 2, FEATURE_DIM)
    assert torch.isfinite(value).all()
    assert value[0, 1, :5].eq(0).all()
    torch.testing.assert_close(value[0, 0, 0], torch.tensor(.3))


def test_clock_availability_and_weekend_identity_are_explicit():
    calendar = torch.tensor([[.5, 0.], [.5, 6.], [.9, 2.]])
    known = torch.tensor([[True, True], [True, True], [False, False]])
    result = clock_features(calendar, known)
    torch.testing.assert_close(result[0, :2], result[1, :2])
    assert not torch.equal(result[0, 2:4], result[1, 2:4])
    assert result[2].eq(0).all()  # Unknown is not silently presented as midnight/Monday.


def test_target_change_does_not_change_context_or_training_statistics(tmp_path):
    path = tmp_path / "data.npz"
    values = np.arange(320*3, dtype=np.float32).reshape(320, 3)
    np.savez(path, values=values, mask=np.ones_like(values, dtype=bool))
    config = {"data": str(path), "context": 12, "horizons": [3], "graph_context": {"enabled": True}}
    before = Data(config, torch.device("cpu"))
    batch_before = before.batch("val", np.arange(2))
    values[before.series.train_end:] += 1e5
    np.savez(path, values=values, mask=np.ones_like(values, dtype=bool))
    after = Data(config, torch.device("cpu"))
    # First validation origin uses only training history. Restrict to that origin.
    batch_after = after.batch("val", np.arange(1))
    torch.testing.assert_close(batch_before["graph_features"][:1], batch_after["graph_features"])
    assert before.pooled_mean == after.pooled_mean and before.pooled_std == after.pooled_std


def test_context_projection_can_transmit_neighbor_level_and_time():
    torch.manual_seed(7)
    module = ContextGraphExchange(8, 4, FEATURE_DIM)
    module.gate.data.fill_(.2)
    module.context_projection.weight.data.normal_(0, .1)
    hidden = torch.randn(1, 3, 2, 8)
    mask = torch.ones(1, 3, 2, dtype=torch.bool)
    adjacency = torch.ones(3, 3)
    context = torch.zeros(1, 3, FEATURE_DIM)
    baseline = module(hidden, adjacency, mask, context)
    for index in (0, 1, 5, 7):  # Level, scale, daily phase, weekly phase.
        altered = context.clone()
        altered[:, 1, index] = 1
        changed = module(hidden, adjacency, mask, altered)
        assert (changed[:, 0]-baseline[:, 0]).abs().max() > 1e-6
    with pytest.raises(ValueError, match="explicit"):
        module(hidden, adjacency, mask, None)


def test_zero_context_projection_preserves_existing_nonzero_graph():
    torch.manual_seed(9)
    original = GraphExchange(8, 4, 0.)
    original.gate.data.fill_(.3)
    extended = ContextGraphExchange(8, 4, FEATURE_DIM)
    missing, unexpected = extended.load_state_dict(original.state_dict(), strict=False)
    assert missing == ["context_projection.weight"] and not unexpected
    hidden = torch.randn(2, 3, 2, 8)
    mask = torch.rand(2, 3, 2) > .2
    graph = torch.rand(3, 3)
    context = torch.randn(2, 3, FEATURE_DIM)
    torch.testing.assert_close(extended(hidden, graph, mask, context), original(hidden, graph, mask))


@pytest.mark.parametrize("layers", [(0, 2), (1, 2)])
def test_native_identity_cache_gradient_and_neighbor_level_sensitivity(layers):
    torch.manual_seed(42)
    model = NativeBolt(random_bolt(), np.ones((3, 3)), 3, layers, rank=4,
                       graph_feature_dim=FEATURE_DIM).train()
    x = torch.randn(1, 12, 3)
    x[:, :, 1] = 30
    mask = torch.ones_like(x, dtype=torch.bool)
    context = features(x, mask)
    original = model.native(context=x.transpose(1, 2).reshape(3, 12)).quantile_preds[:, 1, :3].T[None]
    torch.testing.assert_close(model(x, mask, context), original)
    for module in model.graph.values():
        module.gate.data.fill_(.2)
        module.context_projection.weight.data.normal_(0, .1)
    outputs, gradients = [], []
    for cached in (False, True):
        model.zero_grad(set_to_none=True)
        output = model.from_prefix(model.prefix(x, mask, 1), 1, graph_features=context) if cached else model(
            x, mask, context)
        output.square().mean().backward()
        outputs.append(output.detach())
        gradients.append({n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad})
    torch.testing.assert_close(outputs[0], outputs[1])
    for name in gradients[0]:
        assert torch.isfinite(gradients[0][name]).all()
        torch.testing.assert_close(gradients[0][name], gradients[1][name])
    assert all(gradients[0][f"graph.{layer}.context_projection.weight"].abs().max() > 0 for layer in layers)
    assert all(p.grad is None for p in model.native.parameters())
    shifted = x.clone()
    shifted[:, :, 1] = 65
    changed = model(shifted, mask, features(shifted, mask))
    assert (changed[:, :, 0]-outputs[0][:, :, 0]).abs().max() > 1e-6


def test_context_sidecar_hash_and_dataset_identity(tmp_path):
    path = tmp_path / "calendar.npz"
    np.savez(path, calendar=np.zeros((3, 2), np.float32), known=np.zeros((3, 2), bool))
    path.with_suffix(".metadata.json").write_text(json.dumps({"prepared_sha256": "dataset",
                                                             "sidecar_sha256": file_sha256(path)}))
    calendar, known, _ = load_calendar(path, "dataset", 3)
    assert calendar.shape == (3, 2) and not known.any()
    with pytest.raises(ValueError, match="audit hash"):
        load_calendar(path, "different_dataset", 3)


def test_context_protocol_is_separate_and_keeps_existing_budget_and_models():
    old = stage_config("configs/next_stage.yaml")
    new = stage_config("configs/next_stage_context.yaml")
    assert "graph_context" not in old
    assert new["graph_context"]["enabled"]
    assert old["budget"] == new["budget"] and old["models"] == new["models"]
    assert old["mechanism_report"] != new["mechanism_report"]
    settings = dataset_settings({"data": "data.npz"}, new, "metr_la")
    assert settings["graph_context"]["time_path"].endswith("metr_la.npz")


@pytest.mark.skipif(not Path("external/hsdgnn/model/HSDGNN.py").exists(), reason="Pinned upstream not installed")
def test_hsdgnn_uses_source_clock_and_retains_explicit_unknown_fallback():
    model = HSDGNNAdapter("external/hsdgnn", 8, 3, 3, 24, width=8, embed_dim=4).eval()
    x, mask, origins = torch.randn(2, 8, 3), torch.ones(2, 8, 3, dtype=torch.bool), torch.tensor([8, 20])
    calendar = torch.zeros(2, 8, 2)
    calendar[..., 0], calendar[..., 1] = .5, 6.
    known = torch.zeros_like(calendar, dtype=torch.bool)
    torch.testing.assert_close(model(x, mask, origins, (calendar, known)), model(x, mask, origins))
    captured = []
    hook = model.model.register_forward_pre_hook(lambda _, args: captured.append(args[0].detach()))
    output = model(x, mask, origins, (calendar, ~known))
    hook.remove()
    assert output.shape == (2, 3, 3) and torch.isfinite(output).all()
    assert captured[0][..., 1].eq(.5).all() and captured[0][..., 2].eq(6).all()


def test_resume_fingerprint_changes_if_clock_or_source_data_change(tmp_path):
    prepared = tmp_path / "data.npz"
    prepared.write_bytes(b"prepared")
    sidecar = tmp_path / "toy.npz"
    sidecar.write_bytes(b"clock")
    metadata = sidecar.with_suffix(".metadata.json")
    metadata.write_text("{}")
    config = {"datasets": ["toy"], "graph_context": {"enabled": True, "sidecar_root": str(tmp_path)}}
    spec = {"datasets": {"toy": {"data": str(prepared)}}}
    before = context_inputs(config, spec)
    sidecar.write_bytes(b"different clock")
    assert before != context_inputs(config, spec)
    sidecar.write_bytes(b"clock")
    assert before == context_inputs(config, spec)
    prepared.write_bytes(b"different prepared data")
    assert before != context_inputs(config, spec)
