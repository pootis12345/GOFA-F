from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from gofa_f import encoded_stage, native_stage as runtime
from gofa_f.encoded_context import EncodedHistoryBolt, OrderedData
from gofa_f.graphs import build_graph
from gofa_f.idle_launcher import parse_deadline
from gofa_f.prefix_cache import atomic_json
from test_native import random_bolt
from test_ordered_stage import data_at


def fresh():
    torch.manual_seed(42)
    return EncodedHistoryBolt(random_bolt(), np.eye(3, dtype=np.float32), np.ones((3, 3), np.float32),
        12, "e1_encoded", graph_layers=(1, 2), rank=4, graph_feature_dim=43,
        output_adapter_rank=4, graph_exchange="prior_routed", graph_attention_rank=4,
        graph_prior_floor=.5, history_encoder_width=32)


def test_registered_encoder_and_absolute_cutoff(tmp_path):
    config = encoded_stage.config_file("configs/stage10_metr_encoded.yaml")
    assert config["architecture"]["history_encoder_width"] == 32
    deadline = parse_deadline("2026-10-16T10:58:00+08:00")
    window = encoded_stage.bind_window(tmp_path, now=deadline-3600)
    assert window["wall_deadline"] == deadline
    assert encoded_stage.bind_window(tmp_path, True, now=deadline-1800) == window
    with pytest.raises(ValueError):
        encoded_stage.bind_window(tmp_path/"late", now=deadline-899)


def test_initial_prediction_nonlinear_gradient_cache_and_frozen_backbone():
    model = fresh()
    x = torch.randn(2, 12, 3)
    mask = torch.rand_like(x) > .2
    features = torch.randn(2, 3, 43)
    context = x.transpose(1, 2).reshape(-1, 12).masked_fill(~mask.transpose(1, 2).reshape(-1, 12), torch.nan)
    with torch.no_grad():
        expected = model.native(context=context).quantile_preds[:, model.median_index, :12].reshape(2, 3, 12).transpose(1, 2)
        torch.testing.assert_close(model(x, mask, features), expected)
    for module in model.graph.values():
        module.gate.data.fill_(.1)
        module.context_projection.temporal[-1].weight.data.fill_(.03)
    prefix, records = model.prefix(x, mask, 2), []
    for cached in (False, True):
        model.zero_grad(set_to_none=True)
        prediction = model.from_prefix(prefix, 2, graph_features=features) if cached else model(x, mask, features)
        prediction.square().mean().backward()
        records.append((prediction.detach(), {n: p.grad.clone() for n, p in model.named_parameters()
            if p.requires_grad and p.grad is not None}))
    torch.testing.assert_close(records[0][0], records[1][0])
    for name, gradient in records[0][1].items():
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient, records[1][1][name], atol=1e-5, rtol=1e-4)
    assert sum(g.abs().sum() for n, g in records[0][1].items() if "temporal.0.weight" in n) > 0
    assert all(p.grad is None for p in model.native.parameters())
    mask[:, :, 0] = False
    with torch.no_grad():
        assert torch.isfinite(model(x, mask, features)).all()


@pytest.mark.parametrize("step,selected", [(3.68, "e1_encoded"), (3.69, None)])
def test_no_test_based_selection(tmp_path, step, selected):
    folder = tmp_path/"trials"/"e1_encoded"
    folder.mkdir(parents=True)
    atomic_json(folder/"best_validation.json", {"mae": 3.05, "step12_mae": step})
    result = encoded_stage.select(tmp_path, encoded_stage.config_file("configs/stage10_metr_encoded.yaml"))
    assert result["selected"] == selected
    assert "test" not in str(result).lower()


def test_encoder_optimizer_rng_cursor_restore(tmp_path):
    data = data_at(tmp_path)
    class GPU:
        device = torch.device("cpu")
        micro = 4
        def check(self, budget): pass
        def shrink(self): raise AssertionError("CPU OOM")
    class Budget:
        def remaining(self): return 3600.
        def tick(self, force=False): pass
    gpu, budget = GPU(), Budget()
    options = dict(max_epochs=100, patience=15, adapter_lr=.0001, optimizer_kind="adamw", loss_kind="raw_mae")
    runtime.fit(fresh(), data, tmp_path/"resume", gpu, budget, .001, max_updates=2, **options)
    before = torch.load(tmp_path/"resume/last.pt", weights_only=False)
    runtime.fit(fresh(), data, tmp_path/"resume", gpu, budget, .001, max_updates=2, **options)
    from gofa_f.pure_graph_task import assert_checkpoint_equal
    assert_checkpoint_equal(before, torch.load(tmp_path/"resume/last.pt", weights_only=False))
    resumed, continuous = fresh(), fresh()
    runtime.fit(resumed, data, tmp_path/"resume", gpu, budget, .001, max_updates=3, **options)
    runtime.fit(continuous, data, tmp_path/"continuous", gpu, budget, .001, max_updates=3, **options)
    for name, value in runtime.trained_state(resumed).items():
        torch.testing.assert_close(value, runtime.trained_state(continuous)[name], atol=2e-5, rtol=2e-4)


def test_real_pretrained_weights_cpu():
    weights = Path("data/weights/chronos-bolt-mini-2512683")
    if not weights.exists(): pytest.skip("Real weights live on C")
    from chronos.chronos_bolt import ChronosBoltModelForForecasting
    config = encoded_stage.config_file("configs/stage10_metr_encoded.yaml")
    settings = yaml.safe_load(Path("configs/benchmark.yaml").read_text())["datasets"]["metr_la"]
    settings = dict(settings, validation_only=True, graph_context={"enabled": True,
        "feature_mode": "values_and_clock", "time_path": config["stage10"]["sidecar"]})
    data = OrderedData(settings, torch.device("cpu"))
    graph, _ = build_graph(data.series, settings)
    native = ChronosBoltModelForForecasting.from_pretrained(str(weights), local_files_only=True,
        torch_dtype=torch.float32).requires_grad_(False).eval()
    model = EncodedHistoryBolt(native, graph[:3, :3], graph[:3, :3], 12, "e1_encoded", **config["architecture"])
    batch = data.batch("train", [0])
    x, mask, features = batch["x"][:, :, :3], batch["x_mask"][:, :, :3], batch["graph_features"][:, :3]
    with torch.no_grad():
        direct = model(x, mask, features)
        cached = model.from_prefix(model.prefix(x, mask, 2), 2, graph_features=features)
    torch.testing.assert_close(direct, cached, atol=2e-5, rtol=2e-5)
