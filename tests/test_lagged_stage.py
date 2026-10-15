from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from gofa_f import lagged_stage, native_stage as runtime
from gofa_f.graphs import build_graph
from gofa_f.idle_launcher import parse_deadline
from gofa_f.lagged_context import LaggedSupportBolt, SketchData, lagged_correlation_graph
from gofa_f.prefix_cache import atomic_json
from test_native import random_bolt


def fresh():
    torch.manual_seed(42)
    return LaggedSupportBolt(random_bolt(), np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32),
        np.ones((3, 3), np.float32), 12, "l1_lagged", graph_layers=(1, 2), rank=4,
        graph_feature_dim=19, output_adapter_rank=4, graph_exchange="prior_routed",
        graph_attention_rank=4, graph_prior_floor=.5)


def test_leading_source_direction_and_lag_recovery_train_only():
    rng = np.random.default_rng(42)
    values = rng.normal(size=(140, 3)).astype(np.float32)
    values[2:, 1] = values[:-2, 0]
    mask = np.ones_like(values, bool)
    mask[:2, 1] = False
    graph, lag = lagged_correlation_graph(values, mask, np.zeros(140), 100, k=1)
    assert graph[1, 0] > .999 and lag[1, 0] == 2
    assert graph[0, 1] < .5
    values[100:] = 1e6
    changed, other_lag = lagged_correlation_graph(values, mask, np.zeros(140), 100, k=1)
    np.testing.assert_array_equal(graph, changed)
    np.testing.assert_array_equal(lag, other_lag)


def test_no_crossing_segments_even_when_labels_repeat():
    values = np.random.default_rng(2).normal(size=(21, 2)).astype(np.float32)
    segments = np.r_[np.zeros(10), 1, np.zeros(10)]
    valid = [t for t in range(3, 21) if np.all(segments[t-3:t+1] == segments[t])]
    expected = abs(np.corrcoef(values[valid, 1], values[np.asarray(valid)-3, 0])[0, 1])
    graph, _ = lagged_correlation_graph(values, np.ones_like(values, bool), segments, 21, lags=(3,), k=1, min_overlap=2)
    np.testing.assert_allclose(graph[1, 0], expected, atol=1e-6)


def test_all_missing_and_constant_nodes_finite():
    values = np.ones((40, 3), np.float32)
    mask = np.ones_like(values, bool)
    mask[:, 0] = False
    graph, lag = lagged_correlation_graph(values, mask, np.zeros(40), 40)
    np.testing.assert_array_equal(graph, np.eye(3))
    assert np.count_nonzero(lag) == 0


def test_fixed_cutoff_and_validation_guard(tmp_path):
    config = lagged_stage.config_file("configs/stage11_metr_lagged.yaml")
    deadline = parse_deadline("2026-10-16T10:58:00+08:00")
    window = lagged_stage.bind_window(tmp_path, now=deadline-3600)
    assert window["wall_deadline"] == deadline
    assert lagged_stage.bind_window(tmp_path, True, now=deadline-1800) == window
    folder = tmp_path/"trials/l1_lagged"
    folder.mkdir(parents=True)
    atomic_json(folder/"best_validation.json", {"mae": 3.05, "step12_mae": 3.69})
    assert lagged_stage.select(tmp_path, config)["selected"] is None


def test_native_initialization_cached_gradients_and_frozen_backbone():
    model = fresh()
    x = torch.randn(2, 12, 3)
    mask = torch.rand_like(x) > .15
    features = torch.randn(2, 3, 19)
    context = x.transpose(1, 2).reshape(-1, 12).masked_fill(~mask.transpose(1, 2).reshape(-1, 12), torch.nan)
    with torch.no_grad():
        ref = model.native(context=context).quantile_preds[:, model.median_index, :12].reshape(2, 3, 12).transpose(1, 2)
        torch.testing.assert_close(model(x, mask, features), ref)
    for module in model.graph.values(): module.gate.data.fill_(.1)
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
    assert all(p.grad is None for p in model.native.parameters())


def test_lagged_model_precise_resume(tmp_path):
    values = np.random.default_rng(5).normal(size=(160, 3)).astype(np.float32)
    np.savez(tmp_path/"series.npz", values=values, mask=np.ones_like(values, bool))
    data = SketchData({"data": str(tmp_path/"series.npz"), "context": 12, "horizons": [12],
        "validation_only": True, "graph_context": {"enabled": True, "feature_mode": "values_and_clock"}}, torch.device("cpu"))
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
    config = lagged_stage.config_file("configs/stage11_metr_lagged.yaml")
    settings = yaml.safe_load(Path("configs/benchmark.yaml").read_text())["datasets"]["metr_la"]
    settings = dict(settings, validation_only=True, graph_context={"enabled": True,
        "feature_mode": "values_and_clock", "time_path": config["stage11"]["sidecar"]})
    data = SketchData(settings, torch.device("cpu"))
    graph, _ = build_graph(data.series, settings)
    native = ChronosBoltModelForForecasting.from_pretrained(str(weights), local_files_only=True,
        torch_dtype=torch.float32).requires_grad_(False).eval()
    model = LaggedSupportBolt(native, graph[:3, :3], graph[:3, :3], np.ones((3, 3), np.float32),
        12, "l1_lagged", **config["architecture"])
    batch = data.batch("train", [0])
    x, mask, features = batch["x"][:, :, :3], batch["x_mask"][:, :, :3], batch["graph_features"][:, :3]
    with torch.no_grad():
        direct = model(x, mask, features)
        cached = model.from_prefix(model.prefix(x, mask, 2), 2, graph_features=features)
    torch.testing.assert_close(direct, cached, atol=2e-5, rtol=2e-5)
