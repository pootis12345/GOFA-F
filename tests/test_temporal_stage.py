import subprocess
import sys

import numpy as np
import pytest
import torch

from gofa_f import native_stage as runtime, temporal_stage
from gofa_f.data import correlation_graph
from gofa_f.prefix_cache import atomic_json
from gofa_f.temporal_context import SketchData, TemporalSketchBolt, history_sketch
from test_native import random_bolt


def test_registered_window_and_resume_binding(tmp_path):
    config = temporal_stage.config_file("configs/stage8_metr_temporal.yaml")
    assert config["architecture"]["graph_feature_dim"] == 19
    assert config["resources"]["yield_on_foreign"] is False
    window = temporal_stage.bind_window(tmp_path, now=100)
    assert window == {"started_at": 100, "wall_deadline": 43300,
                      "training_deadline": 42400, "window_seconds": 43200}
    assert temporal_stage.bind_window(tmp_path, resume=True, now=500) == window
    temporal_stage.bind_manifest(tmp_path/"manifest.json", {"window": window, "graph_sha": "a"}, False)
    with pytest.raises(ValueError):
        temporal_stage.bind_manifest(tmp_path/"manifest.json", {"window": window, "graph_sha": "b"}, True)


def test_ordered_sketch_masked_means_and_causal_targets():
    history = torch.arange(12., dtype=torch.float32).reshape(1, 12, 1)
    mask = torch.ones_like(history, dtype=torch.bool)
    mask[:, :3] = False
    result = history_sketch(history, mask, torch.zeros(1), torch.ones(1), 0., 1.)
    torch.testing.assert_close(result[0, 0], torch.tensor([0., 0., 4., 1., 7., 1., 10., 1.]))
    mask[:, 3:6] = False
    result = history_sketch(history, mask, torch.zeros(1), torch.ones(1), 0., 1.)
    assert torch.isfinite(result).all()
    assert torch.count_nonzero(result[0, 0, :4]) == 0


def test_train_prefix_only_correlation_and_dynamic_support():
    train = np.arange(30, dtype=np.float32)[:, None] * np.array([[1, -1, 2]], np.float32)
    tail = np.random.default_rng(1).normal(size=(10, 3)).astype(np.float32)
    mask = np.ones_like(train, bool)
    first = correlation_graph(train, mask, k=1)
    second = correlation_graph(np.vstack((train, tail))[:30], np.ones((30, 3), bool), k=1)
    np.testing.assert_array_equal(first, second)
    physical = np.eye(3, dtype=np.float32)
    native = random_bolt()
    kwargs = dict(graph_layers=(1, 2), rank=4, graph_feature_dim=19,
        output_adapter_rank=4, graph_exchange="prior_routed", graph_attention_rank=4,
        graph_prior_floor=.5)
    torch.manual_seed(42)
    t1 = TemporalSketchBolt(native, physical, first, 12, "t1_sketch", **kwargs)
    torch.manual_seed(42)
    t2 = TemporalSketchBolt(native, physical, first, 12, "t2_correlated", **kwargs)
    for a, b in zip(t1.graph.values(), t2.graph.values()):
        assert torch.equal(a.dynamic_support, torch.as_tensor(physical))
        assert (b.dynamic_support > torch.as_tensor(physical)).any()
        for name, parameter in a.named_parameters():
            torch.testing.assert_close(parameter, dict(b.named_parameters())[name])
    assert torch.equal(t1.adjacency, t2.adjacency)


def test_initial_native_prediction_cache_gradients_and_frozen_backbone():
    native = random_bolt()
    graph = np.ones((3, 3), dtype=np.float32)
    model = TemporalSketchBolt(native, graph, graph, 12, "t2_correlated",
        graph_layers=(1, 2), rank=4, graph_feature_dim=19, output_adapter_rank=4,
        graph_exchange="prior_routed", graph_attention_rank=4, graph_prior_floor=.5)
    x = torch.randn(2, 12, 3)
    observed = torch.rand_like(x) > .15
    features = torch.randn(2, 3, 19)
    context = x.transpose(1, 2).reshape(-1, 12).masked_fill(
        ~observed.transpose(1, 2).reshape(-1, 12), torch.nan)
    with torch.no_grad():
        expected = native(context=context).quantile_preds[:, native.chronos_config.quantiles.index(.5), :12]
        expected = expected.reshape(2, 3, 12).transpose(1, 2)
        direct = model(x, observed, features)
        cached = model.from_prefix(model.prefix(x, observed, 2), 2, graph_features=features)
    torch.testing.assert_close(direct, expected)
    torch.testing.assert_close(cached, direct)
    for module in model.graph.values():
        module.gate.data.fill_(.1)
    results = []
    for cached in (False, True):
        model.zero_grad(set_to_none=True)
        prediction = model.from_prefix(model.prefix(x, observed, 2), 2, graph_features=features) \
            if cached else model(x, observed, features)
        prediction.square().mean().backward()
        results.append({n: p.grad.detach().clone() for n, p in model.named_parameters()
                        if p.requires_grad and p.grad is not None})
    assert results[0].keys() == results[1].keys()
    for name in results[0]:
        torch.testing.assert_close(results[0][name], results[1][name], atol=1e-5, rtol=1e-4)
    assert all(p.grad is None for p in native.parameters())


def test_selection_requires_both_validation_metrics_and_prefers_t1_on_tie(tmp_path):
    config = temporal_stage.config_file("configs/stage8_metr_temporal.yaml")
    for variant, mae, step in (("t1_sketch", 3.120, 3.79), ("t2_correlated", 3.110, 3.78)):
        folder = temporal_stage.task_directory(tmp_path, variant)
        folder.mkdir(parents=True)
        atomic_json(folder/"progress.json", {"epoch": 100, "offset": 0, "updates": 2, "stale": 0})
        atomic_json(folder/"metadata.json", {"train_windows": 100})
        atomic_json(folder/"best_validation.json", {"mae": mae, "step12_mae": step})
    decision = temporal_stage.select(tmp_path, config)
    assert decision["selected"] == "t1_sketch"
    assert "test" not in str(decision).lower()


def test_step12_guard_rejects_mean_improvement(tmp_path):
    config = temporal_stage.config_file("configs/stage8_metr_temporal.yaml")
    for variant in ("t1_sketch", "t2_correlated"):
        folder = temporal_stage.task_directory(tmp_path, variant)
        folder.mkdir(parents=True)
        atomic_json(folder/"best_validation.json", {"mae": 3.0, "step12_mae": 3.80})
    assert temporal_stage.select(tmp_path, config)["selected"] is None


def test_cpu_data_real_shape_and_target_independence(tmp_path):
    values = np.random.default_rng(42).normal(size=(160, 3)).astype(np.float32)
    values[10:13, 0] = np.nan
    mask = np.isfinite(values)
    values = np.nan_to_num(values)
    path = tmp_path/"series.npz"
    np.savez(path, values=values, mask=mask)
    data = SketchData({"data": str(path), "context": 12, "horizons": [12],
        "validation_only": True, "graph_context": {"enabled": True,
        "feature_mode": "values_and_clock"}}, torch.device("cpu"))
    batch = data.batch("train", [0, 1, len(data.windows["train"])-1])
    assert batch["graph_features"].shape == (3, 3, 19)
    assert torch.isfinite(batch["graph_features"]).all()
    saved = batch["graph_features"].clone()
    batch["y_raw"][:] = 1e6
    torch.testing.assert_close(saved, batch["graph_features"])


def test_stage_import_does_not_create_cuda_context():
    subprocess.run([sys.executable, "-c", "import torch,gofa_f.temporal_stage; assert not torch.cuda.is_initialized()"],
                   check=True)


def test_real_metr_cpu_weight_and_history_shape():
    from pathlib import Path
    import yaml
    from gofa_f.graphs import build_graph
    from chronos.chronos_bolt import ChronosBoltModelForForecasting
    weights = Path("data/weights/chronos-bolt-mini-2512683")
    if not weights.exists():
        pytest.skip("Real pretrained weights available on C")
    config = temporal_stage.config_file("configs/stage8_metr_temporal.yaml")
    settings = yaml.safe_load(Path("configs/benchmark.yaml").read_text())["datasets"]["metr_la"]
    settings = dict(settings, validation_only=True, graph_context={"enabled": True,
        "feature_mode": "values_and_clock", "time_path": config["stage8"]["sidecar"]})
    data = SketchData(settings, torch.device("cpu"))
    batch = data.batch("train", [0])
    graph, _ = build_graph(data.series, settings)
    native = ChronosBoltModelForForecasting.from_pretrained(str(weights),
        local_files_only=True, torch_dtype=torch.float32).requires_grad_(False).eval()
    model = TemporalSketchBolt(native, graph[:3, :3], graph[:3, :3], 12, "t1_sketch",
        **config["architecture"])
    x, observed, features = batch["x"][:, :, :3], batch["x_mask"][:, :, :3], batch["graph_features"][:, :3]
    context = x.transpose(1, 2).reshape(-1, 12).masked_fill(
        ~observed.transpose(1, 2).reshape(-1, 12), torch.nan)
    with torch.no_grad():
        reference = native(context=context).quantile_preds[:, model.median_index, :12]
        reference = reference.reshape(1, 3, 12).transpose(1, 2)
        direct = model(x, observed, features)
        cached = model.from_prefix(model.prefix(x, observed, 2), 2, graph_features=features)
    torch.testing.assert_close(direct, reference, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(cached, direct, atol=2e-5, rtol=2e-5)
    assert features.shape == (1, 3, 19)
    assert all(p.grad is None for p in native.parameters())


def test_sketch_training_resume_restores_optimizer_rng_and_cursor(tmp_path):
    values = np.random.default_rng(7).normal(size=(160, 3)).astype(np.float32)
    path = tmp_path/"series.npz"
    np.savez(path, values=values, mask=np.ones_like(values, dtype=bool))
    data = SketchData({"data": str(path), "context": 12, "horizons": [12],
        "validation_only": True, "graph_context": {"enabled": True,
        "feature_mode": "values_and_clock"}}, torch.device("cpu"))

    class GPU:
        device = torch.device("cpu")
        micro = 4
        def check(self, budget):
            pass
        def shrink(self):
            raise AssertionError("Unexpected CPU OOM")

    class Budget:
        def remaining(self):
            return 3600.
        def tick(self, force=False):
            pass

    def fresh():
        torch.manual_seed(42)
        return TemporalSketchBolt(random_bolt(), np.ones((3, 3), np.float32),
            np.ones((3, 3), np.float32), 12, "t1_sketch", graph_layers=(1, 2), rank=4,
            graph_feature_dim=19, output_adapter_rank=4, graph_exchange="prior_routed",
            graph_attention_rank=4, graph_prior_floor=.5)

    options = dict(max_epochs=100, patience=15, adapter_lr=.0001,
                   optimizer_kind="adamw", loss_kind="raw_mae")
    gpu, budget = GPU(), Budget()
    runtime.fit(fresh(), data, tmp_path/"resume", gpu, budget, .001, max_updates=2, **options)
    before = torch.load(tmp_path/"resume/last.pt", map_location="cpu", weights_only=False)
    runtime.fit(fresh(), data, tmp_path/"resume", gpu, budget, .001, max_updates=2, **options)
    after = torch.load(tmp_path/"resume/last.pt", map_location="cpu", weights_only=False)
    from gofa_f.pure_graph_task import assert_checkpoint_equal
    assert_checkpoint_equal(before, after)
    resumed = fresh()
    runtime.fit(resumed, data, tmp_path/"resume", gpu, budget, .001, max_updates=3, **options)
    continuous = fresh()
    runtime.fit(continuous, data, tmp_path/"continuous", gpu, budget, .001, max_updates=3, **options)
    for name, value in runtime.trained_state(resumed).items():
        torch.testing.assert_close(value, runtime.trained_state(continuous)[name], atol=2e-5, rtol=2e-4)
