from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from gofa_f import native_stage as runtime, ordered_stage
from gofa_f.graphs import build_graph
from gofa_f.idle_launcher import parse_deadline
from gofa_f.ordered_context import OrderedData, OrderedHistoryBolt, ordered_history
from gofa_f.temporal_context import history_sketch
from gofa_f.prefix_cache import atomic_json
from test_native import random_bolt


def fresh():
    torch.manual_seed(42)
    return OrderedHistoryBolt(random_bolt(), np.eye(3, dtype=np.float32), np.ones((3, 3), np.float32),
        12, "o1_ordered", graph_layers=(1, 2), rank=4, graph_feature_dim=43,
        output_adapter_rank=4, graph_exchange="prior_routed", graph_attention_rank=4, graph_prior_floor=.5)


def data_at(tmp_path):
    values = np.random.default_rng(42).normal(size=(160, 3)).astype(np.float32)
    path = tmp_path/"series.npz"
    mask = np.ones_like(values, dtype=bool)
    mask[12:15, 0] = False
    np.savez(path, values=values, mask=mask)
    return OrderedData({"data": str(path), "context": 12, "horizons": [12], "validation_only": True,
        "graph_context": {"enabled": True, "feature_mode": "values_and_clock"}}, torch.device("cpu"))


def test_window_ends_at_original_user_cutoff(tmp_path):
    config = ordered_stage.config_file("configs/stage9_metr_ordered.yaml")
    assert config["architecture"]["graph_feature_dim"] == 43
    deadline = parse_deadline("2026-10-16T10:58:00+08:00")
    window = ordered_stage.bind_window(tmp_path, now=deadline-3600)
    assert window["wall_deadline"] == deadline
    assert window["training_deadline"] == deadline-900
    assert ordered_stage.bind_window(tmp_path, True, now=deadline-1800) == window
    with pytest.raises(ValueError):
        ordered_stage.bind_window(tmp_path/"late", now=deadline-899)


def test_preserves_order_lost_by_segment_mean_and_reconstructs_observed_history():
    x = torch.arange(12.).reshape(1, 12, 1)
    reverse = x.reshape(1, 4, 3, 1).flip(2).reshape_as(x)
    mask = torch.ones_like(x, dtype=torch.bool)
    mean, std = torch.zeros(1), torch.ones(1)
    sketch = history_sketch(x, mask, mean, std, 0., 1.)
    torch.testing.assert_close(sketch, history_sketch(reverse, mask, mean, std, 0., 1.))
    detail = ordered_history(x, mask, mean, std, 0., 1.)
    assert not torch.equal(detail, ordered_history(reverse, mask, mean, std, 0., 1.))
    centers = sketch[..., ::2].repeat_interleave(3, -1)
    torch.testing.assert_close(detail[..., ::2]+centers, x.transpose(1, 2))
    mask[:, :, 0] = False
    assert torch.count_nonzero(ordered_history(x, mask, mean, std, 0., 1.)) == 0


def test_causal_data_and_missing_tail(tmp_path):
    data = data_at(tmp_path)
    batch = data.batch("train", [0, len(data.windows["train"])-1])
    assert batch["graph_features"].shape == (2, 3, 43)
    assert torch.isfinite(batch["graph_features"]).all()
    before = batch["graph_features"].clone()
    batch["y_raw"].add_(1000)
    torch.testing.assert_close(before, batch["graph_features"])


def test_native_zero_initialization_cached_gradients_and_freezing():
    model = fresh()
    x = torch.randn(2, 12, 3)
    mask = torch.rand_like(x) > .15
    features = torch.randn(2, 3, 43)
    context = x.transpose(1, 2).reshape(-1, 12).masked_fill(~mask.transpose(1, 2).reshape(-1, 12), torch.nan)
    with torch.no_grad():
        ref = model.native(context=context).quantile_preds[:, model.median_index, :12]
        ref = ref.reshape(2, 3, 12).transpose(1, 2)
        torch.testing.assert_close(model(x, mask, features), ref)
    for module in model.graph.values():
        module.gate.data.fill_(.1)
    records = []
    prefix = model.prefix(x, mask, 2)
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


@pytest.mark.parametrize("step,selected", [(3.68, "o1_ordered"), (3.69, None)])
def test_validation_only_qualification(tmp_path, step, selected):
    folder = tmp_path/"trials"/"o1_ordered"
    folder.mkdir(parents=True)
    atomic_json(folder/"best_validation.json", {"mae": 3.05, "step12_mae": step})
    result = ordered_stage.select(tmp_path, ordered_stage.config_file("configs/stage9_metr_ordered.yaml"))
    assert result["selected"] == selected
    assert "test" not in str(result).lower()


def test_exact_checkpoint_and_continued_training_resume(tmp_path):
    data = data_at(tmp_path)
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
    gpu, budget = GPU(), Budget()
    options = dict(max_epochs=100, patience=15, adapter_lr=.0001, optimizer_kind="adamw", loss_kind="raw_mae")
    runtime.fit(fresh(), data, tmp_path/"resume", gpu, budget, .001, max_updates=2, **options)
    before = torch.load(tmp_path/"resume/last.pt", weights_only=False)
    runtime.fit(fresh(), data, tmp_path/"resume", gpu, budget, .001, max_updates=2, **options)
    after = torch.load(tmp_path/"resume/last.pt", weights_only=False)
    from gofa_f.pure_graph_task import assert_checkpoint_equal
    assert_checkpoint_equal(before, after)
    resumed, continuous = fresh(), fresh()
    runtime.fit(resumed, data, tmp_path/"resume", gpu, budget, .001, max_updates=3, **options)
    runtime.fit(continuous, data, tmp_path/"continuous", gpu, budget, .001, max_updates=3, **options)
    for name, value in runtime.trained_state(resumed).items():
        torch.testing.assert_close(value, runtime.trained_state(continuous)[name], atol=2e-5, rtol=2e-4)


def test_real_weights_and_metr_history_cpu():
    weights = Path("data/weights/chronos-bolt-mini-2512683")
    if not weights.exists():
        pytest.skip("Real weights live on C")
    from chronos.chronos_bolt import ChronosBoltModelForForecasting
    config = ordered_stage.config_file("configs/stage9_metr_ordered.yaml")
    settings = yaml.safe_load(Path("configs/benchmark.yaml").read_text())["datasets"]["metr_la"]
    settings = dict(settings, validation_only=True, graph_context={"enabled": True,
        "feature_mode": "values_and_clock", "time_path": config["stage9"]["sidecar"]})
    data = OrderedData(settings, torch.device("cpu"))
    graph, _ = build_graph(data.series, settings)
    native = ChronosBoltModelForForecasting.from_pretrained(str(weights), local_files_only=True,
        torch_dtype=torch.float32).requires_grad_(False).eval()
    model = OrderedHistoryBolt(native, graph[:3, :3], graph[:3, :3], 12, "o1_ordered", **config["architecture"])
    batch = data.batch("train", [0])
    x, mask, features = batch["x"][:, :, :3], batch["x_mask"][:, :, :3], batch["graph_features"][:, :3]
    with torch.no_grad():
        direct = model(x, mask, features)
        cached = model.from_prefix(model.prefix(x, mask, 2), 2, graph_features=features)
    torch.testing.assert_close(direct, cached, atol=2e-5, rtol=2e-5)
