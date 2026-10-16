from pathlib import Path
import numpy as np
import pytest
import torch
import yaml

from gofa_f import reg_capacity_stage, native_stage as runtime
from gofa_f.reg_capacity import RegCapacityBolt, SketchData
from gofa_f.graphs import build_graph
from gofa_f.idle_launcher import parse_deadline
from gofa_f.prefix_cache import atomic_json
from gofa_f.pure_graph_task import model_counts
from gofa_f.temporal_context import TemporalSketchBolt
from gofa_f.token_exchange import TokenExchangeBolt
from test_native import random_bolt


ARCH = dict(graph_layers=(1, 2), rank=4, graph_feature_dim=19, output_adapter_rank=6,
    graph_exchange="prior_routed", graph_attention_rank=4, graph_prior_floor=.5)


def fresh():
    torch.manual_seed(42)
    return RegCapacityBolt(random_bolt(), np.eye(3, dtype=np.float32), np.ones((3, 3), np.float32),
        12, "r1_reg_rank6", **ARCH)


def test_paired_graph_initialization_and_private_head():
    native = random_bolt()
    graph = np.ones((3, 3), np.float32)
    torch.manual_seed(42)
    reference = TokenExchangeBolt(native, graph, graph, 12, "m2_reg_only", **dict(ARCH, output_adapter_rank=4))
    torch.manual_seed(42)
    larger = RegCapacityBolt(native, graph, graph, 12, "r1_reg_rank6", **ARCH)
    for name, parameter in reference.graph.named_parameters():
        torch.testing.assert_close(parameter, dict(larger.graph.named_parameters())[name], atol=0, rtol=0)
    torch.testing.assert_close(reference.output_adapter.a, larger.output_adapter.a[:4], atol=0, rtol=0)
    x, features, mask = torch.randn(2, 12, 3), torch.randn(2, 3, 19), torch.ones(2, 12, 3, dtype=torch.bool)
    before = {n: p.clone() for n, p in native.named_parameters()}
    torch.testing.assert_close(reference(x, mask, features), larger(x, mask, features))
    optimizer = torch.optim.AdamW([p for p in larger.parameters() if p.requires_grad], lr=1e-3)
    larger(x, mask, features).square().mean().backward()
    optimizer.step()
    assert larger.output_adapter.b.abs().sum() > 0
    assert all(p.grad is None and torch.equal(p, before[n]) for n, p in native.named_parameters())


def test_cached_prediction_gradient_with_active_rank6_head():
    model = fresh()
    x, features = torch.randn(2, 12, 3), torch.randn(2, 3, 19)
    mask = torch.rand_like(x) > .15
    model.output_adapter.b.data.normal_(0, .01)
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
    assert records[0][1]["output_adapter.a"].abs().sum() > 0


def test_reg_only_graph_route_excludes_empty_nodes_and_preserves_attention_mask():
    model = fresh()
    model.graph["1"].gate.data.fill_(.2)
    hidden = torch.randn(2, 3, 2, model.native.config.d_model)
    mask = torch.ones(2, 3, 2, dtype=torch.bool)
    mask[:, 0, 0] = False
    before = mask.clone()
    result = model.exchange(1, hidden, mask, torch.randn(2, 3, 19))
    assert torch.equal(mask, before)
    torch.testing.assert_close(result[..., 0, :], hidden[..., 0, :], rtol=0, atol=0)
    torch.testing.assert_close(result[:, 0], hidden[:, 0], rtol=0, atol=0)
    assert not torch.equal(result[:, 1:, -1], hidden[:, 1:, -1])


def test_cutoff_and_validation_only_qualification(tmp_path):
    config = reg_capacity_stage.config_file("configs/stage15_metr_reg6.yaml")
    deadline = parse_deadline("2026-10-16T10:58:00+08:00")
    assert reg_capacity_stage.bind_window(tmp_path, now=deadline-3600)["wall_deadline"] == deadline
    folder = tmp_path/"trials/r1_reg_rank6"
    folder.mkdir(parents=True)
    atomic_json(folder/"best_validation.json", {"mae": 3.05, "step12_mae": 3.69})
    assert reg_capacity_stage.select(tmp_path, config)["selected"] is None


def test_rank6_optimizer_and_rng_resume(tmp_path):
    values = np.random.default_rng(9).normal(size=(160, 3)).astype(np.float32)
    np.savez(tmp_path/"series.npz", values=values, mask=np.ones_like(values, bool))
    data = SketchData({"data": str(tmp_path/"series.npz"), "context": 12, "horizons": [12], "validation_only": True,
        "graph_context": {"enabled": True, "feature_mode": "values_and_clock"}}, torch.device("cpu"))
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
    from gofa_f.head_capacity import HeadCapacityBolt
    wrong = HeadCapacityBolt(random_bolt(), np.eye(3, dtype=np.float32), np.ones((3, 3), np.float32),
        12, "h1_rank6", **ARCH)
    with pytest.raises(ValueError, match="protocol mismatch"):
        runtime.fit(wrong, data, tmp_path/"resume", gpu, budget, .001, max_updates=2, **options)
    resumed, continuous = fresh(), fresh()
    runtime.fit(resumed, data, tmp_path/"resume", gpu, budget, .001, max_updates=3, **options)
    runtime.fit(continuous, data, tmp_path/"continuous", gpu, budget, .001, max_updates=3, **options)
    for name, value in runtime.trained_state(resumed).items():
        torch.testing.assert_close(value, runtime.trained_state(continuous)[name], atol=2e-5, rtol=2e-4)


def test_real_weights_maximum_rank_within_parameter_budget_cpu():
    weights = Path("data/weights/chronos-bolt-mini-2512683")
    if not weights.exists(): pytest.skip("Real weights live on C")
    from chronos.chronos_bolt import ChronosBoltModelForForecasting
    config = reg_capacity_stage.config_file("configs/stage15_metr_reg6.yaml")
    settings = yaml.safe_load(Path("configs/benchmark.yaml").read_text())["datasets"]["metr_la"]
    settings = dict(settings, validation_only=True, graph_context={"enabled": True,
        "feature_mode": "values_and_clock", "time_path": config["stage15"]["sidecar"]})
    data = SketchData(settings, torch.device("cpu"))
    graph, _ = build_graph(data.series, settings)
    native = ChronosBoltModelForForecasting.from_pretrained(str(weights), local_files_only=True,
        torch_dtype=torch.float32).requires_grad_(False).eval()
    model = RegCapacityBolt(native, graph[:3, :3], graph[:3, :3], 12, "r1_reg_rank6", **config["architecture"])
    assert model_counts(model)["trainable_real_dof"] == 119302
    assert model_counts(model)["point_loss_active_trainable_real_dof"] == 115918
    above = TemporalSketchBolt(native, graph[:3, :3], graph[:3, :3], 12, "t2_correlated",
        **dict(config["architecture"], output_adapter_rank=7))
    limit = 108927*1.1
    assert model_counts(model)["trainable_real_dof"] <= limit < model_counts(above)["trainable_real_dof"]
    batch = data.batch("train", [0])
    x, mask, features = batch["x"][:, :, :3], batch["x_mask"][:, :, :3], batch["graph_features"][:, :3]
    with torch.no_grad():
        direct = model(x, mask, features)
        cached = model.from_prefix(model.prefix(x, mask, 2), 2, graph_features=features)
    torch.testing.assert_close(direct, cached, atol=2e-5, rtol=2e-5)
