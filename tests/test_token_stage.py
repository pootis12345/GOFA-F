from pathlib import Path
import numpy as np
import pytest
import torch
import yaml

from gofa_f import native_stage as runtime, token_stage
from gofa_f.graphs import build_graph
from gofa_f.idle_launcher import parse_deadline
from gofa_f.prefix_cache import atomic_json
from gofa_f.pure_graph_task import model_counts
from gofa_f.token_exchange import SketchData, TokenExchangeBolt
from test_native import random_bolt


VARIANTS = ("m1_data_only", "m2_reg_only")


def fresh(variant):
    torch.manual_seed(42)
    return TokenExchangeBolt(random_bolt(), np.eye(3, dtype=np.float32), np.ones((3, 3), np.float32),
        12, variant, graph_layers=(1, 2), rank=4, graph_feature_dim=19,
        output_adapter_rank=4, graph_exchange="prior_routed", graph_attention_rank=4, graph_prior_floor=.5)


@pytest.mark.parametrize("variant", VARIANTS)
def test_only_selected_token_role_changes_and_native_mask_is_preserved(variant):
    model = fresh(variant)
    module = model.graph["1"]
    module.gate.data.fill_(.2)
    hidden = torch.randn(2, 3, 2, model.native.config.d_model)
    mask = torch.ones(2, 3, 2, dtype=torch.bool)
    mask[:, 0, 0] = False  # Native REG stays valid, but an empty node cannot send graph messages.
    original = mask.clone()
    output = model.exchange(1, hidden, mask, torch.randn(2, 3, 19))
    assert torch.equal(mask, original)
    torch.testing.assert_close(output[:, 0], hidden[:, 0], atol=0, rtol=0)
    untouched = -1 if variant == "m1_data_only" else 0
    changed = 0 if variant == "m1_data_only" else -1
    torch.testing.assert_close(output[..., untouched, :], hidden[..., untouched, :], atol=0, rtol=0)
    assert not torch.equal(output[:, 1:, changed], hidden[:, 1:, changed])


@pytest.mark.parametrize("variant", VARIANTS)
def test_initial_native_prediction_cache_and_frozen_gradients(variant):
    model = fresh(variant)
    x, features = torch.randn(2, 12, 3), torch.randn(2, 3, 19)
    mask = torch.rand_like(x) > .15
    context = x.transpose(1, 2).reshape(-1, 12).masked_fill(~mask.transpose(1, 2).reshape(-1, 12), torch.nan)
    with torch.no_grad():
        expected = model.native(context=context).quantile_preds[:, model.median_index, :12].reshape(2, 3, 12).transpose(1, 2)
        torch.testing.assert_close(model(x, mask, features), expected)
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


@pytest.mark.parametrize("variant", VARIANTS)
def test_checkpoint_resume_and_wrong_token_policy_rejection(tmp_path, variant):
    values = np.random.default_rng(10).normal(size=(160, 3)).astype(np.float32)
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
    runtime.fit(fresh(variant), data, tmp_path/"resume", gpu, budget, .001, max_updates=2, **options)
    before = torch.load(tmp_path/"resume/last.pt", weights_only=False)
    runtime.fit(fresh(variant), data, tmp_path/"resume", gpu, budget, .001, max_updates=2, **options)
    from gofa_f.pure_graph_task import assert_checkpoint_equal
    assert_checkpoint_equal(before, torch.load(tmp_path/"resume/last.pt", weights_only=False))
    wrong = VARIANTS[1] if variant == VARIANTS[0] else VARIANTS[0]
    with pytest.raises(ValueError, match="protocol mismatch"):
        runtime.fit(fresh(wrong), data, tmp_path/"resume", gpu, budget, .001, max_updates=2, **options)
    resumed, continuous = fresh(variant), fresh(variant)
    runtime.fit(resumed, data, tmp_path/"resume", gpu, budget, .001, max_updates=3, **options)
    runtime.fit(continuous, data, tmp_path/"continuous", gpu, budget, .001, max_updates=3, **options)
    for name, value in runtime.trained_state(resumed).items():
        torch.testing.assert_close(value, runtime.trained_state(continuous)[name], atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("first_step,selected", [(3.67, "m1_data_only"), (3.69, "m2_reg_only")])
def test_validation_guard_and_preregistered_tie(tmp_path, first_step, selected):
    for variant, score, step in ((VARIANTS[0], 3.05, first_step), (VARIANTS[1], 3.045, 3.67)):
        folder = tmp_path/"trials"/variant
        folder.mkdir(parents=True)
        atomic_json(folder/"best_validation.json", {"mae": score, "step12_mae": step})
    result = token_stage.select(tmp_path, token_stage.config_file("configs/stage14_metr_tokens.yaml"))
    assert result["selected"] == selected
    assert "test" not in str(result).lower()


def test_absolute_cutoff(tmp_path):
    deadline = parse_deadline("2026-10-16T10:58:00+08:00")
    window = token_stage.bind_window(tmp_path, now=deadline-3600)
    assert window["wall_deadline"] == deadline and window["training_deadline"] == deadline-900
    assert token_stage.bind_window(tmp_path, True, now=deadline-1800) == window


def test_real_weights_and_equal_parameter_budget_cpu():
    weights = Path("data/weights/chronos-bolt-mini-2512683")
    if not weights.exists(): pytest.skip("Real weights live on C")
    from chronos.chronos_bolt import ChronosBoltModelForForecasting
    config = token_stage.config_file("configs/stage14_metr_tokens.yaml")
    settings = yaml.safe_load(Path("configs/benchmark.yaml").read_text())["datasets"]["metr_la"]
    settings = dict(settings, validation_only=True, graph_context={"enabled": True,
        "feature_mode": "values_and_clock", "time_path": config["stage14"]["sidecar"]})
    data = SketchData(settings, torch.device("cpu"))
    graph, _ = build_graph(data.series, settings)
    native = ChronosBoltModelForForecasting.from_pretrained(str(weights), local_files_only=True,
        torch_dtype=torch.float32).requires_grad_(False).eval()
    batch = data.batch("train", [0])
    x, mask, features = batch["x"][:, :, :3], batch["x_mask"][:, :, :3], batch["graph_features"][:, :3]
    for variant in VARIANTS:
        model = TokenExchangeBolt(native, graph[:3, :3], graph[:3, :3], 12, variant, **config["architecture"])
        assert model_counts(model)["trainable_real_dof"] == 115078
        with torch.no_grad():
            prefix = model.prefix(x, mask, 2)
            assert prefix[1].shape[-1] == 2 and prefix[1][..., -1].all()
            direct = model(x, mask, features)
            cached = model.from_prefix(prefix, 2, graph_features=features)
        torch.testing.assert_close(direct, cached, atol=2e-5, rtol=2e-5)
