import subprocess
import sys

import numpy as np
import pytest
import torch

from gofa_f import long_stage
from gofa_f.long_horizon import LongHorizonBolt, TemporalResidual
from gofa_f.prefix_cache import atomic_json
from gofa_f import native_stage as runtime
from test_native import random_bolt


def test_registered_stage_and_six_hour_window(tmp_path):
    config = long_stage.config_file("configs/stage6_metr_long.yaml")
    assert config["datasets"] == ["metr_la"]
    assert config["architecture"]["graph_feature_dim"] == 11
    assert config["resources"]["yield_on_foreign"] is False
    window = long_stage.bind_window(tmp_path, now=100)
    assert window["training_deadline"] == 20800
    assert window["wall_deadline"] == 21700
    assert long_stage.bind_window(tmp_path, resume=True, now=1000) == window


def test_manifest_resume_rejects_changed_clock_or_deadline(tmp_path):
    path = tmp_path/"manifest.json"
    expected = {"git_revision": "fixed", "inputs": {"clock": "sha"},
                "window": {"wall_deadline": 21600}}
    long_stage.bind_manifest(path, expected, False)
    long_stage.bind_manifest(path, expected, True)
    with pytest.raises(ValueError):
        long_stage.bind_manifest(path, dict(expected, inputs={"clock": "changed"}), True)
    with pytest.raises(ValueError):
        long_stage.bind_manifest(path, dict(expected, window={"wall_deadline": 21601}), True)


def test_selection_uses_validation_only_and_prefers_simpler_within_half_percent(tmp_path):
    config = long_stage.config_file("configs/stage6_metr_long.yaml")
    for variant, score in (("c1_clock", 3.20), ("c2_residual", 3.19)):
        folder = long_stage.task_directory(tmp_path, variant)
        folder.mkdir(parents=True)
        atomic_json(folder/"progress.json", {"best_mae": score, "epoch": 1,
            "offset": 0, "updates": 25, "stale": 0})
        atomic_json(folder/"metadata.json", {"train_windows": 100})
    result = long_stage.select(tmp_path, config)
    assert result["selected"] == "c1_clock"
    assert result["control"] == "self_clock"
    assert "test" not in str(result).lower()


def test_selection_rejects_both_below_gain_threshold(tmp_path):
    config = long_stage.config_file("configs/stage6_metr_long.yaml")
    for variant in ("c1_clock", "c2_residual"):
        folder = long_stage.task_directory(tmp_path, variant)
        folder.mkdir(parents=True)
        atomic_json(folder/"progress.json", {"best_mae": 3.27, "epoch": 1,
            "offset": 0, "updates": 25, "stale": 0})
    assert long_stage.select(tmp_path, config)["selected"] is None


def test_residual_zero_initialization_scale_mask_and_quantile_shift():
    head = TemporalResidual()
    x = torch.randn(2, 12, 3)
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[:, :, 0] = False
    loc = torch.zeros(2, 3, 1)
    scale = torch.ones(2, 3, 1)*2
    clock = torch.randn(2, 6)
    assert torch.count_nonzero(head(x, mask, clock, loc, scale)) == 0
    with torch.no_grad():
        head.last.bias.fill_(.25)
    result = head(x, mask, clock, loc, scale)
    assert torch.count_nonzero(result[:, 0]) == 0
    torch.testing.assert_close(result[:, 1:], torch.full((2, 2, 12), .5))
    with pytest.raises(ValueError):
        head(x, mask, torch.randn(2, 5), loc, scale)


@pytest.mark.parametrize("residual", [False, True])
def test_real_model_zero_increment_cache_and_frozen_native(residual):
    torch.manual_seed(42)
    native = random_bolt()
    kwargs = dict(graph_layers=(1, 2), rank=4, graph_feature_dim=11,
                  output_adapter_rank=4, graph_exchange="prior_routed",
                  graph_attention_rank=4, graph_prior_floor=.5)
    model = LongHorizonBolt(native, np.ones((3, 3)), 12,
                            temporal_residual=residual, **kwargs).train()
    x = torch.randn(2, 12, 3)
    mask = torch.rand_like(x) > .1
    features = torch.randn(2, 3, 11)
    features[..., 5:] = features[:, :1, 5:]
    context = x.transpose(1, 2).reshape(-1, 12).masked_fill(
        ~mask.transpose(1, 2).reshape(-1, 12), torch.nan)
    with torch.no_grad():
        native_result = native(context=context).quantile_preds[:, native.chronos_config.quantiles.index(.5), :12]
        native_result = native_result.reshape(2, 3, 12).transpose(1, 2)
        direct = model(x, mask, features)
        cached = model.from_prefix(model.prefix(x, mask, 2), 2,
            graph_features=features, history=x, history_mask=mask)
    torch.testing.assert_close(direct, native_result)
    torch.testing.assert_close(cached, direct)
    for module in model.graph.values():
        module.gate.data.fill_(.1)
    model.zero_grad(set_to_none=True)
    quantiles = model.from_prefix(model.prefix(x, mask, 2), 2,
        return_quantiles=True, graph_features=features, history=x, history_mask=mask)
    assert quantiles.shape == (2, 3, 12, 3)
    (quantiles[:, 1]**2).mean().backward()
    assert all(p.grad is None for p in native.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    if residual:
        before = quantiles.detach()
        with torch.no_grad():
            model.temporal_residual.last.bias.fill_(.2)
            after = model.from_prefix(model.prefix(x, mask, 2), 2,
                return_quantiles=True, graph_features=features, history=x, history_mask=mask)
        difference = after-before
        torch.testing.assert_close(difference[:, 0], difference[:, 1])
        torch.testing.assert_close(difference[:, 1], difference[:, 2])


def test_stage_import_does_not_create_cuda_context():
    subprocess.run([sys.executable, "-c", "import torch,gofa_f.long_stage; assert not torch.cuda.is_initialized()"],
                   check=True)


def test_residual_training_resume_preserves_optimizer_rng_and_cursor(tmp_path):
    values = np.random.default_rng(42).normal(size=(160, 3)).astype(np.float32)
    path = tmp_path/"toy.npz"
    np.savez(path, values=values, mask=np.ones_like(values, dtype=bool))
    data = runtime.Data({"data": str(path), "context": 12, "horizons": [12],
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
        return LongHorizonBolt(random_bolt(), np.ones((3, 3)), 12,
            graph_layers=(1, 2), rank=4, graph_feature_dim=11,
            output_adapter_rank=4, graph_exchange="prior_routed",
            graph_attention_rank=4, graph_prior_floor=.5, temporal_residual=True)

    kwargs = {"max_epochs": 100, "patience": 15, "adapter_lr": .0001,
              "optimizer_kind": "adamw", "loss_kind": "raw_mae"}
    model, gpu, budget = fresh(), GPU(), Budget()
    runtime.fit(model, data, tmp_path/"resume", gpu, budget, .001, max_updates=2, **kwargs)
    saved = torch.load(tmp_path/"resume/last.pt", map_location="cpu", weights_only=False)
    runtime.fit(fresh(), data, tmp_path/"resume", gpu, budget, .001, max_updates=2, **kwargs)
    restored = torch.load(tmp_path/"resume/last.pt", map_location="cpu", weights_only=False)
    from gofa_f.pure_graph_task import assert_checkpoint_equal
    assert_checkpoint_equal(saved, restored)
    model = fresh()
    runtime.fit(model, data, tmp_path/"resume", gpu, budget, .001, max_updates=3, **kwargs)
    continuous = fresh()
    runtime.fit(continuous, data, tmp_path/"continuous", gpu, budget, .001, max_updates=3, **kwargs)
    for name, value in runtime.trained_state(model).items():
        torch.testing.assert_close(value, runtime.trained_state(continuous)[name], rtol=2e-4, atol=2e-5)
