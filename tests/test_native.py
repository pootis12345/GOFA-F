import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gofa_f.native import HSDGNNAdapter, NativeBolt
from gofa_f.models import parameter_counts
from gofa_f.stage_budget import Budget, BudgetExhausted


def random_bolt():
    pytest.importorskip("chronos")
    from chronos.chronos_bolt import ChronosBoltModelForForecasting
    from transformers import T5Config

    config = T5Config(d_model=16, d_ff=32, d_kv=8, num_heads=2, num_layers=4,
                      num_decoder_layers=1, dropout_rate=.1, decoder_start_token_id=0)
    config.chronos_config = {"context_length": 64, "prediction_length": 16,
                            "input_patch_size": 16, "input_patch_stride": 16,
                            "quantiles": [.1, .5, .9], "use_reg_token": True}
    return ChronosBoltModelForForecasting(config).eval()


@pytest.mark.parametrize("length", [8, 12, 24])
@pytest.mark.parametrize("layers", [(), (0, 2), (1, 2)])
def test_native_quantiles_preserved_at_zero_graph_gates(length, layers):
    torch.manual_seed(11)
    native = random_bolt()
    model = NativeBolt(native, np.eye(3), 12, layers, rank=4).train()
    assert not model.native.training
    x = torch.randn(2, length, 3)
    mask = torch.rand_like(x) > .15
    mask[0, :, 0] = False  # Completely unobserved node, including native scaling fallback.
    context = x.transpose(1, 2).reshape(6, length).masked_fill(
        ~mask.transpose(1, 2).reshape(6, length), torch.nan)
    with torch.no_grad():
        expected = native(context=context).quantile_preds.reshape(2, 3, 3, 16)[..., :12].permute(0, 2, 3, 1)
        prefix = model.prefix(x, mask, 1)
        actual = model.from_prefix(prefix, 1, return_quantiles=True)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("layers", [(0, 2), (1, 2)])
def test_cached_prefix_preserves_predictions_and_graph_gradients(layers):
    torch.manual_seed(42)
    model = NativeBolt(random_bolt(), np.ones((3, 3)), 3, layers, rank=4).train()
    for module in model.graph.values():
        module.gate.data.fill_(.1)
    x, mask = torch.randn(2, 24, 3), torch.ones(2, 24, 3, dtype=torch.bool)
    outputs, gradients = [], []
    for prefix_layers in (1, min(layers) + 1):
        model.zero_grad(set_to_none=True)
        prefix = model.prefix(x, mask, prefix_layers)
        assert not prefix[0].requires_grad
        result = model.from_prefix(prefix, prefix_layers)
        result.square().mean().backward()
        outputs.append(result.detach())
        gradients.append({name: p.grad.clone() for name, p in model.named_parameters() if p.requires_grad})
        assert all(p.grad is None for p in model.native.parameters())
        assert all(torch.isfinite(v).all() for v in gradients[-1].values())
        assert all(gradients[-1][f"graph.{i}.gate"].abs() > 0 for i in layers)
    torch.testing.assert_close(outputs[0], outputs[1])
    for name in gradients[0]:
        torch.testing.assert_close(gradients[0][name], gradients[1][name])


def test_budget_resume_preserves_spent_time_and_charges_unrecorded_tail(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("gofa_f.stage_budget.time.monotonic", lambda: clock[0])
    path = tmp_path / "budget.json"
    budget = Budget(path, 100)
    budget.activate("metr_la/gofa", 20)
    clock[0] += 10
    budget.tick(force=True)
    resumed = Budget(path, 100, json.loads(path.read_text()))
    resumed.activate("metr_la/gofa", 20)
    assert resumed.remaining() == 5
    clock[0] += 6
    with pytest.raises(BudgetExhausted):
        resumed.check()
    resumed.close()
    assert json.loads(path.read_text())["active_scope"] is None


@pytest.mark.skipif(not Path("external/hsdgnn/model/HSDGNN.py").exists(), reason="Pinned upstream not installed")
def test_official_hsdgnn_short_history_backward_and_parameter_match():
    torch.manual_seed(42)
    small = HSDGNNAdapter("external/hsdgnn", 8, 3, 3, 24, width=8, embed_dim=4)
    x = torch.randn(2, 8, 3)
    mask = torch.ones_like(x, dtype=torch.bool)
    output = small(x, mask, torch.tensor([8, 20]))
    assert output.shape == (2, 3, 3)
    output.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in small.parameters() if p.grad is not None)
    for nodes, length, horizon, steps in ((207, 12, 12, 288), (325, 12, 3, 288), (437, 24, 6, 24),
                                         (36, 12, 12, 24), (321, 8, 8, 24), (137, 24, 3, 144)):
        with torch.device("meta"):
            official = HSDGNNAdapter("external/hsdgnn", length, horizon, nodes, steps)
        assert abs(parameter_counts(official)["total"] / 21_335_938 - 1) < .10
