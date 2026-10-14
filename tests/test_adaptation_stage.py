import copy
import json

import numpy as np
import pytest
import torch

from gofa_f.adaptation_stage import make_model, select_variant, validate_config
from gofa_f.native_stage import stage_config
from test_native import random_bolt


def test_new_stage_is_strictly_late_and_four_hours():
    config = stage_config("configs/stage3_late_context.yaml")
    validate_config(config)
    wrong = copy.deepcopy(config)
    wrong["models"]["gofa_bolt"]["layout_candidates"] = {"spaced": [0, 2]}
    with pytest.raises(ValueError):
        validate_config(wrong)


def test_shared_graph_initialization_survives_context_and_head_additions():
    native = random_bolt()
    states = []
    for name in ["G0", "G1", "G2"]:
        torch.manual_seed(42)
        model = make_model(native, np.eye(3), 3, name)
        states.append(
            {
                n: p.detach().clone()
                for n, p in model.named_parameters()
                if n.startswith("graph.") and "context_projection" not in n
            }
        )
    for name in states[0]:
        assert torch.equal(states[0][name], states[1][name])
        assert torch.equal(states[0][name], states[2][name])


def test_validation_gate_rejects_missing_regression_and_head_only_advantage():
    values = {
        "metr_la": {"G0": 4.0, "G1": 3.8, "G2": 3.5, "B1": 3.4},
        "electricity": {"G0": 400.0, "G1": 380.0, "G2": 350.0, "B1": 390.0},
    }
    assert select_variant(values)["selected"] == "G1"
    values["metr_la"]["B1"] = 3.6
    assert select_variant(values)["selected"] == "G2"
    values["electricity"]["G1"] = None
    values["electricity"]["G2"] = 401.0
    assert select_variant(values)["selected"] is None


def test_selection_is_unit_invariant_and_ties_favor_smaller_adapter():
    values = {
        "metr_la": {"G0": 4.0, "G1": 3.8, "G2": 3.8, "B1": 3.9},
        "electricity": {"G0": 400.0, "G1": 380.0, "G2": 380.0, "B1": 390.0},
    }
    assert select_variant(values)["selected"] == "G1"
    for key in values["electricity"]:
        values["electricity"][key] *= 1000
    assert select_variant(values)["selected"] == "G1"


def test_resume_rejects_clock_or_adapter_or_optimizer_changes(tmp_path, monkeypatch):
    from gofa_f.adaptation_stage import bind_manifest

    monkeypatch.setattr("gofa_f.adaptation_stage.subprocess.check_output", lambda *a, **k: b"")

    manifest = {"context_inputs": {"time": "a"}, "config": {"rank": 4, "adapter_lr": 0.0001}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    bind_manifest(tmp_path, manifest, True)
    for key, value in [("rank", 0), ("adapter_lr", 0.001)]:
        wrong = copy.deepcopy(manifest)
        wrong["config"][key] = value
        with pytest.raises(ValueError):
            bind_manifest(tmp_path, wrong, True)
    wrong = copy.deepcopy(manifest)
    wrong["context_inputs"]["time"] = "b"
    with pytest.raises(ValueError):
        bind_manifest(tmp_path, wrong, True)


def test_worker_budget_exits_when_supervisor_disappears(tmp_path, monkeypatch):
    from gofa_f.adaptation_stage import ActiveBudget
    from gofa_f.native_stage import StopRequested

    monkeypatch.setenv("GOFA_SUPERVISOR_PID", "123")
    monkeypatch.setattr("gofa_f.adaptation_stage.os.getppid", lambda: 456)
    budget = ActiveBudget(tmp_path / "budget.json", 14400)
    budget.activate("test", 900)
    with pytest.raises(StopRequested):
        budget.check()


def test_worker_honors_parent_absolute_budget_with_save_reserve(tmp_path, monkeypatch):
    from gofa_f.adaptation_stage import ActiveBudget
    from gofa_f.stage_budget import BudgetExhausted

    clock = [100.0]
    monkeypatch.setattr("gofa_f.adaptation_stage.time.monotonic", lambda: clock[0])
    monkeypatch.delenv("GOFA_SUPERVISOR_PID", raising=False)
    monkeypatch.setenv("GOFA_ACTIVE_DEADLINE", "200")
    budget = ActiveBudget(tmp_path / "budget.json", 14400)
    budget.activate("test", 900)
    assert budget.remaining() == 40
    clock[0] = 141
    with pytest.raises(BudgetExhausted):
        budget.check()


def test_evaluation_uses_raw_labels_for_mape_even_if_normalized_label_roundtrips_wrong():
    from types import SimpleNamespace
    from gofa_f.native_stage import evaluate
    from gofa_f.models import Persistence

    batch = {
        "x": torch.zeros(1, 1, 1),
        "x_mask": torch.ones(1, 1, 1, dtype=torch.bool),
        "y": torch.zeros(1, 1, 1),
        "y_raw": torch.zeros(1, 1, 1),
        "y_mask": torch.ones(1, 1, 1, dtype=torch.bool),
    }
    data = SimpleNamespace(
        horizon=1, windows={"val": [0]}, mean=torch.tensor([2e-5]), std=torch.ones(1), batch=lambda *_: batch
    )
    result = evaluate(Persistence(1), data, "val", SimpleNamespace(micro=1, check=lambda _: None), None)
    assert result["raw"]["mape_pct"] is None
    assert result["raw"]["mape_observations"] == 0
    assert result["raw"]["mae"] == pytest.approx(2e-5)


def test_optimizer_groups_and_resume_preserve_updates_without_duplicate_validation(tmp_path):
    from types import SimpleNamespace
    from gofa_f.native_stage import Data, fit, seed, trained_state
    from gofa_f.stage_budget import Budget

    values = np.random.default_rng(1).normal(size=(320, 3)).astype(np.float32)
    path = tmp_path / "data.npz"
    np.savez(path, values=values, mask=np.ones_like(values, bool))
    data = Data({"data": str(path), "context": 12, "horizons": [3]}, "cpu")
    budget = Budget(tmp_path / "budget.json", 100)
    budget.activate("test", 100)
    gpu = SimpleNamespace(micro=64, check=lambda b: b.check())

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.graph = torch.nn.Linear(3, 3)
            self.output_adapter = torch.nn.Linear(3, 3)

        def forward(self, x, mask):
            h = self.graph(x[:, -1])
            return (h + self.output_adapter(h))[:, None, :].expand(-1, 3, -1)

    seed(42)
    model = Toy()
    fit(model, data, tmp_path / "resumed", gpu, budget, 0.001, max_updates=2, adapter_lr=0.0001)
    before = (tmp_path / "resumed/history.jsonl").read_text().splitlines()
    fit(model, data, tmp_path / "resumed", gpu, budget, 0.001, max_updates=2, adapter_lr=0.0001)
    assert (tmp_path / "resumed/history.jsonl").read_text().splitlines() == before
    fit(model, data, tmp_path / "resumed", gpu, budget, 0.001, max_updates=3, adapter_lr=0.0001)
    seed(42)
    continuous = Toy()
    fit(continuous, data, tmp_path / "continuous", gpu, budget, 0.001, max_updates=3, adapter_lr=0.0001)
    for name, value in trained_state(continuous).items():
        torch.testing.assert_close(value, trained_state(model)[name], rtol=0, atol=0)
