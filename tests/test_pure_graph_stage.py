import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gofa_f.native_stage import Data, clip_gradients, fit, seed, trained_state
from gofa_f.pure_graph_stage import config_file, select_lr, task_directory
from gofa_f.stage_budget import Budget


def test_stage4_has_no_global_or_per_job_gpu_activity_cap():
    config = config_file("configs/stage4_pure_graph.yaml")
    assert config["budget"]["total_gpu_seconds"] is None
    assert config["budget"]["per_job_seconds"] is None
    assert "matched_hsdgnn" not in config["models"]
    assert config["models"]["gofa_bolt"]["graph_layers"] == [1, 2]
    assert config["stage4"]["wall_deadline"] == "2026-10-15T11:16:00+08:00"


def test_lr_selection_requires_both_key_datasets_and_never_accepts_test_scores():
    scores = {"metr_la": {"0.001": 3., "0.003": 2.},
              "electricity": {"0.001": 100., "0.003": 180.}}
    reference = {"metr_la": 4., "electricity": 200.}
    assert select_lr(scores, reference)["lr"] == .001
    scores["electricity"]["0.001"] = None
    assert select_lr(scores, reference)["lr"] == .003
    scores["electricity"]["0.003"] = float("nan")
    assert select_lr(scores, reference) is None
    assert select_lr({"test": scores}, reference) is None


def test_lr_selection_unit_invariance_and_lower_lr_tie_break():
    scores = {d: {"0.001": 3., "0.003": 3.00001} for d in ("metr_la", "electricity")}
    assert select_lr(scores, {d: 4. for d in scores})["lr"] == .001
    scores["electricity"] = {k: v*1000 for k, v in scores["electricity"].items()}
    assert select_lr(scores, {"metr_la": 4., "electricity": 4000.})["lr"] == .001


def test_task_directories_preserve_selected_lr_checkpoint():
    assert str(task_directory("out", "electricity", "ufg_s", .003)).replace("\\", "/") == \
        "out/trials/electricity/ufg_s/lr_0.003"


def test_values_only_context_and_training_exclude_test_target_arrays(tmp_path):
    values = np.arange(640, dtype=np.float32).reshape(320, 2)
    np.savez(tmp_path / "toy.npz", values=values, mask=np.ones_like(values, bool))
    data = Data({"data": str(tmp_path / "toy.npz"), "context": 8, "horizons": [3],
                 "validation_only": True, "graph_context": {"enabled": True, "feature_mode": "values_only"}}, "cpu")
    assert set(data.windows) == {"train", "val"}
    assert len(data.raw_values) == data.series.val_end
    batch = data.batch("train", [0, 1])
    assert batch["graph_features"].shape == (2, 2, 5)
    with pytest.raises(KeyError):
        data.batch("test", [0])


def test_complex_gradient_clipping_counts_both_components():
    parameter = torch.nn.Parameter(torch.tensor([3.+4.j]))
    parameter.grad = torch.tensor([3.+4.j])
    assert clip_gradients([parameter], 1).item() == pytest.approx(5)
    assert parameter.grad.abs().item() == pytest.approx(1, abs=1e-6)
    parameter.grad[:] = complex(float("inf"), 0)
    with pytest.raises(FloatingPointError):
        clip_gradients([parameter], 1)


class ComplexToy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(3, dtype=torch.cfloat))

    def forward(self, x, mask):
        value = x[:, -1] * (self.weight.real + self.weight.imag*.3)
        return value[:, None].expand(-1, 3, -1)


def test_complex_rmsprop_resume_and_loss_protocol_are_bound(tmp_path):
    values = np.random.default_rng(3).normal(size=(160, 3)).astype(np.float32)
    observed = np.random.default_rng(4).random(values.shape) > .3
    np.savez(tmp_path / "toy.npz", values=values, mask=observed)
    data = Data({"data": str(tmp_path / "toy.npz"), "context": 8, "horizons": [3],
                 "validation_only": True}, "cpu")
    budget = Budget(tmp_path / "budget.json", 100)
    budget.activate("test", 100)
    gpu = SimpleNamespace(micro=16, check=lambda b: b.check())
    kwargs = {"optimizer_kind": "rmsprop", "loss_kind": "standardized_mse", "max_epochs": 100,
              "patience": 15}
    seed(42)
    resumed = ComplexToy()
    fit(resumed, data, tmp_path / "resumed", gpu, budget, .001, max_updates=2, **kwargs)
    fit(resumed, data, tmp_path / "resumed", gpu, budget, .001, max_updates=3, **kwargs)
    history = (tmp_path / "resumed/history.jsonl").read_text()
    fit(resumed, data, tmp_path / "resumed", gpu, budget, .001, max_updates=3, **kwargs)
    assert history == (tmp_path / "resumed/history.jsonl").read_text()
    seed(42)
    continuous = ComplexToy()
    fit(continuous, data, tmp_path / "continuous", gpu, budget, .001, max_updates=3, **kwargs)
    for name, value in trained_state(continuous).items():
        torch.testing.assert_close(value, trained_state(resumed)[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="protocol"):
        fit(resumed, data, tmp_path / "resumed", gpu, budget, .001, optimizer_kind="rmsprop")
    progress = json.loads((tmp_path / "resumed/progress.json").read_text())
    assert progress["training_protocol"]["loss"] == "standardized_mse"


def test_missing_label_accumulation_matches_full_complex_effective_batch(tmp_path):
    from gofa_f.native_stage import update
    values = np.random.default_rng(3).normal(size=(160, 3)).astype(np.float32)
    observed = np.random.default_rng(4).random(values.shape) > .5
    np.savez(tmp_path / "toy.npz", values=values, mask=observed)
    data = Data({"data": str(tmp_path / "toy.npz"), "context": 8, "horizons": [3]}, "cpu")
    budget = Budget(tmp_path / "budget.json", 100)
    budget.activate("test", 100)
    models = []
    for micro in (64, 4):
        seed(42)
        model = ComplexToy()
        optimizer = torch.optim.RMSprop(model.parameters(), lr=.001)
        update(model, optimizer, data, np.arange(64), SimpleNamespace(micro=micro, check=lambda b: b.check()),
               budget, None, loss_kind="standardized_mse")
        models.append(model)
    torch.testing.assert_close(models[0].weight, models[1].weight, rtol=1e-5, atol=1e-6)


def test_stage_orchestration_locks_before_test_and_keeps_all_domains_when_gnn_wins(tmp_path, monkeypatch):
    from gofa_f import pure_graph_stage as stage
    from gofa_f.idle_launcher import atomic_json, parse_deadline
    config = stage.config_file("configs/stage4_pure_graph.yaml")
    reference, output = tmp_path / "reference", tmp_path / "output"
    output.mkdir()
    config["stage4"]["reference_output"] = str(reference)
    for dataset in stage.KEY_DATASETS:
        path = reference / "reevaluation" / dataset / "persistence/result.json"
        path.parent.mkdir(parents=True)
        atomic_json(path, {"validation": {"raw": {"mae": 4.}}})
    monkeypatch.setattr(stage, "config_file", lambda _: config)
    monkeypatch.setattr(stage, "manifest", lambda _: {"git_revision": "fixed"})
    monkeypatch.setattr(stage.signal, "signal", lambda *a: None)
    monkeypatch.setattr(stage.time, "time", lambda: parse_deadline(config["stage4"]["wall_deadline"])-10000)
    monkeypatch.setenv("GOFA_IDLE_ADMITTED", "1")
    monkeypatch.setenv("GOFA_SUPERVISOR_PID", str(stage.os.getppid()))
    stage.STOP = False
    commands = []
    class Child:
        returncode = 0
        def __init__(self, command, env):
            commands.append(command)
            def option(name): return command[command.index(name)+1]
            phase, dataset, variant = option("--phase"), option("--dataset"), option("--variant")
            lr = float(option("--lr"))
            directory = stage.task_directory(output, dataset, variant, lr)
            directory.mkdir(parents=True, exist_ok=True)
            if phase == "preflight":
                p = output / "mechanisms" / variant
                p.mkdir(parents=True, exist_ok=True)
                atomic_json(p / "mechanisms.json", {"passed": True})
                atomic_json(p / "values_only.json", {"passed": True})
            elif phase == "pilot":
                atomic_json(directory / "pilot_done.json", {"validation_raw_mae": 1. if lr==.001 else 2.})
            elif phase == "train":
                # Pure GNN wins. This must NOT prevent any extension or replace GOFA architecture.
                value = 1. if variant.startswith("ufg") else 3.
                (directory / "best.pt").write_bytes(b"fixed-best")
                atomic_json(directory / "progress.json", {"epoch": 100, "offset": 0, "stale": 0,
                                                           "best_mae": value, "updates": 1000})
                atomic_json(directory / "done.json", {"dataset": dataset, "variant": variant,
                    "complete": True, "epochs": 100, "updates": 1000, "stop_reason": "max_epochs",
                    "parameters": {"registered_real_dof": 100, "active_real_dof": 90,
                                   "trainable_real_dof": 90}})
            elif phase == "diagnose":
                assert (output / "final_selection.json").exists()
                atomic_json(directory / "diagnostics.json", {"splits": {"train": {}, "val": {}}})
            elif phase == "evaluate":
                assert (output / "final_selection.json").exists()
                atomic_json(directory / "result.json", {"dataset": dataset, "variant": variant,
                    "training_complete": True, "test": {"raw": {"mae": 1., "rmse": 2., "mape_pct": 3.,
                    "mape_valid_fraction": 1., "wape_pct": 4.}}})
        def poll(self): return self.returncode
    monkeypatch.setattr(stage.subprocess, "Popen", Child)
    stage.stage("config", output)
    assert json.loads((output / "status.json").read_text())["state"] == "complete"
    assert len(list((output / "trials").rglob("result.json"))) == 21
    assert all(entry["lr"] == .001 for entry in json.loads((output / "selection.json").read_text()).values())
    phases = [c[c.index("--phase")+1] for c in commands]
    assert phases.index("evaluate") > max(i for i, phase in enumerate(phases) if phase == "train")


def test_electricity_diagnostic_groups_never_depend_on_test_labels(tmp_path):
    from gofa_f.pure_graph_task import diagnostics
    from gofa_f.models import Persistence
    values = np.random.default_rng(1).normal(size=(160, 5)).astype(np.float32)
    values *= np.array([1, 2, 3, 4, 100])[None]
    values[::4] = 0
    observed = np.ones_like(values, bool)
    observed[::3, 0] = False
    settings = {"context": 8, "horizons": [3], "validation_only": True}
    budget = Budget(tmp_path / "budget.json", 100)
    budget.activate("test", 100)
    gpu = SimpleNamespace(micro=16, check=lambda b: b.check())
    results = []
    for version in (0, 1):
        changed = values.copy()
        if version:
            changed[128:] += 1e6
        path = tmp_path / f"toy_{version}.npz"
        np.savez(path, values=changed, mask=observed)
        data = Data(dict(settings, data=str(path)), "cpu")
        results.append(diagnostics(Persistence(3), data, gpu, budget))
    assert results[0] == results[1]
    assert set(results[0]["splits"]) == {"train", "val"}
    assert results[0]["high_scale_nodes"] == [4]
    assert results[0]["splits"]["val"]["categories"]["zero"]["observations"] > 0


@pytest.mark.parametrize("self_only", [True, False])
def test_values_only_self_control_isolates_neighbor_messages(self_only):
    from test_native import random_bolt
    from gofa_f.native import NativeBolt
    from gofa_f.graph_context import node_features
    seed(42)
    adjacency = np.eye(3) if self_only else np.ones((3, 3))
    model = NativeBolt(random_bolt(), adjacency, 3, (1, 2), rank=4,
                       graph_feature_dim=5, output_adapter_rank=4).eval()
    for layer in model.graph.values():
        layer.gate.data.fill_(.2)
        layer.context_projection.weight.data.normal_(0, .1)
    x = torch.randn(2, 12, 3)
    observed = torch.ones_like(x, dtype=torch.bool)
    def features(value):
        return node_features(value, observed, torch.zeros(3), torch.ones(3), 0., 1.,
                             torch.zeros(2, 6))[..., :5]
    with torch.no_grad():
        baseline = model(x, observed, features(x))
        shifted = x.clone()
        shifted[:, :, 1] += 100
        changed = model(shifted, observed, features(shifted))
    if self_only:
        torch.testing.assert_close(baseline[:, :, 0], changed[:, :, 0], rtol=0, atol=0)
    else:
        assert (baseline[:, :, 0]-changed[:, :, 0]).abs().max() > 1e-6
