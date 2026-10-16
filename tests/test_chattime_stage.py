import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from gofa_f.chattime import HistoryCodec
from gofa_f.chattime_stage import bind, bind_window, config_file, digest, seal
from gofa_f.chattime_task import (apply_validation, available_metric, initial_state,
    load_window, native_window, optimizer_update, summarize_split, write_window)
from gofa_f.metrics import Metrics
from gofa_f.native_stage import StopRequested
from test_chattime import Tokenizer


def test_window_includes_wait_and_never_extends(tmp_path):
    first = bind_window(tmp_path, now=1000.)
    assert first["training_deadline"] == 1000+9*3600
    assert first["wall_deadline"] == 1000+11*3600
    assert bind_window(tmp_path, resume=True, now=99999.) == first
    with pytest.raises(ValueError):
        bind_window(tmp_path)


def test_resume_binding_rejects_changes(tmp_path):
    path = tmp_path/"manifest.json"
    first = {"weights": "abc", "deadline": 11, "config": {"rank": 64}}
    bind(path, first, False)
    bind(path, first, True)
    changed = copy.deepcopy(first)
    changed["config"]["rank"] = 8
    with pytest.raises(ValueError):
        bind(path, changed, True)


@pytest.mark.parametrize("field,value", [("seed", 43), ("datasets", ["electricity"]),
                                        ("models", ["native_chattime"])])
def test_no_unregistered_candidates(tmp_path, field, value):
    import yaml
    config = config_file("configs/stage16_chattime.yaml")
    config[field] = value
    path = tmp_path/"bad.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        config_file(path)


def test_validation_commit_is_idempotent():
    state = initial_state()
    record = {"key": "epoch1-update2", "best_mae": 3., "best_updates": 2, "stale": 1, "next_epoch": 1}
    apply_validation(state, record)
    first = copy.deepcopy(state)
    apply_validation(state, record)
    assert state == first and state["stale"] == 1


def test_unequal_missing_counts_accumulate_as_global_mean():
    torch.manual_seed(42)
    first = nn.Linear(2, 1)
    second = copy.deepcopy(first)
    optimizers = [torch.optim.AdamW(m.parameters(), lr=.001, weight_decay=.0001) for m in (first, second)]
    x = torch.arange(10, dtype=torch.float32).reshape(5, 2)/10
    y = torch.ones(5, 1)
    mask = torch.tensor([[1], [0], [1], [1], [1]], dtype=torch.bool)
    state = initial_state()
    for sl in (slice(0, 2), slice(2, 5)):
        error = (first(x[sl])-y[sl]).masked_select(mask[sl])
        error.abs().sum().backward()
        state["pending_count"] += int(mask[sl].sum())
        state["pending_loss"] += float(error.detach().abs().sum())
        state["pending_windows"] += 1
    optimizer_update(first, optimizers[0], state)
    (second(x)-y).masked_select(mask).abs().mean().backward()
    torch.nn.utils.clip_grad_norm_(second.parameters(), 1., foreach=False)
    optimizers[1].step()
    for a, b in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-7)


def test_no_valid_generation_is_not_fabricated():
    metric = Metrics(12)
    assert available_metric(metric) is None
    prediction = torch.ones(1, 12, 2)
    mask = torch.ones_like(prediction, dtype=torch.bool)
    mask[:, 0] = False
    metric.update(prediction, torch.zeros_like(prediction), mask)
    result = available_metric(metric)
    assert result["by_horizon"][0]["mae"] is None and result["mape_pct"] is None


class Data:
    def __init__(self):
        self.windows = {"test": type("Windows", (), {"origins": [12, 13], "__len__": lambda self: 2})()}

    def batch(self, split, indices):
        values = torch.zeros(1, 12, 2)
        values[:, :, 1] = 2
        return {"y_raw": values, "y_mask": torch.ones_like(values, dtype=torch.bool)}


def test_partial_evaluation_excluded_and_raw_zeros_retained(tmp_path):
    data = Data()
    pred, valid = np.ones((1, 12, 2), np.float32), np.zeros((1, 12, 2), bool)
    write_window(tmp_path, "native_chattime", "test", 0, 12, pred, valid, "manifest")
    progress, result = summarize_split(tmp_path, "native_chattime", "test", data, "manifest")
    assert not progress["complete"] and result is None
    write_window(tmp_path, "native_chattime", "test", 1, 13, pred, valid, "manifest")
    _, result = summarize_split(tmp_path, "native_chattime", "test", data, "manifest")
    assert result["raw"]["mae"] == 1 and result["raw"]["mape_valid_fraction"] == .5
    assert result["valid_generation"] is None and result["observed_fallback_fraction"] == 1


def test_shard_resume_validates_binding_and_origin(tmp_path):
    write_window(tmp_path, "gofa_chattime", "test", 0, 12, np.ones((1, 12, 2)), np.ones((1, 12, 2), bool), "a")
    path = tmp_path/"gofa_chattime/predictions/test/000000.npz"
    with pytest.raises(ValueError):
        load_window(path, 12, "b")
    with pytest.raises(ValueError):
        load_window(path, 13, "a")


class Generator:
    def generate(self, input_ids, attention_mask, stopping_criteria, **kwargs):
        for criterion in stopping_criteria:
            criterion(input_ids, None)
        return torch.cat((input_ids, torch.ones(len(input_ids), 2, dtype=torch.long)), 1)


def test_native_resume_inside_window(tmp_path):
    codec = HistoryCodec(Tokenizer())
    raw = np.arange(36, dtype=np.float32).reshape(1, 12, 3)
    encoded = codec.encode(raw, np.ones_like(raw, bool), np.zeros(3), padding_side="left")
    calls = [0]

    def stop():
        calls[0] += 1
        if calls[0] == 3:
            raise StopRequested()

    with pytest.raises(StopRequested):
        native_window(Generator(), codec, encoded, tmp_path, 0, stop, node_micro=1)
    partial = torch.load(tmp_path/"partial.pt", weights_only=False)
    assert partial["cursor"] == 1
    resumed = native_window(Generator(), codec, encoded, tmp_path, 0, lambda: None)
    clean = tmp_path/"clean"
    clean.mkdir()
    fresh = native_window(Generator(), codec, encoded, clean, 0, lambda: None, node_micro=1)
    for a, b in zip(resumed, fresh):
        np.testing.assert_array_equal(a, b)


def test_seal_rejects_changed_checkpoint(tmp_path):
    from gofa_f.prefix_cache import atomic_json
    directory = tmp_path/"gofa_chattime"
    directory.mkdir()
    path = directory/"best.pt"
    path.write_bytes(b"original")
    atomic_json(directory/"best_validation.json", {"checkpoint_sha256": digest(path), "validation": {"raw": {"mae": 3.}}})
    path.write_bytes(b"changed")
    with pytest.raises(ValueError):
        seal(tmp_path)


def test_cpu_supervisor_does_not_import_torch():
    # Structural invariant: the waiting launch path has no module-scope torch dependency.
    import ast
    tree = ast.parse(Path("src/gofa_f/chattime_stage.py").read_text())
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert all(not getattr(n, "module", "").startswith("torch") for n in imports if isinstance(n, ast.ImportFrom))
    assert all(a.name != "torch" for n in imports if isinstance(n, ast.Import) for a in n.names)


@pytest.mark.parametrize("mechanism_failure", [False, True])
def test_gpu_phase_order_and_test_seal(tmp_path, monkeypatch, mechanism_failure):
    import os
    import time
    from gofa_f import chattime_stage as stage
    from gofa_f.prefix_cache import atomic_json
    config = config_file("configs/stage16_chattime.yaml")
    window = {"wall_deadline": time.time()+4000, "training_deadline": time.time()+2000}
    expected = {"window": window, "inputs": {}}
    atomic_json(tmp_path/"window.json", window)
    atomic_json(tmp_path/"launch_binding.json", expected)
    atomic_json(tmp_path/"manifest.json", expected)
    monkeypatch.setenv("GOFA_IDLE_ADMITTED", "1")
    monkeypatch.setenv("GOFA_SUPERVISOR_PID", str(os.getppid()))
    monkeypatch.setenv("GOFA_GPU_UUID", "test")
    monkeypatch.setattr(stage, "manifest", lambda *args: expected)
    monkeypatch.setattr(stage, "query_gpu", lambda: {"uuid": "test"})

    class Gate:
        consecutive = 3

        def observe(self, gpu):
            return True

    monkeypatch.setattr(stage, "IdleGate", Gate)
    phases = []

    class Process:
        def __init__(self, command, env):
            phase = command[-1]
            phases.append(phase)
            self.returncode = 1 if phase == "preflight" and mechanism_failure else 0
            if phase == "preflight" and not mechanism_failure:
                atomic_json(tmp_path/"mechanisms.json", {"passed": True})
            if phase == "train":
                folder = tmp_path/"gofa_chattime"
                folder.mkdir()
                (folder/"best.pt").write_bytes(b"registered")
                atomic_json(folder/"best_validation.json", {"validation": {"raw": {"mae": 3.}},
                    "checkpoint_sha256": digest(folder/"best.pt")})
            if phase == "evaluate":
                assert stage.read(tmp_path/"final_selection.json")["gofa"]["checkpoint_sha256"]
                for name in config["models"]:
                    folder = tmp_path/name
                    folder.mkdir(exist_ok=True)
                    atomic_json(folder/"result.json", {"complete": True})

        def poll(self):
            return self.returncode

    monkeypatch.setattr(stage.subprocess, "Popen", Process)
    monkeypatch.setattr(stage, "STOP", False)
    if mechanism_failure:
        with pytest.raises(RuntimeError):
            stage.stage("configs/stage16_chattime.yaml", tmp_path, True)
        assert phases == ["preflight"]
    else:
        stage.stage("configs/stage16_chattime.yaml", tmp_path, True)
        assert phases == ["preflight", "train", "evaluate"]
        assert stage.read(tmp_path/"status.json")["state"] == "complete"
