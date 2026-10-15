from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from gofa_f import routed_stage as stage
from gofa_f.prefix_cache import atomic_json
from gofa_f.routed_task import compatible_cache


def test_stage_config_keeps_registered_six_jobs_and_no_gpu_activity_cap():
    config = stage.config_file("configs/stage5_routed.yaml")
    assert config["architecture"]["graph_exchange"] == "prior_routed"
    assert config["resources"]["activity_limit_seconds"] is None
    assert config["resources"]["wall_window_hours"] == 12
    assert config["stage5"]["loss"] == "raw_mae"
    assert len(config["datasets"]) == 6
    source = Path("src/gofa_f/routed_stage.py").read_text()
    assert "ufg_python" not in source and "--lr" not in source


def test_cpu_stage_import_never_imports_torch():
    subprocess.run([sys.executable, "-c", "import sys; import gofa_f.routed_stage; assert 'torch' not in sys.modules"],
                   check=True)


def test_window_resume_keeps_initial_start_and_deadline(tmp_path):
    first = stage.bind_window(tmp_path, now=100.)
    assert first["wall_deadline"] == 43300.
    assert first["training_deadline"] == 42400.
    assert stage.bind_window(tmp_path, resume=True, now=99999.) == first
    with pytest.raises(ValueError):
        stage.bind_window(tmp_path, now=99999.)
    with pytest.raises(ValueError):
        stage.bind_window(tmp_path/"missing", resume=True)


def test_resume_binding_rejects_code_inputs_architecture_and_window_changes(tmp_path):
    expected = {"code": "fixed", "inputs": {"data": "a"}, "architecture": {"rank": 16}, "deadline": 100.}
    path = tmp_path/"manifest.json"
    stage.bind_manifest(path, expected, False)
    stage.bind_manifest(path, expected, True)
    for key, value in (("code", "new"), ("inputs", {}), ("architecture", {"rank": 32}), ("deadline", 101.)):
        wrong = dict(expected, **{key: value})
        with pytest.raises(ValueError):
            stage.bind_manifest(path, wrong, True)


def test_memory_records_keep_maximum_across_phases_and_interrupted_attempts(tmp_path):
    path = tmp_path/"memory.json"
    stage.record_memory(path, "train", 100, 150)
    stage.record_memory(path, "evaluate", 30, 50)
    stage.record_memory(path, "train", 90, 200)
    result = stage.read(path)
    assert result["peak_allocated"] == 100 and result["peak_reserved"] == 200
    assert result["phases"]["train"] == {"peak_allocated": 100, "peak_reserved": 200, "attempts": 2}
    assert result["phases"]["evaluate"]["peak_allocated"] == 30


def write_cache(path, key, nodes=3):
    path.mkdir()
    atomic_json(path/"metadata.json", {"complete": True, "key": key, "prefix_layers": 2})
    for name, shape, dtype in (("hidden", (4, nodes, 2, 8), np.uint16), ("mask", (4, nodes, 2), bool),
                               ("loc", (4, nodes, 1), np.float32), ("scale", (4, nodes, 1), np.float32)):
        np.save(path/(name+".npy"), np.zeros(shape, dtype=dtype))


def test_historical_cache_is_read_only_and_rejects_key_or_format_changes(tmp_path):
    key = {"context": 12, "revision": "fixed", "origins_hash": "a"}
    folder = tmp_path/"old"
    write_cache(folder, key)
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    cache = compatible_cache(folder, key, 4, 3, 8)
    assert cache is not None and not cache.hidden.flags.writeable
    with pytest.raises(ValueError):
        cache.upgrade(None, 64, lambda: None)
    with pytest.raises(ValueError):
        cache.close(delete=True)
    cache.close()
    assert before == {p.name: p.read_bytes() for p in folder.iterdir()}
    assert compatible_cache(folder, dict(key, origins_hash="other"), 4, 3, 8) is None
    assert compatible_cache(folder, key, 5, 3, 8) is None
    np.save(folder/"loc.npy", np.zeros((4, 3, 1), np.float64))
    assert compatible_cache(folder, key, 4, 3, 8) is None


def test_seal_excludes_jobs_without_complete_validation_and_never_reselects(tmp_path):
    folder = stage.job_directory(tmp_path, "metr_la")
    folder.mkdir(parents=True)
    (folder/"best.pt").write_bytes(b"best-validation")
    atomic_json(folder/"metadata.json", {"training_windows": 640, "parameters": {"trainable": 113286}})
    atomic_json(folder/"progress.json", {"epoch": 99, "offset": 640, "stale": 0, "best_mae": 3., "updates": 1000})
    result = stage.seal(tmp_path)
    assert set(result["jobs"]) == {"metr_la"}
    assert result["jobs"]["metr_la"]["training_complete"]
    assert result["jobs"]["metr_la"]["stop_reason"] == "max_epochs"
    (folder/"best.pt").write_bytes(b"changed-after-seal")
    assert stage.seal(tmp_path) == result  # Worker checks the immutable recorded hash.


def test_new_stage_runs_all_six_and_tests_only_after_global_seal(tmp_path, monkeypatch):
    config = stage.config_file("configs/stage5_routed.yaml")
    output = tmp_path/"run"
    output.mkdir()
    window = stage.bind_window(output, now=100.)
    expected = {"git_revision": "fixed"}
    atomic_json(output/"launch_binding.json", expected)
    monkeypatch.setattr(stage, "config_file", lambda _: config)
    monkeypatch.setattr(stage, "manifest", lambda *a: expected)
    monkeypatch.setattr(stage, "write_report", lambda *a: None)
    monkeypatch.setattr(stage.time, "time", lambda: 200.)
    monkeypatch.setattr(stage.signal, "signal", lambda *a: None)
    monkeypatch.setenv("GOFA_IDLE_ADMITTED", "1")
    monkeypatch.setenv("GOFA_SUPERVISOR_PID", str(stage.os.getppid()))
    monkeypatch.setenv("GOFA_GPU_UUID", "GPU-test")
    monkeypatch.setattr(stage, "query_gpu", lambda: {"uuid": "GPU-test", "pids": [], "utilization": 0,
                        "used_mib": 0, "compute_mode": "Default"})
    monkeypatch.setattr(stage.time, "sleep", lambda _: None)
    stage.STOP = False
    commands = []
    class Child:
        returncode = 0
        def __init__(self, command, env):
            commands.append(command)
            def option(name): return command[command.index(name)+1]
            phase, dataset = option("--phase"), option("--dataset")
            folder = stage.job_directory(output, dataset)
            folder.mkdir(parents=True, exist_ok=True)
            if phase == "preflight":
                (output/"mechanisms").mkdir(exist_ok=True)
                atomic_json(output/"mechanisms/mechanisms.json", {"passed": True})
            elif phase == "train":
                assert not (output/"final_selection.json").exists()
                (folder/"best.pt").write_bytes(dataset.encode())
                atomic_json(folder/"metadata.json", {"training_windows": 640, "parameters": {"trainable": 113286}})
                atomic_json(folder/"progress.json", {"epoch": 100, "offset": 0, "stale": 0,
                                                      "best_mae": 9999., "updates": 1000})
                atomic_json(folder/"done.json", {"complete": True})
            else:
                locked = stage.read(output/"final_selection.json")
                assert len(locked["jobs"]) == 6
                atomic_json(folder/("result.json" if phase == "evaluate" else "routing.json"), {"passed": True})
            expected_boundary = window["training_deadline"] if phase in ("preflight", "train") else window["wall_deadline"]
            assert float(env["GOFA_TASK_DEADLINE"]) <= stage.time.monotonic()+expected_boundary-200+1
        def poll(self): return 0
    monkeypatch.setattr(stage.subprocess, "Popen", Child)
    stage.stage("config", output)
    assert stage.read(output/"status.json")["state"] == "complete"
    assert len([c for c in commands if c[c.index("--phase")+1] == "train"]) == 6
    assert len(list(output.glob("trials/*/routed/result.json"))) == 6


def test_preflight_failure_stops_before_any_training(tmp_path, monkeypatch):
    config = stage.config_file("configs/stage5_routed.yaml")
    window = stage.bind_window(tmp_path, now=100.)
    atomic_json(tmp_path/"launch_binding.json", {"git_revision": "fixed"})
    monkeypatch.setattr(stage, "config_file", lambda _: config)
    monkeypatch.setattr(stage, "manifest", lambda *a: {"git_revision": "fixed"})
    monkeypatch.setattr(stage, "write_report", lambda *a: None)
    monkeypatch.setattr(stage.time, "time", lambda: window["started_at"]+100)
    monkeypatch.setattr(stage.signal, "signal", lambda *a: None)
    monkeypatch.setenv("GOFA_IDLE_ADMITTED", "1")
    monkeypatch.setenv("GOFA_SUPERVISOR_PID", str(stage.os.getppid()))
    monkeypatch.setenv("GOFA_GPU_UUID", "GPU-test")
    monkeypatch.setattr(stage, "query_gpu", lambda: {"uuid": "GPU-test", "pids": [], "utilization": 0,
                        "used_mib": 0, "compute_mode": "Default"})
    monkeypatch.setattr(stage.time, "sleep", lambda _: None)
    stage.STOP = False
    class Child:
        returncode = 1
        def __init__(self, command, env):
            assert command[command.index("--phase")+1] == "preflight"
        def poll(self): return 1
    monkeypatch.setattr(stage.subprocess, "Popen", Child)
    with pytest.raises(RuntimeError, match="mechanisms failed"):
        stage.stage("config", tmp_path)
    assert stage.read(tmp_path/"status.json")["state"] == "failed"


def test_completed_counts_last_epoch_and_early_stop_without_extra_validation():
    assert stage.completed({"epoch": 99, "offset": 640, "stale": 0}, 640) == (100, True)
    assert stage.completed({"epoch": 20, "offset": 64, "stale": 15}, 640) == (20, True)
    assert stage.completed({"epoch": 20, "offset": 64, "stale": 14}, 640) == (20, False)


def test_report_reads_archived_raw_metric_schema_and_leaves_new_results_incomplete(tmp_path):
    config = stage.config_file("configs/stage5_routed.yaml")
    stage.write_report(tmp_path, config)
    text = (tmp_path/"report.md").read_text()
    assert "|metr_la|4.38221|3.66821|incomplete|" in text
    assert "State: not_started" in text
    assert all(stage.read(tmp_path/"report.json")["results"][d] is None for d in stage.DATASETS)
