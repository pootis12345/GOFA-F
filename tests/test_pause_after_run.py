import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys

import pytest


@pytest.fixture
def pause_module():
    spec = importlib.util.spec_from_file_location("pause_after_run", Path("scripts/pause_after_run.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pause_requires_full_matching_result(tmp_path, pause_module):
    path = tmp_path / "result.json"
    expected = {"dataset": "tiny", "model_name": "gofa_chronos", "seed": 42, "horizon": 12}
    assert pause_module.complete_result(path, expected, 123) is None
    path.write_text("{")
    assert pause_module.complete_result(path, expected, 123) is None
    result = {"status": "complete", "config": expected, "environment": {"pid": 123}}
    path.write_text(json.dumps(result))
    assert len(pause_module.complete_result(path, expected, 123)) == 64
    with pytest.raises(ValueError, match="different runner"):
        pause_module.complete_result(path, expected, 456)
    result["status"] = "smoke"
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="full completed"):
        pause_module.complete_result(path, expected, 123)


def test_runner_must_match_own_stage(tmp_path, pause_module):
    import os

    identity = {"uid": os.getuid(), "pid": os.getpid() + 1000, "cwd": str(tmp_path),
                "argv": [b"python", b"-m", b"gofa_f.benchmark", b"--output", b"stage"]}
    pause_module.validate_runner(identity, tmp_path / "stage")
    with pytest.raises(ValueError, match="output"):
        pause_module.validate_runner(identity, tmp_path / "other-stage")
    identity["uid"] += 1
    with pytest.raises(ValueError, match="own stage"):
        pause_module.validate_runner(identity, tmp_path / "stage")


def test_stopping_failure_restores_process(pause_module, monkeypatch):
    calls = []
    monkeypatch.setattr(pause_module, "same_process", lambda identity: True)
    monkeypatch.setattr(pause_module.os, "kill", lambda pid, sig: calls.append((pid, sig)))
    with pytest.raises(RuntimeError):
        with pause_module.stopped_runner({"pid": 123}):
            raise RuntimeError("simulated failure")
    assert calls == [(123, signal.SIGSTOP), (123, signal.SIGCONT)]


def test_completed_stage_runner_is_terminated_without_force_kill(pause_module):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        identity = pause_module.process_info(child.pid)
        pause_module.stop_stage(identity)
        assert child.wait(timeout=5) == -signal.SIGTERM
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=5)
