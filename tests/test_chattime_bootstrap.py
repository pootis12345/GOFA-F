import importlib.util
import json
from types import SimpleNamespace

import pytest


@pytest.fixture
def module():
    spec = importlib.util.spec_from_file_location("stage16_bootstrap", "scripts/bootstrap_chattime_stage.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_failed_cpu_preparation_never_starts_waiter(module, tmp_path):
    output = tmp_path/"out"
    calls = []

    def run(command, **kwargs):
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        calls.append(command)
        return SimpleNamespace(returncode=1)

    def launch(*args, **kwargs):
        raise AssertionError("GPU waiter must not launch on a preparation failure")

    with pytest.raises(RuntimeError):
        module.bootstrap(tmp_path, output, tmp_path/"official.json", run=run, popen=launch)
    assert len(calls) == 1 and not (output/"window.json").exists()
    assert json.loads((output/"bootstrap.json").read_text())["state"] == "failed"


def test_real_cpu_pass_is_required_before_clock(module, tmp_path):
    output = tmp_path/"out"
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if len(calls) == 2:
            (output/"cpu_preflight.json").write_text('{"passed": false}')
        return SimpleNamespace(returncode=0)

    def launch(*args, **kwargs):
        raise AssertionError("Failed CPU preflight must not launch")

    with pytest.raises(RuntimeError):
        module.bootstrap(tmp_path, output, tmp_path/"official.json", run=run, popen=launch)
    assert len(calls) == 2 and not (output/"window.json").exists()


@pytest.mark.parametrize("shortened", [False, True])
def test_successful_preparation_launches_only_registered_waiter(module, tmp_path, shortened):
    output = tmp_path/"out"
    calls, launched = [], []

    def run(command, **kwargs):
        calls.append(command)
        if "check_chattime_cpu.py" in str(command):
            (output/"cpu_preflight.json").write_text('{"passed": true}')
        if "--report-only" in command:
            (output/"report.json").write_text('{"state": "incomplete"}')
        return SimpleNamespace(returncode=0)

    class Child:
        pid = 99999999

        def wait(self):
            return 0

    def launch(command, **kwargs):
        assert (output/"cpu_preflight.json").exists()
        assert "gofa_f.chattime_stage" in command and kwargs["start_new_session"]
        launched.append(command)
        return Child()

    kwargs = {"hard_deadline": "2026-10-16T23:30:00+08:00",
              "adopt_waiting_window": tmp_path/"previous", "shared_gpu": True} if shortened else {}
    module.bootstrap(tmp_path, output, tmp_path/"official.json", run=run, popen=launch, **kwargs)
    assert len(launched) == 1 and not (output/"window.json").exists()
    # The bootstrap never creates the timer; the admitted stage supervisor does.
    assert json.loads((output/"bootstrap.json").read_text())["state"] == "incomplete"
    if shortened:
        assert "--shared-gpu" in launched[0]
        assert launched[0][launched[0].index("--hard-deadline")+1] == kwargs["hard_deadline"]
        assert launched[0][launched[0].index("--adopt-waiting-window")+1] == str(kwargs["adopt_waiting_window"])
