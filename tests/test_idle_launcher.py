import json
import subprocess
import sys

import pytest

from gofa_f import idle_launcher as idle


def xml(pids=(), used=0, utilization=0):
    processes = "".join(f"<process_info><pid>{p}</pid><type>G</type></process_info>" for p in pids)
    return f"""<nvidia_smi_log><gpu><uuid>GPU-test</uuid><compute_mode>Default</compute_mode>
    <fb_memory_usage><used>{used} MiB</used></fb_memory_usage>
    <utilization><gpu_util>{utilization} %</gpu_util></utilization>
    <processes>{processes}</processes></gpu></nvidia_smi_log>"""


def test_idle_launcher_import_never_loads_torch():
    result = subprocess.run(
        [sys.executable, "-c", 'import gofa_f.idle_launcher,sys; assert "torch" not in sys.modules'],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_three_idle_samples_and_busy_or_error_resets_gate():
    gate = idle.IdleGate()
    busy = idle.parse_gpu_xml(xml([999]))
    free = idle.parse_gpu_xml(xml())
    for _ in range(8):
        assert not gate.observe(busy)
    assert not gate.observe(free)
    assert not gate.observe(free)
    assert not gate.observe(None)
    assert not gate.observe(free)
    assert not gate.observe(free)
    assert gate.observe(free)
    assert not gate.observe(idle.parse_gpu_xml(xml(used=257)))
    assert not gate.observe(idle.parse_gpu_xml(xml(utilization=1)))
    assert not gate.observe(idle.parse_gpu_xml(xml().replace("Default", "Exclusive_Process")))


def test_strict_idle_launcher_observes_every_ten_seconds(tmp_path, monkeypatch):
    clock, queries, signals = [0.], [], []
    output = tmp_path/"run"
    monkeypatch.setattr(idle.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(idle.time, "sleep", lambda n: clock.__setitem__(0, clock[0]+n))
    monkeypatch.setattr(idle, "process_identity", lambda _: "start")
    monkeypatch.setattr(idle.subprocess, "check_output", lambda *a, **k: "fixed")
    monkeypatch.setattr(idle.os, "killpg", lambda *a: signals.append(a))
    def query():
        queries.append(clock[0])
        return idle.parse_gpu_xml(xml())
    monkeypatch.setattr(idle, "query_gpu", query)
    class Child:
        pid = 800
        returncode = 0
        def __init__(self, command, env, start_new_session):
            assert queries == [0., 10., 20.]
            idle.atomic_json(output/"status.json", {"state": "complete"})
        def poll(self):
            return 0
    monkeypatch.setattr(idle.subprocess, "Popen", Child)
    idle.launch(output, ["worker"], total=None, wall_deadline=10**15, memory_cap_gib=40,
                yield_on_foreign=False, idle_interval_seconds=10)
    record = json.loads((output/"launcher_manifest.json").read_text())
    assert record["idle_interval_seconds"] == 10 and record["idle_samples"] == 3
    assert record["idle_max_used_mib"] == 256 and not signals


def test_shared_admission_accepts_active_server_with_full_budget_and_margin():
    gpu = {"uuid": "GPU-test", "pids": [999], "utilization": 100, "compute_mode": "Default",
           "used_mib": 20837, "total_mib": 81559, "free_mib": 60253}
    gate = idle.SharedGate(40, 16)
    assert not gate.observe(gpu)
    assert not gate.observe(gpu)
    assert gate.observe(gpu)
    assert not gate.observe({**gpu, "free_mib": 56*1024-1})
    assert not gate.observe(gpu)
    assert not gate.observe(None)
    assert not gate.observe(gpu)
    assert not gate.observe(gpu)
    assert gate.observe(gpu)


def test_cap_only_admission_is_immediate_despite_busy_gpu_or_low_free_memory():
    gpu = {"uuid": "GPU-test", "pids": [999], "utilization": 100, "compute_mode": "Default",
           "used_mib": 80000, "total_mib": 81559, "free_mib": 1500}
    gate = idle.CapOnlyGate()
    assert gate.observe(gpu) and gate.consecutive == 1
    assert not gate.observe(None)
    assert gate.observe(gpu)
    assert not gate.observe({**gpu, "compute_mode": "Exclusive_Process"})
    assert not gate.observe({**gpu, "uuid": "GPU-changed"})


def test_cap_only_launcher_has_no_idle_samples_or_free_memory_delay(tmp_path, monkeypatch):
    clock, signals = [0.], []
    output = tmp_path/"run"
    monkeypatch.setattr(idle.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(idle.time, "sleep", lambda n: clock.__setitem__(0, clock[0]+n))
    monkeypatch.setattr(idle, "process_identity", lambda _: "start")
    monkeypatch.setattr(idle, "own_gpu_pids", lambda pid: {pid})
    monkeypatch.setattr(idle.subprocess, "check_output", lambda *a, **k: "fixed")
    monkeypatch.setattr(idle.os, "killpg", lambda *a: signals.append(a))
    gpu = {"uuid": "GPU-test", "pids": [999], "utilization": 100, "compute_mode": "Default",
           "used_mib": 80000, "total_mib": 81559, "free_mib": 1500, "process_used_mib": {999: 80000}}
    monkeypatch.setattr(idle, "query_gpu", lambda: gpu)
    monkeypatch.setattr(idle, "IdleGate", lambda: pytest.fail("Cap-only mode must not use idle admission"))
    monkeypatch.setattr(idle, "SharedGate", lambda *a: pytest.fail("Cap-only mode must not reserve56GiB"))

    class Child:
        pid = 800
        returncode = None

        def __init__(self, command, env, start_new_session):
            assert clock[0] == 0 and start_new_session and env["CUDA_VISIBLE_DEVICES"] == "GPU-test"
            self.polls = 0

        def poll(self):
            self.polls += 1
            if self.polls > 3:
                self.returncode = 0
                idle.atomic_json(output/"status.json", {"state": "complete"})
            return self.returncode

    monkeypatch.setattr(idle.subprocess, "Popen", Child)
    idle.launch(output, ["worker"], total=None, wall_deadline=10**15, memory_cap_gib=40,
                leave_free_gib=0, yield_on_foreign=False, cap_only_admission=True)
    record = json.loads((output/"launcher_manifest.json").read_text())
    assert record["admission_policy"] == "memory_cap_only" and record["idle_samples"] == 1
    assert record["memory_cap_gib"] == 40 and record["leave_free_gib"] == 0
    assert "idle_max_used_mib" not in record and "admission_min_free_mib" not in record and not signals


@pytest.mark.parametrize("change", [{"memory_cap_gib": None}, {"memory_cap_gib": 41}, {"leave_free_gib": 16},
                                   {"yield_on_foreign": True}, {"shared_admission": True}])
def test_cap_only_policy_requires_cap_and_cannot_restore_other_waiting_policies(tmp_path, change):
    settings = {"total": None, "wall_deadline": 10**15, "memory_cap_gib": 40,
                "leave_free_gib": 0, "yield_on_foreign": False, "cap_only_admission": True}
    with pytest.raises(ValueError):
        idle.launch(tmp_path, ["worker"], **{**settings, **change})


@pytest.mark.parametrize("change", [{"free_mib": None}, {"total_mib": None},
    {"free_mib": 56000}, {"free_mib": 90000}, {"compute_mode": "Exclusive_Process"}])
def test_shared_admission_fails_closed_for_unknown_or_insufficient_actual_free(change):
    # Driver reservations mean total-used may exceed actually usable free memory.
    gpu = {"uuid": "GPU-test", "pids": [999], "utilization": 80, "compute_mode": "Default",
           "used_mib": 20837, "total_mib": 81559, "free_mib": 60253}
    gate = idle.SharedGate(40, 16)
    for _ in range(5):
        assert not gate.observe({**gpu, **change})


def test_shared_admission_launcher_coexists_without_signalling_other_processes(tmp_path, monkeypatch):
    clock, children, signals = [0.], [], []
    output = tmp_path/"run"
    monkeypatch.setattr(idle.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(idle.time, "sleep", lambda n: clock.__setitem__(0, clock[0]+n))
    monkeypatch.setattr(idle, "process_identity", lambda _: "start")
    monkeypatch.setattr(idle, "own_gpu_pids", lambda pid: {pid})
    monkeypatch.setattr(idle.subprocess, "check_output", lambda *a, **k: "fixed")
    monkeypatch.setattr(idle.os, "killpg", lambda *a: signals.append(a))
    gpu = {"uuid": "GPU-test", "pids": [999], "utilization": 80, "compute_mode": "Default",
           "used_mib": 20837, "total_mib": 81559, "free_mib": 60253, "process_used_mib": {}}
    monkeypatch.setattr(idle, "query_gpu", lambda: gpu)
    class Child:
        pid = 800
        returncode = None
        def __init__(self, command, env, start_new_session):
            assert clock[0] >= 60 and env["GOFA_IDLE_ADMITTED"] == "1"
            children.append(self)
            self.polls = 0
        def poll(self):
            self.polls += 1
            if self.polls > 3:
                self.returncode = 0
                idle.atomic_json(output/"status.json", {"state": "complete"})
            return self.returncode
    monkeypatch.setattr(idle.subprocess, "Popen", Child)
    idle.launch(output, ["worker"], total=None, wall_deadline=10**15, memory_cap_gib=40,
                leave_free_gib=16, yield_on_foreign=False, shared_admission=True)
    record = json.loads((output/"launcher_manifest.json").read_text())
    assert record["admission_policy"] == "shared_memory" and record["admission_min_free_mib"] == 56*1024
    assert len(children) == 1 and not signals and "idle_max_used_mib" not in record


@pytest.mark.parametrize(
    "text",
    [
        xml().replace("0 %", "N/A"),
        xml().replace("<processes></processes>", ""),
        xml().replace("GPU-test", "N/A"),
        xml().replace("0 MiB", "N/A"),
    ],
)
def test_unknown_gpu_state_fails_closed(text):
    with pytest.raises(ValueError):
        idle.parse_gpu_xml(text)


def test_foreign_process_including_graphics_requires_yield(monkeypatch):
    monkeypatch.setattr(idle, "own_gpu_pids", lambda _: {100, 101})
    assert not idle.foreign_task(idle.parse_gpu_xml(xml([100, 101])), 100, "GPU-test")
    assert idle.foreign_task(idle.parse_gpu_xml(xml([100, 200])), 100, "GPU-test")
    assert idle.foreign_task(None, 100, "GPU-test")


def test_waiting_is_not_charged_and_resume_keeps_activity(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(idle.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(idle, "process_identity", lambda _: "start")
    ledger = idle.ActivityLedger(tmp_path / "active.json", 14400)
    clock[0] += 1000
    ledger.tick()
    assert ledger.state["spent_seconds"] == 0
    ledger.start(123)
    clock[0] += 12
    ledger.stop()
    clock[0] += 1000
    ledger.tick()
    resumed = idle.ActivityLedger(ledger.path, 14400, json.loads(ledger.path.read_text()))
    assert resumed.remaining == 14388
    resumed.start(124)
    clock[0] += 8
    resumed.stop()
    assert resumed.state["spent_seconds"] == 20


def test_cpu_wait_inside_orchestrator_does_not_consume_activity(tmp_path, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(idle.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(idle, "process_identity", lambda _: "start")
    ledger = idle.ActivityLedger(tmp_path/"ledger.json", None)
    ledger.start(123)
    clock[0] += 10
    ledger.tick(charge=True)
    clock[0] += 600
    ledger.tick(charge=False)
    clock[0] += 100
    ledger.stop()
    assert ledger.state["spent_seconds"] == 10


@pytest.mark.parametrize("query_fails", [False, True])
def test_admission_only_policy_keeps_running_when_foreign_task_joins(tmp_path, monkeypatch, query_fails):
    clock, children, signals = [0.], [], []
    output = tmp_path/"run"
    monkeypatch.setattr(idle.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(idle.time, "sleep", lambda n: clock.__setitem__(0, clock[0]+n))
    monkeypatch.setattr(idle, "process_identity", lambda _: "start")
    monkeypatch.setattr(idle, "own_gpu_pids", lambda pid: {pid})
    monkeypatch.setattr(idle.subprocess, "check_output", lambda *a, **k: "fixed")
    monkeypatch.setattr(idle.os, "killpg", lambda *a: signals.append(a))
    class Child:
        pid = 800
        returncode = None
        def __init__(self, *a, **k):
            children.append(self)
            self.polls = 0
        def poll(self):
            self.polls += 1
            if self.polls > 15:
                self.returncode = 0
                idle.atomic_json(output/"status.json", {"state": "complete"})
            return self.returncode
    def query():
        if children and query_fails:
            raise RuntimeError("query unavailable during admitted task")
        return idle.parse_gpu_xml(xml([800, 999], used=75000, utilization=90) if children else xml())
    monkeypatch.setattr(idle, "query_gpu", query)
    monkeypatch.setattr(idle.subprocess, "Popen", Child)
    idle.launch(output, ["worker"], total=None, wall_deadline=10**15, memory_cap_gib=40,
                leave_free_gib=16, yield_on_foreign=False, worker_wait_state="waiting_for_idle")
    assert len(children) == 1 and not signals
    assert idle.read_status(output)["state"] == "complete"


def test_admission_only_cap_counts_our_group_without_other_users(monkeypatch):
    monkeypatch.setattr(idle, "own_gpu_pids", lambda _: {100, 101})
    gpu = {"pids": [100, 101, 999], "process_used_mib": {100: 1024, 101: 1024, 999: 70000}}
    assert not idle.own_cap_breach(gpu, 100, 40)
    gpu["process_used_mib"][101] = 40*1024
    assert idle.own_cap_breach(gpu, 100, 40)
    assert not idle.own_cap_breach(None, 100, 40)


@pytest.mark.parametrize("query_fails", [False, True])
def test_watchdog_saves_yields_waits_and_resumes_own_worker_only(tmp_path, monkeypatch, query_fails):
    clock = [0.0]
    monkeypatch.setattr(idle.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(idle.time, "sleep", lambda n: clock.__setitem__(0, clock[0] + n))
    monkeypatch.setattr(idle, "process_identity", lambda _: "start")
    monkeypatch.setattr(idle, "own_gpu_pids", lambda pid: {pid})
    output = tmp_path / "run"
    children = []
    signals = []
    queries = [0]

    class Child:
        def __init__(self, command, env, start_new_session):
            assert start_new_session and env["GOFA_IDLE_ADMITTED"] == "1"
            self.pid = 800 + len(children)
            self.returncode = None
            self.queries = 0
            self.started = clock[0]
            self.command = command
            children.append(self)

        def poll(self):
            self.queries += 1
            if len(children) == 2 and self.queries > 2:
                self.returncode = 0
                idle.atomic_json(output / "status.json", {"state": "complete"})
            return self.returncode

    def query():
        queries[0] += 1
        if len(children) == 1 and children[0].returncode is None:
            if query_fails:
                raise RuntimeError("nvidia-smi temporarily unavailable")
            return idle.parse_gpu_xml(xml([children[0].pid, 999]))
        return idle.parse_gpu_xml(xml())

    def killpg(pid, signum):
        signals.append((pid, signum))
        assert pid == 800
        children[0].returncode = 75
        idle.atomic_json(output / "manifest.json", {})
        idle.atomic_json(output / "status.json", {"state": "interrupted"})

    monkeypatch.setattr(idle, "query_gpu", query)
    monkeypatch.setattr(idle.subprocess, "Popen", Child)
    monkeypatch.setattr(idle.subprocess, "check_output", lambda *a, **k: "test-revision")
    monkeypatch.setattr(idle.os, "killpg", killpg)
    idle.launch(output, ["worker", "--output", str(output)])
    assert len(children) == 2 and signals == [(800, idle.signal.SIGTERM)]
    assert children[0].started >= 60 and children[1].started - children[0].started >= 60
    assert children[1].command[-1] == "--resume"
    assert json.loads((output / "launcher.json").read_text())["state"] == "complete"
    assert json.loads((output / "active_budget.json").read_text())["spent_seconds"] < 10


def test_stuck_worker_is_killed_only_after_own_sixty_second_grace(tmp_path, monkeypatch):
    clock = [0.0]
    children = []
    signals = []
    output = tmp_path / "run"
    monkeypatch.setattr(idle.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(idle.time, "sleep", lambda n: clock.__setitem__(0, clock[0] + n))
    monkeypatch.setattr(idle, "process_identity", lambda _: "start")
    monkeypatch.setattr(idle, "own_gpu_pids", lambda p: {p})
    monkeypatch.setattr(idle.subprocess, "check_output", lambda *a, **k: "test-revision")

    class Child:
        pid = 800
        returncode = None

        def __init__(self, *a, **k):
            children.append(self)

        def poll(self):
            return self.returncode

    def query():
        return idle.parse_gpu_xml(xml([800, 999] if children else []))

    def kill(pid, signum):
        assert pid == 800
        signals.append((signum, clock[0]))
        if signum == idle.signal.SIGKILL:
            children[0].returncode = -9
            idle.atomic_json(output / "status.json", {"state": "incomplete"})

    monkeypatch.setattr(idle, "query_gpu", query)
    monkeypatch.setattr(idle.subprocess, "Popen", Child)
    monkeypatch.setattr(idle.os, "killpg", kill)
    idle.launch(output, ["worker"])
    assert [v[0] for v in signals] == [idle.signal.SIGTERM, idle.signal.SIGKILL]
    assert signals[1][1] - signals[0][1] >= 60


def test_exhausted_activity_budget_never_launches_gpu_worker(tmp_path, monkeypatch):
    output = tmp_path / "run"
    output.mkdir()
    idle.atomic_json(output / "active_budget.json", {"spent_seconds": 14400.0, "worker": None})
    monkeypatch.setattr(idle.subprocess, "check_output", lambda *a, **k: "test-revision")
    monkeypatch.setattr(idle, "query_gpu", lambda: pytest.fail("No query or worker after budget exhaustion"))
    idle.launch(output, ["worker"])
    assert json.loads((output / "launcher.json").read_text())["state"] == "budget_exhausted"


def test_unlimited_activity_requires_wall_deadline(tmp_path):
    with pytest.raises(ValueError, match="deadline"):
        idle.launch(tmp_path / "run", ["worker"], total=None)


def test_wall_deadline_ends_busy_wait_without_loading_worker(tmp_path, monkeypatch):
    clock = [100.]
    monkeypatch.setattr(idle.time, "time", lambda: clock[0])
    monkeypatch.setattr(idle.time, "sleep", lambda n: clock.__setitem__(0, clock[0]+n))
    monkeypatch.setattr(idle.subprocess, "check_output", lambda *a, **k: "test-revision")
    monkeypatch.setattr(idle, "query_gpu", lambda: idle.parse_gpu_xml(xml([999])))
    monkeypatch.setattr(idle.subprocess, "Popen", lambda *a, **k: pytest.fail("Busy GPU must not launch"))
    output = tmp_path / "run"
    idle.launch(output, ["worker"], total=None, wall_deadline=200)
    assert clock[0] <= 200
    assert json.loads((output / "launcher.json").read_text())["state"] == "wall_deadline"
    assert json.loads((output / "active_budget.json").read_text())["spent_seconds"] == 0


def test_deadline_timezone_and_unlimited_activity_ledger(tmp_path):
    assert idle.parse_deadline("2026-10-15T11:16:00+08:00") == idle.parse_deadline("2026-10-15T03:16:00Z")
    with pytest.raises(ValueError, match="timezone"):
        idle.parse_deadline("2026-10-15T11:16:00")
    ledger = idle.ActivityLedger(tmp_path / "activity.json", None, {"spent_seconds": 999999., "worker": None})
    assert ledger.remaining == float("inf")


def test_wall_deadline_saves_then_kills_only_own_stuck_worker(tmp_path, monkeypatch):
    clock, signals, children = [0.], [], []
    output = tmp_path / "run"
    monkeypatch.setattr(idle.time, "time", lambda: clock[0])
    monkeypatch.setattr(idle.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(idle.time, "sleep", lambda n: clock.__setitem__(0, clock[0]+n))
    monkeypatch.setattr(idle, "process_identity", lambda _: "start")
    monkeypatch.setattr(idle, "query_gpu", lambda: idle.parse_gpu_xml(xml()))
    monkeypatch.setattr(idle, "own_gpu_pids", lambda _: {800})
    monkeypatch.setattr(idle.subprocess, "check_output", lambda *a, **k: "test-revision")
    class Child:
        pid, returncode = 800, None
        def __init__(self, *a, **k):
            children.append(self)
        def poll(self):
            return self.returncode
    def kill(pid, signum):
        assert pid == 800
        signals.append((signum, clock[0]))
        if signum == idle.signal.SIGKILL:
            children[0].returncode = -9
    monkeypatch.setattr(idle.subprocess, "Popen", Child)
    monkeypatch.setattr(idle.os, "killpg", kill)
    idle.launch(output, ["worker"], total=None, wall_deadline=200)
    assert signals == [(idle.signal.SIGTERM, 140.), (idle.signal.SIGKILL, 200.)]
    assert json.loads((output / "launcher.json").read_text())["state"] == "wall_deadline"


def test_nvml_counts_non_torch_allocations_and_unknown_memory_fails_closed(monkeypatch):
    monkeypatch.setattr(idle, "own_gpu_pids", lambda _: {100, 101})
    gpu = idle.parse_gpu_xml(xml([101], used=35000).replace(
        "<used>35000 MiB</used>", "<used>35000 MiB</used><total>81000 MiB</total>").replace(
        "<pid>101</pid>", "<pid>101</pid><used_memory>35000 MiB</used_memory>"))
    assert not idle.resource_breach(gpu, 100, 40, 16)
    gpu["process_used_mib"][101] = 42000
    assert idle.resource_breach(gpu, 100, 40, 16)
    gpu["process_used_mib"][101] = None
    assert idle.resource_breach(gpu, 100, 40, 16)
    gpu["process_used_mib"][101] = 35000
    gpu["used_mib"] = 70000
    assert idle.resource_breach(gpu, 100, 40, 16)
