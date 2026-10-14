"""CPU-only GPU admission/watchdog. This module deliberately never imports torch.

The launcher owns only the isolated process group that it creates. Waiting between
workers is not charged; the full lifetime of every worker, including saving, is.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import xml.etree.ElementTree as ET


TERMINAL = {"complete", "no_eligible_variant", "budget_exhausted", "incomplete", "failed", "wall_deadline"}


def parse_deadline(value):
    if value is None:
        return None
    instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if instant.tzinfo is None:
        raise ValueError("Wall deadline must have an explicit timezone")
    return instant.astimezone(timezone.utc).timestamp()


def wall_remaining(deadline):
    return float("inf") if deadline is None else deadline - time.time()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def integer_field(text):
    match = re.fullmatch(r"\s*(\d+)\s*(?:MiB|%)?\s*", text or "")
    if match is None:
        raise ValueError("Unavailable nvidia-smi field")
    return int(match[1])


def parse_gpu_xml(text):
    root = ET.fromstring(text)
    gpus = root.findall("gpu")
    if len(gpus) != 1:
        raise ValueError("Expected exactly one selected GPU")
    gpu = gpus[0]
    processes = gpu.find("processes")
    if processes is None:
        raise ValueError("GPU process information unavailable")
    pids, process_memory = [], {}
    for process in processes.findall("process_info"):
        pid = integer_field(process.findtext("pid"))
        if pid <= 0:
            raise ValueError("Invalid GPU process identity")
        pids.append(pid)
        try:
            process_memory[pid] = integer_field(process.findtext("used_memory"))
        except ValueError:
            process_memory[pid] = None
    uuid = gpu.findtext("uuid")
    if not uuid or not uuid.startswith("GPU-"):
        raise ValueError("GPU UUID unavailable")
    return {
        "uuid": uuid,
        "pids": pids,
        "compute_mode": gpu.findtext("compute_mode"),
        "used_mib": integer_field(gpu.findtext("fb_memory_usage/used")),
        "total_mib": integer_field(gpu.findtext("fb_memory_usage/total"))
            if gpu.findtext("fb_memory_usage/total") else None,
        "process_used_mib": process_memory,
        "utilization": integer_field(gpu.findtext("utilization/gpu_util")),
    }


def query_gpu():
    result = subprocess.run(["nvidia-smi", "-q", "-x", "-i", "0"], capture_output=True, text=True, timeout=10)
    if result.returncode:
        raise RuntimeError("nvidia-smi failed: " + result.stderr.strip())
    return parse_gpu_xml(result.stdout)


class IdleGate:
    def __init__(self):
        self.consecutive = 0
        self.uuid = None

    def observe(self, gpu):
        if gpu is None or (self.uuid is not None and gpu["uuid"] != self.uuid):
            self.consecutive = 0
            return False
        self.uuid = gpu["uuid"]
        idle = (
            not gpu["pids"]
            and gpu["utilization"] == 0
            and gpu["used_mib"] <= 256
            and gpu["compute_mode"] == "Default"
        )
        self.consecutive = self.consecutive + 1 if idle else 0
        return self.consecutive >= 3


def process_identity(pid):
    """Linux start ticks avoid treating a recycled PID as one of our workers."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat[stat.rfind(")") + 2 :].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def own_gpu_pids(worker_pid):
    # Include only descendants currently in our isolated process group.
    owned = {worker_pid}
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                pid = int(entry.name)
                if os.getpgid(pid) == worker_pid:
                    owned.add(pid)
            except (ProcessLookupError, PermissionError):
                pass
    return owned


def foreign_task(gpu, worker_pid, uuid):
    return (
        gpu is None
        or gpu["uuid"] != uuid
        or gpu["compute_mode"] != "Default"
        or bool(set(gpu["pids"]) - own_gpu_pids(worker_pid))
    )


def resource_breach(gpu, worker_pid, cap_gib, reserve_gib):
    if cap_gib is None:
        return False
    if gpu is None or gpu.get("total_mib") is None:
        return True
    memory = gpu.get("process_used_mib", {})
    owned = set(gpu["pids"]) & own_gpu_pids(worker_pid)
    if any(memory.get(pid) is None for pid in owned):
        return True
    return sum(memory[pid] for pid in owned) > cap_gib*1024 or \
        gpu["total_mib"] - gpu["used_mib"] < reserve_gib*1024


class ActivityLedger:
    def __init__(self, path, total, state=None):
        self.path, self.total = Path(path), total
        self.state = state or {"spent_seconds": 0.0, "worker": None}
        self.last = None

    def start(self, pid):
        self.last = time.monotonic()
        self.state["worker"] = {"pid": pid, "start_ticks": process_identity(pid), "last_monotonic": self.last}
        self.save()

    def tick(self):
        now = time.monotonic()
        if self.last is not None:
            self.state["spent_seconds"] += max(0, now - self.last)
            self.state["worker"]["last_monotonic"] = now
            self.last = now
        self.save()

    def stop(self):
        self.tick()
        self.last = None
        self.state["worker"] = None
        self.save()

    def save(self):
        atomic_json(self.path, self.state)

    @property
    def remaining(self):
        return float("inf") if self.total is None else self.total - self.state["spent_seconds"]


def read_status(output):
    path = output / "status.json"
    return json.loads(path.read_text()) if path.exists() else {}


def launch(output, command, total=14400, wall_deadline=None, memory_cap_gib=None, leave_free_gib=16):
    if total is None and wall_deadline is None:
        raise ValueError("Unlimited activity still requires an absolute wall deadline")
    if memory_cap_gib is not None and (memory_cap_gib > 40 or memory_cap_gib < 1 or leave_free_gib < 16):
        raise ValueError("Unsafe shared memory policy")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Hold both locks even while waiting, preventing duplicate project launchers.
    locks = []
    for path in (output.with_suffix(".launcher.lock"), output.parent / "native-stage.lock"):
        stream = path.open("a")
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(stream)
    manifest = {
        "command": command,
        "cwd": str(Path.cwd()),
        "total_seconds": total,
        **({"wall_deadline": wall_deadline} if wall_deadline is not None else {}),
        **({"memory_cap_gib": memory_cap_gib, "leave_free_gib": leave_free_gib}
           if memory_cap_gib is not None else {}),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "idle_samples": 3,
        "idle_interval_seconds": 30,
        "watchdog_interval_seconds": 10,
        "idle_max_used_mib": 256,
    }
    manifest_path = output / "launcher_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Launcher resume requires the identical command, snapshot and policy")
    atomic_json(manifest_path, manifest)
    ledger_path = output / "active_budget.json"
    ledger = ActivityLedger(
        ledger_path, total, json.loads(ledger_path.read_text()) if ledger_path.exists() else None
    )
    ledger.save()
    stop = [False]
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.__setitem__(0, True))

    def status(state, **fields):
        atomic_json(
            output / "launcher.json",
            {
                "state": state,
                "pid": os.getpid(),
                "spent_seconds": ledger.state["spent_seconds"],
                "updated_at": time.time(),
                **fields,
            },
        )
        print(json.dumps({"state": state, **fields}), flush=True)

    # A restarted supervisor never launches alongside its orphaned worker.
    orphan = ledger.state.get("worker")
    if orphan:
        pid = orphan["pid"]
        identity = process_identity(pid)
        if identity is not None and identity == orphan["start_ticks"]:
            if os.getpgid(pid) != pid or Path(f"/proc/{pid}/cwd").resolve() != Path.cwd().resolve():
                raise RuntimeError("Refusing to signal a worker with unverified process ownership")
            ledger.last = orphan["last_monotonic"]
            os.killpg(pid, signal.SIGTERM)
            deadline = time.monotonic() + min(60, max(0, wall_remaining(wall_deadline)))
            while process_identity(pid) == identity and time.monotonic() < deadline:
                ledger.tick()
                time.sleep(1)
            if process_identity(pid) == identity:
                os.killpg(pid, signal.SIGKILL)
            ledger.stop()
        else:
            worker_budget = output / "budget.json"
            if worker_budget.exists():
                ledger.state["spent_seconds"] = max(
                    ledger.state["spent_seconds"], json.loads(worker_budget.read_text())["spent_seconds"]
                )
            ledger.state["spent_seconds"] += 5  # Conservative unrecorded crash tail.
            ledger.state["worker"] = None
            ledger.save()
    gate = IdleGate()
    while not stop[0]:
        worker_status = read_status(output).get("state")
        if worker_status in TERMINAL:
            status(worker_status)
            return
        if wall_remaining(wall_deadline) <= 60:
            status("wall_deadline")
            return
        if ledger.remaining <= 60:
            status("budget_exhausted")
            return
        try:
            gpu = query_gpu()
            error = None
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired, ET.ParseError) as exc:
            gpu, error = None, str(exc)
        if not gate.observe(gpu):
            status("waiting_for_idle", consecutive_idle=gate.consecutive, query_error=error)
            for _ in range(30):
                if stop[0]:
                    break
                if wall_remaining(wall_deadline) <= 60:
                    break
                time.sleep(1)
            continue
        if wall_remaining(wall_deadline) <= 60:
            status("wall_deadline")
            return
        environment = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=gate.uuid,
            GOFA_IDLE_ADMITTED="1",
            GOFA_GPU_UUID=gate.uuid,
            GOFA_SUPERVISOR_PID=str(os.getpid()),
            GOFA_ACTIVE_DEADLINE=str(time.monotonic() + min(ledger.remaining, wall_remaining(wall_deadline))),
        )
        child_command = command + (["--resume"] if (output / "manifest.json").exists() else [])
        child = subprocess.Popen(child_command, env=environment, start_new_session=True)
        ledger.start(child.pid)
        status("running", worker_pid=child.pid)
        next_probe, stop_deadline, reason = time.monotonic(), None, None
        while child.poll() is None:
            ledger.tick()
            now = time.monotonic()
            if stop_deadline is None:
                if stop[0] or ledger.remaining <= 60 or wall_remaining(wall_deadline) <= 60:
                    reason = "user_stop" if stop[0] else "wall_deadline" \
                        if wall_remaining(wall_deadline) <= 60 else "budget_exhausted"
                elif now >= next_probe:
                    try:
                        gpu = query_gpu()
                    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired, ET.ParseError):
                        gpu = None
                    if foreign_task(gpu, child.pid, gate.uuid):
                        reason = "yielded_for_other_users"
                    elif resource_breach(gpu, child.pid, memory_cap_gib, leave_free_gib):
                        reason = "yielded_for_resource_margin"
                    next_probe = now + 10
                if reason:
                    status(reason, worker_pid=child.pid)
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    stop_deadline = now + min(60, max(0, ledger.remaining), max(0, wall_remaining(wall_deadline)))
            elif now >= stop_deadline:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            time.sleep(1)
        ledger.stop()
        result = read_status(output).get("state")
        if result in TERMINAL:
            status(result)
            return
        if reason in ("budget_exhausted", "wall_deadline"):
            status(reason)
            return
        if child.returncode not in (0, 75) and reason is None:
            status("failed", worker_returncode=child.returncode)
            return
        gate = IdleGate()
    status("stopped_by_user")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--total-seconds", type=int, default=14400)
    parser.add_argument("--unlimited-active", action="store_true")
    parser.add_argument("--wall-deadline")
    parser.add_argument("--memory-cap-gib", type=int)
    parser.add_argument("--leave-free-gib", type=int, default=16)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or (not args.unlimited_active and args.total_seconds != 14400):
        raise ValueError("Provide a worker command and an authorized budget policy")
    launch(args.output, command, None if args.unlimited_active else args.total_seconds,
           parse_deadline(args.wall_deadline), args.memory_cap_gib, args.leave_free_gib)


if __name__ == "__main__":
    main()
