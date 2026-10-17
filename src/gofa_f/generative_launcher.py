"""Stage17 CPU-only strict-idle supervisor with recorded, unlimited GPU activity.

Legacy launchers and their time budgets are deliberately unchanged.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from gofa_f.idle_launcher import (ActivityLedger, IdleGate, atomic_json, own_gpu_pids,
                                  process_identity, query_gpu)


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def own_usage(gpu, worker, uuid=None):
    if gpu is None or (uuid is not None and gpu["uuid"] != uuid) \
            or gpu.get("compute_mode", "Default") != "Default":
        return None
    pids = set(gpu["pids"]) & own_gpu_pids(worker)
    values = [gpu["process_used_mib"].get(pid) for pid in pids]
    return None if any(v is None for v in values) else sum(values)


def supervise(output, config, snapshot):
    import yaml
    output, config, snapshot = Path(output).resolve(), Path(config).resolve(), Path(snapshot).resolve()
    cfg = yaml.safe_load(config.read_text())
    if cfg["training"]["max_updates"] != 4000 or cfg["resources"]["activity_limit_seconds"] is not None \
            or cfg["resources"]["wall_deadline"] is not None or cfg["data"]["target_evaluation_enabled"]:
        raise ValueError("Stage17 requires bounded updates, unlimited activity and disabled target evaluation")
    output.mkdir(parents=True, exist_ok=True)
    locks = []
    for path in (output/"launcher.lock", output.parent/"stage17-gpu.lock"):
        stream = path.open("a")
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(stream)
    binding = {"snapshot": str(snapshot), "config": str(config), "admission_policy": "strict_idle",
        "idle_check_seconds": 10, "consecutive_idle_samples": 3, "idle_max_used_mib": 256,
        "memory_cap_gib": 40, "torch_cap_gib": 38, "yield_on_foreign": False,
        "activity_limit_seconds": None, "wall_deadline": None, "maximum_updates": 4000}
    if read(output/"launcher_manifest.json", binding) != binding:
        raise ValueError("Launcher recovery policy differs")
    atomic_json(output/"launcher_manifest.json", binding)
    previous = read(output/"activity.json", {})
    old = previous.get("worker") or {}
    if old.get("pid") and process_identity(old["pid"]) == old.get("start_ticks"):
        raise RuntimeError("Previous verified GPU worker still exists")
    # Account an unclean supervisor tail without zeroing previously saved activity.
    if old:
        tail = min(10., max(0., time.monotonic()-old.get("last_monotonic", time.monotonic())))
        previous["spent_seconds"] = previous.get("spent_seconds", 0.)+tail
        previous["conservative_unclean_tail_seconds"] = previous.get("conservative_unclean_tail_seconds", 0.)+tail
        previous["worker"] = None
    ledger = ActivityLedger(output/"activity.json", None, previous or None)
    ledger.save()
    stopped = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))

    def status(state, **fields):
        atomic_json(output/"launcher.json", {"state": state, "pid": os.getpid(),
            "start_ticks": process_identity(os.getpid()), "spent_seconds": ledger.state["spent_seconds"],
            "updated_at": time.time(), **fields})

    gate = IdleGate()
    while not stopped[0]:
        final = read(output/"status.json", {}).get("state")
        if final in ("complete", "failed", "incomplete"):
            status(final)
            return
        try:
            gpu, error = query_gpu(), None
        except Exception as exc:
            gpu, error = None, str(exc)
        admitted = gate.observe(gpu)
        status("waiting_for_idle", consecutive_idle=gate.consecutive, query_error=error)
        if not admitted:
            for _ in range(10):
                if stopped[0]:
                    break
                time.sleep(1)
            continue
        environment = dict(os.environ, PYTHONPATH=str(snapshot/"src"), CUDA_VISIBLE_DEVICES=gate.uuid,
            OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            PYTHONDONTWRITEBYTECODE="1", GOFA_STAGE17_ADMITTED="1", TOKENIZERS_PARALLELISM="false",
            GOFA_STAGE17_SUPERVISOR_PID=str(os.getpid()),
            GOFA_STAGE17_SUPERVISOR_TICKS=str(process_identity(os.getpid())))
        command = [sys.executable, "-u", "-m", "gofa_f.generative_stage", "--config", str(config),
                   "--output", str(output), "--train"]
        if (output/"last.pt").exists():
            command.append("--resume")
        child = subprocess.Popen(command, cwd=snapshot, env=environment, start_new_session=True)
        tick = process_identity(child.pid)
        ledger.start(child.pid)
        status("running", worker_pid=child.pid, worker_start_ticks=tick)
        stop_at, reason, next_check = None, None, time.monotonic()
        peak = previous.get("peak_group_mib", 0)
        while child.poll() is None:
            ledger.tick()
            now = time.monotonic()
            if stop_at is None and (stopped[0] or now >= next_check):
                if stopped[0]:
                    reason = "stopped_by_user"
                else:
                    try:
                        measured = query_gpu()
                        used = own_usage(measured, child.pid, gate.uuid)
                    except Exception:
                        used = None
                    if used is None or used > 40*1024:
                        reason = "resource_paused"
                    else:
                        peak = max(peak, used)
                        ledger.state["peak_group_mib"] = peak
                        status("running", worker_pid=child.pid, group_used_mib=used, peak_group_mib=peak)
                    next_check = now+10
                if reason:
                    status(reason, worker_pid=child.pid)
                    if process_identity(child.pid) == tick:
                        os.killpg(child.pid, signal.SIGTERM)
                    stop_at = now+60
            elif stop_at is not None and now >= stop_at:
                if process_identity(child.pid) == tick:
                    os.killpg(child.pid, signal.SIGKILL)
                status("failed", error="Worker failed to save within 60 seconds")
                child.wait()
                ledger.stop()
                return
            time.sleep(1)
        ledger.stop()
        previous = ledger.state
        final = read(output/"status.json", {}).get("state")
        if reason == "stopped_by_user":
            break
        if final in ("complete", "failed", "incomplete") or (child.returncode not in (0, 75) and reason is None):
            status(final if final in ("complete", "failed", "incomplete") else "failed",
                   worker_returncode=child.returncode)
            return
        gate = IdleGate()  # every resource resume requires three NEW idle observations
    status("stopped_by_user")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--snapshot", required=True)
    args = parser.parse_args()
    supervise(args.output, args.config, args.snapshot)


if __name__ == "__main__":
    main()
