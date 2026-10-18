"""Stage19 CPU proof controller; a verified complete proof is required before idle handoff."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from bootstrap_generative_stage import atomic, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("snapshot", "output", "config"):
        parser.add_argument("--"+field, required=True)
    args = parser.parse_args()
    snapshot, output, config = (Path(x).resolve() for x in (args.snapshot, args.output, args.config))
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/"bootstrap.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stopped = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    def status(state, **extra):
        atomic(output/"bootstrap.json", {"state": state, "pid": os.getpid(),
            "start_ticks": identity(os.getpid()), "updated_at": time.time(), **extra})
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
        PYTHONPATH=str(snapshot/"src"), PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    import yaml
    cfg = yaml.safe_load(config.read_text())
    prior = cfg.get("continuation", {}).get("prior_projection_diagnostic")
    if prior:
        prior = Path(prior)
        launch = json.loads(prior.with_suffix(".launch.json").read_text())
        status("waiting_for_projection_diagnostic", source_output=str(prior))
        while not stopped[0]:
            alive = identity(launch["pid"]) == str(launch["start_ticks"])
            state_path = prior/"status.json"
            prior_state = json.loads(state_path.read_text()) if state_path.exists() else {}
            if not alive:
                if prior_state.get("state") != "complete":
                    status("failed", error="Projection diagnostic incomplete; no CPU handoff")
                    return 1
                break
            time.sleep(1)
        if stopped[0]:
            status("stopped_by_user")
            return 1
    status("cpu_preflight")
    cpu = subprocess.Popen([sys.executable, "-u", str(snapshot/"scripts/check_short_cpu.py"),
        "--config", str(config), "--output", str(output)], cwd=snapshot, env=env, start_new_session=True)
    ticks, deadline = identity(cpu.pid), None
    while cpu.poll() is None:
        if stopped[0] and deadline is None:
            if identity(cpu.pid) == ticks:
                os.killpg(cpu.pid, signal.SIGTERM)
            deadline = time.monotonic()+60
        elif deadline is not None and time.monotonic() >= deadline:
            if identity(cpu.pid) == ticks:
                os.killpg(cpu.pid, signal.SIGKILL)
        time.sleep(1)
    if cpu.returncode or stopped[0]:
        status("stopped_by_user" if stopped[0] else "failed", returncode=cpu.returncode)
        return cpu.returncode or 1
    proof = json.loads((output/"cpu_preflight.json").read_text())
    if not proof.get("passed"):
        raise RuntimeError("Complete bound CPU proof required")
    command = [sys.executable, "-u", "-m", "gofa_f.short_launcher", "--snapshot", str(snapshot),
               "--output", str(output), "--config", str(config)]
    with output.with_suffix(".log").open("a") as log:
        supervisor = subprocess.Popen(command, cwd=snapshot, env=env, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
    launch = {"pid": supervisor.pid, "start_ticks": identity(supervisor.pid), "snapshot": str(snapshot),
        "output": str(output), "command": command, "started_at": time.time(), "admission_policy": "idle",
        "admission_samples": 3, "admission_interval_seconds": 10, "activity_limit_seconds": None, "wall_deadline": None}
    atomic(output.with_suffix(".launch.json"), launch)
    status("supervisor_deployed", supervisor=launch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
