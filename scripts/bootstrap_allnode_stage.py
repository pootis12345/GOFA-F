"""CPU-only Stage18 preflight then a strict-idle supervisor; never touches CUDA."""
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
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    snapshot, output, config = (Path(x).resolve() for x in (args.snapshot, args.output, args.config))
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/"bootstrap.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stopped = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    def state(phase, **fields):
        atomic(output/"bootstrap.json", {"state": phase, "pid": os.getpid(),
            "start_ticks": identity(os.getpid()), "updated_at": time.time(), **fields})
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
        PYTHONPATH=str(snapshot/"src"), PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    state("cpu_preflight")
    cpu = subprocess.Popen([sys.executable, "-u", str(snapshot/"scripts/check_allnode_cpu.py"),
        "--config", str(config), "--output", str(output)], cwd=snapshot, env=env, start_new_session=True)
    cpu_ticks, stop_at = identity(cpu.pid), None
    while cpu.poll() is None:
        if stopped[0] and stop_at is None:
            if identity(cpu.pid) == cpu_ticks:
                os.killpg(cpu.pid, signal.SIGTERM)
            stop_at = time.monotonic()+60
        elif stop_at is not None and time.monotonic() >= stop_at:
            if identity(cpu.pid) == cpu_ticks:
                os.killpg(cpu.pid, signal.SIGKILL)
            cpu.wait()
        time.sleep(1)
    if cpu.returncode or stopped[0]:
        state("stopped_by_user" if stopped[0] else "failed", returncode=cpu.returncode)
        return cpu.returncode or 1
    proof = json.loads((output/"cpu_preflight.json").read_text())
    if not proof.get("passed"):
        raise RuntimeError("Bound complete CPU acceptance did not pass")
    command = [sys.executable, "-u", "-m", "gofa_f.allnode_launcher", "--snapshot", str(snapshot),
               "--output", str(output), "--config", str(config)]
    with output.with_suffix(".log").open("a") as log:
        worker = subprocess.Popen(command, cwd=snapshot, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
    record = {"pid": worker.pid, "start_ticks": identity(worker.pid), "snapshot": str(snapshot),
        "output": str(output), "command": command, "started_at": time.time(),
        "admission_policy": "strict_idle", "activity_limit_seconds": None, "wall_deadline": None}
    atomic(output.with_suffix(".launch.json"), record)
    state("supervisor_deployed", supervisor=record)
    print(json.dumps(record), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
