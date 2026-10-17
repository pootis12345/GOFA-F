"""C-only CPU acceptance then a CPU-only unlimited strict-idle supervisor."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def identity(pid):
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        return text[text.rfind(")")+2:].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def atomic(path, data):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(data, indent=2)+"\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--reuse-real-from")
    parser.add_argument("--reuse-real-snapshot")
    args = parser.parse_args()
    snapshot, output, config = map(lambda x: Path(x).resolve(), (args.snapshot, args.output, args.config))
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
    if args.reuse_real_from:
        source = Path(args.reuse_real_from)
        state("waiting_for_cpu_proof", source=str(source))
        while not stopped[0]:
            path = source/"cpu_preflight.json"
            proof = json.loads(path.read_text()) if path.exists() else {}
            if proof.get("passed"):
                break
            previous = json.loads((source/"bootstrap.json").read_text())
            if previous["state"] != "cpu_preflight":
                state("failed", error="Source real-weight CPU proof did not finish successfully")
                return 1
            time.sleep(10)
        if stopped[0]:
            state("stopped_by_user")
            return 1
    state("cpu_preflight")
    command = [sys.executable, "-u", str(snapshot/"scripts/check_generative_cpu.py"),
               "--config", str(config), "--output", str(output)]
    if args.reuse_real_from or args.reuse_real_snapshot:
        if not (args.reuse_real_from and args.reuse_real_snapshot):
            raise ValueError("Complete CPU proof source binding required")
        command += ["--reuse-real-from", args.reuse_real_from, "--reuse-real-snapshot", args.reuse_real_snapshot]
    result = subprocess.run(command, cwd=snapshot, env=env, check=False)
    if result.returncode or stopped[0]:
        state("stopped_by_user" if stopped[0] else "failed", returncode=result.returncode)
        return result.returncode or 1
    check = json.loads((output/"cpu_preflight.json").read_text())
    if not check.get("passed"):
        raise RuntimeError("Bound complete CPU acceptance did not pass")
    # This process never creates a CUDA context, even after the supervisor starts.
    command = [sys.executable, "-u", "-m", "gofa_f.generative_launcher", "--snapshot", str(snapshot),
               "--output", str(output), "--config", str(config)]
    with output.with_suffix(".log").open("a") as log:
        worker = subprocess.Popen(command, cwd=snapshot, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
    record = {"pid": worker.pid, "start_ticks": identity(worker.pid), "snapshot": str(snapshot),
              "command": command, "output": str(output), "started_at": time.time(),
              "admission_policy": "strict_idle", "activity_limit_seconds": None, "wall_deadline": None}
    atomic(output.with_suffix(".launch.json"), record)
    state("supervisor_deployed", supervisor=record)
    print(json.dumps(record), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
