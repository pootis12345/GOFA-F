"""CPU-only Stage18 preflight then fully-idle admission; never touches CUDA."""
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
    parser.add_argument("--repair-failed-preflight")
    parser.add_argument("--prior-diagnostic", action="append", default=[])
    parser.add_argument("--reuse-real-cpu-from")
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
    if args.reuse_real_cpu_from:
        prior = Path(args.reuse_real_cpu_from).resolve()
        transition = json.loads((prior/"resource_policy_transition.json").read_text())
        old_pid, old_ticks = transition["controller_pid"], transition["start_ticks"]
        if transition["requested"] != "idle" or transition["consecutive_samples"] != 3 \
                or transition["interval_seconds"] != 10:
            raise ValueError("Missing explicit idle-admission transition")
        state("waiting_for_prior_cpu", source_output=str(prior))
        while not stopped[0]:
            proof_path = prior/"cpu_preflight.json"
            prior_proof = json.loads(proof_path.read_text()) if proof_path.exists() else {}
            if prior_proof.get("passed"):
                break
            if prior_proof.get("error_type") or prior_proof.get("phase") == "regression":
                raise RuntimeError("Prior CPU acceptance failed; no proof can be reused")
            if identity(old_pid) != old_ticks:
                raise RuntimeError("Prior CPU controller unexpectedly exited before completing acceptance")
            current_state = Path(f"/proc/{old_pid}/stat").read_text().split(") ", 1)[1].split()[0]
            if current_state not in ("T", "t") or prior.with_suffix(".launch.json").exists():
                raise RuntimeError("Prior shared GPU handoff is not safely suspended")
            time.sleep(1)
        if stopped[0]:
            state("stopped_by_user")
            return 1
        if identity(old_pid) == old_ticks:
            proc = Path(f"/proc/{old_pid}")
            current_state = (proc/"stat").read_text().split(") ", 1)[1].split()[0]
            command = (proc/"cmdline").read_bytes().split(b"\0")
            if current_state not in ("T", "t") or proc.stat().st_uid != os.getuid() \
                    or b"--output" not in command \
                    or Path(command[command.index(b"--output")+1].decode()).resolve() != prior:
                raise RuntimeError("Refusing an unverified old CPU controller")
            # TERM is delivered when CONT resumes the stopped controller, before GPU handoff.
            os.kill(old_pid, signal.SIGTERM)
            os.kill(old_pid, signal.SIGCONT)
            deadline = time.monotonic()+60
            while identity(old_pid) == old_ticks and time.monotonic() < deadline:
                time.sleep(1)
            if identity(old_pid) == old_ticks or prior.with_suffix(".launch.json").exists():
                raise RuntimeError("Old controller did not safely exit without GPU deployment")
    state("cpu_preflight")
    cpu_command = [sys.executable, "-u", str(snapshot/"scripts/check_allnode_cpu.py"),
                   "--config", str(config), "--output", str(output)]
    if args.reuse_real_cpu_from:
        cpu_command.extend(["--reuse-real-cpu-from", args.reuse_real_cpu_from])
    cpu = subprocess.Popen(cpu_command, cwd=snapshot, env=env, start_new_session=True)
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
    if args.repair_failed_preflight:
        from gofa_f.allnode_recovery import migrate_failed_preflight
        migrate_failed_preflight(args.repair_failed_preflight, output, args.prior_diagnostic)
    elif args.prior_diagnostic:
        raise ValueError("Prior diagnostic activity must accompany an audited repair")
    command = [sys.executable, "-u", "-m", "gofa_f.allnode_launcher", "--snapshot", str(snapshot),
               "--output", str(output), "--config", str(config)]
    with output.with_suffix(".log").open("a") as log:
        worker = subprocess.Popen(command, cwd=snapshot, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
    record = {"pid": worker.pid, "start_ticks": identity(worker.pid), "snapshot": str(snapshot),
        "output": str(output), "command": command, "started_at": time.time(),
        "admission_policy": "idle", "admission_samples": 3, "admission_interval_seconds": 10,
        "activity_limit_seconds": None, "wall_deadline": None}
    atomic(output.with_suffix(".launch.json"), record)
    state("supervisor_deployed", supervisor=record)
    print(json.dumps(record), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
