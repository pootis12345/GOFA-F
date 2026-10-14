"""Stop only this stage's runner after a specified complete result has been saved (Linux)."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import time


def process_info(pid):
    root = Path(f"/proc/{pid}")
    fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
    uid_line = next(line for line in (root / "status").read_text().splitlines() if line.startswith("Uid:"))
    return {"pid": pid, "state": fields[0], "ppid": int(fields[1]), "start_ticks": fields[19],
            "uid": int(uid_line.split()[1]), "argv": (root / "cmdline").read_bytes().split(b"\0")[:-1],
            "cwd": str((root / "cwd").resolve())}


def same_process(identity):
    try:
        current = process_info(identity["pid"])
        return current["start_ticks"] == identity["start_ticks"] and current["state"] != "Z"
    except (OSError, StopIteration, IndexError):
        return False


def validate_runner(identity, stage_root):
    if identity["uid"] != os.getuid() or identity["pid"] == os.getpid():
        raise ValueError("Only the user's own stage runner may be stopped")
    argv = identity["argv"]
    if b"gofa_f.benchmark" not in argv or b"--output" not in argv:
        raise ValueError("Target is not a GOFA benchmark runner")
    target = argv[argv.index(b"--output") + 1].decode()
    resolved = (Path(identity["cwd"]) / target).resolve()
    if resolved != Path(stage_root).resolve():
        raise ValueError("Runner output does not match the requested stage")


def complete_result(path, expected, pid):
    try:
        raw = Path(path).read_bytes()
        result = json.loads(raw)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if result["status"] != "complete":
        raise ValueError("Only a full completed result can trigger the stage pause")
    if any(result["config"].get(key) != value for key, value in expected.items()):
        raise ValueError("Completed result does not match the requested job")
    if result["environment"]["pid"] != pid:
        raise ValueError("Completed result belongs to a different runner")
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    temporary = Path(f"{path}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


@contextmanager
def stopped_runner(identity):
    """Prevent subsequent dispatch while terminating; undo a partial stop on errors."""
    if not same_process(identity):
        raise RuntimeError("Runner exited or PID was reused")
    os.kill(identity["pid"], signal.SIGSTOP)
    try:
        yield
    finally:
        if same_process(identity):
            os.kill(identity["pid"], signal.SIGCONT)


def stop_stage(identity):
    with stopped_runner(identity):
        # The current job's full result is already durable. No mid-job recovery is needed.
        os.kill(identity["pid"], signal.SIGTERM)
    deadline = time.monotonic() + 30
    while same_process(identity) and time.monotonic() < deadline:
        time.sleep(.1)
    if same_process(identity):
        raise RuntimeError("Runner did not exit after SIGTERM; no force kill was issued")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--stage", required=True, help="Existing stage output directory")
    parser.add_argument("--result", required=True, help="Full result whose completion triggers the pause")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=12)
    args = parser.parse_args()
    stage, result_path = Path(args.stage).resolve(), Path(args.result).resolve()
    if not stage.is_dir() or not result_path.is_relative_to(stage):
        raise ValueError("Result must be inside the existing stage output directory")
    record_path = Path(f"{stage}.pause.json")
    with Path(f"{stage}.pause.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = process_info(args.pid)
        validate_runner(identity, stage)
        parent = process_info(identity["ppid"])
        record = {"state": "armed", "watcher_pid": os.getpid(), "runner_pid": args.pid,
                  "runner_start_ticks": identity["start_ticks"], "result": str(result_path),
                  "armed_utc": datetime.now(timezone.utc).isoformat(),
                  "expected": {"dataset": args.dataset, "model_name": args.model,
                               "seed": args.seed, "horizon": args.horizon}}
        write_json(record_path, record)
        print(json.dumps(record), flush=True)
        try:
            while True:
                fingerprint = complete_result(result_path, record["expected"], args.pid)
                if fingerprint:
                    stop_stage(identity)
                    # Let run_stage1.sh finish its EXIT trap before labelling this requested pause.
                    deadline = time.monotonic() + 10
                    while same_process(parent) and time.monotonic() < deadline:
                        time.sleep(.1)
                    record.update(state="paused", paused_utc=datetime.now(timezone.utc).isoformat(),
                                  result_sha256=fingerprint, runner_exited=True)
                    write_json(record_path, record)
                    status = Path(f"{stage}.status")
                    temporary = Path(f"{status}.tmp")
                    temporary.write_text(f"paused after {args.dataset}/{args.model}/h{args.horizon}/seed{args.seed}\n")
                    temporary.replace(status)
                    print(json.dumps(record), flush=True)
                    break
                if not same_process(identity):
                    raise RuntimeError("Runner exited before the expected result completed")
                time.sleep(.1)
        except Exception as error:
            record.update(state="error", error=repr(error), updated_utc=datetime.now(timezone.utc).isoformat())
            write_json(record_path, record)
            raise


if __name__ == "__main__":
    main()
