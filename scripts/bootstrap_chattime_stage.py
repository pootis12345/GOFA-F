"""Durable C-only CPU preparation, then the registered eleven-hour idle waiter.

The countdown is created by chattime_stage only AFTER verified downloads and the
complete real-weight CPU preflight. A failed preparation never starts GPU work.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def atomic(path, record):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2)+"\n")
    temporary.replace(path)


def identity(pid):
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        tail = text[text.rfind(")")+2:].split()
        if tail[0] == "Z":
            return None
        return int(tail[19])
    except (OSError, IndexError, ValueError):
        return None


def downloaders(target):
    matches = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            args = (process/"cmdline").read_bytes().decode().split("\0")
            if process.stat().st_uid != os.getuid() or not any(
                    Path(arg).name == "segmented_chattime_download.py" for arg in args):
                continue
            if "--output" in args and Path(args[args.index("--output")+1]).resolve() == Path(target).resolve():
                tick = identity(int(process.name))
                if tick is not None:
                    matches.append((int(process.name), tick))
        except (OSError, ValueError, IndexError):
            pass
    return matches


def bootstrap(root, output, official_manifest, resume=False, run=subprocess.run, popen=subprocess.Popen,
              hard_deadline=None, adopt_waiting_window=None, shared_gpu=False,
              fixed_prompt_kv=False, continue_training_from=None):
    root, output, official_manifest = Path(root).resolve(), Path(output).resolve(), Path(official_manifest).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                   TOKENIZERS_PARALLELISM="false", PYTHONPATH=str(root/"src"), PYTHONDONTWRITEBYTECODE="1")
    python = root/".venv-chat/bin/python"
    lock = (output/"bootstrap.lock").open("a")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("An existing preparation controller owns Stage16") from None
    status_path = output/"bootstrap.json"

    def status(state, **extra):
        atomic(status_path, {"state": state, "pid": os.getpid(), "start_ticks": identity(os.getpid()),
            "updated_at": time.time(), "snapshot": str(root), **extra})
        print(json.dumps({"state": state, **extra}), flush=True)

    window = read(output/"window.json")
    if window and not resume:
        raise ValueError("An existing window requires --resume; no new eleven-hour window")
    launch_record_path = output.parent/(output.name+".launch.json")
    existing = read(launch_record_path, {})
    if existing.get("pid") and existing.get("start_ticks") is not None and \
            identity(existing["pid"]) == existing["start_ticks"]:
        status("following_existing_waiter", launcher_pid=existing["pid"])
        while identity(existing["pid"]) == existing.get("start_ticks"):
            time.sleep(30)
        worker = read(output/"active_budget.json", {}).get("worker") or {}
        if worker.get("pid") and worker.get("start_ticks") is not None and \
                identity(worker["pid"]) == worker["start_ticks"]:
            status("failed", error="Existing waiter exited with a still-live worker; aggregation deferred")
            raise RuntimeError("Still-live GPU worker requires attention")
        run([str(python), "-u", "-m", "gofa_f.chattime_stage", "--output", str(output), "--report-only"],
            cwd=root, env=cpu_env, check=True)
        status(read(output/"report.json", {}).get("state", "failed"), window=read(output/"window.json"))
        return
    try:
        if not window:
            status("preparing_weights", window_started=False)
            target = root/"data/weights/chattime-7b-20baa88"
            active_downloaders = downloaders(target)
            if len(active_downloaders) > 1:
                raise RuntimeError("Multiple CPU downloaders already target this snapshot; no duplicate launch")
            for pid, tick in active_downloaders:
                status("following_existing_download", download_pid=pid, window_started=False)
                while identity(pid) == tick:
                    time.sleep(30)
            command = [str(python), "-u", str(root/"scripts/segmented_chattime_download.py"),
                "--manifest", str(official_manifest), "--output", str(root/"data/weights/chattime-7b-20baa88")]
            with (output/"download.log").open("a") as log:
                result = run(command, cwd=root, env=cpu_env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError("Official-hash weight preparation failed; see download.log")
            status("cpu_preflight", window_started=False)
            command = [str(python), "-u", str(root/"scripts/check_chattime_cpu.py"), "--output", str(output)]
            with (output/"cpu_preflight.log").open("a") as log:
                result = run(command, cwd=root, env=dict(cpu_env, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"),
                             stdout=log, stderr=subprocess.STDOUT)
            if result.returncode or not read(output/"cpu_preflight.json", {}).get("passed"):
                raise RuntimeError("Real-weight CPU preflight failed; GPU launch prohibited")
        elif not read(output/"cpu_preflight.json", {}).get("passed"):
            raise ValueError("Cannot resume without the original successful CPU preflight")
        if continue_training_from is not None and not window:
            previous = Path(continue_training_from).resolve()
            records = [read(previous.parent/(previous.name+suffix), {})
                       for suffix in (".launch.json", ".bootstrap.launch.json")]
            if any(not r.get("pid") or r.get("start_ticks") is None for r in records):
                raise ValueError("Missing continuation source identities")
            status("waiting_for_source_checkpoint", window_started=False,
                   source=str(previous), hard_deadline=hard_deadline)
            from gofa_f.idle_launcher import parse_deadline
            deadline = parse_deadline(hard_deadline)
            if deadline is None:
                raise ValueError("Continuation must keep an explicit absolute deadline")
            while any(identity(r["pid"]) == r["start_ticks"] for r in records):
                if time.time() >= deadline-60:
                    status("wall_deadline", window_started=False)
                    return
                time.sleep(1)
        status("starting_idle_waiter", window_started=bool(window))
        command = [str(python), "-u", "-m", "gofa_f.chattime_stage", "--output", str(output)]
        if hard_deadline is not None:
            command += ["--hard-deadline", hard_deadline]
        if adopt_waiting_window is not None and not window:
            command += ["--adopt-waiting-window", str(adopt_waiting_window)]
        if shared_gpu:
            command.append("--shared-gpu")
        if fixed_prompt_kv:
            command.append("--fixed-prompt-kv")
        if continue_training_from is not None and not window:
            command += ["--continue-training-from", str(continue_training_from)]
        if window:
            command.append("--resume")
        with (output.parent/(output.name+".log")).open("a") as log:
            child = popen(command, cwd=root, env=dict(cpu_env, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"),
                          stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            atomic(launch_record_path, {"pid": child.pid, "start_ticks": identity(child.pid),
                "snapshot": str(root), "output": str(output), "spawned_at": time.time(), "command": command})
            status("idle_waiter_deployed", launcher_pid=child.pid, launcher_start_ticks=identity(child.pid))
            code = child.wait()
        ledger = read(output/"active_budget.json", {})
        worker = ledger.get("worker") or {}
        if worker.get("pid") and worker.get("start_ticks") is not None and \
                identity(worker["pid"]) == worker["start_ticks"]:
            raise RuntimeError("Waiter exited but its worker is still alive; CPU aggregation deferred")
        # No GPU code here: finish aggregate metrics from already-written shards.
        run([str(python), "-u", "-m", "gofa_f.chattime_stage", "--output", str(output), "--report-only"],
            cwd=root, env=cpu_env, stdout=sys.stdout, stderr=sys.stderr, check=True)
        report = read(output/"report.json", {})
        status(report.get("state", "failed"), launcher_returncode=code, window=read(output/"window.json"))
        if code and report.get("state") not in ("complete", "incomplete", "wall_deadline"):
            raise RuntimeError("Stage16 waiter failed; inspect the stage log")
    except Exception as exc:
        status("failed", error=str(exc), window_started=(output/"window.json").exists())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--official-manifest", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hard-deadline")
    parser.add_argument("--adopt-waiting-window")
    parser.add_argument("--shared-gpu", action="store_true")
    parser.add_argument("--fixed-prompt-kv", action="store_true")
    parser.add_argument("--continue-training-from")
    args = parser.parse_args()
    bootstrap(Path(__file__).resolve().parents[1], args.output, args.official_manifest, args.resume,
              hard_deadline=args.hard_deadline, adopt_waiting_window=args.adopt_waiting_window,
              shared_gpu=args.shared_gpu, fixed_prompt_kv=args.fixed_prompt_kv,
              continue_training_from=args.continue_training_from)


if __name__ == "__main__":
    main()
