"""CPU supervisor for one locked Stage5 checkpoint across six target datasets."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import yaml

from gofa_f.data import file_sha256
from gofa_f.idle_launcher import IdleGate, atomic_json, launch as idle_launch, query_gpu
from gofa_f.routed_stage import DATASETS, TASKS, config_file as stage5_config_file, read

STOP = False
TARGETS = ("metr_la", "pems_bay", "aqi", "aqi_36", "electricity", "solar")


def config_file(path):
    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"protocol": "native-v9-locked-transfer", "seed": 42,
                "stage5_config": "configs/stage5_routed.yaml",
                "stage5_result": "docs/results/stage5-titan-six-v1.json",
                "source_dataset": "metr_la",
                "source_checkpoint": "outputs/stage5-titan-six-v1/trials/metr_la/routed/best.pt",
                "targets": list(TARGETS), "wall_window_hours": 8,
                "memory_cap_gib": 40, "leave_free_gib": 16,
                "yield_on_foreign": False, "admission_per_gpu_task": True}
    if config != required:
        raise ValueError("Unregistered locked-transfer protocol")
    return config, stage5_config_file(config["stage5_config"])


def bind_window(output, resume=False, now=None):
    path = Path(output)/"window.json"
    if path.exists():
        if not resume:
            raise ValueError("Existing eight-hour window requires --resume")
        return read(path)
    if resume:
        raise ValueError("No launch window to resume")
    started = time.time() if now is None else now
    window = {"started_at": started, "wall_deadline": started+8*3600,
              "window_seconds": 8*3600}
    atomic_json(path, window)
    return window


def manifest(config, stage5, window, output):
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Transfer must use a clean fixed snapshot")
    preflight_path = Path(output)/"cpu_preflight.json"
    preflight = read(preflight_path)
    if not preflight or preflight.get("passed") is not True or \
            tuple(preflight.get("targets", {})) != TARGETS or \
            preflight.get("weight_updates") != 0:
        raise ValueError("Six-target CUDA-disabled CPU preflight must pass before GPU admission")
    spec = yaml.safe_load(Path("configs/benchmark.yaml").read_text())
    report = read(config["stage5_result"])
    if report["state"] != "complete" or tuple(report["results"]) != DATASETS:
        raise ValueError("Stage5 results are not complete in the registered order")
    expected_checkpoint = report["final_selection"]["jobs"][config["source_dataset"]]["checkpoint_sha256"]
    if file_sha256(config["source_checkpoint"]) != expected_checkpoint:
        raise ValueError("Stage5 source checkpoint differs from sealed SHA")
    if preflight.get("source_checkpoint_sha256") != expected_checkpoint:
        raise ValueError("CPU preflight used a different source checkpoint")
    paths = [Path("configs/benchmark.yaml"), Path(config["stage5_config"]),
             Path("configs/titan_inspired.yaml"), Path(config["stage5_result"]),
             Path(config["source_checkpoint"]), preflight_path]
    for target in TARGETS:
        settings = spec["datasets"][target]
        if (settings["context"], settings["horizons"][0]) != TASKS[target] or len(settings["horizons"]) != 1:
            raise ValueError("Target task changed: " + target)
        paths.append(Path(settings["data"]))
        if settings.get("graph", {}).get("path"):
            graph = Path(settings["graph"]["path"])
            paths.extend((graph, graph.with_suffix(".metadata.json")))
    for name, checksum in stage5["native_file_sha256"].items():
        path = Path(stage5["native_weights"])/name
        if file_sha256(path) != checksum:
            raise ValueError("Frozen native weights changed")
        paths.append(path)
    versions = subprocess.check_output([stage5["stage5"]["native_python"], "-c",
        "import sys,torch,transformers,chronos,json;print(json.dumps([sys.version,torch.__version__,transformers.__version__,chronos.__version__]))"],
        text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    versions = json.loads(versions)
    if versions[2] != "4.46.3" or versions[3] != "2.3.2" or versions[1].split("+")[0] != "2.6.0":
        raise ValueError("Native runtime version changed")
    return {"config": config, "window": window,
            "inputs": {str(path): file_sha256(path) for path in paths},
            "source_checkpoint_sha256": expected_checkpoint, "runtime": versions,
            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "target_statistics": "each target's observed training prefix; no target gradient updates",
            "selection": "source checkpoint fixed before target evaluation; no target-based selection"}


def bind_manifest(path, expected, resume):
    old = read(path)
    if old is None:
        if resume:
            raise ValueError("No bound transfer manifest to resume")
        atomic_json(path, expected)
    elif not resume or old != expected:
        raise ValueError("Transfer manifest changed")


def result_path(output, target):
    return Path(output)/"targets"/target/"result.json"


def write_report(output, config):
    output = Path(output)
    status = read(output/"status.json", {}).get("state", "not_started")
    launcher = read(output/"launcher.json", {}).get("state")
    if launcher in ("wall_deadline", "failed") and status != "complete":
        status = launcher
    results = {target: read(result_path(output, target)) for target in TARGETS}
    reference = read(config["stage5_result"])
    lines = ["# Stage7 one-checkpoint cross-dataset transfer", "", "State: "+status,
             "", "Source: Stage5 METR-LA routed GOFA-F, seed42; zero target weight updates.",
             "Target training-prefix statistics and each audited graph are used as inputs.",
             "", "|Target|Transfer validation MAE|Transfer test MAE|Target-trained Stage5 test MAE|Test ratio|",
             "|---|---:|---:|---:|---:|"]
    for target in TARGETS:
        result = results[target]
        old = reference["results"][target]["test"]["raw"]["mae"]
        if result is None:
            lines.append(f"|{target}|—|—|{old:.6g}|—|")
        else:
            val = result["validation"]["raw"]["mae"]
            test = result["test"]["raw"]["mae"]
            lines.append(f"|{target}|{val:.6g}|{test:.6g}|{old:.6g}|{test/old:.4f}|")
    lines += ["", "No checkpoint switching or target gradient updates. Different histories/horizons,",
              "target-fitted normalization and target-specific prior graphs are disclosed;",
              "this is calibrated weight transfer, not strict cold-start zero-shot forecasting."]
    (output/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    atomic_json(output/"report.json", {"state": status,
        "source": config["source_dataset"], "targets": results,
        "failures": read(output/"failures.json", {}),
        "activity": read(output/"active_budget.json", {}),
        "budget": read(output/"budget.json", {}),
        "manifest": read(output/"manifest.json"),
        "memory": {target: read(output/"targets"/target/"cuda_memory.json") for target in TARGETS}})


def stage(config_path, output, resume=False):
    global STOP
    if os.environ.get("GOFA_IDLE_ADMITTED") != "1":
        raise RuntimeError("Strict idle admission required")
    signal.signal(signal.SIGTERM, lambda *_: globals().__setitem__("STOP", True))
    signal.signal(signal.SIGINT, lambda *_: globals().__setitem__("STOP", True))
    config, stage5 = config_file(config_path)
    output = Path(output).resolve()
    window = read(output/"window.json")
    expected = manifest(config, stage5, window, output)
    if read(output/"launch_binding.json") != expected:
        raise ValueError("Transfer launch binding differs")
    bind_manifest(output/"manifest.json", expected, resume)
    failures = read(output/"failures.json", {})

    def status(state, **fields):
        atomic_json(output/"status.json", {"state": state, "pid": os.getpid(),
            "updated_at": time.time(), "git_revision": expected["git_revision"], **fields})

    def task(target):
        global STOP
        gate = IdleGate()
        while True:
            if STOP or os.getppid() != int(os.environ["GOFA_SUPERVISOR_PID"]):
                raise InterruptedError("Supervisor released")
            if time.time() >= window["wall_deadline"]-60:
                return False
            try:
                gpu = query_gpu()
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired, ET.ParseError):
                gpu = None
            admitted = gate.observe(gpu) and gpu["uuid"] == os.environ.get("GOFA_GPU_UUID")
            status("waiting_for_idle", target=target, consecutive_idle=gate.consecutive)
            if admitted:
                break
            for _ in range(30):
                if STOP or time.time() >= window["wall_deadline"]-60:
                    break
                time.sleep(1)
        command = [stage5["stage5"]["native_python"], "-u", "-m", "gofa_f.transfer_task",
                   "--config", config_path, "--output", str(output), "--target", target]
        environment = dict(os.environ, GOFA_SUPERVISOR_PID=str(os.getpid()),
                           GOFA_TASK_DEADLINE=str(time.monotonic()+max(0, window["wall_deadline"]-time.time())))
        status("running", target=target)
        child = subprocess.Popen(command, env=environment)
        while child.poll() is None:
            if STOP or os.getppid() != int(os.environ["GOFA_SUPERVISOR_PID"]):
                STOP = True
                child.terminate()
            time.sleep(1)
        if STOP or child.returncode == 75:
            raise InterruptedError("Resumable transfer release")
        if child.returncode:
            failures[target] = {"returncode": child.returncode, "time": time.time()}
            atomic_json(output/"failures.json", failures)
            return False
        if not result_path(output, target).exists():
            raise RuntimeError("Evaluation exited without result: " + target)
        return True

    try:
        for target in TARGETS:
            if result_path(output, target).exists() or target in failures:
                continue
            if not task(target):
                if target == config["source_dataset"] and target in failures:
                    raise RuntimeError("Source reproduction failed; no transfer targets evaluated")
                if time.time() >= window["wall_deadline"]-60:
                    break
        complete = all(result_path(output, target).exists() for target in TARGETS)
        status("failed" if failures else "complete" if complete else "incomplete")
    except InterruptedError:
        status("interrupted")
        raise SystemExit(75)
    except Exception as exc:
        status("failed", error=repr(exc))
        raise
    finally:
        write_report(output, config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage7_transfer.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cpu-preflight", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    config, stage5 = config_file(args.config)
    output = Path(args.output).resolve()
    if args.cpu_preflight:
        from gofa_f.transfer_task import preflight_cpu
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output/"cpu_preflight.json", preflight_cpu(config, stage5))
        print(json.dumps({"cpu_preflight": "passed"}), flush=True)
        return
    if args.report_only:
        write_report(output, config)
        return
    if args.worker:
        stage(args.config, output, args.resume)
        return
    output.mkdir(parents=True, exist_ok=True)
    window = bind_window(output, args.resume)
    if time.time() >= window["wall_deadline"]-60:
        atomic_json(output/"launcher.json", {"state": "wall_deadline", "updated_at": time.time()})
        write_report(output, config)
        return
    bind_manifest(output/"launch_binding.json", manifest(config, stage5, window, output), args.resume)
    command = [sys.executable, "-u", "-m", "gofa_f.transfer_stage", "--worker",
               "--config", args.config, "--output", str(output)]
    idle_launch(output, command, total=None, wall_deadline=window["wall_deadline"],
                memory_cap_gib=40, leave_free_gib=16, yield_on_foreign=False,
                worker_wait_state="waiting_for_idle")
    write_report(output, config)


if __name__ == "__main__":
    main()
