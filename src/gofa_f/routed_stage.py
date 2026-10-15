"""CPU-only twelve-hour orchestration. Only the strict idle launcher admits CUDA."""
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

from gofa_f.idle_launcher import IdleGate, atomic_json, launch as idle_launch, query_gpu
from gofa_f.pure_graph_stage import digest, read

DATASETS = ("metr_la", "electricity", "pems_bay", "aqi", "aqi_36", "solar")
TASKS = {"metr_la": (12, 12), "electricity": (8, 8), "pems_bay": (12, 3),
         "aqi": (24, 6), "aqi_36": (12, 12), "solar": (24, 3)}
STOP = False


def config_file(path):
    path = Path(path)
    config = yaml.safe_load(path.read_text())
    base = config.pop("extends", None)
    if base:
        inherited = yaml.safe_load((path.parent / base).read_text())
        inherited["resources"].update(config.pop("resources", {}))
        inherited.update(config)
        config = inherited
    expected_architecture = {"graph_layers": [1, 2], "rank": 64, "graph_feature_dim": 5,
        "output_adapter_rank": 4, "graph_exchange": "prior_routed", "graph_attention_rank": 16,
        "graph_prior_floor": .5}
    expected_stage = {"max_epochs": 100, "patience": 15, "milestones": [10, 30, 100],
        "graph_lr": .001, "adapter_lr": .0001, "optimizer": "adamw", "loss": "raw_mae",
        "final_evaluation_reserve_seconds": 900}
    if config["protocol"] != "native-v7-routed-six" or config["seed"] != 42 or \
            tuple(config["datasets"]) != DATASETS or config["architecture"] != expected_architecture:
        raise ValueError("Unregistered stage5 architecture, dataset order or seed")
    for key, value in expected_stage.items():
        if config["stage5"][key] != value:
            raise ValueError("Unregistered stage5 setting: " + key)
    resources = config["resources"]
    if resources["memory_cap_gib"] > 40 or resources["leave_free_gib"] < 16 or \
            resources["wall_window_hours"] != 12 or resources["activity_limit_seconds"] is not None or \
            resources["effective_batch_size"] != 64 or resources["microbatch_candidates"] != [64, 32, 16, 8, 4, 2, 1]:
        raise ValueError("Shared resource or window policy changed")
    if resources["yield_on_foreign"] is not False or resources["admission_per_gpu_task"] is not True:
        raise ValueError("Use the latest admission-only sharing policy")
    return config


def job_directory(output, dataset):
    return Path(output) / "trials" / dataset / "routed"


def bind_window(output, resume=False, now=None):
    output = Path(output)
    path = output / "window.json"
    if path.exists():
        if not resume:
            raise ValueError("Existing launch window requires --resume")
        return read(path)
    if resume:
        raise ValueError("No launch window to resume")
    start = time.time() if now is None else now
    window = {"started_at": start, "wall_deadline": start+12*3600,
              "training_deadline": start+12*3600-900, "window_seconds": 12*3600}
    atomic_json(path, window)
    return window


def manifest(config, window):
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Use a clean immutable snapshot")
    spec = yaml.safe_load(Path("configs/benchmark.yaml").read_text())
    inputs = {}
    for dataset in DATASETS:
        settings = spec["datasets"][dataset]
        if (settings["context"], settings["horizons"][0]) != TASKS[dataset] or len(settings["horizons"]) != 1:
            raise ValueError("Changed paired history/horizon task")
        paths = [Path(settings["data"])]
        if settings.get("graph", {}).get("path"):
            p = Path(settings["graph"]["path"])
            paths += [p, p.with_suffix(".metadata.json")]
        inputs[dataset] = {str(p): digest(p) for p in paths}
    for name, checksum in config["native_file_sha256"].items():
        if digest(Path(config["native_weights"])/name) != checksum:
            raise ValueError("Pinned pretrained checkpoint changed")
    reference = Path(config["stage5"]["reference_summary"])
    prior = read(reference)
    if prior["state"] != "complete" or prior["git_revision"] != "79a9c86cbb10a4a7118b56235fbaa2aad18303a1":
        raise ValueError("Stage4 reference evidence incompatible")
    old_output = Path(config["stage5"]["reference_output"])
    checkpoint_hashes = {}
    for dataset in DATASETS:
        for variant in ("g2_values", "ufg_s", "ufg_a"):
            key = dataset+"/"+variant
            bound = prior["final_selection"]["jobs"][key]
            folder = old_output / "trials" / dataset / variant
            if variant.startswith("ufg"):
                folder /= "lr_"+str(bound["lr"])
            if digest(folder/"best.pt") != bound["checkpoint_sha256"]:
                raise ValueError("Historical checkpoint changed: " + key)
            checkpoint_hashes[key] = bound["checkpoint_sha256"]
    script = ("import importlib.metadata as m,json,sys;print(json.dumps({'python':sys.version,"
              "'packages':{p:m.version(p) for p in ['torch','transformers','chronos-forecasting']},"
              "'all_packages':[{'name':d.metadata['Name'],'version':d.version,"
              "'location':str(d.locate_file(''))} for d in m.distributions()]}))")
    versions = json.loads(subprocess.check_output([config["stage5"]["native_python"], "-c", script], text=True,
                          env=dict(os.environ, CUDA_VISIBLE_DEVICES="")))
    packages = {k.lower(): v for k, v in versions["packages"].items()}
    for name, value in (("torch", "2.6.0"), ("transformers", "4.46.3"), ("chronos-forecasting", "2.3.2")):
        if packages[name].split("+")[0] != value:
            raise ValueError("Unexpected native runtime: " + name)
    return {"config": config, "window": window, "inputs": inputs, "versions": versions,
            "weights": config["native_file_sha256"], "reference_sha256": digest(reference),
            "historical_checkpoint_sha256": checkpoint_hashes,
            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}


def bind_manifest(path, expected, resume):
    if Path(path).exists():
        if not resume or read(path) != expected:
            raise ValueError("Resume manifest differs: code/data/environment/architecture/window")
    elif resume:
        raise ValueError("Missing resume manifest")
    else:
        atomic_json(path, expected)


def completed(progress, training_windows):
    epochs = progress.get("epoch", 0) + int(progress.get("offset", 0) >= training_windows)
    return epochs, progress.get("stale", 0) >= 15 or epochs >= 100


def record_memory(path, phase, allocated, reserved):
    previous = read(path, {"phases": {}, "peak_allocated": 0, "peak_reserved": 0})
    old = previous["phases"].get(phase, {"peak_allocated": 0, "peak_reserved": 0, "attempts": 0})
    previous["phases"][phase] = {"peak_allocated": max(old["peak_allocated"], allocated),
        "peak_reserved": max(old["peak_reserved"], reserved), "attempts": old["attempts"]+1}
    previous["peak_allocated"] = max(previous["peak_allocated"], allocated)
    previous["peak_reserved"] = max(previous["peak_reserved"], reserved)
    atomic_json(path, previous)


def seal(output):
    path = Path(output)/"final_selection.json"
    if path.exists():
        return read(path)
    locked = {"selection_split": "validation", "sealed_at": time.time(), "jobs": {}}
    for dataset in DATASETS:
        folder = job_directory(output, dataset)
        progress, metadata = read(folder/"progress.json", {}), read(folder/"metadata.json", {})
        if not (folder/"best.pt").exists() or progress.get("best_mae") is None:
            continue
        epochs, complete = completed(progress, metadata["training_windows"])
        locked["jobs"][dataset] = {"checkpoint_sha256": digest(folder/"best.pt"),
            "validation_raw_mae": progress["best_mae"], "updates": progress["updates"], "epochs": epochs,
            "training_complete": complete, "parameters": metadata["parameters"],
            "stop_reason": "early_stopping" if progress["stale"] >= 15 else
                "max_epochs" if complete else "wall_deadline_training"}
    atomic_json(path, locked)
    return locked


def write_report(output, config):
    output = Path(output)
    state = read(output/"status.json", {}).get("state", "not_started")
    launcher = read(output/"launcher.json", {}).get("state")
    if launcher in ("wall_deadline", "failed", "incomplete") and state not in ("complete", "failed", "incomplete"):
        state = launcher
    reference = read(config["stage5"]["reference_summary"])
    results = {d: read(job_directory(output, d)/"result.json") for d in DATASETS}
    locked = read(output/"final_selection.json", {})
    lines = ["# Stage5 TITAN-inspired six datasets", "", "State: "+state, "",
        "Seed42; late_contiguous [1,2]; original-unit MAE; fixed architecture; development evaluation.",
        "|Dataset|Native Bolt|Old G2-V|New routed|UFG-S|UFG-A|Historical limited HSD|",
        "|---|---:|---:|---:|---:|---:|---:|"]
    def number(value):
        return "incomplete" if value is None else f"{value:.6g}"
    for dataset in DATASETS:
        historical = reference["historical_reference"][dataset]
        old = reference["rows"]
        values = [historical["native_bolt"]["test"]["mae"],
            old[dataset+"/g2_values"]["test"]["mae"],
            results[dataset]["test"]["raw"]["mae"] if results[dataset] else None,
            old[dataset+"/ufg_s"]["test"]["mae"], old[dataset+"/ufg_a"]["test"]["mae"],
            historical["matched_hsdgnn"]["test"]["mae"]]
        lines.append("|"+dataset+"|"+"|".join(map(number, values))+"|")
    lines += ["", "|Dataset|Updates|Epochs|Stop reason|Training complete|",
              "|---|---:|---:|---|---|"]
    for dataset in DATASETS:
        r = locked.get("jobs", {}).get(dataset)
        lines.append(f"|{dataset}|{r['updates']}|{r['epochs']}|{r['stop_reason']}|{r['training_complete']}|"
                     if r else f"|{dataset}|—|—|no complete validation|False|")
    lines += ["", "Activity ledger: "+str(read(output/"active_budget.json", {}).get("spent_seconds", 0))+" seconds.",
        "Single seed; AQI overlap; previously observed development tests; different GNN loss/optimizer and pretraining.",
        "Architecture and graph precision changed together; component attribution is unavailable.",
        "21M total-parameter matching, language interaction and one-checkpoint transfer remain untested."]
    (output/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    atomic_json(output/"report.json", {"state": state, "results": results, "final_selection": locked,
        "activity": read(output/"active_budget.json", {}), "budget": read(output/"budget.json", {}),
        "failures": read(output/"failures.json", {})})


def stage(config_path, output, resume=False):
    global STOP
    if os.environ.get("GOFA_IDLE_ADMITTED") != "1":
        raise RuntimeError("Stage must be admitted by the strict idle launcher")
    signal.signal(signal.SIGTERM, lambda *_: globals().__setitem__("STOP", True))
    signal.signal(signal.SIGINT, lambda *_: globals().__setitem__("STOP", True))
    config, output = config_file(config_path), Path(output).resolve()
    window = read(output/"window.json")
    expected = manifest(config, window)
    if read(output/"launch_binding.json") != expected:
        raise ValueError("Launch binding changed")
    bind_manifest(output/"manifest.json", expected, resume)
    failures = read(output/"failures.json", {})

    def status(state, **fields):
        atomic_json(output/"status.json", {"state": state, "pid": os.getpid(), "updated_at": time.time(),
                    "git_revision": expected["git_revision"], **fields})

    def task(phase, dataset="metr_la", milestone=100):
        global STOP
        boundary = window["training_deadline"] if phase in ("preflight", "train") else window["wall_deadline"]
        if time.time() >= boundary-60:
            return False
        if STOP or os.getppid() != int(os.environ["GOFA_SUPERVISOR_PID"]):
            raise InterruptedError("Supervisor requested release")
        command = [config["stage5"]["native_python"], "-u", "-m", "gofa_f.routed_task", "--config", config_path,
                   "--output", str(output), "--phase", phase, "--dataset", dataset, "--milestone", str(milestone)]
        environment = dict(os.environ, GOFA_SUPERVISOR_PID=str(os.getpid()),
                           GOFA_TASK_DEADLINE=str(time.monotonic()+max(0, boundary-time.time())))
        status("running", phase=phase, dataset=dataset, milestone=milestone)
        gate = IdleGate()
        while True:
            if STOP or os.getppid() != int(os.environ["GOFA_SUPERVISOR_PID"]):
                raise InterruptedError("Supervisor stopped during CPU idle wait")
            if time.time() >= boundary-60:
                return False
            try:
                gpu = query_gpu()
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired, ET.ParseError):
                gpu = None
            admitted = gate.observe(gpu) and gpu["uuid"] == os.environ.get("GOFA_GPU_UUID")
            status("waiting_for_idle", phase=phase, dataset=dataset, consecutive_idle=gate.consecutive)
            if admitted:
                break
            for _ in range(30):
                if STOP or time.time() >= boundary-60:
                    break
                time.sleep(1)
        # Compute the monotonic boundary AFTER waiting; waiting never extends it.
        environment["GOFA_TASK_DEADLINE"] = str(time.monotonic()+max(0, boundary-time.time()))
        status("running", phase=phase, dataset=dataset, milestone=milestone)
        child = subprocess.Popen(command, env=environment)
        while child.poll() is None:
            if STOP or os.getppid() != int(os.environ["GOFA_SUPERVISOR_PID"]):
                STOP = True
                child.terminate()
            time.sleep(1)
        if STOP:
            raise InterruptedError("Save and release GPU")
        if child.returncode == 75:
            if time.time() >= boundary-65:
                return False
            raise InterruptedError("Resumable release")
        if child.returncode:
            failures[f"{phase}/{dataset}/{milestone}"] = {"returncode": child.returncode, "time": time.time()}
            atomic_json(output/"failures.json", failures)
            if phase == "preflight":
                raise RuntimeError("GPU mechanisms failed; no training is allowed")
            return False
        return True

    try:
        if not read(output/"mechanisms/mechanisms.json", {}).get("passed"):
            if not task("preflight"):
                status("incomplete", reason="no completed GPU mechanisms before training cutoff")
                return
        if not (output/"final_selection.json").exists():
            for milestone in config["stage5"]["milestones"]:
                for dataset in DATASETS:
                    folder = job_directory(output, dataset)
                    if read(folder/"done.json", {}).get("complete") or (folder/f"milestone_{milestone}.json").exists():
                        continue
                    if any(k.startswith("train/"+dataset+"/") for k in failures):
                        continue
                    task("train", dataset, milestone)
            seal(output)
        locked = read(output/"final_selection.json")
        # Complete final tests before optional diagnostic work if the window is tight.
        for dataset in DATASETS:
            if dataset in locked["jobs"] and not (job_directory(output, dataset)/"result.json").exists():
                task("evaluate", dataset)
        for dataset in DATASETS:
            if dataset in locked["jobs"] and not (job_directory(output, dataset)/"routing.json").exists():
                task("diagnose", dataset)
        all_results = all((job_directory(output, d)/"result.json").exists() for d in DATASETS)
        all_diagnostics = all((job_directory(output, d)/"routing.json").exists() for d in DATASETS)
        complete = len(locked["jobs"]) == 6 and all(r["training_complete"] for r in locked["jobs"].values())
        status("complete" if all_results and all_diagnostics and complete and not failures else "incomplete")
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
    parser.add_argument("--config", default="configs/stage5_routed.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        write_report(args.output, config_file(args.config))
        return
    if args.worker:
        stage(args.config, args.output, args.resume)
        return
    config, output = config_file(args.config), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Refuse an expired restart before importing torch or admitting a worker.
    window = bind_window(output, args.resume)
    if time.time() >= window["wall_deadline"]-60:
        atomic_json(output/"launcher.json", {"state": "wall_deadline", "updated_at": time.time()})
        if (output/"manifest.json").exists():
            seal(output)
        write_report(output, config)
        return
    bind_manifest(output/"launch_binding.json", manifest(config, window), args.resume)
    command = [sys.executable, "-u", "-m", "gofa_f.routed_stage", "--worker",
               "--config", args.config, "--output", str(output)]
    idle_launch(output, command, total=None, wall_deadline=window["wall_deadline"],
                memory_cap_gib=config["resources"]["memory_cap_gib"],
                leave_free_gib=config["resources"]["leave_free_gib"], yield_on_foreign=False,
                worker_wait_state="waiting_for_idle")
    # The launcher has joined its worker. Lock existing validation checkpoints
    # on CPU even if other users prevented final GPU admission/evaluation.
    if (output/"manifest.json").exists():
        seal(output)
    write_report(output, config)


if __name__ == "__main__":
    main()
