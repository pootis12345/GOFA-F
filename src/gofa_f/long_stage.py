"""CPU supervisor for a fixed six-hour METR-LA validation-first experiment."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import yaml

from gofa_f.data import Series, Windows, file_sha256
from gofa_f.idle_launcher import IdleGate, atomic_json, launch as idle_launch, query_gpu
from gofa_f.native_stage import stage_config

STOP = False
CANDIDATES = ("c1_clock", "c2_residual")


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def config_file(path):
    config = stage_config(path)
    architecture = {"graph_layers": [1, 2], "rank": 64, "graph_feature_dim": 11,
        "output_adapter_rank": 4, "graph_exchange": "prior_routed", "graph_attention_rank": 16,
        "graph_prior_floor": .5}
    stage = config["stage6"]
    if config["protocol"] != "native-v8-metr-long" or config["seed"] != 42 or \
            config["datasets"] != ["metr_la"] or config["architecture"] != architecture:
        raise ValueError("Unregistered METR-LA architecture or seed")
    required = {"variants": list(CANDIDATES), "baseline_validation_mae": 3.2724996365709145,
        "minimum_relative_gain": .005, "simpler_tie_tolerance": .005,
        "max_epochs": 100, "patience": 15, "milestones": [10, 30, 100],
        "graph_lr": .001, "adapter_lr": .0001, "loss": "raw_mae", "optimizer": "adamw",
        "final_evaluation_reserve_seconds": 900}
    if any(stage.get(k) != v for k, v in required.items()):
        raise ValueError("Unregistered selection or optimizer protocol")
    resources = config["resources"]
    if resources["wall_window_hours"] != 6 or resources["memory_cap_gib"] > 40 or \
            resources["leave_free_gib"] < 16 or resources["effective_batch_size"] != 64 or \
            resources["microbatch_candidates"] != [64, 32, 16, 8, 4, 2, 1] or \
            resources["yield_on_foreign"] is not False or resources["admission_per_gpu_task"] is not True:
        raise ValueError("Shared-GPU policy changed")
    return config


def bind_window(output, resume=False, now=None):
    path = Path(output)/"window.json"
    if path.exists():
        if not resume:
            raise ValueError("Existing six-hour window requires --resume")
        return read(path)
    if resume:
        raise ValueError("Missing previous launch window")
    start = time.time() if now is None else now
    window = {"started_at": start, "wall_deadline": start+21600,
              "training_deadline": start+20700, "window_seconds": 21600}
    atomic_json(path, window)
    return window


def protocol_audit(config):
    spec = yaml.safe_load(Path("configs/benchmark.yaml").read_text())
    settings = spec["datasets"]["metr_la"]
    if settings["context"] != 12 or settings["horizons"] != [12] or \
            settings["graph"]["kind"] != "road_distance":
        raise ValueError("METR-LA prepared task or physical graph changed")
    series = Series.load(settings["data"])
    normalized, mean, std = series.standardize()
    windows = {part: Windows(series, normalized, part, 12, 12) for part in ("train", "val", "test")}
    graph = Path(settings["graph"]["path"])
    with np.load(graph, allow_pickle=False) as source:
        graph_shapes = {name: list(source[name].shape) for name in source.files}
        graph_node_ids = source["node_ids"].astype(str).tolist()
    with np.load(settings["data"], allow_pickle=False) as source:
        fields = list(source.files)
        prepared_values = source["values"]
        prepared_mask = source["mask"]
    prepared_metadata = read(Path(settings["data"]).with_suffix(".metadata.json"))
    graph_metadata = read(graph.with_suffix(".metadata.json"))
    original = Path("data/raw/tsl/extracted/metr_la.h5")
    if file_sha256(original) != prepared_metadata["hdf_sha256"] or \
            graph_metadata["prepared_sha256"] != series.fingerprint:
        raise ValueError("Source, graph and prepared-series provenance disagree")
    frame = pd.read_hdf(original, key=prepared_metadata["hdf_key"])
    raw_values = frame.to_numpy(dtype=np.float32)
    raw_mask = np.isfinite(raw_values) & (raw_values != 0)
    if graph_node_ids != list(map(str, frame.columns)) or \
            graph_node_ids != graph_metadata["node_ids"] or \
            not np.array_equal(prepared_mask, raw_mask) or \
            not np.array_equal(prepared_values, np.where(raw_mask, raw_values, 0)):
        raise ValueError("Original sensor columns, zeros or graph ordering differ")
    with np.load(config["stage6"]["sidecar"], allow_pickle=False) as source:
        clock_shape = list(source["calendar"].shape)
        clock_known = source["known"].sum(0).tolist()
    boundaries = {name: {"count": len(obj), "first_origin": int(obj.origins[0]),
                         "last_origin": int(obj.origins[-1]),
                         "first_target": int(obj.origins[0]),
                         "last_target": int(obj.origins[-1]+11)} for name, obj in windows.items()}
    assert boundaries["train"]["last_target"] < series.train_end
    assert boundaries["val"]["first_target"] >= series.train_end
    assert boundaries["val"]["last_target"] < series.val_end
    assert boundaries["test"]["first_target"] >= series.val_end
    return {"prepared_rows": len(series.values), "nodes": series.values.shape[1],
        "train_end": series.train_end, "val_end": series.val_end,
        "split_ratios": [series.train_end/len(series.values),
                         (series.val_end-series.train_end)/len(series.values),
                         (len(series.values)-series.val_end)/len(series.values)],
        "windows": boundaries, "prepared_fields": fields,
        "graph_arrays": graph_shapes, "clock_shape": clock_shape, "clock_known_counts": clock_known,
        "sensor_order": "verified exact original HDF columns = prepared array columns = graph node IDs",
        "source_hdf_sha256": prepared_metadata["hdf_sha256"],
        "zero_observed_targets": int((series.values == 0).__and__(series.mask).sum()),
        "mask_rule": "prepared mask AND finite; original zeros retained only if prepared mask says observed",
        "normalization": "per sensor, observed training prefix only; missing filled after normalization",
        "target_rule": "all forecast targets inside their own chronological split; prior split history allowed",
        "step12_rule": "original-unit masked MAE on raw target at origin+11",
        "titan_alignment": "12-to-12 and nominal 70/10/20 only; official sensor order, exact zeros, window and metric code unavailable",
        "titan_step12_external_mae": 3.08, "comparison": "cross-implementation reference, not a strict reproduction"}


def manifest(config, window):
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Use a clean fixed snapshot")
    spec = yaml.safe_load(Path("configs/benchmark.yaml").read_text())
    paths = [Path(spec["datasets"]["metr_la"]["data"]),
             Path(spec["datasets"]["metr_la"]["graph"]["path"]),
             Path(spec["datasets"]["metr_la"]["graph"]["path"]).with_suffix(".metadata.json"),
             Path(config["stage6"]["sidecar"]),
             Path(config["stage6"]["sidecar"]).with_suffix(".metadata.json"),
             Path(config["stage6"]["stage5_result"])]
    for name, checksum in config["native_file_sha256"].items():
        path = Path(config["native_weights"])/name
        if file_sha256(path) != checksum:
            raise ValueError("Pretrained checkpoint changed: "+name)
        paths.append(path)
    prior = read(config["stage6"]["stage5_result"])
    if prior["results"]["metr_la"]["validation"]["raw"]["mae"] != \
            config["stage6"]["baseline_validation_mae"]:
        raise ValueError("Stage5 validation baseline changed")
    versions = subprocess.check_output([config["stage6"]["native_python"], "-c",
        "import sys,torch,transformers,chronos,json;print(json.dumps([sys.version,torch.__version__,transformers.__version__,chronos.__version__]))"],
        text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    versions = json.loads(versions)
    if versions[2] != "4.46.3" or versions[3] != "2.3.2" or versions[1].split("+")[0] != "2.6.0":
        raise ValueError("Native runtime versions changed")
    return {"config": config, "window": window, "inputs": {str(p): file_sha256(p) for p in paths},
        "runtime": versions, "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"],
        text=True).strip(), "protocol_audit": protocol_audit(config)}


def bind_manifest(path, expected, resume):
    old = read(path)
    if old is None:
        if resume:
            raise ValueError("No resume manifest")
        atomic_json(path, expected)
    elif not resume or old != expected:
        raise ValueError("Resume manifest differs: code, data, graph, sidecar or window")


def task_directory(output, variant):
    return Path(output)/"trials"/variant


def complete(progress, metadata):
    if not progress or not metadata:
        return False
    epochs = progress["epoch"] + int(progress["offset"] >= metadata["train_windows"])
    return epochs >= 100 or progress["stale"] >= 15


def select(output, config):
    path = Path(output)/"selection.json"
    if path.exists():
        return read(path)
    records = {}
    for variant in CANDIDATES:
        folder = task_directory(output, variant)
        progress = read(folder/"progress.json", {})
        score = progress.get("best_mae")
        records[variant] = {"validation_raw_mae": score,
            "training_complete": complete(progress, read(folder/"metadata.json", {})),
            "updates": progress.get("updates", 0)}
    baseline = config["stage6"]["baseline_validation_mae"]
    threshold = baseline * (1-config["stage6"]["minimum_relative_gain"])
    both_valid = all(records[v]["validation_raw_mae"] is not None for v in CANDIDATES)
    eligible = [v for v in CANDIDATES if both_valid and records[v]["validation_raw_mae"] <= threshold]
    chosen = None
    if eligible:
        chosen = min(eligible, key=lambda v: records[v]["validation_raw_mae"])
        if len(eligible) == 2 and chosen == "c2_residual":
            c1, c2 = (records[v]["validation_raw_mae"] for v in CANDIDATES)
            if (c1-c2)/c1 < config["stage6"]["simpler_tie_tolerance"]:
                chosen = "c1_clock"
    result = {"selection_split": "validation", "stage5_baseline": baseline,
              "qualification_threshold": threshold, "candidates": records,
              "both_candidates_valid": both_valid, "selected": chosen,
              "control": "self_residual" if chosen == "c2_residual" else "self_clock" if chosen else None}
    atomic_json(path, result)
    return result


def seal(output, selection):
    path = Path(output)/"final_selection.json"
    if path.exists():
        return read(path)
    locked = {"selection_split": "validation", "sealed_at": time.time(),
              "selected": selection["selected"], "jobs": {}}
    for variant in (*CANDIDATES, selection["control"]):
        if variant is None:
            continue
        folder = task_directory(output, variant)
        progress, metadata = read(folder/"progress.json", {}), read(folder/"metadata.json", {})
        if not (folder/"best.pt").exists() or progress.get("best_mae") is None:
            continue
        epochs = progress["epoch"] + int(progress["offset"] >= metadata["train_windows"])
        locked["jobs"][variant] = {"checkpoint_sha256": file_sha256(folder/"best.pt"),
            "validation_raw_mae": progress["best_mae"], "updates": progress["updates"],
            "epochs": epochs, "training_complete": complete(progress, metadata),
            "stop_reason": "early_stopping" if progress["stale"] >= 15 else
                "max_epochs" if epochs >= 100 else "wall_deadline_training",
            "parameters": metadata["parameters"]}
    atomic_json(path, locked)
    return locked


def write_report(output, config):
    output = Path(output)
    selection = read(output/"selection.json", {})
    locked = read(output/"final_selection.json", {})
    results = {v: read(task_directory(output, v)/"result.json") for v in
               ("c1_clock", "c2_residual", "self_clock", "self_residual")}
    status = read(output/"status.json", {}).get("state", "not_started")
    launcher = read(output/"launcher.json", {}).get("state")
    if launcher in ("wall_deadline", "failed") and status not in ("complete", "failed"):
        status = launcher
    lines = ["# Stage6 METR-LA 12-to-12 long-horizon experiment", "", "State: "+status,
        "", "Selection: validation raw MAE across all 12 steps; Stage5 threshold "+
        str(config["stage6"]["baseline_validation_mae"]*(1-config["stage6"]["minimum_relative_gain"])),
        "Chosen: "+str(selection.get("selected")), "", "|Variant|Best validation MAE|Test mean MAE|Test step 12 MAE|Gap to TITAN 3.08|Updates|Stop|",
        "|---|---:|---:|---:|---:|---:|---|"]
    for v in ("c1_clock", "c2_residual", "self_clock", "self_residual"):
        bound, result = locked.get("jobs", {}).get(v, {}), results[v]
        raw = result["test"]["raw"] if result else None
        lines.append("|"+v+"|"+str(bound.get("validation_raw_mae", "—"))+"|"+
            str(raw["mae"] if raw else "—")+"|"+
            str(raw["by_horizon"][11]["mae"] if raw else "—")+"|"+
            str(raw["by_horizon"][11]["mae"]-3.08 if raw else "—")+"|"+
            str(bound.get("updates", "—"))+"|"+str(bound.get("stop_reason", "—"))+"|")
    lines += ["", "TITAN 60-minute MAE 3.08 is a cross-implementation reference; exact protocol equivalence unverified.",
        "Single seed; Stage5 test previously viewed during development; no test-based candidate selection.",
        "Source checkpoint, per-step metrics, memory peaks and active time are in report.json."]
    (output/"report.md").write_text("\n".join(lines)+"\n")
    atomic_json(output/"report.json", {"state": status, "selection": selection, "final_selection": locked,
        "results": results, "protocol_audit": read(output/"protocol_audit.json"),
        "activity": read(output/"active_budget.json", {}),
        "budget": read(output/"budget.json", {}),
        "memory": {v: read(task_directory(output, v)/"cuda_memory.json") for v in results},
        "failures": read(output/"failures.json", {})})


def stage(config_path, output, resume=False):
    global STOP
    if os.environ.get("GOFA_IDLE_ADMITTED") != "1":
        raise RuntimeError("Strict idle admission required")
    signal.signal(signal.SIGTERM, lambda *_: globals().__setitem__("STOP", True))
    signal.signal(signal.SIGINT, lambda *_: globals().__setitem__("STOP", True))
    config, output = config_file(config_path), Path(output).resolve()
    window = read(output/"window.json")
    expected = manifest(config, window)
    if read(output/"launch_binding.json") != expected:
        raise ValueError("Launch binding mismatch")
    bind_manifest(output/"manifest.json", expected, resume)
    failures = read(output/"failures.json", {})

    def status(state, **fields):
        atomic_json(output/"status.json", {"state": state, "pid": os.getpid(),
            "updated_at": time.time(), "git_revision": expected["git_revision"], **fields})

    def task(phase, variant="c1_clock", milestone=100):
        global STOP
        boundary = window["training_deadline"] if phase in ("preflight", "train") else window["wall_deadline"]
        if time.time() >= boundary-60:
            return False
        command = [config["stage6"]["native_python"], "-u", "-m", "gofa_f.long_task",
            "--config", config_path, "--output", str(output), "--phase", phase,
            "--variant", variant, "--milestone", str(milestone)]
        gate = IdleGate()
        while True:
            if STOP or os.getppid() != int(os.environ["GOFA_SUPERVISOR_PID"]):
                raise InterruptedError("Supervisor release")
            if time.time() >= boundary-60:
                return False
            try:
                gpu = query_gpu()
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired, ET.ParseError):
                gpu = None
            admitted = gate.observe(gpu) and gpu["uuid"] == os.environ.get("GOFA_GPU_UUID")
            status("waiting_for_idle", phase=phase, variant=variant, consecutive_idle=gate.consecutive)
            if admitted:
                break
            for _ in range(30):
                if STOP or time.time() >= boundary-60:
                    break
                time.sleep(1)
        environment = dict(os.environ, GOFA_SUPERVISOR_PID=str(os.getpid()),
            GOFA_TASK_DEADLINE=str(time.monotonic()+max(0, boundary-time.time())))
        status("running", phase=phase, variant=variant, milestone=milestone)
        child = subprocess.Popen(command, env=environment)
        while child.poll() is None:
            if STOP or os.getppid() != int(os.environ["GOFA_SUPERVISOR_PID"]):
                STOP = True
                child.terminate()
            time.sleep(1)
        if STOP or child.returncode == 75:
            raise InterruptedError("Resumable checkpoint release")
        if child.returncode:
            failures[f"{phase}/{variant}/{milestone}"] = {"returncode": child.returncode,
                "time": time.time()}
            atomic_json(output/"failures.json", failures)
            if phase == "preflight":
                raise RuntimeError("Mechanism check failed; training prohibited")
            return False
        return True

    try:
        if not read(output/"mechanisms.json", {}).get("passed"):
            if not task("preflight"):
                status("incomplete", reason="mechanism check missed training deadline")
                return
        if not (output/"selection.json").exists():
            for milestone in config["stage6"]["milestones"]:
                for variant in CANDIDATES:
                    folder = task_directory(output, variant)
                    if read(folder/"done.json", {}).get("complete") or (folder/f"milestone_{milestone}.json").exists():
                        continue
                    if any(key.startswith("train/"+variant+"/") for key in failures):
                        continue
                    task("train", variant, milestone)
            select(output, config)
        selection = read(output/"selection.json")
        if selection["control"] and not (output/"final_selection.json").exists():
            variant = selection["control"]
            for milestone in config["stage6"]["milestones"]:
                folder = task_directory(output, variant)
                if read(folder/"done.json", {}).get("complete") or (folder/f"milestone_{milestone}.json").exists():
                    continue
                if any(key.startswith("train/"+variant+"/") for key in failures):
                    continue
                task("train", variant, milestone)
        locked = seal(output, selection)
        if selection["selected"]:
            for variant in (*CANDIDATES, selection["control"]):
                if variant in locked["jobs"] and not (task_directory(output, variant)/"result.json").exists():
                    task("evaluate", variant)
        expected_variants = (*CANDIDATES, selection["control"]) if selection["control"] else ()
        enough = selection["both_candidates_valid"] if selection["selected"] is None else all(
            v in locked["jobs"] and (task_directory(output, v)/"result.json").exists()
            for v in expected_variants)
        state = "failed" if failures else "complete" if enough else "incomplete"
        if selection["selected"] is None and not any(r["validation_raw_mae"] is None
               for r in selection["candidates"].values()):
            state = "no_eligible_variant"
        status(state)
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
    parser.add_argument("--config", default="configs/stage6_metr_long.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    config, output = config_file(args.config), Path(args.output).resolve()
    if args.audit_only:
        audit = protocol_audit(config)
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output/"protocol_audit.json", audit)
        print(json.dumps(audit))
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
        if (output/"manifest.json").exists():
            seal(output, select(output, config))
        write_report(output, config)
        return
    bind_manifest(output/"launch_binding.json", manifest(config, window), args.resume)
    command = [sys.executable, "-u", "-m", "gofa_f.long_stage", "--worker",
               "--config", args.config, "--output", str(output)]
    idle_launch(output, command, total=None, wall_deadline=window["wall_deadline"],
        memory_cap_gib=config["resources"]["memory_cap_gib"],
        leave_free_gib=config["resources"]["leave_free_gib"], yield_on_foreign=False,
        worker_wait_state="waiting_for_idle")
    if (output/"manifest.json").exists():
        seal(output, select(output, config))
    write_report(output, config)


if __name__ == "__main__":
    main()
