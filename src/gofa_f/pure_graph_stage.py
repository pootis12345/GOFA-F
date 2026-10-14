"""CPU orchestration of isolated pure-GNN / frozen-Bolt GPU tasks. No torch imports."""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import yaml

from gofa_f.idle_launcher import atomic_json, parse_deadline

KEY_DATASETS = ("metr_la", "electricity")
EXTENSION_DATASETS = ("pems_bay", "aqi", "aqi_36", "solar")
DATASETS = KEY_DATASETS + EXTENSION_DATASETS
STOP = False


def config_file(path):
    def resolve(path, chain=()):
        path = Path(path).resolve()
        if path in chain:
            raise ValueError("Cyclic configuration")
        override = yaml.safe_load(path.read_text())
        base = override.pop("extends", None)
        result = resolve(path.parent / base, (*chain, path)) if base else {}
        def merge(destination, source):
            for key, value in source.items():
                if isinstance(value, dict) and isinstance(destination.get(key), dict):
                    merge(destination[key], value)
                else:
                    destination[key] = value
        merge(result, override)
        return result
    config = resolve(path)
    config["budget"] = {"total_gpu_seconds": None, "per_job_seconds": None,
                        "wall_deadline": config["stage4"]["wall_deadline"], "waiting_for_idle_counts": False}
    config.pop("adaptation", None)
    config["models"].pop("matched_hsdgnn", None)
    expected = {"pilot_updates": 200, "max_epochs": 100, "patience": 15,
                "target_active_real_dof": 108930, "activity_limit_seconds": None,
                "graph_feature_mode": "values_only", "parameter_tolerance": .10,
                "learning_rates": [.001, .003], "milestones": [10, 30, 100]}
    if config["protocol"] != "native-v6-pure-graph-audit" or config["seed"] != 42:
        raise ValueError("Unexpected stage or seed")
    for key, value in expected.items():
        if config["stage4"][key] != value:
            raise ValueError("Unregistered stage4 setting: " + key)
    if config["resources"]["memory_cap_gib"] > 40 or config["resources"]["leave_free_gib"] < 16:
        raise ValueError("Shared GPU limits relaxed")
    if config["resources"]["effective_batch_size"] != 64:
        raise ValueError("Effective batch must remain 64")
    if config["models"]["gofa_bolt"]["graph_layers"] != [1, 2]:
        raise ValueError("All new GOFA must use late_contiguous")
    parse_deadline(config["stage4"]["wall_deadline"])
    return config


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def task_directory(output, dataset, variant, lr=.001):
    result = Path(output) / "trials" / dataset / variant
    return result / f"lr_{lr}" if variant.startswith("ufg") else result


def select_lr(scores, persistence, rates=(.001, .003)):
    available = []
    for lr in rates:
        values = [scores.get(dataset, {}).get(str(lr)) for dataset in KEY_DATASETS]
        if any(v is None or not math.isfinite(v) or v < 0 for v in values):
            continue
        if any(persistence[d] <= 0 for d in KEY_DATASETS):
            raise ValueError("Nonpositive persistence validation MAE")
        score = math.sqrt(math.prod(v/persistence[d] for d, v in zip(KEY_DATASETS, values)))
        available.append((score, lr))
    if not available:
        return None
    best = min(v[0] for v in available)
    winner = min((v for v in available if math.isclose(v[0], best, rel_tol=1e-4)), key=lambda v: v[1])
    return {"lr": winner[1], "geometric_mean_ratio": winner[0], "validation_scores": scores,
            "selection_split": "validation", "metric": "raw_mae", "test_used": False}


def manifest(config):
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Launch only a clean committed snapshot")
    spec = yaml.safe_load(Path("configs/benchmark.yaml").read_text())
    inputs = {}
    for dataset in config["datasets"]:
        settings = spec["datasets"][dataset]
        paths = [Path(settings["data"])]
        graph_path = settings.get("graph", {}).get("path")
        if graph_path:
            paths += [Path(graph_path), Path(graph_path).with_suffix(".metadata.json")]
        side = Path(config["graph_context"]["sidecar_root"]) / f"{dataset}.npz"
        paths += [side, side.with_suffix(".metadata.json")]
        inputs[dataset] = {str(p): digest(p) for p in paths}
    source_root = Path(config["stage4"]["upstream_root"])
    source = read(source_root / "manifest.json")
    if source["revision"] != config["stage4"]["upstream_revision"]:
        raise ValueError("UFG revision mismatch")
    for name, checksum in source["source_sha256"].items():
        if digest(source_root / name) != checksum:
            raise ValueError("Modified upstream file: " + name)
    weights = config["models"]["native_bolt"]
    for name, checksum in weights["file_sha256"].items():
        if digest(Path(weights["local_weights"]) / name) != checksum:
            raise ValueError("Native checkpoint hash mismatch")
    versions = {}
    script = ("import importlib.metadata as m,json,sys; "
              "print(json.dumps(dict({p:m.version(p) for p in PACKAGES},python=sys.version,"
              "all_packages={d.metadata['Name']:d.version for d in m.distributions()})))")
    for family, packages in (("native", ["torch", "transformers", "chronos-forecasting", "numpy"]),
                             ("ufg", ["torch", "torch-geometric", "dgl", "numpy"])):
        command = [config["stage4"][family + "_python"], "-c", script.replace("PACKAGES", repr(packages))]
        versions[family] = json.loads(subprocess.check_output(command, text=True,
                                        env=dict(os.environ, CUDA_VISIBLE_DEVICES="")))
    expected_versions = {"native": {"torch": "2.6.0", "transformers": "4.46.3", "chronos-forecasting": "2.3.2"},
                         "ufg": {"torch": "2.1.1", "dgl": "2.0.0", "torch-geometric": "2.5.3", "numpy": "1.26.4"}}
    for family, packages in expected_versions.items():
        for name, version in packages.items():
            if versions[family][name].split("+")[0] != version:
                raise ValueError("Unexpected isolated runtime version: " + family + "/" + name)
    reference = Path(config["stage4"]["reference_output"])
    historical = {str(p.relative_to(reference)): digest(p) for p in reference.glob("reevaluation/*/*/result.json")}
    expected_references = {f"reevaluation/{dataset}/{name}/result.json" for dataset in DATASETS
                           for name in ("persistence", "native_bolt", "gofa_bolt", "matched_hsdgnn")}
    if set(historical) != expected_references:
        raise ValueError("Historical raw-target references are missing or unexpected")
    historical_manifest = read(reference / "manifest.json", {})
    if historical_manifest.get("config", {}).get("seed") != 42:
        raise ValueError("Historical reference seed/protocol manifest missing")
    for name in historical:
        result = read(reference / name)
        if result.get("status") != "complete" or result.get("evaluation_protocol") != "raw-target-v2":
            raise ValueError("Incomplete or incompatible historical evaluation: " + name)
    return {"config": config, "inputs": inputs, "source": source, "versions": versions,
            "gpu_uuid": os.environ.get("GOFA_GPU_UUID"),
            "weights": weights["file_sha256"], "historical_reference": historical,
            "historical_manifest_sha256": digest(reference / "manifest.json"),
            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}


def write_report(output, config):
    output = Path(output)
    status = read(output / "status.json", {})
    launcher = read(output / "launcher.json", {})
    if launcher.get("state") in ("wall_deadline", "failed", "incomplete") and status.get("state") in (
            None, "running", "interrupted"):
        status = {**status, "state": launcher["state"]}
    selection = read(output / "selection.json", {})
    ledger = read(output / "active_budget.json", {})
    lines = ["# " + output.name, "", f"State: {status.get('state', 'not_started')}", "",
             "Seed 42; raw-target-v2; total parameter matching deferred.",
             "UFG is the pinned effective forward; inactive branches are frozen and counted separately.",
             "Complex scalar = two real degrees of freedom. Different model-native losses/optimizers are disclosed.",
             "Only validation selects LR/checkpoints. Stage3 tests were already observed: development evidence.", "",
             f"LR selection: `{json.dumps(selection, ensure_ascii=False)}`", "",
             "|Dataset|Variant|Test MAE|RMSE|MAPE %|MAPE fraction|WAPE %|Training complete|",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    results = {}
    for path in sorted((output / "trials").rglob("result.json")):
        result = read(path)
        results[result["dataset"] + "/" + result["variant"]] = result
        raw = result["test"]["raw"]
        lines.append(f"|{result['dataset']}|{result['variant']}|{raw['mae']:.6g}|{raw['rmse']:.6g}|"
                     f"{raw['mape_pct']}|{raw['mape_valid_fraction']:.6g}|{raw['wape_pct']}|"
                     f"{result['training_complete']}|")
    lines += ["", "|Dataset|Native Bolt MAE|G2-V MAE|UFG-S MAE|UFG-A MAE|Historical limited HSD MAE|",
              "|---|---:|---:|---:|---:|---:|"]
    reference = Path(config["stage4"]["reference_output"])
    def score(dataset, variant):
        r = results.get(dataset + "/" + variant)
        return f"{r['test']['raw']['mae']:.6g}" if r else "incomplete"
    for dataset in DATASETS:
        baseline = {}
        for name in ("native_bolt", "matched_hsdgnn"):
            value = read(reference / "reevaluation" / dataset / name / "result.json")
            baseline[name] = f"{value['test']['raw']['mae']:.6g}" if value else "unavailable"
        lines.append(f"|{dataset}|{baseline['native_bolt']}|{score(dataset,'g2_values')}|"
                     f"{score(dataset,'ufg_s')}|{score(dataset,'ufg_a')}|{baseline['matched_hsdgnn']}|")
    lines += ["", "|Job|Updates|Epochs|Reason|Registered real DOF|Active real DOF|Trainable real DOF|",
              "|---|---:|---:|---|---:|---:|---:|"]
    for path in sorted((output / "trials").rglob("done.json")):
        value = read(path)
        locked = read(output / "final_selection.json", {}).get("jobs", {}).get(
            value["dataset"] + "/" + value["variant"], {})
        value = {**value, **{key: locked[key] for key in ("updates", "epochs", "stop_reason") if key in locked}}
        counts = value["parameters"]
        lines.append(f"|{value['dataset']}/{value['variant']}|{value['updates']}|{value['epochs']}|"
                     f"{value['stop_reason']}|{counts['registered_real_dof']}|{counts['active_real_dof']}|"
                     f"{counts['trainable_real_dof']}|")
    lines += ["", f"Supervisor activity ledger: {ledger.get('spent_seconds', 0):.2f} seconds; no activity cap.",
              "Wall deadline: " + config["stage4"]["wall_deadline"],
              "Idle waits excluded. Other users' GPU processes must always be yielded to.", "",
              "Electricity diagnostics (train/validation only) are in each job's diagnostics.json.",
              "Known limitations: one seed, overlapping AQI datasets, prior test exposure, limited convergence.",
              "No language-interaction or single-checkpoint cross-dataset generalization claim.", "",
              "Failures: `" + json.dumps(read(output / "failures.json", {}), ensure_ascii=False) + "`", ""]
    temporary = output / "report.md.tmp"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(output / "report.md")
    atomic_json(output / "report.json", {"state": status.get("state"), "results": results,
                "selection": selection, "activity": ledger, "failures": read(output / "failures.json", {})})


def stage(config_path, output, resume=False):
    global STOP
    if os.environ.get("GOFA_IDLE_ADMITTED") != "1":
        raise RuntimeError("Start only through the idle supervisor")
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: globals().__setitem__("STOP", True))
    config = config_file(config_path)
    output = Path(output).resolve()
    expected = manifest(config)
    old = read(output / "manifest.json")
    if old is not None and (not resume or old != expected):
        raise ValueError("Resume manifest changed: code/data/source/environment/protocol")
    if old is None and resume:
        raise ValueError("No stage manifest to resume")
    atomic_json(output / "manifest.json", expected)
    final_deadline = parse_deadline(config["stage4"]["wall_deadline"])
    training_deadline = final_deadline - config["stage4"]["final_evaluation_reserve_seconds"]
    failures = read(output / "failures.json", {})

    def status(state, **fields):
        atomic_json(output / "status.json", {"state": state, "pid": os.getpid(), "updated_at": time.time(),
                    "git_revision": expected["git_revision"], **fields})

    def task(phase, dataset, variant, lr=.001, milestone=100):
        global STOP
        key = f"{phase}/{dataset}/{variant}/{lr}/{milestone}"
        if key in failures or (phase == "train" and any(
                name.startswith(f"train/{dataset}/{variant}/{lr}/") for name in failures)):
            return False
        boundary = final_deadline if phase in ("evaluate", "diagnose") else training_deadline
        if time.time() >= boundary - 60:
            return False
        if STOP or os.getppid() != int(os.environ["GOFA_SUPERVISOR_PID"]):
            raise InterruptedError("Supervisor requested release or disappeared")
        family = "ufg" if variant.startswith("ufg") else "native"
        command = [config["stage4"][family + "_python"], "-u", "-m", "gofa_f.pure_graph_task",
                   "--config", config_path, "--output", str(output), "--phase", phase,
                   "--dataset", dataset, "--variant", variant, "--lr", str(lr), "--milestone", str(milestone)]
        environment = dict(os.environ, GOFA_SUPERVISOR_PID=str(os.getpid()),
                           GOFA_TASK_DEADLINE=str(time.monotonic()+max(0, boundary-time.time())))
        status("running", phase=phase, dataset=dataset, variant=variant, lr=lr, milestone=milestone)
        child = subprocess.Popen(command, env=environment)
        while child.poll() is None:
            if STOP or os.getppid() != int(os.environ["GOFA_SUPERVISOR_PID"]):
                STOP = True
                try:
                    child.terminate()
                except ProcessLookupError:
                    pass
            time.sleep(1)
        if STOP:
            raise InterruptedError("Save and release GPU")
        if child.returncode == 75:
            if time.time() >= boundary - 65:
                return False
            raise InterruptedError("Child requested resumable release")
        if child.returncode:
            failures[key] = {"returncode": child.returncode, "command": command,
                             "observed_at": datetime.now().astimezone().isoformat()}
            atomic_json(output / "failures.json", failures)
            return False
        return True

    try:
        ready = {}
        for family in ("ufg", "native"):
            p = output / "mechanisms" / family / "mechanisms.json"
            passed = read(p, {}).get("passed", False)
            if family == "native":
                passed = passed and read(p.parent / "values_only.json", {}).get("passed", False)
            ready[family] = passed or task("preflight", "metr_la", family)
        persistence = {d: read(Path(config["stage4"]["reference_output"]) / "reevaluation" / d /
                              "persistence/result.json")["validation"]["raw"]["mae"] for d in KEY_DATASETS}
        for dataset in KEY_DATASETS:
            for variant in ("ufg_s", "ufg_a"):
                for lr in config["stage4"]["learning_rates"]:
                    path = task_directory(output, dataset, variant, lr) / "pilot_done.json"
                    if ready["ufg"] and not path.exists():
                        task("pilot", dataset, variant, lr)
        selection = read(output / "selection.json")
        if selection is None:
            selection = {}
            for variant in ("ufg_s", "ufg_a"):
                scores = {d: {str(lr): read(task_directory(output, d, variant, lr) /
                    "pilot_done.json", {}).get("validation_raw_mae") for lr in config["stage4"]["learning_rates"]}
                    for d in KEY_DATASETS}
                selection[variant] = select_lr(scores, persistence)
            atomic_json(output / "selection.json", selection)
        jobs = []
        for dataset in DATASETS:
            if ready["native"]:
                jobs.append((dataset, "g2_values", .001))
                if dataset in KEY_DATASETS:
                    jobs.append((dataset, "g2_self", .001))
                if dataset == "metr_la":
                    jobs.append((dataset, "g2_clock", .001))
            for variant, selected in selection.items():
                if selected and ready["ufg"]:
                    jobs.append((dataset, variant, selected["lr"]))
        # Epoch milestones allow all domains coverage without changing the 100-epoch/15-patience protocol.
        for milestone in config["stage4"]["milestones"]:
            for dataset, variant, lr in jobs:
                directory = task_directory(output, dataset, variant, lr)
                done = read(directory / "done.json", {})
                if not done.get("complete") and not (directory / f"milestone_{milestone}.json").exists():
                    task("train", dataset, variant, lr, milestone)
        locked = read(output / "final_selection.json")
        if locked is None:
            locked = {"selection_split": "validation", "sealed_at": time.time(), "jobs": {}}
            for dataset, variant, lr in jobs:
                directory = task_directory(output, dataset, variant, lr)
                record = read(directory / "done.json", read(directory / "metadata.json", {}))
                progress = read(directory / "progress.json", {})
                if (directory / "best.pt").exists() and progress.get("best_mae") is not None:
                    complete = progress["stale"] >= 15 or progress["epoch"] >= 100 or (
                        progress["epoch"] == 99 and progress["offset"] >= record.get("training_windows", float("inf")))
                    completed_epochs = progress["epoch"] + int(
                        progress["offset"] >= record.get("training_windows", float("inf")))
                    locked["jobs"][dataset + "/" + variant] = {
                        "lr": lr, "checkpoint_sha256": digest(directory / "best.pt"),
                        "training_complete": complete, "validation_raw_mae": progress["best_mae"],
                        "updates": progress["updates"], "epochs": completed_epochs,
                        "stop_reason": "early_stopping" if progress["stale"] >= 15 else
                        "max_epochs" if complete else "wall_deadline_training", "parameters": record.get("parameters")}
            atomic_json(output / "final_selection.json", locked)
        for dataset, variant, lr in jobs:
            directory = task_directory(output, dataset, variant, lr)
            if dataset == "electricity" and variant != "g2_clock" and dataset + "/" + variant in locked["jobs"]:
                if not (directory / "diagnostics.json").exists():
                    task("diagnose", dataset, variant, lr)
        # Evaluate main rows before optional mechanism controls; never inspect test during selection.
        for dataset, variant, lr in sorted(jobs, key=lambda j: (j[1] in ("g2_self", "g2_clock"), DATASETS.index(j[0]))):
            directory = task_directory(output, dataset, variant, lr)
            if dataset + "/" + variant in locked["jobs"] and not (directory / "result.json").exists():
                task("evaluate", dataset, variant, lr)
        expected_rows = [d + "/" + v for d in DATASETS for v in ("g2_values", "ufg_s", "ufg_a")]
        actual_rows = {read(p)["dataset"] + "/" + read(p)["variant"]
                       for p in (output / "trials").rglob("result.json")}
        state = "complete" if set(expected_rows).issubset(actual_rows) and not failures and all(
            value["training_complete"] for value in locked["jobs"].values()) else "incomplete"
        status(state, evaluated_rows=len(actual_rows), expected_main_rows=len(expected_rows))
        write_report(output, config)
    except InterruptedError:
        status("interrupted")
        write_report(output, config)
        raise SystemExit(75)
    except Exception as exc:
        status("failed", error=repr(exc))
        write_report(output, config)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage4_pure_graph.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        write_report(args.output, config_file(args.config))
    else:
        stage(args.config, args.output, args.resume)


if __name__ == "__main__":
    main()
