"""CPU-only Stage16 supervisor with a persisted user GPU-sharing policy."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import yaml

from gofa_f.idle_launcher import (CapOnlyGate, IdleGate, SharedGate, atomic_json, launch as idle_launch,
    parse_deadline, process_identity, query_gpu)

STOP = False
PROTOCOL = "stage16-chattime-metr-v1"
REVISION = "20baa88d6ab838b34abbd1d2b1f4cea57c8ff870"
SOURCE = "8c2d4c209302d2b2bd6cc3c154586842341e3247"


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def config_file(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config["protocol"] != PROTOCOL or config["seed"] != 42 or config["datasets"] != ["metr_la"] \
            or config["models"] != ["native_chattime", "gofa_chattime"]:
        raise ValueError("Unregistered Stage16 experiment")
    if config["weights"]["revision"] != REVISION or config["weights"]["source_revision"] != SOURCE \
            or config["weights"]["repo"] != "ChengsenWang/ChatTime-1-7B-Chat":
        raise ValueError("Unregistered pretrained checkpoint")
    if config["architecture"] != {"graph_layers": [29, 30], "rank": 64, "attention_rank": 16,
            "feature_dim": 19, "prior_floor": .5, "head": "linear_4096_to_12_zero"}:
        raise ValueError("Unregistered graph/head architecture")
    training = {"context": 12, "horizon": 12, "effective_batch_size": 64, "microbatch": 1,
        "max_epochs": 100, "patience": 15, "graph_lr": .001, "head_lr": .0001,
        "weight_decay": .0001, "gradient_clip": 1., "loss": "raw_mae", "checkpoint_seconds": 30}
    if config["training"] != training or config["native_decoding"] != {
            "do_sample": False, "max_new_tokens": 64, "node_microbatch": 256}:
        raise ValueError("Unregistered optimization or decoding")
    r = config["resources"]
    if r["wall_window_hours"] != 11 or r["evaluation_reserve_hours"] != 2 or \
            r["memory_cap_gib"] != 40 or r["torch_cap_gib"] != 38 or r["leave_free_gib"] != 16 or \
            r["yield_on_foreign"] is not False or r["admission_per_gpu_task"] is not True or \
            config["cache"] != {"disk_limit_gib": 128, "ram_limit_gib": 64, "prefix_layers": 30}:
        raise ValueError("Shared resource/cache/window policy changed")
    return config


def code_files():
    paths = sorted(Path("src/gofa_f").glob("*.py"))
    paths += [Path("configs/stage16_chattime.yaml"), Path("scripts/check_chattime_cpu.py")]
    paths += sorted(Path("tests").glob("test_chattime*.py"))
    return {str(p): digest(p) for p in paths}


def deadline_extension(adopted_from, hard_deadline):
    """Record an explicit user amendment separately from an ordinary resume."""
    inherited = (adopted_from or {}).get("deadline_amendment")
    if inherited is not None and hard_deadline == adopted_from["original_wall_deadline"]:
        if adopted_from.get("kind") != "training_continuation" or not adopted_from.get("window_sha256") \
                or inherited.get("authorization") != "explicit_user_request" or not inherited.get("source_window_sha256") \
                or inherited.get("new_wall_deadline") != hard_deadline \
                or inherited.get("previous_wall_deadline", hard_deadline) >= hard_deadline:
            raise ValueError("Inherited deadline authorization changed")
        return inherited
    if not adopted_from or adopted_from.get("kind") != "training_continuation" \
            or not adopted_from.get("window_sha256") \
            or hard_deadline is None or hard_deadline <= adopted_from["original_wall_deadline"]:
        raise ValueError("An authorized extension requires a saved continuation and a later absolute deadline")
    return {"authorization": "explicit_user_request", "source_window_sha256": adopted_from["window_sha256"],
            "previous_wall_deadline": adopted_from["original_wall_deadline"], "new_wall_deadline": hard_deadline}


def bind_window(output, resume=False, now=None, launch_fingerprint=None, hard_deadline=None,
                adopted_from=None, shared_admission=False, fixed_prompt_kv=False, idle_interval_seconds=30,
                deadline_amendment=None, memory_cap_only=False):
    if shared_admission and memory_cap_only:
        raise ValueError("GPU admission policies are mutually exclusive")
    policy = "memory_cap_only" if memory_cap_only else "shared_memory" if shared_admission else "idle"
    if not isinstance(idle_interval_seconds, int) or not 1 <= idle_interval_seconds <= 60:
        raise ValueError("GPU admission interval must be an integer from1 to60 seconds")
    path = Path(output)/"window.json"
    old = read(path)
    if old is not None:
        if not resume:
            raise ValueError("Existing Stage16 window requires --resume")
        amendment = old.get("deadline_amendment")
        if amendment is not None and amendment != deadline_extension(old.get("adopted_from"), old["wall_deadline"]):
            raise ValueError("Saved deadline amendment changed")
        if not 2*3600 < old["window_seconds"] or (old["window_seconds"] > 11*3600 and amendment is None) \
                or old["wall_deadline"]-old["started_at"] != old["window_seconds"] \
                or old["training_deadline"] != old["wall_deadline"]-2*3600:
            raise ValueError("Saved absolute window changed")
        if amendment is not None and old["started_at"] != old["adopted_from"]["started_at"]:
            raise ValueError("An amended window must preserve its original start")
        if deadline_amendment is not None:
            raise ValueError("Resume cannot create a new deadline amendment")
        if hard_deadline is not None and hard_deadline != old["wall_deadline"]:
            raise ValueError("Resume must retain the saved shortened deadline")
        if old.get("admission_policy", "idle") != policy:
            raise ValueError("Resume must retain the saved GPU admission policy")
        if memory_cap_only and old.get("leave_free_gib") != 0:
            raise ValueError("Cap-only policy cannot restore a free-memory margin")
        if bool(old.get("fixed_prompt_kv")) != bool(fixed_prompt_kv):
            raise ValueError("Resume must retain the fixed prompt cache policy")
        if old.get("idle_interval_seconds", 30) != idle_interval_seconds:
            raise ValueError("Resume must retain the saved admission interval")
        if launch_fingerprint is not None and old.get("launch_fingerprint") != launch_fingerprint:
            raise ValueError("Saved launch identity changed")
        return old
    if resume:
        raise ValueError("Missing persisted Stage16 window")
    now = time.time() if now is None else now
    hard_deadline = now+11*3600 if hard_deadline is None else hard_deadline
    if deadline_amendment is not None:
        if deadline_amendment != deadline_extension(adopted_from, hard_deadline) or now != adopted_from["started_at"]:
            raise ValueError("Deadline amendment must retain the exact source window and original start")
    if not now+2*3600 < hard_deadline or (hard_deadline > now+11*3600 and deadline_amendment is None):
        raise ValueError("An explicit deadline can only shorten the eleven-hour maximum")
    result = {"started_at": now, "training_deadline": hard_deadline-2*3600,
              "wall_deadline": hard_deadline, "window_seconds": hard_deadline-now}
    if adopted_from is not None:
        result["adopted_from"] = adopted_from
    if deadline_amendment is not None:
        result["deadline_amendment"] = deadline_amendment
    if shared_admission:
        result["admission_policy"] = "shared_memory"
    if memory_cap_only:
        result.update(admission_policy="memory_cap_only", leave_free_gib=0)
    if idle_interval_seconds != 30:
        result["idle_interval_seconds"] = idle_interval_seconds
    if fixed_prompt_kv:
        result["fixed_prompt_kv"] = True
    if launch_fingerprint is not None:
        result["launch_fingerprint"] = launch_fingerprint
    atomic_json(path, result)
    return result


def adopt_waiting_window(previous, config, prepared, identity=process_identity):
    """Move only an exited CPU waiter with zero GPU activity; never migrate training."""
    previous = Path(previous).resolve()
    launcher = read(previous/"launcher.json", {})
    ledger = read(previous/"active_budget.json", {})
    budget = read(previous/"budget.json", {})
    if launcher.get("state") != "stopped_by_user" or ledger.get("spent_seconds") != 0 \
            or ledger.get("worker") is not None or budget.get("spent_seconds", 0) != 0:
        raise ValueError("Only an explicitly stopped, zero-activity CPU waiter can be adopted")
    for suffix in (".launch.json", ".bootstrap.launch.json"):
        record = read(previous.parent/(previous.name+suffix), {})
        if not record.get("pid") or record.get("start_ticks") is None:
            raise ValueError("Missing previous process identity")
        actual = identity(record["pid"])
        if actual is not None and str(actual) == str(record["start_ticks"]):
            raise ValueError("Previous CPU controller/waiter is still alive")
    if any((previous/name).exists() for name in ("status.json", "mechanisms.json", "parameters.json")):
        raise ValueError("GPU stage artifacts prohibit waiting-only adoption")
    for variant in ("gofa_chattime", "native_chattime"):
        directory = previous/variant
        # CPU --report-only creates these empty summary records even when no
        # CUDA child was ever admitted. They contain no predictions or training.
        files = [p for p in directory.rglob("*") if p.is_file()]
        if any(p != directory/"evaluation_progress.json" for p in files):
            raise ValueError("Checkpoint/prediction artifacts prohibit waiting-only adoption")
        summaries = read(directory/"evaluation_progress.json", {})
        if any(row.get("complete", False) or row.get("completed_windows", 0) != 0
               for row in summaries.values()):
            raise ValueError("Completed evaluation prohibits waiting-only adoption")
    selection = read(previous/"final_selection.json", {})
    if selection.get("gofa") is not None or selection.get("updates", 0) != 0:
        raise ValueError("A trained selection prohibits waiting-only adoption")
    original = read(previous/"manifest.json", {})
    if original.get("config") != config:
        raise ValueError("Only deadline/source infrastructure may change, not the experiment")
    def inputs(record, output):
        return {("local_correlation_graph" if Path(k).resolve() == output/"correlation_graph.npz" else k): v
                for k, v in record.items()}
    old_inputs = original.get("inputs", {})
    if not old_inputs or old_inputs != {p: digest(p) for p in old_inputs}:
        raise ValueError("Previous immutable inputs changed")
    current_output = Path(prepared["output"]).resolve()
    if inputs(old_inputs, previous) != inputs(prepared["inputs"], current_output):
        raise ValueError("Waiting adoption changed data, weights, features or graph")
    window = read(previous/"window.json")
    if not window or original.get("window") != window:
        raise ValueError("Previous window binding differs")
    return {"output": str(previous), "window_sha256": digest(previous/"window.json"),
        "manifest_sha256": digest(previous/"manifest.json"), "started_at": window["started_at"],
        "original_wall_deadline": window["wall_deadline"], "gpu_activity_seconds": 0}


def bind(path, expected, resume):
    old = read(path)
    if old is None:
        if resume:
            raise ValueError("Missing resume binding")
        atomic_json(path, expected)
    elif not resume or old != expected:
        raise ValueError("Resume differs: source/data/weights/features/environment/window")


def prepare(config, output):
    """CPU preparation, called before the eleven-hour window is created."""
    import numpy as np
    from gofa_f.data import Series, Windows
    from gofa_f.graphs import build_graph
    from gofa_f.graph_context import load_calendar
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    spec = yaml.safe_load(Path(config["data"]["benchmark"]).read_text())
    settings = spec["datasets"]["metr_la"]
    if settings["context"] != 12 or settings["horizons"] != [12] or settings["graph"]["kind"] != "road_distance":
        raise ValueError("Existing METR-LA task or physical graph changed")
    series = Series.load(settings["data"])
    normalized, _, _ = series.standardize()
    windows = {part: Windows(series, normalized, part, 12, 12) for part in ("train", "val", "test")}
    graph, graph_record = build_graph(series, settings)
    _, _, calendar_record = load_calendar(config["data"]["calendar"], series.fingerprint, len(series.values))
    correlation = Path(config["data"]["correlation"])
    if digest(correlation) != config["data"]["correlation_sha256"]:
        raise ValueError("Locked Stage8 graph changed")
    with np.load(correlation, allow_pickle=False) as source:
        if source["adjacency"].shape != graph.shape or int(source["train_end"]) != series.train_end \
                or str(source["prepared_sha256"]) != series.fingerprint:
            raise ValueError("Correlation is not bound to the training prefix")
    destination = output/"correlation_graph.npz"
    if destination.exists() and digest(destination) != digest(correlation):
        raise ValueError("Existing Stage16 graph changed")
    if not destination.exists():
        shutil.copyfile(correlation, destination)
    weights = Path(config["weights"]["directory"])
    snapshot = read(weights/"snapshot.json")
    if not snapshot or snapshot["revision"] != REVISION or snapshot["source_revision"] != SOURCE:
        raise ValueError("Pinned pretrained download is not complete")
    files = [Path(settings["data"]), Path(settings["graph"]["path"]),
             Path(settings["graph"]["path"]).with_suffix(".metadata.json"),
             Path(config["data"]["calendar"]), Path(config["data"]["calendar"]).with_suffix(".metadata.json"),
             destination, weights/"snapshot.json"]
    for name, record in snapshot["files"].items():
        path = weights/name
        if path.stat().st_size != record["size"] or digest(path) != record["sha256"]:
            raise ValueError("Pinned weight file mismatch: "+name)
        files.append(path)
    upstream = Path(config["weights"]["source_directory"])
    for name, checksum in snapshot["source_files"].items():
        if digest(upstream/name) != checksum:
            raise ValueError("Pinned official source mismatch: "+name)
        files.append(upstream/name)
    model_config = read(weights/"config.json")
    if {key: model_config[key] for key in ("num_hidden_layers", "hidden_size", "vocab_size")} != {
            "num_hidden_layers": 32, "hidden_size": 4096, "vocab_size": 42001}:
        raise ValueError("Unexpected ChatTime model shape")
    report = {"inputs": {str(p): digest(p) for p in files}, "data_fingerprint": series.fingerprint,
        "nodes": series.values.shape[1], "train_end": series.train_end, "val_end": series.val_end,
        "windows": {part: {"count": len(w), "first_origin": int(w.origins[0]), "last_origin": int(w.origins[-1])}
                    for part, w in windows.items()}, "physical_graph": graph_record,
        "calendar": calendar_record, "model_config": model_config,
        "statistics_fit": "observed training prefix only", "pretraining_overlap": "not fully auditable; no strict unseen-data claim"}
    atomic_json(output/"preparation.json", report)
    return report


def manifest(config, output, window):
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Use a clean fixed snapshot")
    prepared = read(Path(output)/"preparation.json")
    cpu = read(Path(output)/"cpu_preflight.json")
    if not prepared or not cpu or not cpu.get("passed") or cpu["code_files"] != code_files():
        raise ValueError("CPU preflight is missing or no longer matches the code")
    if prepared["inputs"] != {p: digest(p) for p in prepared["inputs"]}:
        raise ValueError("Prepared immutable input changed")
    result = {"config": config, "window": window, "inputs": prepared["inputs"],
        "code_files": code_files(), "cpu_preflight_sha256": digest(Path(output)/"cpu_preflight.json"),
        "runtime": cpu["runtime"], "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    if window and window.get("admission_policy") == "shared_memory":
        result["admission_policy"] = "shared_memory"
    if window and window.get("admission_policy") == "memory_cap_only":
        result.update(admission_policy="memory_cap_only", memory_cap_gib=40, torch_cap_gib=38, leave_free_gib=0)
    if window and window.get("fixed_prompt_kv"):
        result["fixed_prompt_kv"] = True
    if window and window.get("idle_interval_seconds", 30) != 30:
        result["idle_interval_seconds"] = window["idle_interval_seconds"]
    return result


def seal(output):
    output = Path(output)
    existing = read(output/"final_selection.json")
    if existing is not None:
        return existing
    progress = read(output/"gofa_chattime"/"progress.json", {})
    best = read(output/"gofa_chattime"/"best_validation.json")
    result = {"selection_split": "complete validation raw MAE", "sealed_at": time.time(),
              "native": {"revision": REVISION, "do_sample": False, "max_new_tokens": 64},
              "gofa": None, "updates": progress.get("updates", 0),
              "stop_reason": progress.get("stop_reason", "training_deadline")}
    if best:
        path = output/"gofa_chattime"/"best.pt"
        if digest(path) != best["checkpoint_sha256"]:
            raise ValueError("Best validation checkpoint hash changed")
        result["gofa"] = {**best, "path": str(path)}
    atomic_json(output/"final_selection.json", result)
    return result


def write_report(output):
    output = Path(output)
    state = read(output/"status.json", {}).get("state", "not_started")
    launcher = read(output/"launcher.json", {}).get("state")
    if launcher in ("wall_deadline", "failed") and state != "complete":
        state = launcher
    results = {name: read(output/name/"result.json") for name in ("native_chattime", "gofa_chattime")}
    if all(record and record.get("complete") for record in results.values()):
        state = "complete"
    progress = {name: read(output/name/"evaluation_progress.json", {}) for name in results}
    archived = read("docs/results/stage4-pure-graph-v2.json", {})
    old = archived.get("historical_reference", {}).get("metr_la", {})
    historical = {"stage4_native_bolt_g2_hsd": old,
        "stage4_gnn_rows": {name: row for name, row in archived.get("rows", {}).items()
                            if name.startswith("metr_la/")},
        "stage8": read("docs/results/stage8-metr-temporal-v1.json", {}).get("results", {}),
        "stage15_validation_only": read("docs/results/stage15-metr-reg6-v1.json", {}).get("final_selection"),
        "note": "References only; no new training, no parameter matching, Stage15 test remains unavailable"}
    report = {"state": state, "results": results, "evaluation_progress": progress,
        "historical_references": historical,
        "final_selection": read(output/"final_selection.json"),
        "training": read(output/"gofa_chattime"/"progress.json"),
        "parameters": read(output/"parameters.json"), "mechanisms": read(output/"mechanisms.json"),
        "failures": read(output/"failures.json", {}), "cache": read(output/"cache.json"),
        "activity": read(output/"active_budget.json", {}), "budget": read(output/"budget.json", {}),
        "memory": read(output/"cuda_memory.json", {}), "window": read(output/"window.json"),
        "limitations": ["single seed; development tests previously viewed", "native greedy differs from official 8 samples",
            "GOFA supervised continuous head and 19 history/clock features; no no-graph adaptation control",
            "historical UFG/HSD are references, not matched to the new parameter count",
            "pretraining data overlap not fully auditable; TITAN 3.08 is a cross-implementation reference"]}
    atomic_json(output/"report.json", report)
    lines = ["# Stage16 ChatTime / GOFA-F(ChatTime), METR-LA 12-to-12", "", "State: "+state, "",
        "|Model|Complete validation MAE|Complete test MAE|Test step12 MAE|Observed-target fallback rate|",
        "|---|---:|---:|---:|---:|"]
    for name, record in results.items():
        validation = record.get("validation") if record else None
        test = record.get("test") if record else None
        bound = report["final_selection"] or {}
        if name == "gofa_chattime" and validation is None and bound.get("gofa"):
            validation = bound["gofa"]["validation"]
        lines.append("|"+name+"|"+str(validation["raw"]["mae"] if validation else "—")+"|"+
            str(test["raw"]["mae"] if test else "—")+"|"+
            str(test["raw"]["by_horizon"][11]["mae"] if test else "—")+"|"+
            str(test.get("observed_fallback_fraction", 0) if test else "—")+"|")
    lines += ["", "Incomplete evaluation is excluded from the main metric table.",
              "Native full-coverage scores include the preregistered Persistence fallback; valid-generation scores are separate."]
    lines += ["", *report["limitations"]]
    (output/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def stage(config_path, output, resume):
    global STOP
    if os.environ.get("GOFA_IDLE_ADMITTED") != "1":
        raise RuntimeError("Registered GPU admission launcher required")
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: globals().__setitem__("STOP", True))
    config, output = config_file(config_path), Path(output)
    window = read(output/"window.json")
    shared = window.get("admission_policy") == "shared_memory"
    cap_only = window.get("admission_policy") == "memory_cap_only"
    admission_interval = window.get("idle_interval_seconds", 30)
    expected = manifest(config, output, window)
    if read(output/"launch_binding.json") != expected:
        raise ValueError("Launch binding differs")
    bind(output/"manifest.json", expected, resume)
    failures = read(output/"failures.json", {})

    def status(state, **fields):
        atomic_json(output/"status.json", {"state": state, "pid": os.getpid(), "updated_at": time.time(), **fields})

    def task(phase):
        boundary = window["wall_deadline"] if phase == "evaluate" else window["training_deadline"]
        gate = CapOnlyGate() if cap_only else SharedGate(40, 16) if shared else IdleGate()
        while time.time() < boundary-60:
            if STOP or os.getppid() != int(os.environ["GOFA_SUPERVISOR_PID"]):
                raise InterruptedError("Supervisor released")
            try:
                gpu = query_gpu()
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired, ET.ParseError):
                gpu = None
            admitted = gate.observe(gpu) and gpu["uuid"] == os.environ.get("GOFA_GPU_UUID")
            status("waiting_for_gpu_query" if cap_only else "waiting_for_shared_capacity" if shared else "waiting_for_idle",
                   phase=phase, consecutive_idle=gate.consecutive)
            if admitted:
                break
            for _ in range(admission_interval):
                if STOP or time.time() >= boundary-60:
                    break
                time.sleep(1)
        else:
            return False
        if time.time() >= boundary-60:
            return False
        status("running", phase=phase)
        environment = dict(os.environ, GOFA_SUPERVISOR_PID=str(os.getpid()),
                           GOFA_TASK_DEADLINE=str(time.monotonic()+boundary-time.time()),
                           TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                           HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        command = [config["resources"]["python"], "-u", "-m", "gofa_f.chattime_task",
                   "--config", config_path, "--output", str(output), "--phase", phase]
        child = subprocess.Popen(command, env=environment)
        while child.poll() is None:
            if STOP or os.getppid() != int(os.environ["GOFA_SUPERVISOR_PID"]):
                child.terminate()
            time.sleep(1)
        if STOP:
            raise InterruptedError("Checkpoint release")
        if child.returncode == 75:
            return False
        if child.returncode:
            failures[phase] = {"returncode": child.returncode, "time": time.time()}
            atomic_json(output/"failures.json", failures)
            raise RuntimeError("GPU task failed: "+phase)
        return True

    try:
        if not read(output/"mechanisms.json", {}).get("passed") and not task("preflight"):
            status("incomplete", reason="mechanisms missed training deadline")
            return
        if not (output/"final_selection.json").exists():
            task("train")
            seal(output)
        task("evaluate")
        enough = all(read(output/name/"result.json", {}).get("complete") for name in config["models"])
        status("complete" if enough else "incomplete")
    except InterruptedError:
        status("interrupted")
        raise SystemExit(75)
    except Exception as exc:
        status("failed", error=repr(exc))
        raise
    finally:
        write_report(output)


def main():
    # The parent retains this Linux lock throughout waiting and all child phases.
    # It is independent of stale PID numbers and is automatically released on exit.
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage16_chattime.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hard-deadline", help="Explicit timezone; may shorten but never extend the original window")
    parser.add_argument("--adopt-waiting-window", help="Exited CPU-only output whose original start time must be retained")
    parser.add_argument("--shared-gpu", action="store_true", help="Admit other applications with at least56GiB actually free")
    parser.add_argument("--memory-cap-only", action="store_true", help="Start sharing immediately; project cap40GiB, no free-memory margin")
    parser.add_argument("--fixed-prompt-kv", action="store_true")
    parser.add_argument("--idle-check-seconds", type=int, default=30)
    parser.add_argument("--authorized-deadline-extension", action="store_true",
                        help="Explicit user amendment; fresh continuation only, never an ordinary resume")
    parser.add_argument("--repair-failed-preflight", action="store_true")
    parser.add_argument("--continue-training-from", help="Stopped checkpoint to migrate without resetting training or time")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.shared_gpu and args.memory_cap_only:
        raise ValueError("GPU admission policies are mutually exclusive")
    if args.authorized_deadline_extension and (args.resume or not args.continue_training_from):
        raise ValueError("A user deadline extension requires a fresh training continuation")
    if args.repair_failed_preflight and (args.resume or not args.continue_training_from):
        raise ValueError("Preflight repair requires a fresh saved-state continuation")
    config, output = config_file(args.config), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.prepare_only:
        prepare(config, output)
        return
    if args.report_only:
        if (output/"final_selection.json").exists() and (output/"manifest.json").exists():
            # Aggregate only already-written forecasts. This path never loads weights
            # or creates a CUDA context, even after the wall deadline.
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["OMP_NUM_THREADS"] = "2"
            os.environ["MKL_NUM_THREADS"] = "2"
            from gofa_f.chattime_task import data_for, finalize_evaluation
            manifest_record = read(output/"manifest.json")
            if manifest_record["inputs"] != {p: digest(p) for p in manifest_record["inputs"]}:
                raise ValueError("Immutable data/source changed before CPU aggregation")
            binding = hashlib.sha256(json.dumps(manifest_record, sort_keys=True).encode()).hexdigest()
            data, _ = data_for(config, evaluation=True)
            finalize_evaluation(output, data, binding, read(output/"final_selection.json"))
        write_report(output)
        return
    if args.worker:
        stage(args.config, output, args.resume)
        return
    owner = (output/"owner.lock").open("a")
    try:
        fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("An existing Stage16 supervisor owns this output") from None
    # All checks occur BEFORE the first persisted countdown starts.
    cpu = read(output/"cpu_preflight.json", {})
    if not cpu.get("passed") or cpu.get("code_files") != code_files():
        raise ValueError("Current CPU preflight required before starting the clock")
    expected = manifest(config, output, None)
    if args.shared_gpu:
        expected["admission_policy"] = "shared_memory"
    if args.memory_cap_only:
        expected.update(admission_policy="memory_cap_only", memory_cap_gib=40, torch_cap_gib=38, leave_free_gib=0)
    if args.fixed_prompt_kv:
        expected["fixed_prompt_kv"] = True
    if args.idle_check_seconds != 30:
        expected["idle_interval_seconds"] = args.idle_check_seconds
    fingerprint = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
    cutoff = parse_deadline(args.hard_deadline)
    adopted = None
    amendment = None
    if args.adopt_waiting_window:
        if args.resume or (output/"window.json").exists() or cutoff is None:
            raise ValueError("Adoption requires a fresh output and an explicit shortened deadline")
        prepared = {**read(output/"preparation.json"), "output": str(output)}
        adopted = adopt_waiting_window(args.adopt_waiting_window, config, prepared)
        if cutoff > adopted["original_wall_deadline"]:
            raise ValueError("Adoption cannot extend the original deadline")
    if args.continue_training_from:
        if args.resume or args.adopt_waiting_window or (output/"window.json").exists() \
                or cutoff is None or not args.fixed_prompt_kv:
            raise ValueError("Cache continuation requires a fresh output and explicit cutoff")
        from gofa_f.chattime_continuation import source_record
        prepared = {**read(output/"preparation.json"), "output": str(output)}
        adopted = source_record(args.continue_training_from, config, prepared,
                                allow_failed_preflight=args.repair_failed_preflight,
                                allow_expired_waiter=args.authorized_deadline_extension)
        if cutoff > adopted["original_wall_deadline"] and not args.authorized_deadline_extension:
            raise ValueError("Continuation cannot extend the original deadline")
        if args.authorized_deadline_extension:
            amendment = deadline_extension(adopted, cutoff)
        elif adopted.get("deadline_amendment"):
            amendment = deadline_extension(adopted, cutoff)
    window = bind_window(output, args.resume, launch_fingerprint=fingerprint, hard_deadline=cutoff,
        now=adopted["started_at"] if adopted else None, adopted_from=adopted,
        shared_admission=args.shared_gpu, fixed_prompt_kv=args.fixed_prompt_kv,
        idle_interval_seconds=args.idle_check_seconds, deadline_amendment=amendment,
        memory_cap_only=args.memory_cap_only)
    if time.time() >= window["wall_deadline"]-60:
        atomic_json(output/"launcher.json", {"state": "wall_deadline"})
        if (output/"manifest.json").exists():
            seal(output)
        write_report(output)
        return
    expected["window"] = window
    # A crash immediately after window creation may leave no other bindings.
    # The persisted fingerprint permits their reconstruction without a new timer.
    bind(output/"launch_binding.json", expected, args.resume and (output/"launch_binding.json").exists())
    bind(output/"manifest.json", expected, args.resume and (output/"manifest.json").exists())
    continuation = window.get("adopted_from", {})
    if continuation.get("kind") == "training_continuation":
        from gofa_f.chattime_continuation import copy_continuation
        binding = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
        copy_continuation(continuation, output, binding)
    command = [sys.executable, "-u", "-m", "gofa_f.chattime_stage", "--worker", "--config", args.config,
               "--output", str(output), "--resume"]
    idle_launch(output, command, total=None, wall_deadline=window["wall_deadline"], memory_cap_gib=40,
                leave_free_gib=0 if args.memory_cap_only else 16, yield_on_foreign=False,
                worker_wait_state="waiting_for_gpu_query" if args.memory_cap_only else
                    "waiting_for_shared_capacity" if args.shared_gpu else "waiting_for_idle",
                shared_admission=args.shared_gpu, idle_interval_seconds=args.idle_check_seconds,
                cap_only_admission=args.memory_cap_only)
    if (output/"manifest.json").exists():
        seal(output)
    write_report(output)


if __name__ == "__main__":
    main()
