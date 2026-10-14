"""Four-hour validation-gated late GOFA-F adaptation, admitted by the idle launcher.

Historical checkpoints are re-evaluated without retraining or rewriting their
results. New test targets are read only after the global variant is locked.
"""

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

import numpy as np
import torch
import yaml

from gofa_f import native_stage as runtime
from gofa_f.data import file_sha256
from gofa_f.graph_context import FEATURE_DIM
from gofa_f.graphs import build_graph
from gofa_f.models import Persistence, parameter_counts
from gofa_f.native import NativeBolt, HSDGNNAdapter, late_contiguous_layers
from gofa_f.prefix_cache import PrefixCache, atomic_json
from gofa_f.stage_budget import Budget, BudgetExhausted


VARIANTS = {
    "G0": {"graph": True, "context": False, "output_rank": 0},
    "G1": {"graph": True, "context": True, "output_rank": 0},
    "G2": {"graph": True, "context": True, "output_rank": 4},
    "B1": {"graph": False, "context": False, "output_rank": 4},
}
KEY_DATASETS = ["metr_la", "electricity"]
EXTENSION_DATASETS = ["pems_bay", "aqi", "aqi_36", "solar"]
EVALUATION_PROTOCOL = "raw-target-v2"


class IncompleteTrial(RuntimeError):
    pass


class ActiveBudget(Budget):
    def check(self):
        if runtime.STOP:
            raise runtime.StopRequested("Save and release GPU")
        supervisor = os.environ.get("GOFA_SUPERVISOR_PID")
        if supervisor and os.getppid() != int(supervisor):
            raise runtime.StopRequested("CPU supervisor exited; save and release GPU")
        super().check()

    def remaining(self):
        remaining = super().remaining()
        deadline = float(os.environ.get("GOFA_ACTIVE_DEADLINE", "inf"))
        # Reserve enough parent-owned active time for saving and CUDA teardown.
        return min(remaining, deadline - time.monotonic() - 60)


def validate_config(config):
    a = config["adaptation"]
    if (
        config["protocol"] != "native-v5-late-context-adapter"
        or config["seed"] != 42
        or a["layout"] != "late_contiguous"
        or a["variants"] != VARIANTS
        or config["models"]["gofa_bolt"]["graph_layers"] != [1, 2]
        or config["models"]["gofa_bolt"].get("layout_candidates") is not None
    ):
        raise ValueError("This stage requires the registered late-only four-variant protocol")
    if a["key_datasets"] != KEY_DATASETS or a["extension_datasets"] != EXTENSION_DATASETS:
        raise ValueError("Unexpected validation/extension datasets")
    if (
        config["budget"]["total_gpu_seconds"] != 14400
        or a["evaluation_protocol"] != EVALUATION_PROTOCOL
        or a["graph_lr"] != 0.001
        or a["adapter_lr"] != 0.0001
        or a["selection_relative_tolerance"] != 0.0001
    ):
        raise ValueError("Unexpected budget, optimizer or selection protocol")
    if config["resources"]["memory_cap_gib"] > 40 or config["resources"]["leave_free_gib"] < 16:
        raise ValueError("Shared GPU limits were relaxed")
    for field, expected in {
        "mechanism_and_reevaluation_seconds": 900,
        "key_trial_seconds": 900,
        "extension_trial_seconds": 1200,
        "final_evaluation_seconds": 900,
        "recovery_reserve_seconds": 600,
        "waiting_for_idle_counts": False,
    }.items():
        if config["budget"][field] != expected:
            raise ValueError("Unexpected registered phase budget")
    if config["resources"]["effective_batch_size"] != 64:
        raise ValueError("Paired effective batch must remain 64")


def make_model(native, adjacency, horizon, variant):
    options = VARIANTS[variant]
    return NativeBolt(
        native,
        adjacency,
        horizon,
        late_contiguous_layers(len(native.encoder.block)) if options["graph"] else (),
        rank=64,
        graph_feature_dim=FEATURE_DIM if options["context"] else 0,
        output_adapter_rank=options["output_rank"],
    )


def select_variant(scores, tolerance=1e-4):
    """Only validation scalars are accepted; test fields never enter this function."""
    eligible, rejected = [], {}
    for variant in ("G1", "G2"):
        reasons, ratios = [], []
        for dataset in KEY_DATASETS:
            values = scores.get(dataset, {})
            required = ["G0", variant] + (["B1"] if variant == "G2" else [])
            if any(values.get(name) is None or not math.isfinite(values[name]) for name in required):
                reasons.append(dataset + ": missing complete validation")
                continue
            baseline, value = values["G0"], values[variant]
            if baseline <= 0 or value >= baseline * (1 - tolerance):
                reasons.append(dataset + ": did not improve G0")
            if variant == "G2" and value >= values["B1"] * (1 - tolerance):
                reasons.append(dataset + ": did not improve head-only B1")
            ratios.append(value / baseline if baseline > 0 else float("inf"))
        if reasons:
            rejected[variant] = reasons
        else:
            eligible.append((math.sqrt(math.prod(ratios)), variant))
    if not eligible:
        return {"selected": None, "scores": scores, "rejected": rejected, "eligible": []}
    eligible.sort()
    best_score = eligible[0][0]
    tied = [item for item in eligible if math.isclose(item[0], best_score, rel_tol=tolerance)]
    chosen = min(tied, key=lambda item: (VARIANTS[item[1]]["output_rank"], item[0]))
    return {
        "selected": chosen[1],
        "geometric_mean_ratio": chosen[0],
        "scores": scores,
        "rejected": rejected,
        "eligible": [v for _, v in eligible],
    }


def checkpoint_path(historical, dataset, model, result):
    if model == "gofa_bolt":
        return historical / dataset / "gofa_bolt" / result["selection"]["name"] / "best.pt"
    if model == "matched_hsdgnn":
        return historical / dataset / "hsdgnn" / f"lr_{result['selection']['lr']}" / "best.pt"
    return None


def make_manifest(config, spec):
    historical = Path(config["adaptation"]["historical_output"])
    sources = {}
    for dataset in config["datasets"]:
        for model in ("persistence", "native_bolt", "gofa_bolt", "matched_hsdgnn"):
            path = historical / dataset / model / "result.json"
            result = json.loads(path.read_text())
            if result["status"] != "complete" or result["seed"] != 42:
                raise ValueError("Historical reference is incomplete or has another seed")
            ckpt = checkpoint_path(historical, dataset, model, result)
            sources[f"{dataset}/{model}"] = {
                "result_sha256": file_sha256(path),
                "checkpoint_sha256": file_sha256(ckpt) if ckpt else None,
            }
    graphs = {}
    for dataset, settings in spec["datasets"].items():
        if settings.get("graph", {}).get("path"):
            path = Path(settings["graph"]["path"])
            graphs[dataset] = {"graph_sha256": file_sha256(path)}
            metadata = path.with_suffix(".metadata.json")
            if metadata.exists():
                graphs[dataset]["metadata_sha256"] = file_sha256(metadata)
    return {
        "config": config,
        "datasets": spec["datasets"],
        "historical_inputs": sources,
        "context_inputs": runtime.context_inputs(config, spec),
        "spatial_inputs": graphs,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
    }


def bind_manifest(output, manifest, resume):
    if subprocess.check_output(["git", "status", "--porcelain"]).strip():
        raise ValueError("Launch/resume only a clean committed snapshot")
    path = output / "manifest.json"
    if path.exists():
        if not resume or json.loads(path.read_text()) != manifest:
            raise ValueError("Resume requires the same immutable code, data, clock, heads and optimizer")
    else:
        if resume:
            raise ValueError("No worker manifest to resume")
        atomic_json(path, manifest)


def new_data(config, spec, dataset, device, context=True):
    settings = runtime.dataset_settings(spec["datasets"][dataset], config, dataset)
    if not context:
        settings.pop("graph_context", None)
    return runtime.Data(settings, device), settings


def prefix_key(data, revision):
    return {
        "data_sha256": data.series.fingerprint,
        "revision": revision,
        "preprocessing_hash": hashlib.sha256(
            data.mean.cpu().numpy().tobytes() + data.std.cpu().numpy().tobytes()
        ).hexdigest(),
        "context": data.context,
        "horizon": data.horizon,
        "split": "train",
        "prefix_layers": 2,
        "dtype": "bfloat16",
        "native_eval": True,
        "origins_hash": hashlib.sha256(
            np.asarray(data.windows["train"].origins, np.int64).tobytes()
        ).hexdigest(),
    }


def trial(output, dataset, variant, native, data, graph, gpu, budget, config, cache):
    directory = output / "trials" / dataset / variant
    done = directory / "done.json"
    if done.exists():
        return json.loads(done.read_text())
    runtime.seed(42)
    model = make_model(native, graph, data.horizon, variant).to(gpu.device)
    state = runtime.fit(
        model,
        data,
        directory,
        gpu,
        budget,
        config["adaptation"]["graph_lr"],
        cache,
        adapter_lr=config["adaptation"]["adapter_lr"],
    )
    record = {
        "variant": variant,
        "validation_raw_mae": state["best_mae"],
        "parameters": parameter_counts(model),
        "updates": state["updates"],
        "epochs": state["epoch"],
        "stale": state["stale"],
        "stop_reason": "max_epochs"
        if state["epoch"] >= 30
        else "early_stopping"
        if state["stale"] >= 5
        else "time_budget",
        "valid": state["best_mae"] is not None,
    }
    atomic_json(done, record)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return record


def train_dataset(output, dataset, variants, limit, native, gpu, budget, config, spec, status):
    if all((output / "trials" / dataset / v / "done.json").exists() for v in variants):
        return
    data, settings = new_data(config, spec, dataset, gpu.device)
    graph, graph_info = build_graph(data.series, settings)
    directory = output / "trials" / dataset
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(
        directory / "preprocessing.json",
        {
            "data_sha256": data.series.fingerprint,
            "context": data.context,
            "horizon": data.horizon,
            "graph": graph_info,
            "graph_context": data.context_provenance,
            "windows": {k: len(v) for k, v in data.windows.items()},
        },
    )
    cache = None
    try:
        for variant in variants:
            if (directory / variant / "done.json").exists():
                continue
            budget.activate(f"trial/{dataset}/{variant}", limit)
            status("running", dataset=dataset, variant=variant, phase="cache")
            if cache is None:
                runtime.seed(42)
                model = make_model(native, graph, data.horizon, "G0").to(gpu.device)
                cache = PrefixCache.build(
                    directory / "_cache",
                    prefix_key(data, config["models"]["native_bolt"]["revision"]),
                    model,
                    data.batches("train", gpu.micro),
                    len(data.windows["train"]),
                    lambda: gpu.check(budget),
                )
                del model
            status("running", dataset=dataset, variant=variant, phase="training")
            record = trial(output, dataset, variant, native, data, graph, gpu, budget, config, cache)
            if dataset not in KEY_DATASETS and not record["valid"]:
                raise IncompleteTrial(f"{dataset}/{variant} has no complete validation checkpoint")
    finally:
        if cache is not None:
            complete = all((directory / v / "done.json").exists() for v in variants)
            cache.close(delete=complete)
        del data
        gc.collect()
        torch.cuda.empty_cache()


def load_best(path, model):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    runtime.load_trained(model, saved["trained"])
    return saved


@torch.no_grad()
def reevaluate_historical(output, config, spec, native, gpu, budget, status):
    historical = Path(config["adaptation"]["historical_output"])
    for dataset in config["datasets"]:
        data, _ = new_data(config, spec, dataset, gpu.device, context=False)
        with np.load(historical / dataset / "preprocessing.npz") as saved:
            graph = saved["adjacency"]
            np.testing.assert_array_equal(saved["mean"], data.mean.cpu().numpy())
            np.testing.assert_array_equal(saved["std"], data.std.cpu().numpy())
        for name in ("persistence", "native_bolt", "gofa_bolt", "matched_hsdgnn"):
            target = output / "reevaluation" / dataset / name / "result.json"
            if target.exists():
                continue
            gpu.check(budget)
            source = historical / dataset / name / "result.json"
            result = json.loads(source.read_text())
            if name == "persistence":
                model = Persistence(data.horizon).to(gpu.device)
            elif name == "matched_hsdgnn":
                steps = {
                    "metr_la": 288,
                    "pems_bay": 288,
                    "aqi": 24,
                    "aqi_36": 24,
                    "electricity": 24,
                    "solar": 144,
                }[dataset]
                model = HSDGNNAdapter(
                    config["models"]["matched_hsdgnn"]["root"],
                    data.context,
                    data.horizon,
                    data.series.values.shape[1],
                    steps,
                ).to(gpu.device)
            else:
                layers = result["selection"]["layers"] if name == "gofa_bolt" else ()
                model = NativeBolt(native, graph, data.horizon, layers).to(gpu.device)
            ckpt = checkpoint_path(historical, dataset, name, result)
            if ckpt:
                load_best(ckpt, model)
            status("running", phase="historical_reevaluation", dataset=dataset, model=name)
            corrected = {
                "status": "complete",
                "dataset": dataset,
                "model": name,
                "source_result_sha256": file_sha256(source),
                "source_checkpoint": str(ckpt) if ckpt else None,
                "source_git_revision": result["git_revision"],
                "source_selection": result["selection"],
                "evaluation_protocol": EVALUATION_PROTOCOL,
                "validation": runtime.evaluate(model, data, "val", gpu, budget),
                "test": runtime.evaluate(model, data, "test", gpu, budget),
                "parameters": parameter_counts(model),
            }
            corrected["original_test_mape_pct"] = result["test"]["raw"]["mape_pct"]
            corrected["test_raw_mae_delta"] = corrected["test"]["raw"]["mae"] - result["test"]["raw"]["mae"]
            target.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(target, corrected)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        del data


def mechanism_check(output, native, config, spec, gpu, budget):
    path = output / "mechanisms.json"
    if path.exists():
        return
    data, _ = new_data(config, spec, "metr_la", gpu.device)
    batch = data.batch("train", [0, 1])
    x, mask, features = batch["x"][:, :, :3], batch["x_mask"][:, :, :3], batch["graph_features"][:, :3]
    runtime.seed(42)
    model = make_model(native, np.ones((3, 3), np.float32), 3, "G2").to(gpu.device)
    context = x.transpose(1, 2).reshape(6, -1).masked_fill(~mask.transpose(1, 2).reshape(6, -1), torch.nan)
    gpu.check(budget)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = (
            native(context=context).quantile_preds[:, model.median_index, :3].reshape(2, 3, 3).transpose(1, 2)
        )
        actual = model(x, mask, features)
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    for layer in model.graph.values():
        layer.gate.data.fill_(0.05)
        layer.context_projection.weight.data.normal_(0, 0.01)
    model.output_adapter.b.data.normal_(0, 0.001)
    disk_cache = PrefixCache.build(
        output / "_mechanism_cache",
        {"prefix_layers": 2},
        model,
        [{"x": x, "x_mask": mask}],
        2,
        lambda: gpu.check(budget),
    )
    outputs, gradients = [], []
    for cached in (False, True):
        gpu.check(budget)
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = (
                model.from_prefix(disk_cache.take(np.arange(2), gpu.device), 2, graph_features=features)
                if cached
                else model(x, mask, features)
            )
        y.float().square().mean().backward()
        outputs.append(y.detach())
        gradients.append({n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad})
    torch.testing.assert_close(*outputs, rtol=2e-3, atol=2e-3)
    for name in gradients[0]:
        torch.testing.assert_close(gradients[0][name], gradients[1][name], rtol=2e-3, atol=2e-3)
        if not torch.isfinite(gradients[0][name]).all():
            raise FloatingPointError("Nonfinite adapter/graph mechanism gradient")
    if any(p.grad is not None for p in native.parameters()):
        raise AssertionError("Native backbone received gradients")
    disk_cache.close(delete=True)
    # Actual fused CUDA AdamW restoration with BOTH parameter groups, using only training data.
    with tempfile.TemporaryDirectory(prefix="resume-check-", dir=output) as temporary:
        temporary = Path(temporary)
        values = data.series.values[:128, :3]
        observed = data.series.mask[:128, :3]
        np.savez(temporary / "train.npz", values=values, mask=observed)
        small = runtime.Data(
            {
                "data": str(temporary / "train.npz"),
                "context": 12,
                "horizons": [3],
                "graph_context": {"enabled": True},
            },
            gpu.device,
        )
        runtime.seed(42)
        resumed = make_model(native, np.eye(3), 3, "G2").to(gpu.device)
        runtime.fit(
            resumed, small, temporary / "resumed", gpu, budget, 0.001, max_updates=2, adapter_lr=0.0001
        )
        runtime.fit(
            resumed, small, temporary / "resumed", gpu, budget, 0.001, max_updates=3, adapter_lr=0.0001
        )
        runtime.seed(42)
        continuous = make_model(native, np.eye(3), 3, "G2").to(gpu.device)
        runtime.fit(
            continuous, small, temporary / "continuous", gpu, budget, 0.001, max_updates=3, adapter_lr=0.0001
        )
        for name, value in runtime.trained_state(continuous).items():
            torch.testing.assert_close(value, runtime.trained_state(resumed)[name], rtol=1e-6, atol=1e-7)
        del resumed, continuous, small
    atomic_json(
        path,
        {
            "passed": True,
            "checks": [
                "real BF16 zero identity",
                "E2 prediction/gradient parity",
                "private FP32 head delta gradients finite",
                "native parameters frozen",
                "fused CUDA graph/head optimizer 2+1 resumed updates equal 3 uninterrupted updates",
            ],
            "parameters": parameter_counts(model),
        },
    )
    model.zero_grad(set_to_none=True)
    del model, data
    torch.cuda.empty_cache()


def evaluate_final(output, selected, config, spec, native, gpu, budget, status):
    for dataset in config["datasets"]:
        data, settings = new_data(config, spec, dataset, gpu.device)
        graph, graph_info = build_graph(data.series, settings)
        variants = [selected] + (["G0", "B1"] if dataset in KEY_DATASETS else [])
        for variant in variants:
            target = output / "results" / dataset / variant / "result.json"
            if target.exists():
                continue
            status("running", phase="final_evaluation", dataset=dataset, variant=variant)
            runtime.seed(42)
            model = make_model(native, graph, data.horizon, variant).to(gpu.device)
            saved = load_best(output / "trials" / dataset / variant / "best.pt", model)
            result = {
                "status": "complete",
                "dataset": dataset,
                "variant": variant,
                "context": data.context,
                "horizon": data.horizon,
                "seed": 42,
                "evaluation_protocol": EVALUATION_PROTOCOL,
                "layout": "late_contiguous" if variant != "B1" else None,
                "parameters": parameter_counts(model),
                "graph": graph_info if variant != "B1" else None,
                "graph_context": data.context_provenance if VARIANTS[variant]["context"] else None,
                "checkpoint_updates": saved["updates"],
                "validation": runtime.evaluate(model, data, "val", gpu, budget),
                "test": runtime.evaluate(model, data, "test", gpu, budget),
            }
            target.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(target, result)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        del data


def write_report(output, selection, state):
    lines = [
        "# stage3-late-context-v1",
        "",
        "State: " + state,
        "",
        "Validation selection: " + str(selection.get("selected")),
        "",
        "Seed 42; raw-target-v2; total time bounded. Historical HSD is a limited adaptation reference.",
        "",
        "|Dataset|Variant|Validation MAE|",
        "|---|---|---:|",
    ]
    for dataset, scores in selection.get("scores", {}).items():
        for variant, value in scores.items():
            lines.append(f"|{dataset}|{variant}|{value if value is not None else 'incomplete'}|")
    lines += ["", "|Dataset|Variant|Test MAE|Test RMSE|Test MAPE %|WAPE %|", "|---|---|---:|---:|---:|---:|"]
    for p in sorted((output / "results").glob("*/*/result.json")):
        d = json.loads(p.read_text())
        r = d["test"]["raw"]
        lines.append(
            f"|{d['dataset']}|{d['variant']}|{r['mae']:.6g}|{r['rmse']:.6g}|{r['mape_pct']}|{r['wape_pct']}|"
        )
    lines += [
        "",
        "Historical reevaluation is stored separately under reevaluation/. Old results were not overwritten.",
        "Known air-quality dataset overlap remains. This is development evidence, not an independent confirmation.",
    ]
    lines += [
        "",
        "|Dataset|Persistence MAE|Native Bolt MAE|Selected GOFA-F MAE|Historical HSD MAE|",
        "|---|---:|---:|---:|---:|",
    ]
    wins_native = wins_hsd = complete = 0
    selected = selection.get("selected")
    for dataset in KEY_DATASETS + EXTENSION_DATASETS:
        references = {}
        for name in ["persistence", "native_bolt", "matched_hsdgnn"]:
            path = output / "reevaluation" / dataset / name / "result.json"
            references[name] = json.loads(path.read_text())["test"]["raw"]["mae"] if path.exists() else None
        path = output / "results" / dataset / str(selected) / "result.json"
        value = json.loads(path.read_text())["test"]["raw"]["mae"] if path.exists() else None
        cells = [references["persistence"], references["native_bolt"], value, references["matched_hsdgnn"]]
        lines.append(
            "|" + dataset + "|" + "|".join(f"{v:.6g}" if v is not None else "incomplete" for v in cells) + "|"
        )
        if value is not None and all(v is not None for v in cells):
            complete += 1
            wins_native += value < references["native_bolt"]
            wins_hsd += value < references["matched_hsdgnn"]
    lines += [
        "",
        f"Complete datasets: {complete}/6; beats native: {wins_native}/6; beats limited HSD: {wins_hsd}/6.",
        "All-six goal achieved in this protocol: " + str(complete == wins_native == wins_hsd == 6),
        "",
        "|Trial|Updates|Epochs|Stop reason|Trainable / total parameters|",
        "|---|---:|---:|---|---:|",
    ]
    for path in sorted((output / "trials").glob("*/*/done.json")):
        d = json.loads(path.read_text())
        p = d["parameters"]
        lines.append(
            f"|{path.parent.parent.name}/{d['variant']}|{d['updates']}|{d['epochs']}|{d['stop_reason']}|{p['trainable']} / {p['total']}|"
        )
    for name in ["active_budget.json", "budget.json"]:
        path = output / name
        if path.exists():
            d = json.loads(path.read_text())
            lines += ["", f"{name}: {d['spent_seconds']:.1f} active seconds."]
    lines += [
        "",
        "GPU-idle waits are excluded. active_budget.json is the supervisor-owned total activity ledger.",
    ]
    temporary = output / "report.md.tmp"
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(output / "report.md")


def stage(config_path, output, resume=False):
    if os.environ.get("GOFA_IDLE_ADMITTED") != "1" or "GOFA_ACTIVE_DEADLINE" not in os.environ:
        raise RuntimeError("GPU work must be admitted by the CPU idle launcher")
    runtime.STOP = False
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, runtime.stop_handler)
    config = runtime.stage_config(config_path)
    validate_config(config)
    spec = yaml.safe_load(Path("configs/benchmark.yaml").read_text())
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = make_manifest(config, spec)
    bind_manifest(output, manifest, resume)
    ledger = output / "budget.json"
    budget = ActiveBudget(ledger, 14400, json.loads(ledger.read_text()) if ledger.exists() else None)
    selection = {}

    def status(state, **fields):
        atomic_json(
            output / "status.json",
            {
                "state": state,
                "pid": os.getpid(),
                "spent_seconds": budget.state["spent_seconds"],
                "git_revision": manifest["git_revision"],
                **fields,
            },
        )

    try:
        budget.activate("preflight", config["budget"]["mechanism_and_reevaluation_seconds"])
        status("running", phase="preflight")
        budget.check()
        gpu = runtime.GPU(config["resources"])
        import chronos
        import transformers
        from chronos.chronos_bolt import ChronosBoltModelForForecasting

        if chronos.__version__ != "2.3.2" or transformers.__version__ != "4.46.3":
            raise RuntimeError("Unexpected native runtime versions")
        weights = Path(config["models"]["native_bolt"]["local_weights"])
        for name, digest in config["models"]["native_bolt"]["file_sha256"].items():
            if file_sha256(weights / name) != digest:
                raise ValueError("Pinned native weights mismatch")
        native = (
            ChronosBoltModelForForecasting.from_pretrained(
                weights, local_files_only=True, torch_dtype=torch.bfloat16
            )
            .to(gpu.device)
            .eval()
            .requires_grad_(False)
        )
        if not (output / "preflight_done.json").exists():
            mechanism_check(output, native, config, spec, gpu, budget)
            reevaluate_historical(output, config, spec, native, gpu, budget, status)
            atomic_json(output / "preflight_done.json", {"complete": True})
        for dataset in KEY_DATASETS:
            train_dataset(
                output,
                dataset,
                list(VARIANTS),
                config["budget"]["key_trial_seconds"],
                native,
                gpu,
                budget,
                config,
                spec,
                status,
            )
        selection_path = output / "selection.json"
        if selection_path.exists():
            selection = json.loads(selection_path.read_text())
        else:
            scores = {
                ds: {
                    v: json.loads((output / "trials" / ds / v / "done.json").read_text())[
                        "validation_raw_mae"
                    ]
                    for v in VARIANTS
                }
                for ds in KEY_DATASETS
            }
            selection = select_variant(scores, config["adaptation"]["selection_relative_tolerance"])
            selection.update(
                selection_split="validation",
                metric="raw_mae",
                tolerance=config["adaptation"]["selection_relative_tolerance"],
            )
            atomic_json(selection_path, selection)
        if selection["selected"] is None:
            status("no_eligible_variant")
            write_report(output, selection, "no_eligible_variant")
            return
        for dataset in EXTENSION_DATASETS:
            train_dataset(
                output,
                dataset,
                [selection["selected"]],
                config["budget"]["extension_trial_seconds"],
                native,
                gpu,
                budget,
                config,
                spec,
                status,
            )
        budget.activate("final_evaluation", config["budget"]["final_evaluation_seconds"])
        evaluate_final(output, selection["selected"], config, spec, native, gpu, budget, status)
        write_report(output, selection, "complete")
        status("complete")
    except runtime.StopRequested:
        status("interrupted")
        raise
    except IncompleteTrial as exc:
        status("incomplete", reason=str(exc))
        write_report(output, selection, "incomplete")
    except BudgetExhausted as exc:
        scope = str(exc)
        state = (
            "budget_exhausted"
            if budget.state["spent_seconds"] >= 14400
            or float(os.environ.get("GOFA_ACTIVE_DEADLINE", "inf")) - time.monotonic() <= 60
            else "incomplete"
        )
        status(state, scope=scope)
        write_report(output, selection, state)
    except Exception as exc:
        status("failed", error=str(exc))
        raise
    finally:
        budget.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage3_late_context.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        stage(args.config, args.output, args.resume)
    except runtime.StopRequested:
        raise SystemExit(75)


if __name__ == "__main__":
    main()
