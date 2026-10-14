"""One admitted GPU task, using the model family's isolated Python environment."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import signal
import tempfile
import time

import numpy as np
import torch
import yaml

from gofa_f import native_stage as runtime
from gofa_f.adaptation_stage import ActiveBudget, prefix_key
from gofa_f.graphs import build_graph
from gofa_f.models import parameter_counts
from gofa_f.native import NativeBolt
from gofa_f.prefix_cache import PrefixCache, atomic_json
from gofa_f.pure_graph_stage import config_file, task_directory
from gofa_f.stage_budget import BudgetExhausted
from gofa_f.data import file_sha256


class TaskBudget(ActiveBudget):
    def remaining(self):
        remaining = super().remaining()
        deadline = os.environ.get("GOFA_TASK_DEADLINE")
        return min(remaining, float(deadline) - time.monotonic() - 60) if deadline else remaining


def data_for(config, spec, dataset, device, variant, evaluation=False):
    settings = runtime.dataset_settings(spec["datasets"][dataset], config, dataset)
    settings["validation_only"] = not evaluation
    if variant.startswith("ufg"):
        settings.pop("graph_context", None)
    else:
        settings["graph_context"]["feature_mode"] = "values_and_clock" if variant == "g2_clock" else "values_only"
    return runtime.Data(settings, device), settings


def load_native(config, device):
    from chronos.chronos_bolt import ChronosBoltModelForForecasting
    return ChronosBoltModelForForecasting.from_pretrained(
        config["models"]["native_bolt"]["local_weights"], local_files_only=True,
        torch_dtype=torch.bfloat16).to(device).requires_grad_(False).eval()


def make_model(config, data, settings, variant, device, native=None):
    runtime.seed(42)
    if variant.startswith("ufg"):
        from gofa_f.ufg import UFGTime
        return UFGTime(config["stage4"]["upstream_root"], data.context, data.horizon,
                       data.values.shape[1], scale=variant[-1].upper(), device=device).to(device)
    graph, _ = build_graph(data.series, settings)
    if variant == "g2_self":
        graph = np.eye(len(graph), dtype=np.float32)
    return NativeBolt(native, graph, data.horizon, (1, 2), rank=64,
                      graph_feature_dim=11 if variant == "g2_clock" else 5,
                      output_adapter_rank=4).to(device)


def model_counts(model):
    if isinstance(model, NativeBolt):
        counts = parameter_counts(model)
        point_dof = counts["trainable"]
        if model.output_adapter is not None and not model.native.output_patch_embedding.use_layer_norm:
            point_dof -= model.output_adapter.b.numel() - model.horizon * model.output_adapter.b.shape[1]
        return {**counts, "registered_real_dof": counts["total"], "active_real_dof": None,
                "trainable_real_dof": counts["trainable"], "complex_counting": "all parameters real",
                "point_loss_active_trainable_real_dof": point_dof,
                "note": "Trainable tensor count; median/horizon-only loss may not update all head rows. "
                        "Frozen backbone scalar usage not separately counted"}
    from gofa_f.ufg import audit_parameters
    return audit_parameters(model)


def checkpoint_record(directory, model, state, data):
    completed_epochs = state["epoch"] + int(state["offset"] >= len(data.windows["train"]))
    complete = completed_epochs >= 100 or state["stale"] >= 15
    return {"valid": state["best_mae"] is not None, "validation_raw_mae": state["best_mae"],
            "updates": state["updates"], "epochs": completed_epochs, "stale": state["stale"],
            "complete": complete, "stop_reason": "early_stopping" if state["stale"] >= 15 else
            "max_epochs" if completed_epochs >= 100 else "milestone", "parameters": model_counts(model),
            "directory": str(directory), "training_windows": len(data.windows["train"])}


def diagnostics(model, data, gpu, budget):
    """Training/validation only. Groups and high-scale set are fitted on training statistics."""
    model.eval()
    nodes, horizon = data.values.shape[1], data.horizon
    order = np.argsort(data.std.cpu().numpy(), kind="stable")
    groups = np.array_split(order, 5)
    high = order[-max(1, math.ceil(nodes * .05)):]
    output = {"group_basis": "training standard deviation", "training_std": data.std.cpu().tolist(),
              "high_scale_nodes": high.tolist(), "splits": {}}
    for split in ("train", "val"):
        absolute = np.zeros((horizon, nodes), np.float64)
        square = np.zeros_like(absolute)
        count = np.zeros_like(absolute)
        missing = np.zeros_like(absolute)
        categories = {key: [0., 0] for key in ("zero", "nonzero", "history_missing", "history_complete")}
        offset = 0
        while offset < len(data.windows[split]):
            gpu.check(budget)
            end = min(offset + gpu.micro, len(data.windows[split]))
            batch = data.batch(split, np.arange(offset, end))
            try:
                with torch.no_grad():
                    pred = runtime.prediction(model, batch) * data.std + data.mean
                    error = (pred - batch["y_raw"]).abs().double()
                observed = batch["y_mask"]
                masked = error.masked_fill(~observed, 0)
                absolute += masked.sum(0).cpu().numpy()
                square += masked.square().sum(0).cpu().numpy()
                count += observed.sum(0).cpu().numpy()
                missing += (~observed).sum(0).cpu().numpy()
                zero = batch["y_raw"].abs() <= 1e-5
                history_missing = (~batch["x_mask"]).any(1)[:, None].expand_as(observed)
                for name, mask in (("zero", zero), ("nonzero", ~zero),
                                   ("history_missing", history_missing), ("history_complete", ~history_missing)):
                    mask = mask & observed
                    categories[name][0] += error.masked_select(mask).sum().item()
                    categories[name][1] += mask.sum().item()
            except torch.OutOfMemoryError:
                del batch
                if "pred" in locals():
                    del pred, error
                gpu.shrink()
                continue
            offset = end
        total = absolute.sum()
        def aggregate(indices):
            denominator = count[:, indices].sum()
            return {"nodes": np.asarray(indices).tolist(), "observations": int(denominator),
                    "mae": float(absolute[:, indices].sum() / denominator) if denominator else None,
                    "absolute_error_share": float(absolute[:, indices].sum() / total) if total else None}
        output["splits"][split] = {
            "mae": float(total / count.sum()), "rmse": float(np.sqrt(square.sum() / count.sum())),
            "by_node": [{"node": i, "mae": float(absolute[:, i].sum()/count[:, i].sum())
                         if count[:, i].sum() else None, "observations": int(count[:, i].sum()),
                         "missing_labels": int(missing[:, i].sum()),
                         "absolute_error_share": float(absolute[:, i].sum()/total) if total else None}
                        for i in range(nodes)],
            "by_horizon": [{"step": i+1, "mae": float(absolute[i].sum()/count[i].sum())
                            if count[i].sum() else None}
                           for i in range(horizon)],
            "scale_quintiles": [aggregate(group) for group in groups], "top_five_percent_scale": aggregate(high),
            "categories": {name: {"mae": value[0]/value[1] if value[1] else None,
                                  "observations": value[1]} for name, value in categories.items()},
        }
    return output


def assert_checkpoint_equal(first, second):
    """Check serialized state exactly, independently of CUDA reduction roundoff."""
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, np.ndarray):
        np.testing.assert_array_equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_checkpoint_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert type(first) is type(second) and len(first) == len(second)
        for a, b in zip(first, second):
            assert_checkpoint_equal(a, b)
    else:
        assert first == second


def mechanism(config, spec, output, family, gpu, budget):
    directory = output / "mechanisms" / family
    directory.mkdir(parents=True, exist_ok=True)
    if family == "native":
        from gofa_f.adaptation_stage import mechanism_check
        native = load_native(config, gpu.device)
        mechanism_check(directory, native, config, spec, gpu, budget)
        data, settings = data_for(config, spec, "metr_la", gpu.device, "g2_values")
        model = make_model(config, data, settings, "g2_values", gpu.device, native)
        if model_counts(model)["trainable_real_dof"] != 108930:
            raise AssertionError("G2-V parameter audit mismatch")
        batch = data.batch("train", [0, 1])
        with torch.no_grad():
            pred = runtime.prediction(model, batch)
        if not torch.isfinite(pred).all() or batch["graph_features"].shape[-1] != 5:
            raise AssertionError("Value-only BF16 path failed")
        atomic_json(directory / "values_only.json", {"passed": True, "parameters": model_counts(model)})
    else:
        from gofa_f.ufg import UFGTime, audit_parameters, runtime_versions, portable_knn_graph
        points = torch.randn(2, 7, 2, generator=torch.Generator().manual_seed(42))
        cpu_graph = portable_knn_graph(points, 2)
        gpu.check(budget)
        cuda_graph = portable_knn_graph(points.to(gpu.device), 2)
        def edges(graph):
            return sorted(zip(*(edge.cpu().tolist() for edge in graph.edges())))
        if edges(cpu_graph) != edges(cuda_graph):
            raise AssertionError("CPU reference and official CUDA kNN differ on untied distances")
        for length in (8, 12, 24):
            runtime.seed(42)
            model = UFGTime(config["stage4"]["upstream_root"], length, 3, 3,
                            device=gpu.device, width=8).to(gpu.device)
            x = torch.randn(2, length, 3, device=gpu.device)
            observed = torch.ones_like(x, dtype=torch.bool)
            gpu.check(budget)
            result = model(x, observed)
            together = result.detach()
            with torch.no_grad():
                alone = torch.cat([model(x[i:i+1], observed[i:i+1]) for i in range(2)])
            torch.testing.assert_close(together, alone, rtol=2e-4, atol=2e-4)
            result.square().mean().backward()
            for name, parameter in model.named_parameters():
                if parameter.requires_grad and (parameter.grad is None or not torch.isfinite(parameter.grad).all()):
                    raise AssertionError("Invalid active UFG gradient: " + name)
            del model, x, observed, result
        # Real CUDA RMSprop/complex checkpoint restoration, not a timing benchmark.
        with tempfile.TemporaryDirectory(prefix="ufg-resume-", dir=directory) as temporary:
            temporary = Path(temporary)
            values = np.random.default_rng(42).normal(size=(128, 3)).astype(np.float32)
            np.savez(temporary / "toy.npz", values=values, mask=np.ones_like(values, bool))
            data = runtime.Data({"data": str(temporary / "toy.npz"), "context": 8, "horizons": [3],
                                 "validation_only": True}, gpu.device)
            def new():
                runtime.seed(42)
                return UFGTime(config["stage4"]["upstream_root"], 8, 3, 3,
                               device=gpu.device, width=8).to(gpu.device)
            resumed = new()
            kwargs = {"optimizer_kind": "rmsprop", "loss_kind": "standardized_mse", "max_epochs": 100,
                      "patience": 15}
            runtime.fit(resumed, data, temporary / "resume", gpu, budget, .001, max_updates=2, **kwargs)
            checkpoint = torch.load(temporary / "resume/last.pt", map_location="cpu", weights_only=False)
            resumed = new()
            runtime.fit(resumed, data, temporary / "resume", gpu, budget, .001, max_updates=2, **kwargs)
            restored = torch.load(temporary / "resume/last.pt", map_location="cpu", weights_only=False)
            assert_checkpoint_equal(checkpoint, restored)
            runtime.fit(resumed, data, temporary / "resume", gpu, budget, .001, max_updates=3, **kwargs)
            continuous = new()
            runtime.fit(continuous, data, temporary / "continuous", gpu, budget, .001, max_updates=3, **kwargs)
            maximum_difference = 0.
            for name, value in runtime.trained_state(continuous).items():
                other = runtime.trained_state(resumed)[name]
                maximum_difference = max(maximum_difference, float((value-other).abs().max()))
                torch.testing.assert_close(value, other, rtol=2e-4, atol=2e-5)
            counts = audit_parameters(continuous)
        atomic_json(directory / "mechanisms.json", {"passed": True, "versions": runtime_versions(),
                    "restore_exact": True, "trajectory_rtol": 2e-4, "trajectory_atol": 2e-5,
                    "trajectory_max_absolute_difference": maximum_difference,
                    "parameters": counts, "checks": ["complex CUDA finite gradients", "batch graph isolation",
                    "short histories 8/12/24", "CPU brute-force reference equals official CUDA kNN",
                    "exact model/optimizer/scheduler/RNG/sampling checkpoint restore",
                    "2+1 RMSprop resume agrees with 3 uninterrupted updates within CUDA numeric tolerance"]})


def run(args):
    if os.environ.get("GOFA_IDLE_ADMITTED") != "1":
        raise RuntimeError("GPU task must be admitted by the CPU idle supervisor")
    config = config_file(args.config)
    spec = yaml.safe_load(Path("configs/benchmark.yaml").read_text())
    output = Path(args.output).resolve()
    binding = json.loads((output / "manifest.json").read_text())
    for name, checksum in binding["inputs"][args.dataset].items():
        if file_sha256(name) != checksum:
            raise ValueError("Stage input changed after manifest binding: " + name)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, runtime.stop_handler)
    budget_path = output / "budget.json"
    budget = TaskBudget(budget_path, float("inf"), json.loads(budget_path.read_text())
                        if budget_path.exists() else None)
    budget.activate(f"{args.phase}/{args.dataset}/{args.variant}", float("inf"))
    cache = None
    try:
        resources = dict(config["resources"])
        # Leave headroom inside the 40 GiB process cap for driver and DGL allocations outside torch's pool.
        resources["memory_cap_gib"] = 36 if args.variant.startswith("ufg") else 38
        gpu = runtime.GPU(resources)
        if args.phase == "preflight":
            mechanism(config, spec, output, args.variant, gpu, budget)
            return
        directory = task_directory(output, args.dataset, args.variant, args.lr)
        directory.mkdir(parents=True, exist_ok=True)
        data, settings = data_for(config, spec, args.dataset, gpu.device, args.variant, args.phase == "evaluate")
        native = None if args.variant.startswith("ufg") else load_native(config, gpu.device)
        model = make_model(config, data, settings, args.variant, gpu.device, native)
        metadata = {"dataset": args.dataset, "variant": args.variant, "lr": args.lr,
                    "data_sha256": data.series.fingerprint, "training_windows": len(data.windows["train"]),
                    "parameters": model_counts(model), "context_provenance": data.context_provenance,
                    "provenance": getattr(model, "provenance", None)}
        metadata = json.loads(json.dumps(metadata, allow_nan=False))
        metadata_path = directory / "metadata.json"
        if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
            raise ValueError("Per-job input/parameter manifest mismatch")
        atomic_json(metadata_path, metadata)
        if args.phase in ("pilot", "train"):
            if isinstance(model, NativeBolt):
                key = prefix_key(data, config["models"]["native_bolt"]["revision"])
                cache = PrefixCache.build(output / "cache" / args.dataset, key, model,
                    data.batches("train", gpu.micro), len(data.windows["train"]), lambda: gpu.check(budget))
            kwargs = {"max_epochs": 100, "patience": 15}
            if args.variant.startswith("ufg"):
                kwargs.update(optimizer_kind="rmsprop", loss_kind="standardized_mse")
            else:
                kwargs.update(adapter_lr=.0001)
            limit = 200 if args.phase == "pilot" else args.milestone * math.ceil(len(data.windows["train"])/64)
            state = runtime.fit(model, data, directory, gpu, budget, args.lr, cache,
                                max_updates=limit, **kwargs)
            if budget.remaining() <= 0:
                raise runtime.StopRequested("Training/evaluation wall boundary reached")
            record = checkpoint_record(directory, model, state, data)
            record.update(dataset=args.dataset, variant=args.variant, lr=args.lr,
                          training_loss="standardized_mse" if args.variant.startswith("ufg") else "raw_mae")
            atomic_json(directory / ("pilot_done.json" if args.phase == "pilot" else "done.json"), record)
            if args.phase == "train":
                atomic_json(directory / f"milestone_{args.milestone}.json", record)
        else:
            selected = json.loads((output / "final_selection.json").read_text())
            key = args.dataset + "/" + args.variant
            bound = selected["jobs"][key]
            if file_sha256(directory / "best.pt") != bound["checkpoint_sha256"]:
                raise ValueError("Final checkpoint changed after the evaluation lock")
            saved = runtime.best(directory, model)
            if args.phase == "diagnose":
                result = diagnostics(model, data, gpu, budget)
                result["checkpoint_sha256"] = bound["checkpoint_sha256"]
                atomic_json(directory / "diagnostics.json", result)
            else:
                result = {"status": "complete", "dataset": args.dataset, "variant": args.variant,
                          "seed": 42, "context": data.context, "horizon": data.horizon,
                          "checkpoint_sha256": bound["checkpoint_sha256"], "checkpoint_updates": saved["updates"],
                          "parameters": model_counts(model), "test": runtime.evaluate(model, data, "test", gpu, budget),
                          "validation": saved["validation"], "training_complete": bound["training_complete"],
                          "evaluation_protocol": "raw-target-v2", "layout": "late_contiguous" if native else None,
                          "provenance": getattr(model, "provenance", data.context_provenance)}
                atomic_json(directory / "result.json", result)
        atomic_json(directory / "cuda_memory.json", {"peak_allocated": torch.cuda.max_memory_allocated(),
                                                     "peak_reserved": torch.cuda.max_memory_reserved()})
    finally:
        if cache is not None:
            cache.close(delete=False)
        budget.close()
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", choices=["preflight", "pilot", "train", "diagnose", "evaluate"], required=True)
    parser.add_argument("--dataset", default="metr_la")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--milestone", type=int, default=100)
    args = parser.parse_args()
    try:
        run(args)
    except (runtime.StopRequested, BudgetExhausted):
        raise SystemExit(75)


if __name__ == "__main__":
    main()
