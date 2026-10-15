"""One admitted METR-LA long-horizon mechanism, training or final evaluation task."""
import argparse
import gc
import math
import os
from pathlib import Path
import signal

import numpy as np
import torch
import yaml

from gofa_f import native_stage as runtime
from gofa_f.adaptation_stage import prefix_key
from gofa_f.data import file_sha256
from gofa_f.graphs import build_graph
from gofa_f.head_capacity import SketchData, HeadCapacityBolt
from gofa_f.prefix_cache import PrefixCache, atomic_json
from gofa_f.pure_graph_task import TaskBudget, model_counts
from gofa_f.routed_stage import record_memory
from gofa_f.routed_task import compatible_cache
from gofa_f.stage_budget import BudgetExhausted

VARIANTS = ("h1_rank6",)


def data_for(config, device, evaluation=False):
    spec = yaml.safe_load(Path("configs/benchmark.yaml").read_text())
    settings = dict(spec["datasets"]["metr_la"], validation_only=not evaluation,
                    graph_context={"enabled": True, "feature_mode": "values_and_clock",
                                   "time_path": config["stage13"]["sidecar"]})
    return SketchData(settings, device), settings


def new_model(config, native, data, settings, variant, output):
    if variant not in VARIANTS:
        raise ValueError("Unregistered candidate")
    runtime.seed(42)
    graph, _ = build_graph(data.series, settings)
    with np.load(Path(output)/"correlation_graph.npz", allow_pickle=False) as source:
        correlation = source["adjacency"]
        if int(source["train_end"]) != data.series.train_end or \
                str(source["prepared_sha256"]) != data.series.fingerprint:
            raise ValueError("Correlation graph does not match training-prefix data")
    model = HeadCapacityBolt(native, graph, correlation, data.horizon, variant,
                             **config["architecture"]).to(data.device)
    if model_counts(model)["trainable_real_dof"] > config["stage13"]["parameter_reference_dof"]*config["stage13"]["maximum_parameter_ratio"]:
        raise ValueError("Private head exceeds the registered trainable-parameter budget")
    return model


def task_directory(output, variant):
    return Path(output) / "trials" / variant


def training_cache(config, output, model, data, gpu, budget):
    key = prefix_key(data, config["native_revision"])
    old = config["stage13"]["stage5_cache"]
    cache = compatible_cache(old, key, len(data.windows["train"]),
                             data.values.shape[1], model.native.config.d_model)
    reused = cache is not None
    if cache is not None:
        indices = np.unique([0, len(data.windows["train"])//2, len(data.windows["train"])-1])
        batch = data.batch("train", indices)
        gpu.check(budget)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            expected = model.prefix(batch["x"], batch["x_mask"], 2)
        try:
            for actual, wanted in zip(cache.take(indices, gpu.device), expected):
                torch.testing.assert_close(actual, wanted, rtol=2e-3, atol=2e-3)
        except AssertionError:
            cache.close()
            cache, reused = None, False
    if cache is None:
        cache = PrefixCache.build(Path(output)/"cache"/"metr_la", key, model,
            data.batches("train", gpu.micro), len(data.windows["train"]), lambda: gpu.check(budget))
    atomic_json(Path(output)/"cache_source.json", {"reused_read_only": reused,
        "directory": str(cache.directory), "key": key})
    return cache


def mechanisms(config, output, native, gpu, budget):
    data, settings = data_for(config, gpu.device)
    batch = data.batch("train", [0, len(data.windows["train"])-1])
    if not torch.equal(batch["graph_features"][:, 0, 5:11], batch["graph_features"][:, -1, 5:11]):
        raise AssertionError("Forecast clock must be shared across nodes")
    records = {}
    for variant in VARIANTS:
        gpu.check(budget)
        model = new_model(config, native, data, settings, variant, output)
        context = batch["x"].transpose(1, 2).reshape(-1, 12).masked_fill(
            ~batch["x_mask"].transpose(1, 2).reshape(-1, 12), torch.nan)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            reference = native(context=context).quantile_preds[:, model.median_index, :12]
            reference = reference.reshape(2, -1, 12).transpose(1, 2)
            result = model(batch["x"], batch["x_mask"], batch["graph_features"])
        torch.testing.assert_close(result, reference, rtol=2e-3, atol=2e-3)
        # Causal clock: changing targets cannot change any model input or prediction.
        perturbed = dict(batch, y=batch["y"] + 1000, y_raw=batch["y_raw"] + 1000)
        assert torch.equal(batch["graph_features"], perturbed["graph_features"])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            prefix = model.prefix(batch["x"], batch["x_mask"], 2)
        for module in model.graph.values():
            module.gate.data.fill_(.1)
        values, gradients = [], []
        for cached in (False, True):
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model.from_prefix(prefix, 2, graph_features=batch["graph_features"]) if cached else model(
                    batch["x"], batch["x_mask"], batch["graph_features"])
            prediction.float().square().mean().backward()
            values.append(prediction.detach())
            gradients.append({n: p.grad.detach().clone() for n, p in model.named_parameters()
                              if p.requires_grad and p.grad is not None})
        torch.testing.assert_close(values[0], values[1], rtol=2e-3, atol=2e-3)
        if gradients[0].keys() != gradients[1].keys():
            raise AssertionError("Cache changed active gradient set")
        for name in gradients[0]:
            assert torch.isfinite(gradients[0][name]).all()
            torch.testing.assert_close(gradients[0][name], gradients[1][name], rtol=2e-3, atol=2e-3)
        assert all(p.grad is None for p in native.parameters())
        missing = batch["x_mask"].clone()
        missing[:, :, 0] = False
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            assert torch.isfinite(model(batch["x"], missing, batch["graph_features"])).all()
        records[variant] = {"initial_max_difference": float((result-reference).abs().max()),
                            "cache_prediction_and_gradient": True, "frozen_native": True,
                            "missing_node_finite": True, "parameters": model_counts(model)}
        del model
        gc.collect()
    atomic_json(Path(output)/"mechanisms.json", {"passed": True, "variants": records})


def run(args):
    if os.environ.get("GOFA_IDLE_ADMITTED") != "1":
        raise RuntimeError("Only the strict idle launcher may admit GPU tasks")
    from gofa_f.head_stage import config_file, read
    config, output = config_file(args.config), Path(args.output).resolve()
    binding = read(output/"manifest.json")
    if binding["inputs"] != {str(p): file_sha256(p) for p in map(Path, binding["inputs"])}:
        raise ValueError("Pinned input changed")
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, runtime.stop_handler)
    budget_path = output/"budget.json"
    budget = TaskBudget(budget_path, float("inf"), read(budget_path))
    budget.activate(args.phase+"/"+args.variant, float("inf"))
    gpu = runtime.GPU(dict(config["resources"], memory_cap_gib=38))
    torch.cuda.reset_peak_memory_stats()
    directory = output/"mechanisms" if args.phase == "preflight" else task_directory(output, args.variant)
    directory.mkdir(parents=True, exist_ok=True)
    cache = None
    try:
        from chronos.chronos_bolt import ChronosBoltModelForForecasting
        native = ChronosBoltModelForForecasting.from_pretrained(config["native_weights"],
            local_files_only=True, torch_dtype=torch.bfloat16).to(gpu.device).requires_grad_(False).eval()
        if args.phase == "preflight":
            mechanisms(config, output, native, gpu, budget)
            return
        data, settings = data_for(config, gpu.device, evaluation=args.phase == "evaluate")
        model = new_model(config, native, data, settings, args.variant, output)
        metadata = {"variant": args.variant, "seed": 42, "data_sha256": data.series.fingerprint,
            "train_windows": len(data.windows["train"]), "context": 12, "horizon": 12,
            "architecture": model.graph_architecture, "parameters": model_counts(model),
            "optimizer": "adamw", "loss": "raw_mae", "graph_lr": .001, "adapter_lr": .0001}
        if (directory/"metadata.json").exists() and read(directory/"metadata.json") != metadata:
            raise ValueError("Task metadata changed")
        atomic_json(directory/"metadata.json", metadata)
        if args.phase == "train":
            cache = training_cache(config, output, model, data, gpu, budget)
            state = runtime.fit(model, data, directory, gpu, budget, .001, cache,
                max_updates=args.milestone*math.ceil(len(data.windows["train"])/64),
                max_epochs=100, patience=15, adapter_lr=.0001, optimizer_kind="adamw", loss_kind="raw_mae")
            if (directory/"best.pt").exists():
                best = torch.load(directory/"best.pt", map_location="cpu", weights_only=True)
                atomic_json(directory/"best_validation.json", {
                    "mae": best["validation"]["raw"]["mae"],
                    "step12_mae": best["validation"]["raw"]["by_horizon"][11]["mae"],
                    "updates": best["updates"], "checkpoint_sha256": file_sha256(directory/"best.pt")})
            epochs = state["epoch"] + int(state["offset"] >= len(data.windows["train"]))
            record = {"complete": epochs >= 100 or state["stale"] >= 15,
                "epochs": epochs, "updates": state["updates"], "validation_raw_mae": state["best_mae"]}
            atomic_json(directory/"done.json", record)
            atomic_json(directory/f"milestone_{args.milestone}.json", record)
        else:
            selection = read(output/"final_selection.json")
            if selection["selected"] != args.variant:
                raise ValueError("Only the validation-qualified locked candidate may be tested")
            locked = selection["jobs"][args.variant]
            if file_sha256(directory/"best.pt") != locked["checkpoint_sha256"]:
                raise ValueError("Best checkpoint changed after seal")
            best = runtime.best(directory, model)
            result = {"status": "complete", "variant": args.variant, "seed": 42,
                "parameters": metadata["parameters"], "checkpoint_sha256": locked["checkpoint_sha256"],
                "checkpoint_updates": best["updates"], "validation": best["validation"],
                "test": runtime.evaluate(model, data, "test", gpu, budget),
                "evaluation_protocol": "raw-target-v2"}
            atomic_json(directory/"result.json", result)
    finally:
        record_memory(directory/"cuda_memory.json", args.phase, torch.cuda.max_memory_allocated(),
                      torch.cuda.max_memory_reserved())
        if cache is not None:
            cache.close()
        budget.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", choices=("preflight", "train", "evaluate"), required=True)
    parser.add_argument("--variant", choices=VARIANTS, default="h1_rank6")
    parser.add_argument("--milestone", type=int, default=100)
    args = parser.parse_args()
    try:
        run(args)
    except (runtime.StopRequested, BudgetExhausted):
        raise SystemExit(75)


if __name__ == "__main__":
    main()
