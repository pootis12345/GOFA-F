"""One admitted BF16 routed-GOFA task; no GNN training or hyperparameter search."""
import argparse
import gc
import math
import os
from pathlib import Path
import signal
import tempfile

import numpy as np
import torch
import yaml

from gofa_f import native_stage as runtime
from gofa_f.adaptation_stage import prefix_key
from gofa_f.data import file_sha256
from gofa_f.graphs import build_graph
from gofa_f.native import NativeBolt
from gofa_f.prefix_cache import PrefixCache, atomic_json
from gofa_f.pure_graph_task import TaskBudget, assert_checkpoint_equal, diagnostics, model_counts
from gofa_f.routed_stage import DATASETS, completed, config_file, job_directory, read, record_memory
from gofa_f.stage_budget import BudgetExhausted


def data_for(spec, dataset, device, evaluation=False):
    settings = dict(spec["datasets"][dataset], validation_only=not evaluation,
                    graph_context={"enabled": True, "feature_mode": "values_only"})
    return runtime.Data(settings, device), settings


def new_model(config, native, data, settings):
    runtime.seed(42)
    graph, _ = build_graph(data.series, settings)
    return NativeBolt(native, graph, data.horizon, **config["architecture"]).to(data.device)


def compatible_cache(directory, key, count, nodes, width):
    """Open old data read-only only when its complete key and array format agree."""
    metadata = read(Path(directory)/"metadata.json", {})
    if not metadata.get("complete") or metadata.get("key") != key or metadata.get("prefix_layers") != 2:
        return None
    cache = None
    try:
        cache = PrefixCache(directory, read_only=True)
        patches = math.ceil(key["context"]/16)+1
        if cache.hidden.shape != (count, nodes, patches, width) or cache.hidden.dtype != np.uint16 or \
                cache.mask.shape != (count, nodes, patches) or cache.mask.dtype != bool or \
                cache.loc.shape != (count, nodes, 1) or cache.scale.shape != (count, nodes, 1) or \
                cache.loc.dtype != np.float32 or cache.scale.dtype != np.float32:
            cache.close()
            return None
        return cache
    except (OSError, ValueError, KeyError):
        if cache is not None:
            cache.close()
        return None


def training_cache(config, output, dataset, model, data, gpu, budget):
    key = prefix_key(data, config["native_revision"])
    old = Path(config["stage5"]["reference_output"])/"cache"/dataset
    cache = compatible_cache(old, key, len(data.windows["train"]), data.values.shape[1], native_width(model))
    reused = cache is not None
    if cache is not None:
        # Validate actual activations as well as metadata; never repair the old files.
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
        cache = PrefixCache.build(Path(output)/"cache"/dataset, key, model,
            data.batches("train", gpu.micro), len(data.windows["train"]), lambda: gpu.check(budget))
    atomic_json(job_directory(output, dataset)/"cache.json", {"reused_read_only": reused,
        "directory": str(cache.directory), "key": key, "format_verified": True})
    return cache


def native_width(model):
    return model.native.config.d_model


def preflight(config, spec, output, native, gpu, budget):
    folder = Path(output)/"mechanisms"
    folder.mkdir(parents=True, exist_ok=True)
    frozen = {n: p.detach().cpu().clone() for n, p in native.named_parameters()}
    records = []
    for dataset in DATASETS:
        gpu.check(budget)
        data, settings = data_for(spec, dataset, gpu.device)
        model = new_model(config, native, data, settings)
        assert model_counts(model)["trainable_real_dof"] == 113286
        batch = data.batch("train", [0])
        context = batch["x"].transpose(1, 2).reshape(-1, data.context).masked_fill(
            ~batch["x_mask"].transpose(1, 2).reshape(-1, data.context), torch.nan)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            expected = native(context=context).quantile_preds[:, model.median_index, :data.horizon]
            expected = expected.reshape(1, -1, data.horizon).transpose(1, 2)
            actual = model(batch["x"], batch["x_mask"], batch["graph_features"])
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
        initial_difference = float((actual-expected).abs().max())
        for module in model.graph.values():
            module.gate.data.fill_(.1)
            module.key.weight.data.normal_(0, .01)
            module.router.weight.data.normal_(0, .01)
        results, gradients = [], []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            prefix = model.prefix(batch["x"], batch["x_mask"], 2)
        for cached in (False, True):
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = model.from_prefix(prefix, 2, graph_features=batch["graph_features"]) if cached else model(
                    batch["x"], batch["x_mask"], batch["graph_features"])
            assert torch.isfinite(result).all()
            result.float().square().mean().backward()
            results.append(result.detach())
            gradients.append({n: p.grad.detach().clone() for n, p in model.named_parameters() if p.requires_grad})
        torch.testing.assert_close(results[0], results[1], rtol=2e-3, atol=2e-3)
        for name in gradients[0]:
            assert torch.isfinite(gradients[0][name]).all()
            torch.testing.assert_close(gradients[0][name], gradients[1][name], rtol=2e-3, atol=2e-3)
        assert all(p.grad is None and torch.equal(p.detach().cpu(), frozen[n]) for n, p in native.named_parameters())
        missing = batch["x_mask"].clone()
        missing[:, :, 0] = False
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            assert torch.isfinite(model(batch["x"], missing, batch["graph_features"])).all()
        records.append({"dataset": dataset, "initial_max_difference": initial_difference,
                        "bf16_cached_predictions_and_gradients": True, "missing_nodes_finite": True})
        del data, model, batch, prefix, results, gradients, result, actual, expected, context
        gc.collect()
    with tempfile.TemporaryDirectory(prefix="resume-", dir=folder) as scratch:
        scratch = Path(scratch)
        values = np.random.default_rng(42).normal(size=(160, 3)).astype(np.float32)
        np.savez(scratch/"toy.npz", values=values, mask=np.ones_like(values, bool))
        settings = {"data": str(scratch/"toy.npz"), "context": 12, "horizons": [3], "validation_only": True,
                    "graph_context": {"enabled": True, "feature_mode": "values_only"}}
        data = runtime.Data(settings, gpu.device)
        kwargs = {"max_epochs": 100, "patience": 15, "adapter_lr": .0001, "optimizer_kind": "adamw",
                  "loss_kind": "raw_mae"}
        def fresh():
            runtime.seed(42)
            return NativeBolt(native, np.ones((3, 3)), 3, **config["architecture"]).to(gpu.device)
        model = fresh()
        runtime.fit(model, data, scratch/"resume", gpu, budget, .001, max_updates=2, **kwargs)
        saved = torch.load(scratch/"resume/last.pt", map_location="cpu", weights_only=False)
        model = fresh()
        runtime.fit(model, data, scratch/"resume", gpu, budget, .001, max_updates=2, **kwargs)
        restored = torch.load(scratch/"resume/last.pt", map_location="cpu", weights_only=False)
        assert_checkpoint_equal(saved, restored)
        runtime.fit(model, data, scratch/"resume", gpu, budget, .001, max_updates=3, **kwargs)
        continuous = fresh()
        runtime.fit(continuous, data, scratch/"continuous", gpu, budget, .001, max_updates=3, **kwargs)
        difference = 0.
        for name, value in runtime.trained_state(continuous).items():
            other = runtime.trained_state(model)[name]
            difference = max(difference, float((value-other).abs().max()))
            torch.testing.assert_close(value, other, rtol=2e-4, atol=2e-5)
    atomic_json(folder/"mechanisms.json", {"passed": True, "records": records, "restore_exact": True,
        "cache_rtol": 2e-3, "cache_atol": 2e-3, "trajectory_rtol": 2e-4, "trajectory_atol": 2e-5,
        "trajectory_max_absolute_difference": difference})


def routing_diagnostics(model, data, gpu, budget, initial):
    """Validation histories only; collect summaries, never persist dense attention matrices."""
    summaries, originals = {}, {}
    for name, module in model.graph.items():
        totals = summaries[name] = {"nodes": 0, "dynamic_fraction_sum": 0., "gate_sum": 0.,
            "gate_square_sum": 0., "saturated_gate_count": 0, "attention_l1_sum": 0.}
        originals[name] = module.routing
        def traced(z, adjacency, mask, base=module.routing, totals=totals):
            attention, fraction, amplitude = base(z, adjacency, mask)
            with torch.no_grad():
                valid = mask.any(2)
                prior = adjacency[None] * valid[:, None].float()
                prior = prior / prior.sum(-1, keepdim=True).clamp_min(1e-6)
                gates = amplitude.squeeze(-1).masked_select(valid)
                totals["nodes"] += int(valid.sum())
                totals["dynamic_fraction_sum"] += float(fraction.squeeze(-1).masked_select(valid).sum())
                totals["gate_sum"] += float(gates.sum())
                totals["gate_square_sum"] += float(gates.square().sum())
                totals["saturated_gate_count"] += int((gates.abs() > .95).sum())
                totals["attention_l1_sum"] += float((attention-prior).abs().sum(-1).masked_select(valid).sum())
            return attention, fraction, amplitude
        module.routing = traced
    try:
        validation = runtime.evaluate(model, data, "val", gpu, budget)
    finally:
        for name, base in originals.items():
            model.graph[name].routing = base
    for totals in summaries.values():
        count = max(totals["nodes"], 1)
        totals["dynamic_fraction_mean"] = totals["dynamic_fraction_sum"]/count
        totals["gate_mean"] = totals["gate_sum"]/count
        totals["gate_std"] = max(0, totals["gate_square_sum"]/count-totals["gate_mean"]**2)**.5
        totals["attention_l1_mean"] = totals["attention_l1_sum"]/count
    deltas = {n: float((p.detach().cpu()-initial[n]).norm()) for n, p in model.named_parameters()
              if p.requires_grad and any(part in n for part in (".query.", ".key.", ".router.", ".gate"))}
    return {"split": "validation", "layers": summaries, "parameter_delta_l2": deltas, "validation": validation}


def run(args):
    if os.environ.get("GOFA_IDLE_ADMITTED") != "1":
        raise RuntimeError("Only the strict idle launcher may admit GPU tasks")
    config, output = config_file(args.config), Path(args.output).resolve()
    binding = read(output/"manifest.json")
    for path, checksum in binding["inputs"][args.dataset].items():
        if file_sha256(path) != checksum:
            raise ValueError("Input changed: "+path)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, runtime.stop_handler)
    budget_path = output/"budget.json"
    budget = TaskBudget(budget_path, float("inf"), read(budget_path))
    budget.activate(args.phase+"/"+args.dataset, float("inf"))
    resources = dict(config["resources"], memory_cap_gib=38)
    gpu = runtime.GPU(resources)
    torch.cuda.reset_peak_memory_stats()
    cache = None
    directory = output/"mechanisms" if args.phase == "preflight" else job_directory(output, args.dataset)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        from chronos.chronos_bolt import ChronosBoltModelForForecasting
        native = ChronosBoltModelForForecasting.from_pretrained(config["native_weights"], local_files_only=True,
                    torch_dtype=torch.bfloat16).to(gpu.device).requires_grad_(False).eval()
        spec = yaml.safe_load(Path("configs/benchmark.yaml").read_text())
        if args.phase == "preflight":
            preflight(config, spec, output, native, gpu, budget)
            return
        data, settings = data_for(spec, args.dataset, gpu.device, args.phase == "evaluate")
        model = new_model(config, native, data, settings)
        initial = runtime.trained_state(model)
        metadata = {"dataset": args.dataset, "architecture": config["architecture"], "seed": 42,
            "data_sha256": data.series.fingerprint, "training_windows": len(data.windows["train"]),
            "context": data.context, "horizon": data.horizon, "parameters": model_counts(model),
            "loss": "raw_mae", "graph_lr": .001, "adapter_lr": .0001}
        if (directory/"metadata.json").exists() and read(directory/"metadata.json") != metadata:
            raise ValueError("Task manifest changed")
        atomic_json(directory/"metadata.json", metadata)
        if args.phase == "train":
            cache = training_cache(config, output, args.dataset, model, data, gpu, budget)
            state = runtime.fit(model, data, directory, gpu, budget, .001, cache,
                max_updates=args.milestone*math.ceil(len(data.windows["train"])/64), max_epochs=100,
                patience=15, adapter_lr=.0001, optimizer_kind="adamw", loss_kind="raw_mae")
            if budget.remaining() <= 0:
                raise runtime.StopRequested("Training cutoff")
            epochs, complete = completed(state, len(data.windows["train"]))
            record = {"complete": complete, "epochs": epochs, "updates": state["updates"],
                      "validation_raw_mae": state["best_mae"], "parameters": metadata["parameters"]}
            atomic_json(directory/"done.json", record)
            atomic_json(directory/f"milestone_{args.milestone}.json", record)
        else:
            locked = read(output/"final_selection.json")["jobs"][args.dataset]
            if file_sha256(directory/"best.pt") != locked["checkpoint_sha256"]:
                raise ValueError("Best checkpoint changed after global seal")
            selected = runtime.best(directory, model)
            if args.phase == "evaluate":
                result = {"status": "complete", "dataset": args.dataset, "variant": "routed", "seed": 42,
                    "context": data.context, "horizon": data.horizon, "parameters": metadata["parameters"],
                    "checkpoint_sha256": locked["checkpoint_sha256"], "checkpoint_updates": selected["updates"],
                    "training_complete": locked["training_complete"], "validation": selected["validation"],
                    "test": runtime.evaluate(model, data, "test", gpu, budget), "evaluation_protocol": "raw-target-v2",
                    "layout": "late_contiguous", "graph_precision": "float32"}
                atomic_json(directory/"result.json", result)
            else:
                routes = routing_diagnostics(model, data, gpu, budget, initial)
                routes["checkpoint_sha256"] = locked["checkpoint_sha256"]
                if args.dataset == "electricity":
                    atomic_json(directory/"diagnostics.json", diagnostics(model, data, gpu, budget))
                atomic_json(directory/"routing.json", routes)
    finally:
        record_memory(directory/"cuda_memory.json", args.phase, torch.cuda.max_memory_allocated(),
                      torch.cuda.max_memory_reserved())
        if cache is not None:
            cache.close()
        budget.close()
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", required=True, choices=("preflight", "train", "evaluate", "diagnose"))
    parser.add_argument("--dataset", default="metr_la", choices=DATASETS)
    parser.add_argument("--milestone", type=int, default=100)
    args = parser.parse_args()
    try:
        run(args)
    except (runtime.StopRequested, BudgetExhausted):
        raise SystemExit(75)


if __name__ == "__main__":
    main()
