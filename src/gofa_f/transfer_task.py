"""One admitted evaluation of a locked Stage5 checkpoint on a target dataset."""
import argparse
import gc
import os
from pathlib import Path
import signal

import numpy as np
import torch
import yaml

from gofa_f import native_stage as runtime
from gofa_f.data import file_sha256
from gofa_f.graphs import build_graph
from gofa_f.native import NativeBolt
from gofa_f.prefix_cache import atomic_json
from gofa_f.pure_graph_task import TaskBudget, model_counts
from gofa_f.routed_stage import record_memory
from gofa_f.stage_budget import BudgetExhausted


def target_data(config, target, device, evaluation=True):
    spec = yaml.safe_load(Path("configs/benchmark.yaml").read_text())
    settings = dict(spec["datasets"][target], validation_only=not evaluation,
                    graph_context={"enabled": True, "feature_mode": "values_only"})
    return runtime.Data(settings, device), settings


def target_model(stage5, native, data, settings, trained):
    runtime.seed(42)
    graph, _ = build_graph(data.series, settings)
    model = NativeBolt(native, graph, data.horizon, **stage5["architecture"]).to(data.device)
    runtime.load_trained(model, trained)
    if model_counts(model)["trainable_real_dof"] != 113286:
        raise ValueError("Transfer model trainable shape changed")
    return model.eval()


def load_source(config):
    path = Path(config["source_checkpoint"])
    report = __import__("json").loads(Path(config["stage5_result"]).read_text())
    expected = report["final_selection"]["jobs"][config["source_dataset"]]["checkpoint_sha256"]
    if file_sha256(path) != expected:
        raise ValueError("Locked Stage5 source checkpoint changed")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint.get("trained"), dict):
        raise ValueError("Source checkpoint lacks private trainable weights")
    return checkpoint, report, expected


def preflight_cpu(config, stage5):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU preflight requires CUDA_VISIBLE_DEVICES empty")
    from chronos.chronos_bolt import ChronosBoltModelForForecasting
    checkpoint, _, expected = load_source(config)
    native = ChronosBoltModelForForecasting.from_pretrained(stage5["native_weights"],
        local_files_only=True, torch_dtype=torch.float32).requires_grad_(False).eval()
    records = {}
    for target in config["targets"]:
        data, settings = target_data(config, target, torch.device("cpu"), evaluation=False)
        model = target_model(stage5, native, data, settings, checkpoint["trained"])
        batch = data.batch("val", [0, len(data.windows["val"])-1])
        with torch.no_grad():
            prediction = model(batch["x"], batch["x_mask"], batch["graph_features"])
        if prediction.shape != batch["y"].shape or not torch.isfinite(prediction).all():
            raise AssertionError("Transfer shape or finite CPU prediction failed: " + target)
        if any(p.grad is not None for p in native.parameters()):
            raise AssertionError("Transfer modified frozen backbone")
        records[target] = {"nodes": data.values.shape[1], "context": data.context,
                           "horizon": data.horizon, "validation_windows": len(data.windows["val"]),
                           "finite": True, "parameters": model_counts(model)}
        del model, data
        gc.collect()
    if torch.cuda.is_initialized():
        raise AssertionError("CPU preflight initialized CUDA")
    return {"passed": True, "source_checkpoint_sha256": expected,
            "target_train_statistics": True, "weight_updates": 0, "targets": records}


def evaluate_target(config, stage5, output, target):
    if os.environ.get("GOFA_IDLE_ADMITTED") != "1":
        raise RuntimeError("Only strict idle admission may start GPU evaluation")
    from gofa_f.transfer_stage import read
    binding = read(Path(output)/"manifest.json")
    if binding["git_revision"] != __import__("subprocess").check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip():
        raise ValueError("Transfer source revision changed")
    if any(file_sha256(path) != checksum for path, checksum in binding["inputs"].items()):
        raise ValueError("Transfer input changed after binding")
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, runtime.stop_handler)
    directory = Path(output)/"targets"/target
    directory.mkdir(parents=True, exist_ok=True)
    budget_path = Path(output)/"budget.json"
    budget = TaskBudget(budget_path, float("inf"), read(budget_path))
    budget.activate("evaluate/"+target, float("inf"))
    try:
        gpu = runtime.GPU(dict(stage5["resources"], memory_cap_gib=38))
        torch.cuda.reset_peak_memory_stats()
        from chronos.chronos_bolt import ChronosBoltModelForForecasting
        checkpoint, report, expected = load_source(config)
        native = ChronosBoltModelForForecasting.from_pretrained(stage5["native_weights"],
            local_files_only=True, torch_dtype=torch.bfloat16).to(gpu.device).requires_grad_(False).eval()
        data, settings = target_data(config, target, gpu.device)
        model = target_model(stage5, native, data, settings, checkpoint["trained"])
        batch = data.batch("val", [0, len(data.windows["val"])-1])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            probe = model(batch["x"], batch["x_mask"], batch["graph_features"])
        if probe.shape != batch["y"].shape or not torch.isfinite(probe).all():
            raise AssertionError("BF16 transfer mechanism failed: " + target)
        if any(p.grad is not None for p in native.parameters()):
            raise AssertionError("Frozen native backbone received gradient")
        validation = runtime.evaluate(model, data, "val", gpu, budget)
        if target == config["source_dataset"]:
            old = report["results"][target]["validation"]["raw"]["mae"]
            if not np.isclose(validation["raw"]["mae"], old, atol=1e-4, rtol=0):
                raise AssertionError("Locked source did not reproduce Stage5 validation")
        test = runtime.evaluate(model, data, "test", gpu, budget)
        result = {"status": "complete", "source": config["source_dataset"], "target": target,
                  "source_checkpoint_sha256": expected, "weight_updates": 0,
                  "target_train_statistics": True, "target_graph": settings["graph"],
                  "context": data.context, "horizon": data.horizon,
                  "parameters": model_counts(model), "validation": validation, "test": test,
                  "evaluation_protocol": "raw-target-v2"}
        atomic_json(directory/"result.json", result)
    finally:
        if torch.cuda.is_initialized():
            record_memory(directory/"cuda_memory.json", "evaluate", torch.cuda.max_memory_allocated(),
                          torch.cuda.max_memory_reserved())
        budget.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    from gofa_f.transfer_stage import config_file
    config, stage5 = config_file(args.config)
    if args.target not in config["targets"]:
        raise ValueError("Unregistered transfer target")
    try:
        evaluate_target(config, stage5, args.output, args.target)
    except (runtime.StopRequested, BudgetExhausted):
        raise SystemExit(75)


if __name__ == "__main__":
    main()
