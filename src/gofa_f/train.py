"""Single-GPU reproducible training. Run this module only on the remote server."""
import argparse
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import transformers
import yaml

from gofa_f.baseline import DSTSGNNAdapter
from gofa_f.data import Series, Windows, correlation_graph
from gofa_f.metrics import Metrics, masked_loss
from gofa_f.models import ForecastEncoder, Persistence, parameter_counts


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def clip_gradients(parameters, max_norm):
    """Float64 norm accumulation avoids overflow for large finite spectral gradients."""
    gradients = [p.grad for p in parameters if p.grad is not None]
    if not gradients:
        raise ValueError("No gradients to clip")
    norm = torch.linalg.vector_norm(torch.stack([
        torch.linalg.vector_norm(g.detach(), dtype=torch.float64) for g in gradients]))
    if not torch.isfinite(norm):
        raise FloatingPointError("Nonfinite gradient values; run retained for diagnosis")
    scale = (max_norm / (norm + 1e-6)).clamp_max(1)
    for gradient in gradients:
        gradient.mul_(scale.to(gradient.dtype))
    return norm


def build_model(config, adjacency, nodes):
    m = dict(config["model"])
    family = m.pop("family", "encoder")
    context, horizon = config["context"], config["horizon"]
    if family == "persistence":
        return Persistence(horizon)
    if family == "dst_sgnn":
        return DSTSGNNAdapter(context, horizon, nodes, **m)
    if family != "encoder":
        raise ValueError(f"Unknown model family: {family}")
    return ForecastEncoder(context, horizon, adjacency, **m)


def evaluate(model, loader, device, mean, std, horizon, amp=False):
    model.eval()
    raw, normalized = Metrics(horizon), Metrics(horizon)
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                pred = model(batch["x"], batch["x_mask"]).float()
            normalized.update(pred, batch["y"], batch["y_mask"])
            raw.update(pred * std + mean, batch["y"] * std + mean, batch["y_mask"])
    return {"raw": raw.result(), "standardized": normalized.result()}


def run(config, output):
    evaluation_split = config.get("evaluation_split", "test")
    if evaluation_split not in ("validation", "test"):
        raise ValueError("evaluation_split must be validation or test")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    seed_everything(config.get("seed", 42))
    series = Series.load(config["data"], config.get("train_ratio", .7), config.get("val_ratio", .1))
    normalized, mean, std = series.standardize()
    adjacency = correlation_graph(series.values[:series.train_end], series.mask[:series.train_end],
                                    k=config.get("graph_k", 8))
    np.savez(output / "preprocessing.npz", mean=mean, std=std, adjacency=adjacency,
             train_end=series.train_end, val_end=series.val_end)
    split_names = ("train", "val", "test") if evaluation_split == "test" else ("train", "val")
    splits = {s: Windows(series, normalized, s, config["context"], config["horizon"],
                         stride=config.get("train_stride", 1) if s == "train" else 1)
              for s in split_names}
    if config.get("max_eval_windows"):
        for split in split_names[1:]:
            splits[split] = Subset(splits[split], range(min(len(splits[split]), config["max_eval_windows"])))
    device = torch.device(config.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(config, adjacency, series.values.shape[1]).to(device)
    counts = parameter_counts(model)
    match = config.get("parameter_match")
    if match:
        relative_error = abs(counts[match["basis"]] / match["target"] - 1)
        if relative_error > match["tolerance"]:
            raise ValueError(f"Parameter match failed: {counts}, relative_error={relative_error:.3f}")
    worker_count = config.get("workers", 0)
    loaders = {s: DataLoader(ds, batch_size=config.get("batch_size", 2), shuffle=s == "train",
                             num_workers=worker_count, pin_memory=device.type == "cuda") for s, ds in splits.items()}
    mean, std = torch.tensor(mean, device=device), torch.tensor(std, device=device)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    env = {"git_revision": revision, "git_dirty": dirty, "python": platform.python_version(),
           "torch": torch.__version__, "transformers": transformers.__version__, "device": str(device),
           "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
           "backbone_revision": getattr(model, "resolved_revision", None),
           "baseline": getattr(model, "provenance", None), "parameters": counts,
           "data_sha256": series.fingerprint,
           "cold_start_nodes": np.flatnonzero(series.mask[:series.train_end].sum(0) < 2).tolist(),
           "split": {"train_end": series.train_end, "val_end": series.val_end, "length": len(series.values)},
           "windows": {s: len(ds) for s, ds in splits.items()},
           "pid": os.getpid()}
    (output / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    print(json.dumps(env), flush=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    amp = bool(config.get("amp", False) and device.type == "cuda"
               and config["model"].get("family") != "dst_sgnn")
    start, best, best_epoch, stale, steps = time.monotonic(), float("inf"), -1, 0, 0
    best_validation = None
    if trainable:
        optimizer = torch.optim.AdamW(trainable, lr=config.get("lr", 1e-4),
                                      weight_decay=config.get("weight_decay", 1e-4))
        accumulation = config.get("accumulation", 1)
        for epoch in range(config.get("epochs", 30)):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss, batches = 0.0, 0
            for index, batch in enumerate(loaders["train"]):
                batch = {k: v.to(device) for k, v in batch.items()}
                group_size = min(accumulation, len(loaders["train"]) - (index // accumulation) * accumulation)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                    pred = model(batch["x"], batch["x_mask"])
                    loss = masked_loss(pred.float(), batch["y"], batch["y_mask"], config.get("loss", "mae"))
                (loss / group_size).backward()
                total_loss += loss.item()
                batches += 1
                steps += 1
                if (index + 1) % accumulation == 0 or index + 1 == len(loaders["train"]):
                    clip_gradients(trainable, config.get("clip_grad", 1.0))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                if config.get("max_steps") and steps >= config["max_steps"]:
                    break
            val = evaluate(model, loaders["val"], device, mean, std, config["horizon"], amp)
            score = val[config.get("selection_scale", "standardized")][config.get("selection_metric", "mae")]
            log = {"epoch": epoch, "train_loss": total_loss / batches, "validation": val, "steps": steps}
            with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(log) + "\n")
            print(json.dumps(log), flush=True)
            if score < best:
                best, best_epoch, stale = score, epoch, 0
                best_validation = val
                torch.save({"model": model.state_dict(), "epoch": epoch, "config": config}, output / "best.pt")
            else:
                stale += 1
            if stale >= config.get("patience", 5) or (config.get("max_steps") and steps >= config["max_steps"]):
                break
        saved = torch.load(output / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(saved["model"])
    else:
        best_validation = evaluate(model, loaders["val"], device, mean, std, config["horizon"], amp)
    result = {"config": config, "environment": env, "best_epoch": best_epoch,
              "validation": best_validation,
              "status": "complete" if evaluation_split == "test" else "validated"}
    # Tuning runs never construct a test loader or evaluate test targets.
    if evaluation_split == "test":
        if device.type == "cuda":
            torch.cuda.synchronize()
        test_start = time.monotonic()
        result["test"] = evaluate(model, loaders["test"], device, mean, std, config["horizon"], amp)
        if device.type == "cuda":
            torch.cuda.synchronize()
        result["test_seconds"] = time.monotonic() - test_start
    if device.type == "cuda":
        torch.cuda.synchronize()
    result.update(elapsed_seconds=time.monotonic() - start,
                  peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0)
    if config.get("max_steps") or config.get("max_eval_windows"):
        result["status"] = "smoke"
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--validation-only", action="store_true", help="Do not construct or evaluate test windows")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.validation_only:
        config["evaluation_split"] = "validation"
    run(config, args.output)


if __name__ == "__main__":
    main()
