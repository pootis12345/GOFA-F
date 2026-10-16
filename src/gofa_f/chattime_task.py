"""One strictly admitted Stage16 CUDA mechanism, training, or evaluation task."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList
import yaml

from gofa_f.chattime import ChatTimeGOFA, HistoryCodec, PrefixLRU
from gofa_f.chattime_stage import config_file, digest, read
from gofa_f.graphs import build_graph
from gofa_f.metrics import Metrics
from gofa_f.native_stage import GPU, atomic_torch
from gofa_f import native_stage as runtime
from gofa_f.prefix_cache import atomic_json
from gofa_f.pure_graph_task import TaskBudget
from gofa_f.routed_stage import record_memory
from gofa_f.stage_budget import BudgetExhausted
from gofa_f.temporal_context import SketchData


class StageBudget(TaskBudget):
    def check(self):
        parent = os.environ.get("GOFA_SUPERVISOR_PID")
        if parent and os.getppid() != int(parent):
            raise runtime.StopRequested("CPU supervisor disappeared")
        super().check()


def seed(value=42):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(value)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])


def metric_state(metric):
    return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in vars(metric).items()}


def metric_from(state, horizon=12):
    metric = Metrics(horizon)
    for key, value in state.items():
        old = getattr(metric, key)
        setattr(metric, key, np.asarray(value, dtype=old.dtype) if isinstance(old, np.ndarray) else value)
    return metric


def available_metric(metric):
    """Valid-generation horizons may legitimately have no valid answers."""
    count = int(metric.count.sum())
    if count == 0:
        return None
    if (metric.count > 0).all():
        return metric.result()
    ape = int(metric.ape_count.sum())
    return {"mae": float(metric.abs.sum()/count), "rmse": float(np.sqrt(metric.square.sum()/count)),
        "mape_pct": float(100*metric.ape.sum()/ape) if ape else None,
        "mape_valid_fraction": ape/count, "observations": count,
        "wape_pct": float(100*metric.abs.sum()/metric.target_abs.sum()) if metric.target_abs.sum() else None,
        "by_horizon": [{"step": i+1, "mae": float(a/c) if c else None,
            "rmse": float(np.sqrt(s/c)) if c else None, "observations": int(c)}
            for i, (a, s, c) in enumerate(zip(metric.abs, metric.square, metric.count))]}


def data_for(config, evaluation=False):
    spec = yaml.safe_load(Path(config["data"]["benchmark"]).read_text())
    settings = dict(spec["datasets"]["metr_la"], validation_only=not evaluation,
                    graph_context={"enabled": True, "feature_mode": "values_and_clock",
                                   "time_path": config["data"]["calendar"]})
    return SketchData(settings, torch.device("cpu")), settings


def encoded_batch(codec, data, batch, device):
    # Retrieve raw observed history, never reconstruct rounded labels from z scores.
    origins = batch["origins"].cpu().numpy()
    history = origins[:, None]+np.arange(-12, 0)
    return codec.encode(data.series.values[history], data.series.mask[history],
                        data.mean.cpu().numpy()).to(device)


def new_model(native, data, settings, output):
    seed()
    physical, _ = build_graph(data.series, settings)
    with np.load(Path(output)/"correlation_graph.npz", allow_pickle=False) as source:
        correlation = source["adjacency"]
    return ChatTimeGOFA(native, physical, correlation)


def counts(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    native = sum(p.numel() for p in model.native.parameters())
    head = sum(p.numel() for p in model.native.lm_head.parameters())
    return {"registered_real_dof": total, "trainable_real_dof": trainable,
        "continuous_forward_module_real_dof": total-head, "native_registered_real_dof": native,
        "new_graph_and_router": sum(p.numel() for p in model.graph.parameters()),
        "new_continuous_head": sum(p.numel() for p in model.head.parameters()),
        "note": "Forward module count excludes original unused LM head; embedding tensor counted in full. "
                "Frozen parameters and compute are not matched to historical UFG-A."}


def make_optimizer(model):
    return torch.optim.AdamW([{"params": model.graph.parameters(), "lr": .001},
                             {"params": model.head.parameters(), "lr": .0001}], weight_decay=.0001)


def gradients(model):
    return {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters()
            if p.requires_grad and p.grad is not None}


def restore_gradients(model, state):
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.grad = state[name].to(param).clone() if name in state else None


def initial_state():
    return {"epoch": 0, "offset": 0, "updates": 0, "stale": 0, "best_mae": None,
        "best_updates": None, "phase": "train", "pending_windows": 0, "pending_count": 0,
        "pending_loss": 0., "validation_offset": 0, "validation_metrics": {},
        "last_validated_key": None, "suffix_checkpointing": False, "prefix_node_chunk": 64}


def save_checkpoint(directory, model, optimizer, state, binding):
    directory = Path(directory)
    state["suffix_checkpointing"] = model.suffix_checkpointing
    atomic_torch(directory/"last.pt", {"binding": binding, "model": model.trainable_state(),
        "optimizer": optimizer.state_dict(), "gradients": gradients(model), "rng": rng_state(), "state": state})
    public = {k: v for k, v in state.items() if k != "validation_metrics"}
    public.update({"updated_at": time.time(), "checkpoint_sha256": digest(directory/"last.pt")})
    atomic_json(directory/"progress.json", public)


def load_checkpoint(directory, model, optimizer, binding):
    path = Path(directory)/"last.pt"
    if not path.exists():
        return initial_state()
    saved = torch.load(path, map_location="cpu", weights_only=False)  # trusted own snapshot artifact
    if saved["binding"] != binding:
        raise ValueError("Resume checkpoint differs from immutable manifest")
    model.load_trainable(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    restore_gradients(model, saved["gradients"])
    restore_rng(saved["rng"])
    model.suffix_checkpointing = saved["state"]["suffix_checkpointing"]
    return saved["state"]


def optimizer_update(model, optimizer, state):
    count = state["pending_count"]
    if count:
        parameters = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
        for param in parameters:
            param.grad.div_(count)
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True, foreach=False)
        optimizer.step()
        state["updates"] += 1
        state["last_gradient_norm"] = float(norm)
        state["last_training_raw_mae"] = state["pending_loss"]/count
    optimizer.zero_grad(set_to_none=True)
    state.update(pending_count=0, pending_windows=0, pending_loss=0.)


def apply_validation(state, record):
    """Journal makes a completed validation idempotent across interrupted saves."""
    state.update(best_mae=record["best_mae"], best_updates=record["best_updates"],
                 stale=record["stale"], last_validated_key=record["key"], epoch=record["next_epoch"],
                 offset=0, phase="train", validation_offset=0, validation_metrics={})


def train(config, output, model, codec, data, gpu, budget, binding):
    directory = Path(output)/"gofa_chattime"
    directory.mkdir(parents=True, exist_ok=True)
    (directory/"validations").mkdir(exist_ok=True)
    optimizer = make_optimizer(model)
    state = load_checkpoint(directory, model, optimizer, binding)
    probe = encoded_batch(codec, data, data.batch("train", [0]), torch.device("cpu"))
    estimate = len(data.windows["train"])*probe.input_ids.numel()*(model.native.config.hidden_size*2+1)
    disk = Path(output)/"cache" if estimate <= 128*2**30 else None
    cache = PrefixLRU(binding, disk_directory=disk)
    atomic_json(Path(output)/"cache.json", {"estimated_full_bytes": estimate, "disk_enabled": disk is not None,
        "disk_limit_bytes": 128*2**30, "ram_limit_bytes": cache.max_bytes, "prefix_layers": 30,
        "dtype": "bfloat16, lossless", "key_binding": binding})
    saved_at = time.monotonic()
    recovered_oom = False

    def guard():
        gpu.check(budget)

    try:
        while state["epoch"] < 100 and state["stale"] < 15:
            guard()
            if state["phase"] == "train":
                order = np.random.default_rng(42+state["epoch"]).permutation(len(data.windows["train"]))
                model.train()
                while state["offset"] < len(order):
                    guard()
                    if recovered_oom:
                        gc.collect()
                        torch.cuda.empty_cache()
                        recovered_oom = False
                    batch = data.batch("train", [int(order[state["offset"]])])
                    encoded = encoded_batch(codec, data, batch, gpu.device)
                    before_rng, before_grad = rng_state(), gradients(model)
                    try:
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            prediction = model(encoded, batch["graph_features"].to(gpu.device), cache,
                                               state["prefix_node_chunk"], guard)
                        mask, target = batch["y_mask"].to(gpu.device), batch["y_raw"].to(gpu.device)
                        error = (prediction-target).masked_select(mask)
                        if not torch.isfinite(error).all():
                            raise FloatingPointError("Nonfinite raw-unit training error")
                        loss = error.abs().sum()
                        loss.backward()
                    except torch.OutOfMemoryError:
                        restore_rng(before_rng)
                        restore_gradients(model, before_grad)
                        gc.collect()
                        torch.cuda.empty_cache()
                        if not model.suffix_checkpointing:
                            model.suffix_checkpointing = True
                            state["suffix_checkpointing"] = True
                            atomic_json(directory/"oom_recovery.json", {"suffix_checkpointing": True,
                                "microbatch": 1, "updates": state["updates"], "offset": state["offset"]})
                            for name in ("prediction", "loss", "error"):
                                if name in locals():
                                    # Drop graph references before the next retry.
                                    if name == "prediction":
                                        del prediction
                                    elif name == "loss":
                                        del loss
                                    else:
                                        del error
                            recovered_oom = True
                            continue
                        raise RuntimeError("OOM at graph microbatch1 with suffix checkpointing; no architecture fallback")
                    state["pending_count"] += int(mask.sum())
                    state["pending_windows"] += 1
                    state["pending_loss"] += float(loss.detach())
                    state["offset"] += 1
                    del prediction, loss, error, encoded, before_grad, batch
                    if state["pending_windows"] == 64 or state["offset"] == len(order):
                        optimizer_update(model, optimizer, state)
                    if time.monotonic()-saved_at >= 30:
                        save_checkpoint(directory, model, optimizer, state, binding)
                        saved_at = time.monotonic()
                state["phase"] = "validate"
                save_checkpoint(directory, model, optimizer, state, binding)

            key = f"epoch-{state['epoch']+1:03d}-update-{state['updates']:08d}"
            journal = directory/"validations"/(key+".json")
            record = read(journal)
            if record is None:
                model.eval()
                metric = metric_from(state["validation_metrics"]) if state["validation_metrics"] else Metrics(12)
                while state["validation_offset"] < len(data.windows["val"]):
                    guard()
                    batch = data.batch("val", [state["validation_offset"]])
                    encoded = encoded_batch(codec, data, batch, gpu.device)
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        prediction = model(encoded, batch["graph_features"].to(gpu.device), cache,
                                           state["prefix_node_chunk"], guard)
                    metric.update(prediction.cpu(), batch["y_raw"], batch["y_mask"])
                    state["validation_offset"] += 1
                    state["validation_metrics"] = metric_state(metric)
                    if time.monotonic()-saved_at >= 30:
                        save_checkpoint(directory, model, optimizer, state, binding)
                        saved_at = time.monotonic()
                validation = {"raw": metric.result()}
                score = validation["raw"]["mae"]
                improved = state["best_mae"] is None or score < state["best_mae"]
                if improved:
                    atomic_torch(directory/"best.pt", {"model": model.trainable_state(), "validation": validation,
                        "updates": state["updates"], "validation_key": key, "binding": binding})
                    atomic_json(directory/"best_validation.json", {"validation": validation,
                        "updates": state["updates"], "validation_key": key,
                        "checkpoint_sha256": digest(directory/"best.pt")})
                record = {"key": key, "epoch": state["epoch"]+1, "updates": state["updates"],
                    "validation": validation, "improved": improved,
                    "best_mae": score if improved else state["best_mae"],
                    "best_updates": state["updates"] if improved else state["best_updates"],
                    "stale": 0 if improved else state["stale"]+1, "next_epoch": state["epoch"]+1}
                atomic_json(journal, record)
                print(json.dumps({"phase": "validation", "epoch": record["epoch"], "updates": state["updates"],
                    "raw_mae": score, "step12_mae": validation["raw"]["by_horizon"][11]["mae"]}), flush=True)
            apply_validation(state, record)
            history = [read(p) for p in sorted((directory/"validations").glob("*.json"))]
            atomic_json(directory/"history.json", history)
            save_checkpoint(directory, model, optimizer, state, binding)
        state["stop_reason"] = "early_stopping" if state["stale"] >= 15 else "max_epochs"
    except (runtime.StopRequested, BudgetExhausted):
        state["stop_reason"] = "training_deadline_or_signal"
        raise
    finally:
        save_checkpoint(directory, model, optimizer, state, binding)
        cache_record = read(Path(output)/"cache.json")
        cache_record.update(ram_peak_bytes=cache.peak, hits=cache.hits, misses=cache.misses,
                            disk_bytes=cache.disk_used)
        atomic_json(Path(output)/"cache.json", cache_record)


class GuardGeneration(StoppingCriteria):
    def __init__(self, check):
        self.check = check

    def __call__(self, input_ids, scores, **kwargs):
        self.check()
        return torch.zeros(len(input_ids), dtype=torch.bool, device=input_ids.device)


def native_window(native, codec, encoded, directory, index, check, node_micro=64):
    directory = Path(directory)
    partial = directory/"partial.pt"
    rows = len(encoded.input_ids)
    predictions = np.repeat(encoded.last.cpu().numpy().reshape(-1, 1), 12, axis=1)
    valid = np.zeros((rows, 12), dtype=bool)
    cursor = 0
    if partial.exists():
        old = torch.load(partial, map_location="cpu", weights_only=False)
        if old["index"] != index:
            if old["index"] < index and (directory/f"{old['index']:06d}.npz").exists():
                partial.unlink()  # Completed-shard commit survived before partial cleanup.
            else:
                raise ValueError("Native evaluation cursor changed")
        else:
            if old["rows"] != rows:
                raise ValueError("Native evaluation node count changed")
            cursor, predictions, valid, node_micro = old["cursor"], old["prediction"], old["valid"], old["node_micro"]
    while cursor < rows:
        check()
        end = min(rows, cursor+node_micro)
        ids, mask = encoded.input_ids[cursor:end], encoded.attention_mask[cursor:end]
        try:
            with torch.no_grad(), torch.autocast(encoded.input_ids.device.type, dtype=torch.bfloat16):
                generated = native.generate(input_ids=ids, attention_mask=mask,
                    do_sample=False, max_new_tokens=64, use_cache=True,
                    temperature=1.0, top_p=1.0,
                    pad_token_id=codec.tokenizer.eos_token_id, eos_token_id=codec.tokenizer.eos_token_id,
                    stopping_criteria=StoppingCriteriaList([GuardGeneration(check)]))
            texts = codec.tokenizer.batch_decode(generated[:, ids.shape[1]:], skip_special_tokens=False)
        except torch.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            if node_micro == 1:
                raise RuntimeError("Native generation OOM at node microbatch1")
            node_micro = max(1, node_micro//2)
            continue
        for row, text in enumerate(texts, cursor):
            predictions[row], valid[row] = codec.parse(text, encoded.inverse_scale[row],
                encoded.inverse_offset[row], float(encoded.last.reshape(-1)[row]))
            if not bool(encoded.present.reshape(-1)[row]):
                predictions[row] = float(encoded.last.reshape(-1)[row])
                valid[row] = False
        cursor = end
        atomic_torch(partial, {"index": index, "rows": rows, "cursor": cursor,
            "prediction": predictions, "valid": valid, "node_micro": node_micro})
    b, n = encoded.shape
    return predictions.reshape(b, n, 12).transpose(0, 2, 1), valid.reshape(b, n, 12).transpose(0, 2, 1)


def atomic_npz(path, **values):
    temporary = Path(str(path)+".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def prediction_path(output, variant, split, index):
    return Path(output)/variant/"predictions"/split/f"{index:06d}.npz"


def write_window(output, variant, split, index, origin, prediction, valid, binding):
    path = prediction_path(output, variant, split, index)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_npz(path, prediction=prediction, valid=valid, origin=origin, binding=binding)
    partial = path.parent/"partial.pt"
    if partial.exists():
        partial.unlink()


def load_window(path, origin, binding):
    with np.load(path, allow_pickle=False) as saved:
        if int(saved["origin"]) != int(origin) or str(saved["binding"]) != binding:
            raise ValueError("Evaluation shard source/cursor changed")
        prediction, valid = saved["prediction"], saved["valid"]
    if prediction.shape != valid.shape or prediction.shape[0:2] != (1, 12) or not np.isfinite(prediction).all():
        raise ValueError("Invalid evaluation shard")
    return prediction, valid


def summarize_split(output, variant, split, data, binding):
    raw, generated = Metrics(12), Metrics(12)
    completed, fallback, total_forecasts = 0, 0, 0
    for i, origin in enumerate(data.windows[split].origins):
        path = prediction_path(output, variant, split, i)
        if not path.exists():
            continue
        prediction, valid = load_window(path, origin, binding)
        batch = data.batch(split, [i])
        p, v = torch.from_numpy(prediction), torch.from_numpy(valid)
        raw.update(p, batch["y_raw"], batch["y_mask"])
        generated.update(p, batch["y_raw"], batch["y_mask"] & v)
        completed += 1
        fallback += int((batch["y_mask"] & ~v).sum())
        total_forecasts += valid.size
    progress = {"completed_windows": completed, "total_windows": len(data.windows[split]),
                "complete": completed == len(data.windows[split])}
    if not progress["complete"]:
        return progress, None
    result = {"raw": raw.result(), "valid_generation": available_metric(generated),
              "observed_fallback_fraction": fallback/int(raw.count.sum()),
              "observed_valid_generation_fraction": int(generated.count.sum())/int(raw.count.sum()),
              "total_forecast_points": total_forecasts, "complete": True}
    if variant == "gofa_chattime":
        result.pop("valid_generation")
    return progress, result


def finalize_evaluation(output, data, binding, locked):
    enough = True
    for variant in ("native_chattime", "gofa_chattime"):
        directory = Path(output)/variant
        directory.mkdir(parents=True, exist_ok=True)
        progress, results = {}, {}
        for split in ("val", "test"):
            if variant == "gofa_chattime" and split == "val":
                results[split] = locked["gofa"]["validation"] if locked.get("gofa") else None
                progress[split] = {"complete": bool(locked.get("gofa")), "source": "locked complete validation"}
            else:
                progress[split], results[split] = summarize_split(output, variant, split, data, binding)
        atomic_json(directory/"evaluation_progress.json", progress)
        complete = all(v is not None for v in results.values())
        enough &= complete
        if complete:
            result = {"complete": True, "model": variant, "validation": results["val"],
                "test": results["test"], "binding": binding,
                "checkpoint": locked["native"] if variant == "native_chattime" else locked["gofa"],
                "metr_minutes": {str(minute): results["test"]["raw"]["by_horizon"][step-1]
                                 for minute, step in ((5, 1), (15, 3), (30, 6), (60, 12))}}
            atomic_json(directory/"result.json", result)
    return enough


def evaluate(config, output, model, codec, data, gpu, budget, binding):
    locked = read(Path(output)/"final_selection.json")
    if not locked or locked["native"]["revision"] != config["weights"]["revision"]:
        raise ValueError("Test access requires sealed weights and decoding")
    if locked.get("gofa"):
        path = Path(locked["gofa"]["path"])
        if digest(path) != locked["gofa"]["checkpoint_sha256"]:
            raise ValueError("Selected checkpoint changed after seal")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["binding"] != binding:
            raise ValueError("Best checkpoint source binding differs")
        model.load_trainable(saved["model"])
    model.eval()

    def guard():
        gpu.check(budget)

    try:
        for split in ("val", "test"):
            for i, origin in enumerate(data.windows[split].origins):
                variants = ("native_chattime",) if split == "val" else ("gofa_chattime", "native_chattime")
                for variant in variants:
                    if variant == "gofa_chattime" and not locked.get("gofa"):
                        continue
                    path = prediction_path(output, variant, split, i)
                    if path.exists():
                        load_window(path, origin, binding)
                        continue
                    guard()
                    path.parent.mkdir(parents=True, exist_ok=True)
                    batch = data.batch(split, [i])
                    if variant == "native_chattime":
                        history = np.asarray(data.windows[split].origins)[[i]][:, None]+np.arange(-12, 0)
                        encoded = codec.encode(data.series.values[history], data.series.mask[history],
                                               data.mean.numpy(), padding_side="left").to(gpu.device)
                        prediction, valid = native_window(model.native, codec, encoded, path.parent,
                                                         i, guard)
                    else:
                        encoded = encoded_batch(codec, data, batch, gpu.device)
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                            prediction = model(encoded, batch["graph_features"].to(gpu.device), check=guard).cpu().numpy()
                        valid = np.ones_like(prediction, dtype=bool)
                    write_window(output, variant, split, i, origin, prediction, valid, binding)
                if (i+1) % 32 == 0:
                    # Cheap cursor progress; aggregate metrics only at completion/release.
                    for variant in variants:
                        directory = Path(output)/variant
                        directory.mkdir(parents=True, exist_ok=True)
                        old = read(directory/"evaluation_progress.json", {})
                        old[split] = {"last_completed_origin_index": i, "total_windows": len(data.windows[split]),
                                      "complete": False}
                        atomic_json(directory/"evaluation_progress.json", old)
    finally:
        finalize_evaluation(output, data, binding, locked)


def mechanisms(config, output, model, codec, data, gpu, budget, binding):
    from gofa_f.chattime_checks import check_model
    batch = data.batch("train", [0])
    encoded = encoded_batch(codec, data, batch, gpu.device)
    result = check_model(model, encoded, batch["graph_features"].to(gpu.device),
                         check=lambda: gpu.check(budget), gpu=True, binding=binding)
    atomic_json(Path(output)/"mechanisms.json", {"passed": True, **result})


def run(args):
    if os.environ.get("GOFA_IDLE_ADMITTED") != "1":
        raise RuntimeError("GPU tasks require strict idle admission")
    config, output = config_file(args.config), Path(args.output).resolve()
    manifest = read(output/"manifest.json")
    if not manifest or manifest["inputs"] != {p: digest(p) for p in manifest["inputs"]}:
        raise ValueError("Pinned Stage16 inputs changed")
    if args.phase == "evaluate" and not (output/"final_selection.json").exists():
        raise ValueError("Missing test-access seal")
    for path, checksum in manifest["runtime"]["critical_files"].items():
        if digest(path) != checksum:
            raise ValueError("CPU-verified dependency source changed")
    binding = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, runtime.stop_handler)
    runtime.STOP = False
    budget_path = output/"budget.json"
    budget = StageBudget(budget_path, float("inf"), read(budget_path))
    budget.activate(args.phase, float("inf"))
    gpu = GPU({"memory_cap_gib": 38, "leave_free_gib": 16, "allow_tf32": True, "microbatch_candidates": [1]})
    torch.cuda.reset_peak_memory_stats()
    try:
        gpu.check(budget)
        versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy")}
        if versions != {k: manifest["runtime"]["packages"][k] for k in versions}:
            raise ValueError("Actual task dependencies differ from CPU-verified environment")
        seed()
        native = LlamaForCausalLM.from_pretrained(config["weights"]["directory"], local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(gpu.device).requires_grad_(False).eval()
        tokenizer = LlamaTokenizer.from_pretrained(config["weights"]["directory"], local_files_only=True)
        codec = HistoryCodec(tokenizer)
        # Construction of test windows/targets is prohibited until the seal exists.
        data, settings = data_for(config, evaluation=args.phase == "evaluate")
        model = new_model(native, data, settings, output).to(gpu.device)
        parameters = counts(model)
        if parameters["trainable_real_dof"] != 1120914:
            raise ValueError("Registered trainable parameter count changed")
        old = read(output/"parameters.json")
        if old is not None and old != parameters:
            raise ValueError("Parameter audit changed")
        atomic_json(output/"parameters.json", parameters)
        if args.phase == "preflight":
            mechanisms(config, output, model, codec, data, gpu, budget, binding)
        elif args.phase == "train":
            if not read(output/"mechanisms.json", {}).get("passed"):
                raise ValueError("Passed CUDA mechanisms required")
            train(config, output, model, codec, data, gpu, budget, binding)
        else:
            evaluate(config, output, model, codec, data, gpu, budget, binding)
    finally:
        record_memory(output/"cuda_memory.json", args.phase,
                      torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
        budget.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", choices=("preflight", "train", "evaluate"), required=True)
    args = parser.parse_args()
    try:
        run(args)
    except (runtime.StopRequested, BudgetExhausted):
        raise SystemExit(75)


if __name__ == "__main__":
    main()
