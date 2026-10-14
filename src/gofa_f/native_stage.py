"""Single-seed, time-bounded native Bolt / GOFA-Bolt / HSDGNN stage.

Run only on C. Dataset windows and train-only preprocessing are shared across
models. Validation selects layouts/LRs; test targets are read only afterwards.
"""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import time

import numpy as np
import torch
import yaml

from gofa_f.data import Series, Windows, file_sha256
from gofa_f.graphs import build_graph
from gofa_f.graph_context import FEATURE_DIM, FEATURE_NAMES, clock_features, load_calendar, node_features
from gofa_f.metrics import Metrics
from gofa_f.models import Persistence, parameter_counts
from gofa_f.native import HSDGNNAdapter, NativeBolt
from gofa_f.prefix_cache import PrefixCache, atomic_json
from gofa_f.stage_budget import Budget, BudgetExhausted


STOP = False


def stage_config(path, chain=()):
    path = Path(path).resolve()
    if path in chain:
        raise ValueError("Cyclic stage configuration inheritance")
    override = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = override.pop("extends", None)
    if base is None:
        return override
    config = stage_config(path.parent / base, (*chain, path))

    def merge(destination, source):
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(destination.get(key), dict):
                merge(destination[key], value)
            else:
                destination[key] = value
    merge(config, override)
    return config


def dataset_settings(settings, config, name):
    result = copy.deepcopy(settings)
    if config.get("graph_context", {}).get("enabled"):
        result["graph_context"] = {"enabled": True,
                                   "time_path": str(Path(config["graph_context"]["sidecar_root"]) / f"{name}.npz")}
    return result


def context_inputs(config, spec):
    """Bind resume to exact data and independent clock sidecars before any GPU work."""
    if not config.get("graph_context", {}).get("enabled"):
        return None
    inputs = {}
    for name in config["datasets"]:
        settings = dataset_settings(spec["datasets"][name], config, name)
        path = Path(settings["graph_context"]["time_path"])
        inputs[name] = {"prepared_sha256": file_sha256(settings["data"]),
                        "sidecar_sha256": file_sha256(path),
                        "metadata_sha256": file_sha256(path.with_suffix(".metadata.json"))}
    return inputs


class StopRequested(RuntimeError):
    pass


def stop_handler(signum, frame):
    global STOP
    STOP = True


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])


def trained_state(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def load_trained(model, state):
    parameters = dict(model.named_parameters())
    expected = {name for name, p in parameters.items() if p.requires_grad}
    if set(state) != expected:
        raise ValueError("Trainable checkpoint does not match model layout")
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name]))


def atomic_torch(path, value):
    temporary = Path(str(path) + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


class GPU:
    def __init__(self, settings):
        if not torch.cuda.is_available():
            raise RuntimeError("This runner requires CUDA; no silent CPU fallback")
        self.device = torch.device("cuda:0")
        free, total = torch.cuda.mem_get_info()
        self.reserve = int(settings["leave_free_gib"] * 2**30)
        self.cap = min(int(settings["memory_cap_gib"] * 2**30), free - self.reserve)
        if self.cap < 2**30:
            raise RuntimeError("Less than 1 GiB available after preserving shared GPU margin")
        torch.cuda.set_per_process_memory_fraction(self.cap / total)
        torch.backends.cuda.matmul.allow_tf32 = settings["allow_tf32"]
        torch.backends.cudnn.allow_tf32 = settings["allow_tf32"]
        self.micro = settings["microbatch_candidates"][0]
        self.last_check = 0

    def check(self, budget):
        budget.check()
        if STOP:
            raise StopRequested("User signal; checkpoint and stop")
        if time.monotonic() - self.last_check < 10:
            return
        self.last_check = time.monotonic()
        free, _ = torch.cuda.mem_get_info()
        if free < self.reserve:
            torch.cuda.empty_cache()
            while torch.cuda.mem_get_info()[0] < self.reserve:
                budget.check()
                if STOP:
                    raise StopRequested("User signal while waiting for shared GPU")
                print(json.dumps({"state": "waiting_for_shared_gpu", "leave_free_gib": self.reserve / 2**30}), flush=True)
                time.sleep(10)

    def shrink(self):
        if self.micro == 1:
            raise RuntimeError("OOM at microbatch=1; shared GPU budget cannot accommodate this model")
        self.micro //= 2
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps({"state": "reduced_microbatch", "microbatch": self.micro}), flush=True)


class Data:
    def __init__(self, config, device):
        self.series = Series.load(config["data"])
        normalized, mean, std = self.series.standardize()
        self.windows = {split: Windows(self.series, normalized, split, config["context"], config["horizons"][0])
                        for split in ("train", "val", "test")}
        self.values = torch.from_numpy(normalized).to(device)
        self.raw_values = torch.from_numpy(self.series.values).to(device)
        self.mask = torch.from_numpy(self.series.mask).to(device)
        self.mean, self.std = (torch.from_numpy(v).to(device) for v in (mean, std))
        self.context, self.horizon = config["context"], config["horizons"][0]
        self.device = device
        self.context_enabled = config.get("graph_context", {}).get("enabled", False)
        self.context_provenance = None
        if self.context_enabled:
            training = self.series.values[:self.series.train_end][self.series.mask[:self.series.train_end]]
            self.pooled_mean = float(training.mean(dtype=np.float64))
            self.pooled_std = max(float(training.std(dtype=np.float64)), 1e-5)
            length = len(self.series.values)
            if config["graph_context"].get("time_path"):
                calendar, known, provenance = load_calendar(config["graph_context"]["time_path"],
                                                            self.series.fingerprint, length)
            else:
                calendar, known = np.zeros((length, 2), np.float32), np.zeros((length, 2), bool)
                provenance = {"known_calendar": False, "note": "No timestamp sidecar provided"}
            self.calendar = torch.from_numpy(calendar).to(device)
            self.calendar_known = torch.from_numpy(known).to(device)
            self.clock = clock_features(self.calendar, self.calendar_known)
            self.context_provenance = {"feature_names": FEATURE_NAMES, "clock": provenance,
                                       "pooled_mean": self.pooled_mean, "pooled_std": self.pooled_std,
                                       "statistics_fit": "observed training prefix only"}

    def batch(self, split, indices):
        origins = torch.as_tensor(np.asarray(self.windows[split].origins)[indices], device=self.device)
        history = origins[:, None] + torch.arange(-self.context, 0, device=self.device)
        future = origins[:, None] + torch.arange(self.horizon, device=self.device)
        batch = {"x": self.values[history], "x_mask": self.mask[history],
                 "y": self.values[future], "y_raw": self.raw_values[future],
                 "y_mask": self.mask[future], "origins": origins}
        if self.context_enabled:
            # Forecast timestamp is known in advance; no forecast values enter these features.
            batch["graph_features"] = node_features(batch["x"], batch["x_mask"], self.mean, self.std,
                                                    self.pooled_mean, self.pooled_std, self.clock[origins])
            batch["history_calendar"] = (self.calendar[history], self.calendar_known[history])
        return batch

    def batches(self, split, size):
        for i in range(0, len(self.windows[split]), size):
            yield self.batch(split, np.arange(i, min(i + size, len(self.windows[split]))))


def prediction(model, batch, cache=None, indices=None):
    if isinstance(model, NativeBolt):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if cache is None:
                return model(batch["x"], batch["x_mask"], graph_features=batch.get("graph_features")).float()
            return model.from_prefix(cache.take(indices, batch["x"].device),
                                     cache.metadata["prefix_layers"], graph_features=batch.get("graph_features")).float()
    if isinstance(model, HSDGNNAdapter):
        return model(batch["x"], batch["x_mask"], batch["origins"], calendar=batch.get("history_calendar"))
    return model(batch["x"], batch["x_mask"])


@torch.no_grad()
def evaluate(model, data, split, gpu, budget, deadline=None):
    model.eval()
    raw, normalized = Metrics(data.horizon), Metrics(data.horizon)
    offset = 0
    while offset < len(data.windows[split]):
        if deadline is not None and time.monotonic() >= deadline:
            raise BudgetExhausted("candidate_validation_time")
        gpu.check(budget)
        end = min(offset + gpu.micro, len(data.windows[split]))
        batch = data.batch(split, np.arange(offset, end))
        try:
            pred = prediction(model, batch)
        except torch.OutOfMemoryError:
            del batch
            gpu.shrink()
            continue
        normalized.update(pred, batch["y"], batch["y_mask"])
        raw.update(pred * data.std + data.mean, batch["y_raw"], batch["y_mask"])
        offset = end
    return {"raw": raw.result(), "standardized": normalized.result()}


def update(model, optimizer, data, indices, gpu, budget, cache, gradient_clip=1.0):
    before_rng = rng_state()
    while True:
        optimizer.zero_grad(set_to_none=True)
        restore_rng(before_rng)
        model.train()
        # One loss over the entire effective batch, including unequal missing counts.
        count = sum(int(data.batch("train", indices[i:i+gpu.micro])["y_mask"].sum())
                    for i in range(0, len(indices), gpu.micro))
        try:
            loss_sum = 0.0
            for i in range(0, len(indices), gpu.micro):
                gpu.check(budget)
                subset = indices[i:i+gpu.micro]
                batch = data.batch("train", subset)
                pred = prediction(model, batch, cache, subset)
                errors = ((pred - batch["y"]) * data.std).masked_select(batch["y_mask"])
                if not torch.isfinite(errors).all():
                    raise FloatingPointError("Nonfinite training prediction")
                loss = errors.abs().sum() / count
                loss.backward()
                loss_sum += loss.detach().item()
        except torch.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            # Free activations referenced by this frame before retrying the same effective batch.
            if "pred" in locals():
                del pred, errors, loss
            if "batch" in locals():
                del batch
            gpu.shrink()
            continue
        except (BudgetExhausted, StopRequested):
            optimizer.zero_grad(set_to_none=True)
            restore_rng(before_rng)
            raise
        # An optimizer OOM must fail safely; never retry a potentially partially applied update.
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                      gradient_clip, error_if_nonfinite=True)
        optimizer.step()
        return loss_sum


def fit(model, data, directory, gpu, budget, lr, cache=None, max_updates=None, candidate_seconds=None,
        adapter_lr=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer_type = torch.optim.Adam if isinstance(model, HSDGNNAdapter) else torch.optim.AdamW
    if adapter_lr is not None:
        graph_parameters = [p for name, p in model.named_parameters()
                            if p.requires_grad and not name.startswith("output_adapter.")]
        head_parameters = [p for name, p in model.named_parameters()
                           if p.requires_grad and name.startswith("output_adapter.")]
        parameters = ([{"params": graph_parameters, "lr": lr}] if graph_parameters else [])
        if head_parameters:
            parameters.append({"params": head_parameters, "lr": adapter_lr})
    optimizer = optimizer_type(parameters, lr=lr, weight_decay=1e-4, fused=True)
    state = {"epoch": 0, "offset": 0, "updates": 0, "best_mae": None, "stale": 0,
              "validation": None, "microbatch": gpu.micro, "lr": lr, "adapter_lr": adapter_lr,
              "last_validation_updates": None}
    path = directory / "last.pt"
    if path.exists():
        # Only trusted locally produced checkpoints are loaded; RNG tuples require weights_only=False.
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["state"]["lr"] != lr or saved["state"].get("adapter_lr") != adapter_lr:
            raise ValueError("Checkpoint learning rate mismatch")
        state = saved["state"]
        load_trained(model, saved["trained"])
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
        gpu.micro = min(gpu.micro, state["microbatch"])
    start, last_save = time.monotonic(), time.monotonic()

    def save():
        state["microbatch"] = gpu.micro
        atomic_torch(path, {"state": state, "trained": trained_state(model),
                            "optimizer": optimizer.state_dict(), "rng": rng_state()})
        budget.tick(force=True)
        atomic_json(directory / "progress.json", dict(state, remaining_seconds=max(0, budget.remaining())))

    def validation():
        if state.get("last_validation_updates") == state["updates"]:
            return
        deadline = start + candidate_seconds if candidate_seconds else None
        val = evaluate(model, data, "val", gpu, budget, deadline)
        score = val["raw"]["mae"]
        state["validation"] = val
        state["last_validation_updates"] = state["updates"]
        if state["best_mae"] is None or score < state["best_mae"]:
            state["best_mae"], state["stale"] = score, 0
            atomic_torch(directory / "best.pt", {"trained": trained_state(model), "validation": val,
                                                 "updates": state["updates"], "epoch": state["epoch"]})
        else:
            state["stale"] += 1
        record = {"epoch": state["epoch"], "updates": state["updates"], "validation_raw_mae": score,
                  "best_raw_mae": state["best_mae"], "microbatch": gpu.micro}
        with (directory / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(dict(record, path=str(directory))), flush=True)

    try:
        while state["epoch"] < 30 and state["stale"] < 5:
            gpu.check(budget)
            if max_updates is not None and state["updates"] >= max_updates:
                break
            if candidate_seconds and time.monotonic() - start >= max(0, candidate_seconds - 60):
                break
            # Independent per-epoch RNG gives identical sample order across layouts and LR candidates.
            generator = torch.Generator().manual_seed(42 + state["epoch"])
            order = torch.randperm(len(data.windows["train"]), generator=generator).numpy()
            if state["offset"] >= len(order):
                validation()
                state["epoch"] += 1
                state["offset"] = 0
                save()
                continue
            end = min(state["offset"] + 64, len(order))
            loss = update(model, optimizer, data, order[state["offset"]:end], gpu, budget, cache)
            state["offset"], state["updates"] = end, state["updates"] + 1
            if state["updates"] % 25 == 0:
                print(json.dumps({"path": str(directory), "epoch": state["epoch"],
                                  "updates": state["updates"], "loss": loss, "microbatch": gpu.micro,
                                  "remaining_seconds": max(0, budget.remaining())}), flush=True)
            if time.monotonic() - last_save >= 30:
                save()
                last_save = time.monotonic()
        # Candidate selection always uses a complete validation pass, never test.
        if budget.remaining() > 0:
            validation()
    except BudgetExhausted:
        pass
    finally:
        save()
    return state


def best(directory, model):
    saved = torch.load(Path(directory) / "best.pt", map_location="cpu", weights_only=True)
    load_trained(model, saved["trained"])
    return saved


def select_candidates(candidates):
    available = [(entry["state"]["best_mae"], i, entry) for i, entry in enumerate(candidates)
                 if entry["state"]["best_mae"] is not None]
    if not available:
        raise RuntimeError("No candidate completed a full validation pass within budget")
    return min(available, key=lambda item: (item[0], item[1]))[-1]


def stage(config_path, output, resume=False):
    from chronos.chronos_bolt import ChronosBoltModelForForecasting
    import chronos
    import transformers

    if chronos.__version__ != "2.3.2" or transformers.__version__ != "4.46.3":
        raise RuntimeError("Native adapter requires chronos 2.3.2 and transformers 4.46.3")
    config = stage_config(config_path)
    context_enabled = config.get("graph_context", {}).get("enabled", False)
    graph_options = {"graph_feature_dim": FEATURE_DIM} if context_enabled else {}
    spec = yaml.safe_load(Path("configs/benchmark.yaml").read_text())
    if config["seed"] != 42 or spec["seeds"] != [42]:
        raise ValueError("Only seed 42 is authorized for this stage")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=resume)
    manifest = {"config": config, "datasets": spec["datasets"],
                "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()}
    if context_enabled:
        manifest["context_inputs"] = context_inputs(config, spec)
    manifest_path = output / "manifest.json"
    if resume:
        old = json.loads(manifest_path.read_text())
        if old != manifest:
            raise ValueError("Resume requires the same immutable snapshot and full manifest")
    else:
        if subprocess.check_output(["git", "status", "--porcelain"]).strip():
            raise RuntimeError("Refusing to train from a dirty snapshot")
        atomic_json(manifest_path, manifest)
    preflight = json.loads(Path(config["mechanism_report"]).read_text())
    if not preflight["passed"]:
        raise RuntimeError("Native mechanisms have not passed")
    if preflight["config_hash"] != hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest():
        raise RuntimeError("Stage configuration changed since the native mechanism check")
    for name, expected in preflight["code_sha256"].items():
        if file_sha256(name) != expected:
            raise RuntimeError(f"Code changed since the native mechanism check: {name}")
    budget_state = json.loads((output / "budget.json").read_text()) if resume else {
        "spent_seconds": preflight["gpu_seconds"],
        "scopes": {"mechanism_and_setup": preflight["gpu_seconds"]}, "active_scope": None}
    budget = Budget(output / "budget.json", config["budget"]["total_gpu_seconds"], budget_state)
    gpu = GPU(config["resources"])
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    provenance = {"pid": os.getpid(), "torch": torch.__version__, "transformers": transformers.__version__,
                  "chronos": chronos.__version__, "gpu": torch.cuda.get_device_name(),
                  "memory_cap_bytes": gpu.cap, "tf32": True, "bolt_dtype": "bfloat16",
                  "hsdgnn_dtype": "float32", "graph_parameters_dtype": "float32",
                  "effective_batch": 64, "gradient_checkpointing": False,
                  "budget_accounting": "active process wall time; includes cache, validation, IO and checkpoints",
                  "input_pipeline": "prepared arrays resident on GPU; direct indexed windows; no DataLoader workers"}
    atomic_json(output / "environment.json", provenance)
    print(json.dumps(provenance), flush=True)
    weights = Path(config["models"]["native_bolt"]["local_weights"])
    weight_revision = config["models"]["native_bolt"]["revision"]
    if json.loads((weights / "provenance.json").read_text())["revision"] != weight_revision:
        raise ValueError("Wrong Bolt weight revision")
    for name, expected in config["models"]["native_bolt"]["file_sha256"].items():
        if file_sha256(weights / name) != expected:
            raise ValueError(f"Bolt weight hash mismatch: {name}")
    native = None
    status_path = Path(str(output) + ".status")

    def status(state, **fields):
        budget.tick(force=True)
        record = dict(state=state, pid=os.getpid(), spent_seconds=budget.state["spent_seconds"],
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(), **fields)
        atomic_json(output / "status.json", record)
        status_path.write_text(json.dumps(record) + "\n")

    try:
        budget.activate("mechanism_and_setup", config["budget"]["mechanism_seconds"])
        native = ChronosBoltModelForForecasting.from_pretrained(weights, local_files_only=True,
                                                               torch_dtype=torch.bfloat16).to(gpu.device).eval()
        for dataset in config["datasets"]:
            budget.activate("mechanism_and_setup", config["budget"]["mechanism_seconds"])
            directory = output / dataset
            directory.mkdir(exist_ok=True)
            if (directory / "done.json").exists():
                continue
            settings = dataset_settings(spec["datasets"][dataset], config, dataset)
            seed(42)
            data = Data(settings, gpu.device)
            graph, graph_info = build_graph(data.series, settings)
            atomic_json(directory / "preprocessing.json", {"data_sha256": data.series.fingerprint,
                        "graph": graph_info, "train_end": data.series.train_end, "val_end": data.series.val_end,
                        "length": len(data.series.values), "context": data.context, "horizon": data.horizon,
                        "windows": {name: len(windows) for name, windows in data.windows.items()},
                        "graph_context": data.context_provenance})
            np.savez(directory / "preprocessing.npz", mean=data.mean.cpu().numpy(), std=data.std.cpu().numpy(),
                     adjacency=graph)
            layouts = config["models"]["gofa_bolt"]["layout_candidates"]
            gofa_limit = config["budget"]["gofa_seconds_per_dataset"]
            budget.activate(dataset + "/gofa_bolt", gofa_limit)
            gofa_ready = directory / "gofa_ready.json"
            if not gofa_ready.exists():
                status("running", dataset=dataset, model="gofa_bolt", phase="cache")
                seed(42)
                cache_model = NativeBolt(native, graph, data.horizon, layouts["late_contiguous"], **graph_options).to(gpu.device)
                key = {"data_sha256": data.series.fingerprint, "revision": weight_revision,
                       "preprocessing_hash": hashlib.sha256(data.mean.cpu().numpy().tobytes() +
                                                             data.std.cpu().numpy().tobytes()).hexdigest(),
                       "context": data.context, "horizon": data.horizon, "split": "train",
                       "prefix_layers": 1, "dtype": "bfloat16", "native_eval": True,
                       "origins_hash": hashlib.sha256(np.asarray(data.windows["train"].origins, dtype=np.int64).tobytes()).hexdigest()}
                cache = PrefixCache.build(directory / "_cache", key, cache_model,
                                          data.batches("train", gpu.micro), len(data.windows["train"]),
                                          lambda: gpu.check(budget))
                del cache_model
                selection_path = directory / "gofa_selection.json"
                if not selection_path.exists():
                    candidates = []
                    for name, layers in layouts.items():
                        seed(42)
                        model = NativeBolt(native, graph, data.horizon, layers, **graph_options).to(gpu.device)
                        candidate = directory / "gofa_bolt" / name
                        status("running", dataset=dataset, model="gofa_bolt", phase=name)
                        state = fit(model, data, candidate, gpu, budget, .001, cache,
                                    config["models"]["gofa_bolt"]["comparison_optimizer_steps"],
                                    config["models"]["gofa_bolt"]["comparison_max_seconds_per_candidate"])
                        candidates.append({"name": name, "layers": layers, "state": state})
                        del model
                    selected = select_candidates(candidates)
                    atomic_json(selection_path, {"selected": selected, "candidates": candidates,
                                "selection_split": "validation", "metric": "raw_mae",
                                "equal_updates_completed": all(c["state"]["updates"] == 200 for c in candidates)})
                selected = json.loads(selection_path.read_text())["selected"]
                seed(42)
                gofa = NativeBolt(native, graph, data.horizon, selected["layers"], **graph_options).to(gpu.device)
                if selected["name"] == "late_contiguous" and budget.remaining() > 60:
                    cache.upgrade(gofa, gpu.micro, lambda: gpu.check(budget))
                gofa_path = directory / "gofa_bolt" / selected["name"]
                status("running", dataset=dataset, model="gofa_bolt", phase="selected_training")
                fit(gofa, data, gofa_path, gpu, budget, .001, cache)
                best(gofa_path, gofa)
                target_count = parameter_counts(gofa)
                atomic_json(gofa_ready, {"ready": True})
                cache.close(delete=True)
                del cache
            else:
                selected = json.loads((directory / "gofa_selection.json").read_text())["selected"]
                seed(42)
                gofa = NativeBolt(native, graph, data.horizon, selected["layers"], **graph_options).to(gpu.device)
                gofa_path = directory / "gofa_bolt" / selected["name"]
                best(gofa_path, gofa)
                target_count = parameter_counts(gofa)
            # Keep only the small graph checkpoint on CPU while training the GNN.
            gofa.cpu()
            native.to(gpu.device)  # gofa.cpu() also moved its shared frozen backbone.
            gpu.micro = config["resources"]["microbatch_candidates"][0]
            budget.activate(dataset + "/hsdgnn", config["budget"]["hsdgnn_seconds_per_dataset"])
            hsd_settings = config["models"]["matched_hsdgnn"]
            steps = {"metr_la": 288, "pems_bay": 288, "aqi": 24, "aqi_36": 24,
                     "electricity": 24, "solar": 144}[dataset]

            def new_hsd(dataset_data=data):
                seed(42)
                model = HSDGNNAdapter(hsd_settings["root"], dataset_data.context, dataset_data.horizon,
                                     dataset_data.series.values.shape[1], steps, hsd_settings["rnn_units"],
                                     hsd_settings["embed_dim"]).to(gpu.device)
                if dataset_data.context_enabled:
                    model.provenance["time_features"] = "source clock where available; sample-index cyclic fallback otherwise"
                return model

            hsd_selection = directory / "hsd_selection.json"
            hsd_ready = directory / "hsd_ready.json"
            if not hsd_ready.exists():
                if not hsd_selection.exists():
                    candidates = []
                    for lr in hsd_settings["learning_rates"]:
                        model = new_hsd()
                        counts = parameter_counts(model)
                        if abs(counts["total"] / target_count["total"] - 1) > hsd_settings["parameter_tolerance"]:
                            raise ValueError("Official HSDGNN is outside the registered parameter tolerance")
                        status("running", dataset=dataset, model="hsdgnn", phase=f"lr_{lr}")
                        state = fit(model, data, directory / "hsdgnn" / f"lr_{lr}", gpu, budget, lr,
                                    max_updates=200, candidate_seconds=300)
                        candidates.append({"lr": lr, "state": state, "parameters": counts})
                        del model
                        gc.collect()
                        torch.cuda.empty_cache()
                    atomic_json(hsd_selection, {"selected": select_candidates(candidates), "candidates": candidates,
                                               "selection_split": "validation", "metric": "raw_mae"})
                chosen_hsd = json.loads(hsd_selection.read_text())["selected"]
                hsd = new_hsd()
                hsd_path = directory / "hsdgnn" / f"lr_{chosen_hsd['lr']}"
                status("running", dataset=dataset, model="hsdgnn", phase="selected_training")
                fit(hsd, data, hsd_path, gpu, budget, chosen_hsd["lr"])
                best(hsd_path, hsd)
                atomic_json(hsd_ready, {"ready": True})
            else:
                chosen_hsd = json.loads(hsd_selection.read_text())["selected"]
                hsd = new_hsd()
                hsd_path = directory / "hsdgnn" / f"lr_{chosen_hsd['lr']}"
                best(hsd_path, hsd)
            budget.activate(dataset + "/evaluation", config["budget"]["final_evaluation_seconds"] / len(config["datasets"]))
            gofa.to(gpu.device)
            models = {"persistence": Persistence(data.horizon).to(gpu.device),
                      "native_bolt": NativeBolt(native, graph, data.horizon).to(gpu.device),
                      "gofa_bolt": gofa, "matched_hsdgnn": hsd}
            for name, model in models.items():
                result_path = directory / name / "result.json"
                if result_path.exists():
                    continue
                result_path.parent.mkdir(exist_ok=True)
                status("running", dataset=dataset, model=name, phase="final_evaluation")
                val = evaluate(model, data, "val", gpu, budget)
                test = evaluate(model, data, "test", gpu, budget)
                atomic_json(result_path, {"status": "complete", "dataset": dataset, "model": name,
                            "context": data.context, "horizon": data.horizon, "seed": 42,
                            "validation": val, "test": test, "parameters": parameter_counts(model),
                            "graph": graph_info if name == "gofa_bolt" else getattr(model, "provenance", None),
                            "graph_context": data.context_provenance if name in ("gofa_bolt", "matched_hsdgnn") else None,
                            "data_sha256": data.series.fingerprint, "git_revision": manifest["git_revision"],
                            "training_protocol": "time-bounded, at most 30 epochs, patience 5; native models untrained",
                            "scope_seconds": dict(budget.state["scopes"]),
                            "selection": selected if name == "gofa_bolt" else chosen_hsd if name == "matched_hsdgnn" else None})
            atomic_json(directory / "done.json", {"complete": True})
            del models, model, gofa, hsd, data
            gc.collect()
            torch.cuda.empty_cache()
        status("complete")
    except BudgetExhausted as error:
        status("budget_exhausted", scope=str(error))
    except StopRequested:
        status("stopped_by_user")
    except Exception as error:
        status("failed", error=repr(error))
        raise
    finally:
        budget.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/next_stage.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    stage(args.config, args.output, args.resume)


if __name__ == "__main__":
    main()
