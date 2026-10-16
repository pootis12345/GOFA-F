"""Audited same-experiment continuation into a new immutable implementation."""
import hashlib
import json
from pathlib import Path
import shutil

import torch

from gofa_f.chattime_stage import digest, read
from gofa_f.idle_launcher import process_identity
from gofa_f.prefix_cache import atomic_json


def source_record(previous, config, prepared, identity=process_identity):
    previous = Path(previous).resolve()
    ledger = read(previous/"active_budget.json", {})
    budget = read(previous/"budget.json", {})
    if read(previous/"launcher.json", {}).get("state") != "stopped_by_user" \
            or ledger.get("worker") is not None or budget.get("active_scope") is not None:
        raise ValueError("Continuation requires a stopped and fully saved source")
    for suffix in (".launch.json", ".bootstrap.launch.json"):
        record = read(previous.parent/(previous.name+suffix), {})
        actual = identity(record.get("pid")) if record.get("pid") else None
        if not record.get("pid") or record.get("start_ticks") is None \
                or (actual is not None and str(actual) == str(record["start_ticks"])):
            raise ValueError("Source process identity is missing or still alive")
    original = read(previous/"manifest.json", {})
    if original.get("config") != config:
        raise ValueError("Continuation cannot change the scientific configuration")
    old_inputs = original.get("inputs", {})
    if not old_inputs or old_inputs != {p: digest(p) for p in old_inputs}:
        raise ValueError("Source immutable inputs changed")
    def normalize(inputs, folder):
        return {("local_graph" if Path(k).resolve() == folder/"correlation_graph.npz" else k): v
                for k, v in inputs.items()}
    if normalize(old_inputs, previous) != normalize(prepared["inputs"], Path(prepared["output"]).resolve()):
        raise ValueError("Data, graph, clock or weights changed")
    for name in ("gofa_chattime", "native_chattime"):
        directory = previous/name
        if (directory/"result.json").exists() or any((directory/"predictions").rglob("*.npz")):
            raise ValueError("Cannot resume adaptive training after prediction evaluation")
    window = read(previous/"window.json")
    if not window or original.get("window") != window or window.get("admission_policy", "idle") not in ("idle", "shared_memory"):
        raise ValueError("Source window/policy differs")
    source_binding = hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
    path = previous/"gofa_chattime/last.pt"
    saved = torch.load(path, map_location="cpu", weights_only=False)
    progress = read(previous/"gofa_chattime/progress.json", {})
    if saved["binding"] != source_binding or progress.get("checkpoint_sha256") != digest(path):
        raise ValueError("Source final checkpoint binding/hash differs")
    if saved["state"].get("stop_reason") != "training_deadline_or_signal":
        raise ValueError("Source is not an interrupted training checkpoint")
    return {"kind": "training_continuation", "output": str(previous), "source_binding": source_binding,
        "window_sha256": digest(previous/"window.json"), "manifest_sha256": digest(previous/"manifest.json"),
        "checkpoint_sha256": digest(path), "started_at": window["started_at"],
        "original_wall_deadline": window["wall_deadline"], "gpu_activity_seconds": ledger["spent_seconds"],
        "updates": saved["state"]["updates"], "offset": saved["state"]["offset"],
        "pending_windows": saved["state"]["pending_windows"], "pending_count": saved["state"]["pending_count"]}


def copy_continuation(record, output, target_binding):
    output = Path(output)
    marker = output/"migration.json"
    old = read(marker)
    if old:
        if old["target_binding"] != target_binding or old["source"] != record:
            raise ValueError("Existing continuation identity changed")
        return old
    previous = Path(record["output"])
    source = previous/"gofa_chattime"
    target = output/"gofa_chattime"
    target.mkdir(parents=True, exist_ok=True)
    path = source/"last.pt"
    if digest(path) != record["checkpoint_sha256"]:
        raise ValueError("Source changed during continuation")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["binding"] != record["source_binding"]:
        raise ValueError("Source checkpoint identity changed")
    saved["binding"] = target_binding
    from gofa_f.native_stage import atomic_torch
    from gofa_f.chattime_checks import assert_tree_equal
    atomic_torch(target/"last.pt", saved)
    restored = torch.load(target/"last.pt", map_location="cpu", weights_only=False)
    # Model, optimizer, RNG, pending gradients/counters and cursor restore bitwise.
    assert_tree_equal(saved, restored)
    progress = read(source/"progress.json")
    progress["checkpoint_sha256"] = digest(target/"last.pt")
    atomic_json(target/"progress.json", progress)
    for name in ("history.json",):
        if (source/name).exists():
            shutil.copyfile(source/name, target/name)
    if (source/"validations").exists():
        shutil.copytree(source/"validations", target/"validations", dirs_exist_ok=True)
    if (source/"best.pt").exists():
        best = torch.load(source/"best.pt", map_location="cpu", weights_only=False)
        if best["binding"] != record["source_binding"]:
            raise ValueError("Source best checkpoint binding differs")
        best["binding"] = target_binding
        atomic_torch(target/"best.pt", best)
        metadata = read(source/"best_validation.json")
        metadata["checkpoint_sha256"] = digest(target/"best.pt")
        atomic_json(target/"best_validation.json", metadata)
    for name in ("budget.json", "cuda_memory.json", "parameters.json"):
        if (previous/name).exists():
            shutil.copyfile(previous/name, output/name)
    ledger = read(previous/"active_budget.json")
    if ledger["worker"] is not None:
        raise ValueError("Source GPU worker must have exited")
    atomic_json(output/"active_budget.json", ledger)
    result = {"source": record, "target_binding": target_binding,
              "target_checkpoint_sha256": digest(target/"last.pt"), "state_restore": "exact",
              "cache": "ephemeral caches rebuilt; old caches not modified", "original_deadline_retained": True}
    atomic_json(marker, result)
    return result
