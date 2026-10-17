"""Audited migration of an untrained, failed Stage18 GPU preflight only."""
import copy
import json
from pathlib import Path

import torch

from gofa_f.generative_data import sha
from gofa_f.idle_launcher import process_identity
from gofa_f.prefix_cache import atomic_json


def read(path):
    return json.loads(Path(path).read_text())


def require_exited(output):
    for suffix in (".bootstrap.launch.json", ".launch.json"):
        path = output.with_suffix(suffix)
        if path.exists():
            identity = read(path)
            if process_identity(identity["pid"]) == str(identity["start_ticks"]):
                raise ValueError("Source controller/supervisor is still alive")
    activity = read(output/"activity.json")
    if activity.get("worker") is not None:
        raise ValueError("Source worker scope is not cleared")
    return activity


def science(cfg):
    result = copy.deepcopy(cfg)
    result.pop("protocol")
    result["architecture"].pop("numerical_execution")
    return result


def migrate_failed_preflight(source, output, diagnostics=()):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError("A repair requires a new output")
    old, new = read(source/"manifest.json"), read(output/"manifest.json")
    failure = read(source/"failures.json")
    if failure.get("type") != "AssertionError" or failure.get("updates") != 0 \
            or failure.get("target_predictions") != 0 \
            or read(source/"status.json").get("state") != "failed" \
            or (source/"mechanisms.json").exists():
        raise ValueError("Only an untrained GPU mechanism failure can use this migration")
    if any((source/name).exists() for name in ("best.pt", "best_validation.json", "final_selection.json")) \
            or any((source/"evaluations").glob("*.json")) or any((source/"candidates").glob("*.pt")):
        raise ValueError("Evaluation/training outputs require a different recovery audit")
    if science(old["configuration"]) != science(new["configuration"]):
        raise ValueError("A preflight repair cannot change the scientific protocol")
    for key in ("weights_sha256", "source_data", "pools", "dependencies", "prompt_sha256", "decode"):
        if old[key] != new[key]:
            raise ValueError("Repair changed a fixed data/weight/decode binding: "+key)
    evidence = [(source, require_exited(source))]
    for path in diagnostics:
        path = Path(path).resolve()
        if path in [item[0] for item in evidence]:
            raise ValueError("Duplicate activity source")
        status = read(path/"status.json")
        if status.get("state") != "complete" or not status.get("diagnostic_only"):
            raise ValueError("Diagnostic is not complete or is a training run")
        evidence.append((path, require_exited(path)))
    saved = torch.load(source/"last.pt", map_location="cpu", weights_only=False)
    state = saved["state"]
    if saved["binding"] != sha(source/"manifest.json") or saved["parameter_version"] != 0 \
            or any(state[k] != 0 for k in ("updates", "pending", "pending_tokens", "pending_loss",
                                          "last_validated_update", "bad_validations")) \
            or state["best_score"] is not None or state["early_reference"] is not None \
            or saved["optimizer"]["state"] or any(g is not None for g in saved["gradients"].values()):
        raise ValueError("Checkpoint is not the exact untrained preflight state")
    receipt = {"source": str(source), "source_manifest_sha256": sha(source/"manifest.json"),
        "source_checkpoint_sha256": sha(source/"last.pt"), "new_manifest_sha256": sha(output/"manifest.json"),
        "source_code": old["code_revision"], "new_code": new["code_revision"], "formal_updates": 0,
        "state_restore": "exact_except_manifest_rebinding", "no_predictions_or_selection_imported": True,
        "reason": "shape_stable_rmsnorm_full_incremental_repair",
        "original_started_at": read(source.with_suffix(".bootstrap.launch.json")).get("started_at"),
        "activity_sources": [{"output": str(path), "activity_sha256": sha(path/"activity.json"),
                              **activity} for path, activity in evidence]}
    marker = output/"migration.json"
    if marker.exists():
        previous = read(marker)
        if any(previous[k] != value for k, value in receipt.items()):
            raise ValueError("Existing migration binding changed")
        return previous  # never overwrite a subsequently advanced checkpoint or ledger
    if (output/"last.pt").exists() or (output/"activity.json").exists():
        raise ValueError("Destination has existing training/activity state")
    saved["binding"] = receipt["new_manifest_sha256"]
    temporary = output/"last.tmp"
    torch.save(saved, temporary)
    temporary.replace(output/"last.pt")
    receipt["migrated_checkpoint_sha256"] = sha(output/"last.pt")
    ledger = {"worker": None,
        "spent_seconds": sum(item["spent_seconds"] for _, item in evidence),
        "waiting_seconds": sum(item.get("waiting_seconds", 0.) for _, item in evidence),
        "resource_pause_count": sum(item.get("resource_pause_count", 0) for _, item in evidence),
        "peak_group_mib": max(item.get("peak_group_mib", 0) for _, item in evidence),
        "inherited_activity": receipt["activity_sources"]}
    atomic_json(output/"activity.json", ledger)
    atomic_json(marker, receipt)
    return receipt
