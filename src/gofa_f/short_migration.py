"""Exact stopped or failed-acceptance continuation; never accepts a live source."""
import copy
from pathlib import Path

import torch

from gofa_f.allnode_stage import read
from gofa_f.generative_data import sha
from gofa_f.idle_launcher import process_identity
from gofa_f.prefix_cache import atomic_json


INFERENCE_FILES = ("allnode_model.py", "allnode_codec.py", "allnode_graph.py", "allnode_kernel.py",
    "generative_model.py", "generative_kernel.py", "generative_graph.py", "generative_data.py",
    "chattime.py", "short_codec.py", "short_forecast.py", "short_data.py")


def assert_equal(a, b):
    if torch.is_tensor(a):
        if not torch.is_tensor(b) or a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            raise ValueError("Migration tensor changed")
    elif isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            raise ValueError("Migration dictionary changed")
        for k in a:
            assert_equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            raise ValueError("Migration sequence changed")
        for x, y in zip(a, b):
            assert_equal(x, y)
    elif hasattr(a, "dtype"):
        import numpy as np
        if not np.array_equal(a, b):
            raise ValueError("Migration RNG array changed")
    elif a != b:
        raise ValueError("Migration scalar changed")


def reusable_cpu_proof(cfg, output):
    """Reuse FP32 only when every tensor execution module and input is unchanged."""
    output = Path(output)
    migration = read(output/"migration.json", {})
    if not migration.get("failed_acceptance_repair"):
        return None
    source = Path(cfg["continuation"]["source_output"])
    before, after = read(source/"manifest.json"), read(output/"manifest.json")
    old_cfg, new_cfg = copy.deepcopy(before["configuration"]), copy.deepcopy(cfg)
    for value in (old_cfg, new_cfg):
        value.pop("protocol", None)
        value.pop("continuation", None)
    if old_cfg != new_cfg or set(before["code"]) != set(after["code"]):
        return None
    diagnostic_only = {"short_migration.py", "short_checks.py", "short_training_checks.py"}
    unchanged = set(before["code"])-diagnostic_only
    if "short_training.py" not in unchanged or any(before["code"][n] != after["code"][n] for n in unchanged):
        return None
    proof = read(source/"cpu_preflight.json", {})
    if not proof.get("passed") or proof.get("manifest_sha256") != sha(source/"manifest.json") \
            or not proof.get("real_weights", {}).get("passed") \
            or not proof.get("continued_training_optimizations", {}).get("passed"):
        return None
    for key in ("weights_sha256", "raw_hashes", "query_sha256", "epsilon_sha256", "dependencies"):
        if before[key] != after[key]:
            return None
    record = {"source_proof": str(source/"cpu_preflight.json"),
        "source_proof_sha256": sha(source/"cpu_preflight.json"),
        "unchanged_tensor_code": {n: after["code"][n] for n in sorted(unchanged)},
        "reason": "only acceptance value upcast/diagnostics and exact migration changed; FP32 execution unchanged",
        "current_equivalent_checkpoint_sha256": sha(output/"last.pt"),
        "gpu_acceptance_reused": False}
    result = {k: copy.deepcopy(proof[k]) for k in ("alphabet", "real_weights", "continued_training_optimizations")}
    result["cpu_proof_reuse"] = record
    return result


def projection_diagnostic(cfg):
    """Retain the rejected compact-tail diagnostic and all of its resource costs."""
    path = cfg.get("continuation", {}).get("prior_projection_diagnostic")
    if not path:
        return None
    path = Path(path)
    launch = read(path.with_suffix(".launch.json"))
    if process_identity(launch["pid"]) == str(launch["start_ticks"]):
        raise ValueError("Projection diagnostic supervisor is still alive")
    ledger, launcher, state = (read(path/name) for name in ("activity.json", "launcher.json", "status.json"))
    report, policy = read(path/"projection_probe.json"), read(path/"launcher_manifest.json")
    if ledger.get("worker") is not None or launcher["state"] != "complete" or state["state"] != "complete" \
            or report["candidate_passed"] is not False or report["tolerance"] != .002:
        raise ValueError("Stopped diagnostic rejecting the compact-tail candidate required")
    required = {"worker_mode": "projection_probe", "admission_policy": "idle",
        "resource_check_seconds": 10, "admission_samples": 3, "memory_cap_gib": 40, "torch_cap_gib": 38}
    if any(policy.get(k) != v for k, v in required.items()) or \
            report["finite_tokens"] != 10000 or report["native_vocabulary"] != 42001 \
            or report["candidate"] != "finite_plus_one_zero_work_row":
        raise ValueError("Projection diagnostic policy or output layout differs")
    root = Path(cfg["weights"]["directory"])
    index = read(root/"model.safetensors.index.json")
    shard = index["weight_map"]["lm_head.weight"]
    source = Path(cfg["continuation"]["source_output"])
    if report["weight_shard_sha256"] != read(root/"snapshot.json")["files"][shard]["sha256"] or \
            report["alphabet_sha256"] != read(source/"cpu_preflight.json")["alphabet"]["alphabet_sha256"] or \
            read(path/"gpu_identity.json")["uuid"] != read(source/"gpu_identity.json")["uuid"]:
        raise ValueError("Projection probe head, alphabet or GPU binding changed")
    if set(report["cases"]) != {"1", "25", "48"} or not any(
            item["violations"] for case in report["cases"].values() for item in case["tail_one_10001"].values()):
        raise ValueError("Projection probe rejection has no measured discrepancy")
    return {"output": str(path), "report_sha256": sha(path/"projection_probe.json"),
        "activity_sha256": sha(path/"activity.json"), "activity": ledger, "source_revision": launch["code_revision"],
        "candidate_rejected": True, "selected_fix": "observed_positions_native_head_width"}


def migrate(cfg, output):
    output = Path(output)
    source = Path(cfg["continuation"]["source_output"]).resolve()
    expected = cfg["continuation"]["checkpoint_sha256"]
    if source == output.resolve() or sha(source/"last.pt") != expected:
        raise ValueError("Stopped source checkpoint SHA differs")
    for suffix in (".bootstrap.launch.json", ".launch.json"):
        identity = read(Path(str(source)+suffix), {})
        if identity.get("pid") and process_identity(identity["pid"]) == str(identity["start_ticks"]):
            raise ValueError("Source controller is still alive")
    ledger = read(source/"activity.json")
    launcher, status = read(source/"launcher.json"), read(source/"status.json")
    stopped = launcher["state"] == "stopped_by_user" and status["state"] == "resource_paused"
    failed = launcher["state"] == status["state"] == "failed" and \
        read(source/"failures.json", {}).get("phase") == "gpu_mechanisms"
    if ledger.get("worker") is not None or not (stopped or failed):
        raise ValueError("Source did not safely stop or fail only acceptance")
    existing = read(output/"migration.json")
    if existing is not None:
        if existing["source_checkpoint_sha256"] != expected or \
                existing["target_checkpoint_sha256"] != sha(output/"last.pt"):
            raise ValueError("Do not migrate over an already progressed destination")
        return existing
    diagnostic = projection_diagnostic(cfg)
    if diagnostic:
        ledger = copy.deepcopy(ledger)
        for key in ("spent_seconds", "waiting_seconds", "resource_pause_count"):
            ledger[key] = ledger.get(key, 0)+diagnostic["activity"].get(key, 0)
        ledger["peak_group_mib"] = max(ledger.get("peak_group_mib", 0),
                                      diagnostic["activity"].get("peak_group_mib", 0))
    if (source/"final_selection.json").exists() or (source/"best_validation.json").exists():
        raise ValueError("This migration requires the current pre-first-validation training state")
    before, after = read(source/"manifest.json"), read(output/"manifest.json")
    converted = copy.deepcopy(before["configuration"])
    converted["protocol"] = cfg["protocol"]
    converted["training"]["microbatch"] = 2
    converted["optimizations"] = cfg["optimizations"]
    converted["continuation"] = cfg["continuation"]
    if converted != cfg:
        raise ValueError("Migration changed the scientific protocol beyond two training optimizations")
    for key in ("weights_sha256", "raw_hashes", "query_sha256", "epsilon_sha256",
                "source_range_audit_sha256", "source_target_audit_sha256", "dependencies"):
        if before[key] != after[key]:
            raise ValueError("Migration data/weight/dependency binding changed: "+key)
    for name in INFERENCE_FILES:
        if before["code"][name] != after["code"][name]:
            raise ValueError("Cannot import baseline after inference code changed: "+name)
    old_binding, new_binding = sha(source/"manifest.json"), sha(output/"manifest.json")
    payload = torch.load(source/"last.pt", map_location="cpu", weights_only=False)
    if payload["binding"] != old_binding or payload["state"]["last_validated_update"] != 0 \
            or not 0 < payload["state"]["updates"] < 500 or not 0 <= payload["state"]["pending"] <= 64:
        raise ValueError("Unexpected trained source state")
    if failed:
        ancestor = read(source/"migration.json")
        if not ancestor or not read(source/"cpu_preflight.json", {}).get("passed"):
            raise ValueError("Failed acceptance source lacks validated migration")
        ancestor_path = Path(ancestor["source"])/"last.pt"
        if sha(ancestor_path) != ancestor["source_checkpoint_sha256"]:
            raise ValueError("Ancestor checkpoint changed")
        reference = torch.load(ancestor_path, map_location="cpu", weights_only=False)
        for key in ("adapter", "optimizer", "gradients", "rng", "parameter_version"):
            assert_equal(payload[key], reference[key])
        for key in ("updates", "pending", "pending_tokens", "pending_loss", "best_score",
                    "early_reference", "bad_validations", "last_validated_update"):
            assert_equal(payload["state"][key], reference["state"][key])
        if any((source/"evaluations").glob("update-*")) or any((source/"evaluations").glob("target-*")):
            raise ValueError("Failed preflight migration cannot contain new evaluations")
    original = copy.deepcopy(payload)
    payload["binding"], payload["state"]["microbatch"] = new_binding, 2
    payload["progress"]["throughput_anchor_updates"] = payload["progress"]["measured_updates"]
    payload["progress"]["throughput_anchor_seconds"] = payload["progress"]["train_seconds"]
    temporary = output/"last.tmp"
    torch.save(payload, temporary)
    restored = torch.load(temporary, map_location="cpu", weights_only=False)
    assert_equal(payload, restored)
    reference = copy.deepcopy(original)
    reference["binding"], reference["state"]["microbatch"] = new_binding, 2
    reference["progress"] = payload["progress"]
    assert_equal(reference, restored)
    temporary.replace(output/"last.pt")
    baseline = source/"evaluations/native"
    complete = read(baseline/"complete.json")
    alphabet = read(source/"cpu_preflight.json")["alphabet"]["alphabet_sha256"]
    old_eval = old_binding+":native:"+alphabet
    new_eval = new_binding+":native:"+alphabet
    queries = read(output/"pools/queries.json")["validation"]
    if not complete["complete"] or complete["queries"] != len(queries) or complete["binding"] != old_eval:
        raise ValueError("Complete native baseline required")
    destination = output/"evaluations/native"
    destination.mkdir(parents=True, exist_ok=True)
    for i, query in enumerate(queries):
        item = read(baseline/f"{i:06d}.json")
        if item["query"] != query or item["binding"] != old_eval or item["mae"] != complete["errors"][i]:
            raise ValueError("Source baseline shard binding changed")
        item["binding"] = new_eval
        atomic_json(destination/f"{i:06d}.json", item)
    complete["binding"] = new_eval
    atomic_json(destination/"complete.json", complete)
    atomic_json(output/"activity.json", ledger)
    for name in ("gpu_identity.json",):
        atomic_json(output/name, read(source/name))
    record = {"source": str(source), "source_checkpoint_sha256": expected,
        "source_manifest_sha256": old_binding, "target_manifest_sha256": new_binding,
        "target_checkpoint_sha256": sha(output/"last.pt"), "state_restore": "exact",
        "operational_changes": ["microbatch1_to2", "compact_training_projection", "post_migration_eta_anchor"]+
            (["observed_positions_native_projection_width", "isolated_projection_diagnostic_costs"] if diagnostic else []),
        "updates": payload["state"]["updates"], "pending": payload["state"]["pending"],
        "pending_tokens": payload["state"]["pending_tokens"],
        "source_baseline_sha256": sha(baseline/"complete.json"), "baseline_queries_preserved": len(queries),
        "baseline_predictions_changed": False, "inference_code_sha_unchanged": list(INFERENCE_FILES),
        "activity_retained": ledger, "old_processes_exited": True,
        "failed_acceptance_repair": failed, "prior_projection_diagnostic": diagnostic,
        "source_migration": read(source/"migration.json") if failed else None}
    atomic_json(output/"migration.json", record)
    return record
