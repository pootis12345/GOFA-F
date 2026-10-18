"""Stage19: fresh short-source training, sealed checkpoint, then nine zero-shot targets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import numpy as np
import torch
import yaml

from gofa_f.allnode_kernel import EXECUTION
from gofa_f.allnode_model import frozen_cache_binding
from gofa_f.allnode_stage import (Paused, apply_update, read, rng_state, restore_rng, save_checkpoint,
    restore_checkpoint, schedule, supervisor_alive, verify_weights, load_model as base_load_model)
from gofa_f.generative_data import SOURCES, Series, sha
from gofa_f.generative_model import GenerativePrefixLRU
from gofa_f.prefix_cache import atomic_json
from gofa_f.short_codec import ShortCodec
from gofa_f.short_data import TEMPLATES, TARGETS, build_queries, relative_score
from gofa_f.short_forecast import evaluate
from gofa_f.short_progress import Progress
from gofa_f.short_training import training_batch


def config_check(cfg):
    a, t, r = cfg["architecture"], cfg["training"], cfg["resources"]
    if (a["graph_layers"], a["prefix_layers"], a["injection_side"], a["numerical_execution"]) != \
            ([30, 31], 30, "before", EXECUTION):
        raise ValueError("Fixed Stage19 interleaving/precision boundary changed")
    fixed = {"graph_width": 64, "attention_rank": 16, "prior_fraction": .5,
             "graph_top_k": 8, "max_subgraph_nodes": 9}
    if any(a[k] != v for k, v in fixed.items()):
        raise ValueError("Fixed graph architecture changed")
    fixed = {"max_updates": 4000, "effective_batch": 64, "microbatch": 2,
        "learning_rate": 2e-4, "weight_decay": 1e-4, "gradient_clip": 1., "warmup_updates": 100,
        "validation_every": 500, "patience": 5, "minimum_updates": 1000, "relative_min_delta": .001}
    if any(t[k] != v for k, v in fixed.items()):
        raise ValueError("Fixed optimization protocol changed")
    if cfg.get("optimizations") != {"training_logits": "observed_numeric_slots_finite_vocabulary_v1",
                                    "teacher_attention": "isolated_rows_original_capacity_v1"}:
        raise ValueError("Training-only optimization binding changed")
    if tuple(map(tuple, cfg["data"]["templates"])) != TEMPLATES or cfg["seed"] != 42 \
            or cfg["data"]["sources"] != list(SOURCES) or cfg["data"]["target_training_enabled"] \
            or cfg["data"]["target_evaluation_after_lock"] is not True:
        raise ValueError("Fixed short-source-only training changed")
    if cfg["sampling"] != {"source_train_queries": 8192, "source_origins_per_variable_template": 1,
                           "target_queries_minimum": 2048, "target_origins_per_variable_minimum": 1}:
        raise ValueError("User-approved original query budget changed")
    if cfg["decode"] != {"maximum_numeric_slots": 24, "block_size": 24,
        "grammar": "finite_numeric_tokens_with_canonical_delimiters", "scale": "immutable_origin",
        "missing_teacher_inputs": "causal_forward_fill", "loss": "observed_numeric_slots_conditional_cross_entropy"}:
        raise ValueError("Fixed numeric grammar/loss changed")
    fixed = {"admission_policy": "idle", "resource_check_seconds": 10, "admission_samples": 3,
        "memory_cap_gib": 40, "torch_cap_gib": 38, "yield_on_foreign": False, "leave_free_gib": 0,
        "activity_limit_seconds": None, "wall_deadline": None}
    if any(r[k] != v for k, v in fixed.items()):
        raise ValueError("Idle admission or own-memory-only runtime cap changed")


def load_model(cfg, device, dtype=torch.bfloat16):
    model, old_codec = base_load_model(cfg, device, dtype)
    return model, ShortCodec(old_codec.tokenizer)


def prepare(cfg, output):
    config_check(cfg)
    root, output = Path(cfg["data"]["directory"]), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    series = {name: Series(root/"sources"/(name+".npz")) for name in SOURCES}
    for name, digest in read(root/"pools.json")["hashes"].items():
        if sha(root/name) != digest:
            raise ValueError("Previously audited source query/epsilon binding changed")
    raw_hashes = {}
    for name, data in series.items():
        raw_hashes[name] = sha(data.path)
        if raw_hashes[name] != data.meta["sha256"]:
            raise ValueError("Source observations changed")
    for name in TARGETS:
        metadata = read(root/"targets"/(name+".json"))
        digest = sha(root/"targets"/(name+".npz"))
        if digest != metadata["sha256"]:
            raise ValueError("Target observations changed")
        raw_hashes["target:"+name] = digest
    pools = build_queries(cfg, output, series)
    repository = Path(__file__).resolve().parents[2]
    revision = repository/"CODE_REVISION"
    manifest = {"protocol": cfg["protocol"], "configuration": cfg, "code_revision":
        revision.read_text().strip() if revision.exists() else subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
        "code": {p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        "weights_sha256": verify_weights(cfg), "raw_hashes": raw_hashes,
        "query_sha256": sha(output/"pools/queries.json"),
        "epsilon_sha256": sha(output/"pools/metric_epsilons.json"),
        "source_range_audit_sha256": sha(output/"pools/source_range_audit.json"),
        "source_target_audit_sha256": sha(repository/"docs/results/stage17-target-protocol-audit.json"),
        "source_validation_queries": len(pools["validation"]),
        "target_queries": {k: len(v) for k, v in pools["targets"].items()},
        "dependencies": {"torch": torch.__version__, "transformers": __import__("transformers").__version__,
                         "numpy": np.__version__},
        "old_adapter_imported": False, "baseline_imported": bool(cfg.get("continuation")),
        "continuation_source": cfg.get("continuation"),
        "target_training_enabled": False, "target_evaluation_requires_sealed_source_checkpoint": True,
        "stop_rule": "4000_updates_or_complete_source_patience_or_user_or_unrecoverable_failure"}
    if read(output/"manifest.json", manifest) != manifest:
        raise ValueError("Immutable Stage19 manifest changed")
    atomic_json(output/"manifest.json", manifest)
    if not (output/"progress.json").exists():
        Progress(output).tick("prepared", force=True)
    return manifest


def record_validation(state, score, update):
    if not np.isfinite(score) or score <= 0 or update <= 0 or update % 500:
        raise ValueError("Complete finite source validation at a 500-update boundary required")
    if state["last_validated_update"] == update:
        return False
    if state["last_validated_update"] > update:
        raise ValueError("Validation order reversed")
    best = state["best_score"] is None or score < state["best_score"]
    if best:
        state["best_score"] = score
    if state["early_reference"] is None or score <= state["early_reference"]*.999:
        state["early_reference"], state["bad_validations"] = score, 0
    else:
        state["bad_validations"] += 1
    state["last_validated_update"] = update
    return best


def run(cfg, output, resume):
    if os.environ.get("GOFA_STAGE19_ADMITTED") != "1":
        raise RuntimeError("Stage19 GPU work requires its CPU idle-admission supervisor")
    config_check(cfg)
    manifest = read(output/"manifest.json")
    binding = sha(output/"manifest.json")
    if not read(output/"cpu_preflight.json", {}).get("passed") or \
            read(output/"cpu_preflight.json")["manifest_sha256"] != binding or \
            manifest["configuration"] != cfg or manifest["code"] != \
            {p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))}:
        raise ValueError("Successful code/config-bound CPU proof required")
    torch.cuda.set_per_process_memory_fraction(38*2**30/torch.cuda.get_device_properties(0).total_memory)
    device = torch.device("cuda:0")
    model, codec = load_model(cfg, device)
    if codec.alphabet_sha != read(output/"cpu_preflight.json")["alphabet"]["alphabet_sha256"]:
        raise ValueError("CPU/GPU numeric alphabet or grammar differs")
    root = Path(cfg["data"]["directory"])
    series = {name: Series(root/"sources"/(name+".npz")) for name in SOURCES}
    pools = read(output/"pools/queries.json")
    optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=2e-4, weight_decay=1e-4)
    initial_adapter = model.trainable_state()
    state = {"updates": 0, "pending": 0, "pending_tokens": 0, "pending_loss": 0., "microbatch": 2,
        "best_score": None, "early_reference": None, "bad_validations": 0, "last_validated_update": 0,
        "peak_memory": {}, "stop_reason": None}
    saved_progress = None
    if resume and (output/"last.pt").exists():
        state, saved_progress = restore_checkpoint(output/"last.pt", model, optimizer, binding)
        saved = torch.load(output/"last.pt", map_location="cpu", weights_only=False)
        model.parameter_version = saved["parameter_version"]
        del saved
    elif (output/"last.pt").exists():
        raise ValueError("Existing training state requires --resume")
    progress = Progress(output, saved_progress)
    stopped = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    def check():
        if stopped[0]:
            raise Paused("cooperative_resource_or_user_stop")
        if not supervisor_alive(os.environ["GOFA_STAGE19_SUPERVISOR_PID"], os.environ["GOFA_STAGE19_SUPERVISOR_TICKS"]):
            raise Paused("verified_supervisor_exited")
    def status(kind, **extra):
        atomic_json(output/"status.json", {"state": kind, "updates": state["updates"],
            "pending_queries": state["pending"], "updated_at": time.time(), **extra})
    def memory(phase):
        item = state["peak_memory"].setdefault(phase, {"allocated_bytes": 0, "reserved_bytes": 0})
        item["allocated_bytes"] = max(item["allocated_bytes"], torch.cuda.max_memory_allocated())
        item["reserved_bytes"] = max(item["reserved_bytes"], torch.cuda.max_memory_reserved())
        atomic_json(output/"cuda_memory.json", state["peak_memory"])
        state["cache_totals"] = {"hits": cache_base["hits"]+cache.hits,
            "misses": cache_base["misses"]+cache.misses,
            "peak_bytes": max(cache_base["peak_bytes"], cache.peak)}
        record = read(output/"cache.json", {})
        record.update(state["cache_totals"], current_bytes=cache.used,
                      hit_fraction=(state["cache_totals"]["hits"]/max(1, state["cache_totals"]["hits"]+state["cache_totals"]["misses"])))
        atomic_json(output/"cache.json", record)
    cache = GenerativePrefixLRU(frozen_cache_binding(model, manifest["weights_sha256"]), max_bytes=32*2**30)
    cache_base = dict(state.get("cache_totals", {"hits": 0, "misses": 0, "peak_bytes": 0}))
    atomic_json(output/"cache.json", {"prefix_layers": 30, "ram_limit_gib": 32,
        "disk_limit_gib": 128, "disk_cache_enabled": False,
        "conservative_complete_bytes": 32768*(9*160+200)*4096*2,
        "graph_kv_cross_update": False, "role_and_input_binding": True})
    try:
        if not read(output/"mechanisms.json", {}).get("passed"):
            from gofa_f.short_checks import mechanisms
            status("running", phase="gpu_mechanisms")
            progress.tick("gpu_mechanisms", force=True)
            continued_adapter, continued_version = model.trainable_state(), model.parameter_version
            model.load_trainable(initial_adapter)
            model.parameter_version = 0
            try:
                proof = mechanisms(model, codec, series, pools["validation"], 2e-3, check=check,
                    diagnostic=lambda v: atomic_json(output/"mechanism_diagnostics.json", v))
            finally:
                model.load_trainable(continued_adapter)
                model.parameter_version = continued_version
            from gofa_f.short_training_checks import optimization_checks
            proof["training_optimizations"] = optimization_checks(model, codec, series,
                pools["validation"], 2e-3, check=check,
                checkpoint=output/"last.pt" if (output/"last.pt").exists() else None,
                diagnostic=lambda v: atomic_json(output/"optimization_diagnostics.json", v))
            atomic_json(output/"mechanisms.json", proof)
            memory("mechanisms")
            torch.cuda.reset_peak_memory_stats()
        atomic_json(output/"parameters.json", model.parameter_counts())
        if (output/"final_selection.json").exists():
            sealed = read(output/"final_selection.json")
            if sealed["manifest_sha256"] != binding or sha(output/"best.pt") != sealed["sha256"]:
                raise ValueError("Sealed checkpoint changed")
            model.load_trainable(torch.load(output/"best.pt", map_location="cpu", weights_only=False))
            for name in TARGETS:
                check()
                data = {name: Series(root/"targets"/(name+".npz"))}
                for native in (True, False):
                    label = f"target-{name}-{'native' if native else 'gofa'}"
                    status("running", phase=label)
                    progress.tick("target_evaluation", force=True)
                    result = evaluate(model, codec, data, pools["targets"][name], output, label,
                        binding+":"+sealed["sha256"]+":"+label, progress, check, native)
                    memory(label)
                    atomic_json(output/(label+"-result.json"), result)
            status("complete", stop_reason=sealed["stop_reason"], target_datasets_complete=9)
            progress.tick("complete", force=True)
            report(output)
            return
        status("running", phase="native_source_baseline")
        progress.tick("native_source_baseline", force=True)
        baseline = evaluate(model, codec, series, pools["validation"], output, "native",
            binding+":native:"+codec.alphabet_sha, progress, check, native=True)
        memory("native_source_baseline")
        eps = read(output/"pools/metric_epsilons.json")
        last_save = time.monotonic()
        while True:
            check()
            if state["updates"] and state["updates"] % 500 == 0 and state["last_validated_update"] != state["updates"]:
                status("running", phase="source_validation")
                progress.tick("source_validation", force=True)
                candidate = output/"candidates"/f"update-{state['updates']:06d}.pt"
                candidate.parent.mkdir(exist_ok=True)
                if not candidate.exists():
                    torch.save(model.trainable_state(), candidate)
                digest = sha(candidate)
                result = evaluate(model, codec, series, pools["validation"], output,
                    f"update-{state['updates']:06d}", binding+":"+digest, progress, check)
                scores = relative_score(result["errors"], baseline["errors"], pools["validation"], eps, 1)
                record = {"updates": state["updates"], "checkpoint_sha256": digest,
                          "checkpoint": str(candidate), **scores}
                atomic_json(output/f"validation-{state['updates']:06d}.json", record)
                if record_validation(state, scores["score"], state["updates"]):
                    shutil.copyfile(candidate, output/"best.pt")
                    atomic_json(output/"best_validation.json", record)
                progress.state["validation_runs"] += 1
                memory("source_validation")
                save_checkpoint(output/"last.pt", model, optimizer, state, progress, binding)
            if state["updates"] >= 4000 or (state["updates"] >= 1000 and state["bad_validations"] >= 5):
                state["stop_reason"] = "max_updates" if state["updates"] >= 4000 else "source_patience"
                break
            status("running", phase="train")
            progress.tick("train", updates=state["updates"], pending_queries=state["pending"])
            if state["pending"] == 64:
                apply_update(model, optimizer, state, progress)
                continue
            size = min(state["microbatch"], 64-state["pending"])
            queries = [schedule(pools["train"], state["updates"]*64+state["pending"]+i, cfg["seed"])
                       for i in range(size)]
            progress.tick(current_microbatch=size, current_history=[q.length for q in queries],
                          current_horizon=[q.horizon for q in queries])
            gradients = {n: p.grad.detach().clone() if p.grad is not None else None for n, p in model.adapter.named_parameters()}
            before_rng = rng_state()
            try:
                loss, count = training_batch(model, codec, series, queries, cache, check)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite numeric loss")
                loss.backward()
            except torch.cuda.OutOfMemoryError:
                for name, p in model.adapter.named_parameters():
                    p.grad = gradients[name]
                restore_rng(before_rng)
                if size > 1:
                    state["microbatch"] = 1
                    state.setdefault("microbatch_fallbacks", []).append({"updates": state["updates"],
                        "pending": state["pending"], "from": size, "to": 1, "reason": "cuda_oom_rolled_back"})
                    torch.cuda.empty_cache()
                    continue
                raise
            state["pending"] += size
            state["pending_tokens"] += count
            state["pending_loss"] += float(loss.detach())
            if state["pending"] == 64:
                apply_update(model, optimizer, state, progress)
            if time.monotonic()-last_save >= 30 or state["updates"] % 500 == 0 and state["pending"] == 0:
                memory("train")
                save_checkpoint(output/"last.pt", model, optimizer, state, progress, binding)
                last_save = time.monotonic()
        save_checkpoint(output/"last.pt", model, optimizer, state, progress, binding)
        best = read(output/"best_validation.json")
        if best is None:
            raise RuntimeError("Cannot seal without complete source validation")
        atomic_json(output/"final_selection.json", {"checkpoint": str(output/"best.pt"), "sha256": sha(output/"best.pt"),
            "manifest_sha256": binding, "best_validation": best, "stop_reason": state["stop_reason"],
            "trained_source_only": True, "source_queries_complete": len(pools["validation"])})
        status("source_locked", stop_reason=state["stop_reason"])
        progress.tick("source_locked", force=True)
    except Paused as exc:
        memory("paused")
        if not (output/"final_selection.json").exists():
            save_checkpoint(output/"last.pt", model, optimizer, state, progress, binding)
        status("resource_paused", reason=str(exc))
        progress.tick("resource_paused", force=True)
    except Exception as exc:
        memory("failed")
        if not (output/"final_selection.json").exists():
            save_checkpoint(output/"last.pt", model, optimizer, state, progress, binding)
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            from gofa_f.idle_launcher import own_gpu_pids, query_gpu
            try:
                external = bool(set(query_gpu()["pids"])-own_gpu_pids(os.getpid()))
            except Exception:
                external = True
            if external:
                status("resource_paused", reason="cuda_oom_with_external_or_unknown_allocations; exact state saved")
                progress.tick("resource_paused", force=True)
                return
        atomic_json(output/"failures.json", {"type": type(exc).__name__, "error": str(exc),
            "updates": state["updates"], "phase": progress.phase})
        status("failed", error=str(exc))
        raise


def report(output):
    targets = {name: {mode: read(output/f"target-{name}-{mode}-result.json") for mode in ("native", "gofa")}
               for name in TARGETS}
    result = {"status": read(output/"status.json"), "best_validation": read(output/"best_validation.json"),
        "final_selection": read(output/"final_selection.json"), "activity": read(output/"activity.json"),
        "cuda_memory": read(output/"cuda_memory.json"), "targets": targets,
        "cache": read(output/"cache.json"),
        "complete_target_datasets": sum(all(v.values()) for v in targets.values()),
        "strict_table4_reproduction": False,
        "limitations": ["single seed; pretrained row exposure unverified", "finite grammar BF16 greedy single-block differs from author protocol",
                        "graph-enhanced QA unverified; native QA bypass retained", "no trained no-graph adapter control"]}
    atomic_json(output/"report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    for mode in ("prepare", "audit-only", "train", "resume", "report-only", "evaluate-targets"):
        parser.add_argument("--"+mode, action="store_true")
    args = parser.parse_args()
    cfg, output = yaml.safe_load(Path(args.config).read_text()), Path(args.output)
    if args.report_only:
        print(json.dumps(report(output), indent=2))
        return
    if args.prepare or args.audit_only:
        prepare(cfg, output)
    if args.train or args.evaluate_targets:
        if args.evaluate_targets and not (output/"final_selection.json").exists():
            raise ValueError("Target predictions require a source-only sealed checkpoint")
        run(cfg, output, args.resume)


if __name__ == "__main__":
    main()
