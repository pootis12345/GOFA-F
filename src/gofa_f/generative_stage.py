"""Stage17 source-only training. This entry point never evaluates target datasets."""
import argparse
import copy
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import torch
import yaml
from transformers import LlamaForCausalLM, LlamaTokenizer

from gofa_f.chattime import REVISION, SOURCE_REVISION
from gofa_f.generative_data import (Query, SOURCES, Series, build_pools, prepare_sources,
                                    relative_score, sha)
from gofa_f.generative_forecast import evaluate_sources
from gofa_f.generative_model import (GenerativeCodec, GenerativeGOFA, GenerativePrefixLRU,
                                     completion_loss, frozen_cache_binding)
from gofa_f.generative_progress import Progress
from gofa_f.prefix_cache import atomic_json


class Paused(RuntimeError):
    pass


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def config_check(cfg):
    a, t, r = cfg["architecture"], cfg["training"], cfg["resources"]
    if a["graph_layers"] != [30, 31] or a["prefix_layers"] != 30 or a["injection_side"] != "before":
        raise ValueError("Stage17 interleaving boundary changed")
    if any(a[k] != v for k, v in {"graph_width": 64, "attention_rank": 16,
                                  "prior_fraction": .5, "graph_top_k": 8}.items()):
        raise ValueError("Fixed graph architecture changed")
    if a.get("numerical_execution") != "bf16_rows128_sdpa_fixed_prompt_capacity_v1":
        raise ValueError("Shape-stable execution binding changed")
    if cfg["weights"]["revision"] != REVISION or cfg["weights"]["source_revision"] != SOURCE_REVISION:
        raise ValueError("Official ChatTime snapshot changed")
    fixed = {"max_updates": 4000, "effective_batch": 64, "microbatch": 4,
        "learning_rate": 2e-4, "weight_decay": 1e-4, "gradient_clip": 1., "warmup_updates": 100,
        "validation_every": 250, "patience": 5, "minimum_updates": 1000, "relative_min_delta": .001}
    if any(t[key] != value for key, value in fixed.items()):
        raise ValueError("Preregistered optimization protocol changed")
    if r["activity_limit_seconds"] is not None or r["wall_deadline"] is not None \
            or r["admission_policy"] != "strict_idle" or r["memory_cap_gib"] != 40 \
            or r["torch_cap_gib"] != 38 or r["idle_check_seconds"] != 10:
        raise ValueError("Preregistered unlimited/strict-idle policy changed")
    if r["consecutive_idle_samples"] != 3 or r["idle_max_used_mib"] != 256 \
            or r["yield_on_foreign"] or r["leave_free_gib"] != 0:
        raise ValueError("Strict admission and runtime coexistence changed")
    if cfg["data"]["target_evaluation_enabled"] or cfg["data"]["sources"] != list(SOURCES):
        raise ValueError("Target evaluation/training is forbidden")
    if cfg["seed"] != 42 or cfg["data"]["train_queries_per_source"] != 8192:
        raise ValueError("Seed and source pool size changed")


def verify_weights(cfg):
    root = Path(cfg["weights"]["directory"])
    manifest = read(root/"snapshot.json")
    if manifest["revision"] != REVISION or manifest["source_revision"] != SOURCE_REVISION:
        raise ValueError("Wrong cached ChatTime revision")
    for name, record in manifest["files"].items():
        if (root/name).stat().st_size != record["size"] or sha(root/name) != record["sha256"]:
            raise ValueError(f"Official weight hash mismatch: {name}")
    return sha(root/"snapshot.json")


def prepare(cfg, output):
    config_check(cfg)
    root = Path(cfg["data"]["directory"])
    root.mkdir(parents=True, exist_ok=True)
    records = prepare_sources(root/"sources", root/"raw")
    atomic_json(root/"source_audit.json", records)
    from gofa_f.generative_audit import audit_targets
    audit = audit_targets(root, output/"target_protocol")
    series = {name: Series(root/"sources"/(name+".npz")) for name in SOURCES}
    pool_path = root/"pools.json"
    if pool_path.exists():
        pools = read(pool_path)
        if any(sha(root/name) != value for name, value in pools["hashes"].items()):
            raise ValueError("Fixed query pool changed")
    else:
        pools = build_pools(series, root, cfg["data"]["train_queries_per_source"], cfg["seed"])
        atomic_json(pool_path, pools)
    weight_sha = verify_weights(cfg)
    source_code = Path(__file__).parent
    code = {p.name: sha(p) for p in sorted(source_code.glob("*.py"))}
    config_sha = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    manifest = {"protocol": cfg["protocol"], "configuration": cfg, "config_sha256": config_sha,
        "weights_sha256": weight_sha, "source_data": records, "pools": pools, "code": code,
        "dependencies": {"torch": torch.__version__, "transformers": __import__("transformers").__version__,
                         "numpy": np.__version__, "python": sys.version},
        "targets_evaluated": False, "target_evaluation_enabled": False,
        "stop_rule": "4000_updates_or_source_patience_or_unrecoverable_failure"}
    repository = Path(__file__).resolve().parents[2]
    revision_file = repository/"CODE_REVISION"
    manifest["code_revision"] = revision_file.read_text().strip() if revision_file.exists() else \
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    manifest["source_target_exclusion"] = audit["source_exclusion"]
    old = read(output/"manifest.json")
    if old is not None and old != manifest:
        raise ValueError("Immutable run manifest changed")
    atomic_json(output/"manifest.json", manifest)
    if not (output/"progress.json").exists():
        Progress(output).tick("prepared", force=True)
    return manifest


def load_model(cfg, device, dtype=torch.bfloat16):
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    native = LlamaForCausalLM.from_pretrained(cfg["weights"]["directory"],
        torch_dtype=dtype, attn_implementation="sdpa", local_files_only=True)
    if native.config.num_hidden_layers != 32 or native.config.hidden_size != 4096 \
            or native.config.vocab_size != 42001:
        raise ValueError("Official ChatTime architecture differs")
    tokenizer = LlamaTokenizer.from_pretrained(cfg["weights"]["directory"], local_files_only=True)
    model = GenerativeGOFA(native, stable_fp32=dtype == torch.float32).to(device)
    return model, GenerativeCodec(tokenizer)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, model, optimizer, state, progress, binding):
    progress.tick(force=True)
    payload = {"binding": binding, "adapter": model.trainable_state(), "optimizer": optimizer.state_dict(),
        "gradients": {n: p.grad.detach().cpu().clone() if p.grad is not None else None
                      for n, p in model.adapter.named_parameters()}, "state": copy.deepcopy(state),
        "progress": copy.deepcopy(progress.state), "rng": rng_state(),
        "parameter_version": model.parameter_version}
    temporary = Path(path).with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_checkpoint(path, model, optimizer, binding):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["binding"] != binding:
        raise ValueError("Checkpoint data/code/config/weight binding differs")
    model.load_trainable(saved["adapter"])
    model.parameter_version = saved["parameter_version"]
    optimizer.load_state_dict(saved["optimizer"])
    for n, p in model.adapter.named_parameters():
        p.grad = saved["gradients"][n].to(p.device) if saved["gradients"][n] is not None else None
    restore_rng(saved["rng"])
    return saved["state"], saved["progress"]


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def apply_update(model, optimizer, state, progress):
    """One atomic update; an OOM must not save half an AdamW transaction."""
    if state["pending"] != 64:
        raise ValueError("Only a complete effective batch can update")
    adapter = model.trainable_state()
    moments, version = cpu_copy(optimizer.state_dict()), model.parameter_version
    gradients = {n: cpu_copy(p.grad) for n, p in model.adapter.named_parameters()}
    try:
        for p in model.adapter.parameters():
            if p.grad is not None:
                p.grad.div_(64)
        norm = torch.nn.utils.clip_grad_norm_(model.adapter.parameters(), 1., error_if_nonfinite=True)
        lr = 2e-4*min(1., (state["updates"]+1)/100)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
    except Exception:
        model.load_trainable(adapter)
        model.parameter_version = version
        optimizer.state.clear()
        optimizer.load_state_dict(moments)
        for n, p in model.adapter.named_parameters():
            p.grad = gradients[n].to(p.device) if gradients[n] is not None else None
        raise
    optimizer.zero_grad(set_to_none=True)
    model.mark_updated()
    state["updates"] += 1
    mean_loss = state["pending_loss"]/64
    state["pending"], state["pending_tokens"], state["pending_loss"] = 0, 0, 0.
    progress.state["measured_updates"] += 1
    progress.tick(updates=state["updates"], pending_queries=0, loss=mean_loss,
                  learning_rate=lr, gradient_norm=float(norm))


@lru_cache(maxsize=64)
def source_permutation(seed, epoch, source_index):
    return np.random.default_rng(np.random.SeedSequence([seed, epoch, source_index])).permutation(8192)


def schedule(pool, consumed, seed):
    names = list(SOURCES)
    epoch, offset = divmod(consumed, 32768)
    source_index = offset % 4
    source = names[source_index]
    permutation = source_permutation(seed, epoch, source_index)
    return Query(**pool[source][int(permutation[offset//4])])


def train(cfg, output, resume):
    if os.environ.get("GOFA_STAGE17_ADMITTED") != "1":
        raise RuntimeError("GPU training must enter through the CPU strict-idle launcher")
    manifest = read(output/"manifest.json")
    config_check(cfg)
    binding = sha(output/"manifest.json")
    preflight = read(output/"cpu_preflight.json", {})
    code = {p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    if manifest["configuration"] != cfg or not preflight.get("passed") \
            or preflight.get("manifest_sha256") != binding or manifest["code"] != code:
        raise ValueError("Successful bound CPU preflight required")
    torch.cuda.set_per_process_memory_fraction(38*2**30/torch.cuda.get_device_properties(0).total_memory)
    device = torch.device("cuda:0")
    root = Path(cfg["data"]["directory"])
    series = {name: Series(root/"sources"/(name+".npz")) for name in SOURCES}
    pools, validation, eps = (read(root/name) for name in
        ("train_queries.json", "validation_queries.json", "metric_epsilons.json"))
    model, codec = load_model(cfg, device)
    optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=2e-4, weight_decay=1e-4)
    state = {"updates": 0, "pending": 0, "pending_tokens": 0, "pending_loss": 0., "microbatch": 4,
        "best_score": None, "early_reference": None, "bad_validations": 0, "last_validated_update": 0,
        "peak_memory": {}, "stop_reason": None}
    saved_progress = None
    if resume:
        state, saved_progress = restore_checkpoint(output/"last.pt", model, optimizer, binding)
    elif (output/"last.pt").exists():
        raise ValueError("Existing state requires --resume")
    progress = Progress(output, saved_progress)
    stopped = [False]
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stopped.__setitem__(0, True))

    def check():
        if stopped[0]:
            raise Paused("cooperative_resource_or_user_stop")
        parent = os.environ.get("GOFA_STAGE17_SUPERVISOR_PID")
        if parent:
            from gofa_f.idle_launcher import process_identity
            if process_identity(int(parent)) != int(os.environ["GOFA_STAGE17_SUPERVISOR_TICKS"]):
                raise Paused("verified_supervisor_exited")

    def status(kind, **fields):
        atomic_json(output/"status.json", {"state": kind, "updates": state["updates"],
            "pending_queries": state["pending"], "updated_at": time.time(), **fields})

    def memory(phase):
        item = state["peak_memory"].setdefault(phase, {"allocated_bytes": 0, "reserved_bytes": 0})
        item["allocated_bytes"] = max(item["allocated_bytes"], torch.cuda.max_memory_allocated())
        item["reserved_bytes"] = max(item["reserved_bytes"], torch.cuda.max_memory_reserved())
        atomic_json(output/"cuda_memory.json", state["peak_memory"])

    cache = GenerativePrefixLRU(frozen_cache_binding(model, manifest["weights_sha256"]),
                      max_bytes=cfg["cache"]["ram_limit_gib"]*2**30)
    atomic_json(output/"cache.json", {"prefix_layers": 30, "ram_limit_gib": 32,
        "disk_limit_gib": 128, "mode": "bounded_lossless_ram_lru", "graph_kv_cross_update": False,
        "conservative_complete_bytes": 32768*1024*4096*2,
        "disk_cache_enabled": False, "reason": "conservative complete estimate exceeds 128GiB"})
    try:
        if not read(output/"mechanisms.json", {}).get("passed"):
            from gofa_f.generative_checks import mechanisms
            status("running", phase="gpu_mechanisms")
            progress.tick("gpu_mechanisms", force=True)
            result = mechanisms(model, codec, series, validation, 2e-3,
                diagnostic=lambda result: atomic_json(output/"mechanism_diagnostics.json", result), check=check)
            atomic_json(output/"mechanisms.json", result)
            memory("mechanisms")
            torch.cuda.reset_peak_memory_stats()
        atomic_json(output/"parameters.json", model.parameter_counts())
        status("running", phase="native_source_baseline")
        progress.tick("native_source_baseline", force=True)
        baseline = evaluate_sources(model, codec, {"data": series, "epsilons": eps}, validation,
            output, "native", binding+":native_greedy_bf16", progress, check)
        memory("native_source_baseline")
        torch.cuda.reset_peak_memory_stats()
        last_save = time.monotonic()
        while True:
            check()
            # A restored uncommitted validation is completed before any further update.
            if state["updates"] and state["updates"] % 250 == 0 \
                    and state["last_validated_update"] != state["updates"]:
                progress.tick("source_validation", force=True)
                status("running", phase="source_validation")
                candidate = output/"candidates"/f"update-{state['updates']:06d}.pt"
                candidate.parent.mkdir(exist_ok=True)
                if not candidate.exists():
                    torch.save(model.trainable_state(), candidate)
                checkpoint_sha = sha(candidate)
                current = evaluate_sources(model, codec, {"data": series, "epsilons": eps}, validation,
                    output, f"update-{state['updates']:06d}", binding+":"+checkpoint_sha, progress, check)
                result = relative_score(current, baseline, eps)
                record = {"updates": state["updates"], "checkpoint_sha256": checkpoint_sha,
                          "checkpoint": str(candidate), **result, **current}
                atomic_json(output/f"validation-{state['updates']:06d}.json", record)
                score = result["score"]
                if state["best_score"] is None or score < state["best_score"]:
                    state["best_score"] = score
                    shutil.copyfile(candidate, output/"best.pt")
                    atomic_json(output/"best_validation.json", record)
                if state["early_reference"] is None or score <= state["early_reference"]*(1-.001):
                    state["early_reference"], state["bad_validations"] = score, 0
                else:
                    state["bad_validations"] += 1
                state["last_validated_update"] = state["updates"]
                progress.state["validation_runs"] += 1
                memory("source_validation")
                torch.cuda.reset_peak_memory_stats()
                save_checkpoint(output/"last.pt", model, optimizer, state, progress, binding)
            if state["updates"] >= 4000 or (state["updates"] >= 1000 and state["bad_validations"] >= 5):
                state["stop_reason"] = "max_updates" if state["updates"] >= 4000 else "source_patience"
                break
            status("running", phase="train")
            progress.tick("train", updates=state["updates"], pending_queries=state["pending"])
            if state["pending"] == 64:
                apply_update(model, optimizer, state, progress)
                continue
            consumed = state["updates"]*64+state["pending"]
            count = min(state["microbatch"], 64-state["pending"])
            queries = [schedule(pools, consumed+i, cfg["seed"]) for i in range(count)]
            records = [codec.record(*series[q.source].training_record(q)) for q in queries]
            tokens = codec.batch(records).to(device)
            graphs = [series[q.source].graph(q).to(device) for q in queries]
            # Snapshot partial gradients/RNG so OOM never doubles accumulated examples.
            gradients = {n: p.grad.detach().clone() if p.grad is not None else None
                         for n, p in model.adapter.named_parameters()}
            before_rng = rng_state()
            try:
                key = cache.key(tokens)
                prefix = cache.get(key, device)
                if prefix is None:
                    prefix = model.prefix(tokens)
                    cache.put(key, prefix)
                loss, valid_tokens = completion_loss(model(tokens, graphs, prefix=prefix).logits, tokens.labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite completion loss")
                loss.backward()
            except torch.cuda.OutOfMemoryError:
                for n, p in model.adapter.named_parameters():
                    p.grad = gradients[n]
                restore_rng(before_rng)
                if state["microbatch"] == 1:
                    raise
                state["microbatch"] = max(1, state["microbatch"]//2)
                loss, prefix, tokens, graphs = None, None, None, None
                torch.cuda.empty_cache()
                continue
            state["pending"] += count
            state["pending_tokens"] += valid_tokens
            state["pending_loss"] += float(loss.detach())
            if state["pending"] == 64:
                apply_update(model, optimizer, state, progress)
            else:
                progress.tick(pending_queries=state["pending"], microbatch=state["microbatch"])
            if time.monotonic()-last_save >= 30 or state["updates"] % 250 == 0 and state["pending"] == 0:
                memory("train")
                save_checkpoint(output/"last.pt", model, optimizer, state, progress, binding)
                last_save = time.monotonic()
        progress.tick("complete", force=True)
        save_checkpoint(output/"last.pt", model, optimizer, state, progress, binding)
        best = read(output/"best_validation.json")
        if best is None:
            raise RuntimeError("Cannot lock a checkpoint without complete source validation")
        atomic_json(output/"final_selection.json", {"checkpoint": str(output/"best.pt"),
            "sha256": sha(output/"best.pt"), "best_validation": best,
            "stop_reason": state["stop_reason"], "target_evaluation_enabled": False})
        status("complete", stop_reason=state["stop_reason"])
    except Paused as exc:
        memory("paused")
        progress.tick("resource_paused", force=True)
        save_checkpoint(output/"last.pt", model, optimizer, state, progress, binding)
        status("resource_paused", reason=str(exc))
        return 75
    except Exception as exc:
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            from gofa_f.idle_launcher import query_gpu, own_gpu_pids
            try:
                other_allocations = bool(set(query_gpu()["pids"])-own_gpu_pids(os.getpid()))
            except Exception:
                other_allocations = True  # unknown resource state must re-enter strict idle
            if other_allocations:
                memory("resource_paused")
                progress.tick("resource_paused", force=True)
                save_checkpoint(output/"last.pt", model, optimizer, state, progress, binding)
                status("resource_paused", reason="cuda_oom_with_external_or_unknown_allocations")
                return 75
        progress.tick("failed", force=True)
        save_checkpoint(output/"last.pt", model, optimizer, state, progress, binding)
        atomic_json(output/"failures.json", {"type": type(exc).__name__, "error": str(exc),
            "updates": state["updates"], "target_predictions": 0})
        status("failed", error=str(exc))
        raise
    finally:
        atomic_json(output/"cache_usage.json", {"hits": cache.hits, "misses": cache.misses,
            "peak_ram_bytes": cache.peak, "disk_bytes": cache.disk_used})
    return 0


def report(output):
    status = read(output/"status.json", {"state": "not_started"})
    final = read(output/"final_selection.json")
    result = {"status": status, "final_selection": final, "target_predictions": 0,
              "activity": read(output/"activity.json"), "progress": read(output/"progress.json"),
              "parameters": read(output/"parameters.json"), "cuda_memory": read(output/"cuda_memory.json")}
    atomic_json(output/"report.json", result)
    (output/"report.md").write_text("# Stage17 source-only run\n\nState: "+status["state"]+
        "\n\nTarget tasks evaluated: **0**.\n\n"+json.dumps(result, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage17_generative_chattime.yaml")
    parser.add_argument("--output", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--train", action="store_true")
    group.add_argument("--report-only", action="store_true")
    group.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config).read_text())
    config_check(cfg)
    if args.prepare:
        prepare(cfg, output)
    elif args.audit_only:
        from gofa_f.generative_audit import audit_targets
        audit_targets(cfg["data"]["directory"], output/"target_protocol")
    elif args.report_only:
        report(output)
    else:
        if (output/"final_selection.json").exists():
            raise ValueError("Source checkpoint already sealed; do not restart a completed stage")
        raise SystemExit(train(cfg, output, args.resume))


if __name__ == "__main__":
    main()
