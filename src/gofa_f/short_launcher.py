"""CPU-only resource supervisor with unlimited recorded GPU activity.

Legacy launchers and their time budgets are deliberately unchanged.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from gofa_f.idle_launcher import (ActivityLedger, CapOnlyGate, IdleGate, atomic_json, own_gpu_pids,
                                  process_identity, query_gpu)


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def own_usage(gpu, worker, uuid=None):
    if gpu is None or (uuid is not None and gpu["uuid"] != uuid) \
            or gpu.get("compute_mode", "Default") != "Default":
        return None
    pids = set(gpu["pids"]) & own_gpu_pids(worker)
    values = [gpu["process_used_mib"].get(pid) for pid in pids]
    return None if any(v is None for v in values) else sum(values)


def admission_gate(resources, worker_mode):
    """The Stage24 override never changes legacy workers' idle-only policy."""
    policy = resources['admission_policy']
    if resources['resource_check_seconds'] != 10:
        raise ValueError('Resource audits must remain ten seconds apart')
    if worker_mode in ('robust_source_adaptation', 'weather_prefix_adaptation') and policy == 'memory_cap_only':
        if resources['admission_samples'] != 1 or resources['memory_cap_gib'] != 40 \
                or resources['torch_cap_gib'] != 38 or resources['yield_on_foreign'] \
                or resources['leave_free_gib'] != 0:
            raise ValueError('Shared Stage24 admission requires unchanged project/Torch caps')
        return CapOnlyGate()
    if policy != 'idle' or resources['admission_samples'] != 3:
        raise ValueError('Legacy workers require three fully-idle observations')
    return IdleGate()


def supervise(output, config, snapshot, worker_mode="train"):
    import yaml
    output, config, snapshot = Path(output).resolve(), Path(config).resolve(), Path(snapshot).resolve()
    cfg = yaml.safe_load(config.read_text())
    update_cap = 100 if worker_mode == "weather_prefix_adaptation" else 250 if worker_mode == "robust_source_adaptation" else 1000 if worker_mode == "source_retraining" else 4000
    if cfg["training"]["max_updates"] != update_cap or cfg["resources"]["activity_limit_seconds"] is not None \
            or cfg["resources"]["wall_deadline"] is not None or cfg["data"]["target_training_enabled"]:
        raise ValueError("Stage19 requires bounded updates, unlimited activity and disabled target training")
    gate = admission_gate(cfg['resources'], worker_mode)
    cap_only = cfg['resources']['admission_policy'] == 'memory_cap_only'
    output.mkdir(parents=True, exist_ok=True)
    locks = []
    for path in (output/"launcher.lock", output.parent/"stage19-gpu.lock"):
        stream = path.open("a")
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(stream)
    if worker_mode not in ("train", "projection_probe", "source_diagnostic", "native_case_probe",
                           "sampling_diagnostic", "stability_diagnostic", "sampled_continuation", "sampled_targets",
                           "target_calibration", "graph_benchmark", "source_reliability_gate", "source_retraining", "robust_source_adaptation", "weather_prefix_adaptation"):
        raise ValueError("Unknown supervised worker mode")
    binding = {"snapshot": str(snapshot), "config": str(config), "admission_policy": cfg['resources']['admission_policy'],
        "resource_check_seconds": 10, "admission_samples": cfg['resources']['admission_samples'],
        "memory_cap_gib": 40, "torch_cap_gib": 38, "yield_on_foreign": False,
        "activity_limit_seconds": None, "wall_deadline": None, "maximum_updates": update_cap,
        "worker_mode": worker_mode}
    if cap_only:
        binding['leave_free_gib'] = 0
    else:
        binding['idle_max_used_mib'] = 256
    if worker_mode == "sampled_continuation":
        binding.update(authorized_review_update=cfg.get('review_at_update',1500),target_evaluation_enabled=False)
    if worker_mode == "sampled_targets":
        binding.update(target_evaluation_enabled=True,optimizer_updates=0,maximum_updates=None,
            adapter_sha256=cfg['adapter_sha256'],expected_target_queries=2121)
    if worker_mode == "target_calibration":
        binding.update(target_evaluation_enabled=True,target_label_calibration_enabled=True,
            strict_zero_shot=False,optimizer_updates=0,maximum_updates=None,
            adapter_sha256=cfg['adapter_sha256'],expected_validation_queries=1152,
            cpu_finalization_after_gpu_exit=True)
    if worker_mode == "graph_benchmark":
        binding.update(target_evaluation_enabled=True, target_label_calibration_enabled=False,
            optimizer_updates=0, maximum_updates=None, adapter_sha256=cfg['adapter_sha256'],
            expected_target_queries=768, expected_model_queries=1536)
    if worker_mode == "source_reliability_gate":
        binding.update(target_evaluation_enabled='only_after_two_source_checks',
            target_label_calibration_enabled=False, optimizer_updates=0, maximum_updates=None,
            adapter_sha256=cfg['adapter_sha256'], source_fitting_queries=128,
            source_confirmation_queries=128, maximum_target_queries=768)
    if worker_mode == "source_retraining":
        binding.update(target_evaluation_enabled=False, target_label_calibration_enabled=False,
            parent_update=3500, adapter_sha256=cfg['adapter_sha256'], source_validation_queries=384,
            source_confirmation_queries=384, new_optimizer=True)
    if worker_mode == "robust_source_adaptation":
        binding.update(target_evaluation_enabled='only_after_locked_source_confirmation',
            target_label_calibration_enabled=False, parent_update=3500, adapter_sha256=cfg['adapter_sha256'],
            source_validation_queries=384, source_confirmation_queries=192, maximum_target_queries=768,
            new_optimizer=True, loss=dict(ce=1., cdf=1., parent_kl=.1))
    if worker_mode == "weather_prefix_adaptation":
        binding.update(target_evaluation_enabled='only_after_locked_source_confirmation',
            target_label_calibration_enabled=False,parent_update=3500,adapter_sha256=cfg['adapter_sha256'],
            source_validation_queries=384,source_confirmation_queries=192,maximum_target_queries=768,
            new_optimizer=True,training_random_sampling=False,parent_greedy_queries=1024,
            primary_metric='variable_equal_raw_mae')
    if read(output/"launcher_manifest.json", binding) != binding:
        raise ValueError("Launcher recovery policy differs")
    atomic_json(output/"launcher_manifest.json", binding)
    previous = read(output/"activity.json", {})
    old = previous.get("worker") or {}
    if old.get("pid") and process_identity(old["pid"]) == old.get("start_ticks"):
        raise RuntimeError("Previous verified GPU worker still exists")
    # Account an unclean supervisor tail without zeroing previously saved activity.
    if old:
        tail = min(10., max(0., time.monotonic()-old.get("last_monotonic", time.monotonic())))
        previous["spent_seconds"] = previous.get("spent_seconds", 0.)+tail
        previous["conservative_unclean_tail_seconds"] = previous.get("conservative_unclean_tail_seconds", 0.)+tail
        previous["worker"] = None
    ledger = ActivityLedger(output/"activity.json", None, previous or None)
    ledger.state.setdefault("waiting_seconds", 0.)
    ledger.state.setdefault("resource_pause_count", 0)
    ledger.save()
    stopped = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))

    last_state = [None]
    def status(state, **fields):
        if state != last_state[0]:
            with (output/"lifecycle.jsonl").open("a") as stream:
                stream.write(json.dumps({"state": state, "at": time.time(),
                    "spent_seconds": ledger.state["spent_seconds"],
                    "waiting_seconds": ledger.state["waiting_seconds"], **fields})+"\n")
            last_state[0] = state
        atomic_json(output/"launcher.json", {"state": state, "pid": os.getpid(),
            "start_ticks": process_identity(os.getpid()), "spent_seconds": ledger.state["spent_seconds"],
            "waiting_seconds": ledger.state["waiting_seconds"], "updated_at": time.time(), **fields})

    identity_path = output/"gpu_identity.json"
    pinned_uuid = read(identity_path, {}).get("uuid")
    gate.uuid = pinned_uuid
    def finish_calibration_cpu():
        # No live GPU child, no activity charge, no CUDA context for coefficient
        # fitting or frozen-test replay. Recovery here needs no GPU admission.
        environment = dict(os.environ,PYTHONPATH=str(snapshot/"src"),CUDA_VISIBLE_DEVICES="",
            OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",PYTHONDONTWRITEBYTECODE="1",
            HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false")
        status("cpu_finalization")
        result = subprocess.run([sys.executable,"-u","-m","gofa_f.target_calibration",
            "--config",str(config),"--output",str(output),"--finish-cpu"],cwd=snapshot,env=environment)
        final = read(output/"status.json",{})
        status("complete" if result.returncode == 0 and final.get("state") == "complete" else "failed",
            cpu_returncode=result.returncode,goal_achieved=final.get("goal_achieved"))
    while not stopped[0]:
        final = read(output/"status.json", {}).get("state")
        if worker_mode == "target_calibration" and final in ("validation_complete","calibrated"):
            finish_calibration_cpu()
            return
        if final in ("complete", "failed", "incomplete"):
            status(final)
            return
        try:
            gpu, error = query_gpu(), None
        except Exception as exc:
            gpu, error = None, str(exc)
        if gpu is not None and pinned_uuid is None:
            pinned_uuid = gpu["uuid"]
            gate.uuid = pinned_uuid
            atomic_json(identity_path, {"uuid": pinned_uuid, "first_observed_at": time.time()})
        admitted = gate.observe(gpu)
        status("device_checked" if admitted else "waiting_for_gpu_query" if cap_only else "waiting_for_idle", query_error=error,
               consecutive=gate.consecutive)
        if not admitted:
            wait_started = time.monotonic()
            for _ in range(10):
                if stopped[0]:
                    break
                time.sleep(1)
            ledger.state["waiting_seconds"] += time.monotonic()-wait_started
            ledger.save()
            continue
        environment = dict(os.environ, PYTHONPATH=str(snapshot/"src"), CUDA_VISIBLE_DEVICES=gate.uuid,
            OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            PYTHONDONTWRITEBYTECODE="1", GOFA_STAGE19_ADMITTED="1", TOKENIZERS_PARALLELISM="false",
            GOFA_STAGE19_SUPERVISOR_PID=str(os.getpid()),
            GOFA_STAGE19_SUPERVISOR_TICKS=str(process_identity(os.getpid())))
        module = {"train": "gofa_f.short_generative_stage", "projection_probe": "gofa_f.short_projection_probe",
                  "source_diagnostic": "gofa_f.short_source_diagnostic",
                  "native_case_probe": "gofa_f.short_native_case_probe",
                  "sampling_diagnostic": "gofa_f.short_sampling_diagnostic",
                  "stability_diagnostic": "gofa_f.short_stability_diagnostic",
                  "sampled_continuation": "gofa_f.short_sampled_continuation",
                  "sampled_targets": "gofa_f.short_sampled_targets",
                  "target_calibration": "gofa_f.target_calibration",
                  "graph_benchmark": "gofa_f.graph_benchmark",
                  "source_reliability_gate": "gofa_f.source_reliability_gate",
                  "source_retraining": "gofa_f.source_retraining",
                  "robust_source_adaptation": "gofa_f.robust_source_adaptation",
                  "weather_prefix_adaptation": "gofa_f.weather_prefix_adaptation"}[worker_mode]
        command = [sys.executable, "-u", "-m", module,
                   "--config", str(config), "--output", str(output)]
        if worker_mode == "train":
            command.append("--train")
            if (output/"last.pt").exists():
                command.append("--resume")
        elif worker_mode in ("graph_benchmark", "source_reliability_gate"):
            command.extend(("--predict", "--resume"))
        elif worker_mode == "stability_diagnostic" and any((output/"teachers").glob("*.json")):
            command.append("--resume")
        elif worker_mode in ("sampled_continuation", "sampled_targets", "target_calibration", "source_retraining", "robust_source_adaptation", "weather_prefix_adaptation"):
            command.append("--resume")
        child = subprocess.Popen(command, cwd=snapshot, env=environment, start_new_session=True)
        tick = process_identity(child.pid)
        ledger.start(child.pid)
        status("running", worker_pid=child.pid, worker_start_ticks=tick)
        stop_at, reason, next_check = None, None, time.monotonic()
        peak = previous.get("peak_group_mib", 0)
        while child.poll() is None:
            ledger.tick()
            now = time.monotonic()
            if stop_at is None and (stopped[0] or now >= next_check):
                if stopped[0]:
                    reason = "stopped_by_user"
                else:
                    try:
                        measured = query_gpu()
                        used = own_usage(measured, child.pid, gate.uuid)
                    except Exception:
                        used = None
                    if used is None or used > 40*1024:
                        reason = "resource_paused"
                    else:
                        peak = max(peak, used)
                        ledger.state["peak_group_mib"] = peak
                        status("running", worker_pid=child.pid, group_used_mib=used, peak_group_mib=peak,
                               gpu_utilization=measured["utilization"], total_used_mib=measured["used_mib"])
                    next_check = now+10
                if reason:
                    if reason == "resource_paused":
                        ledger.state["resource_pause_count"] += 1
                    status(reason, worker_pid=child.pid)
                    if process_identity(child.pid) == tick:
                        os.killpg(child.pid, signal.SIGTERM)
                    stop_at = now+60
            elif stop_at is not None and now >= stop_at:
                if process_identity(child.pid) == tick:
                    os.killpg(child.pid, signal.SIGKILL)
                status("failed", error="Worker failed to save within 60 seconds")
                child.wait()
                ledger.stop()
                return
            time.sleep(1)
        ledger.stop()
        previous = ledger.state
        final = read(output/"status.json", {}).get("state")
        if worker_mode in ("sampled_continuation", "sampled_targets", "target_calibration", "graph_benchmark",
                            "source_reliability_gate", "source_retraining", "robust_source_adaptation", "weather_prefix_adaptation") and final == "resource_paused" and reason is None:
            ledger.state["resource_pause_count"] += 1
            ledger.save()
        if reason == "stopped_by_user":
            break
        if worker_mode == "target_calibration" and final == "validation_complete" and child.returncode == 0:
            finish_calibration_cpu()
            return
        if final in ("complete", "failed", "incomplete") or (child.returncode not in (0, 75) and reason is None):
            status(final if final in ("complete", "failed", "incomplete") else "failed",
                   worker_returncode=child.returncode)
            return
        # Recovery rechecks the same authorized policy; shared mode has no idle gate.
        wait_started = time.monotonic()
        for _ in range(10):
            if stopped[0]:
                break
            time.sleep(1)
        ledger.state["waiting_seconds"] += time.monotonic()-wait_started
        ledger.save()
        gate = admission_gate(cfg['resources'], worker_mode)
        gate.uuid = pinned_uuid
    status("stopped_by_user")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--worker-mode", choices=("train", "projection_probe", "source_diagnostic",
                                                "native_case_probe", "sampling_diagnostic",
                                                "stability_diagnostic", "sampled_continuation", "sampled_targets",
                                                "target_calibration", "graph_benchmark",
                                                "source_reliability_gate", "source_retraining", "robust_source_adaptation", "weather_prefix_adaptation"), default="train")
    args = parser.parse_args()
    supervise(args.output, args.config, args.snapshot, args.worker_mode)


if __name__ == "__main__":
    main()
