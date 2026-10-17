import copy
import json
from pathlib import Path

import pytest
import torch
import yaml

from gofa_f.allnode_recovery import migrate_failed_preflight
from gofa_f.generative_data import sha


def write(path, value):
    path.write_text(json.dumps(value))


def failed_run(tmp_path):
    source, output, diagnostic = (tmp_path/name for name in ("source", "repair", "diagnostic"))
    for path in (source, output, diagnostic):
        path.mkdir()
    cfg = yaml.safe_load(Path("configs/stage18_allnode_chattime.yaml").read_text())
    manifest = dict(configuration=cfg, weights_sha256="weights", source_data={}, pools={},
        dependencies={}, prompt_sha256="official", decode={}, code_revision="new")
    old = copy.deepcopy(manifest)
    old["configuration"]["protocol"] = "stage18-allnode-chattime-v2"
    old["configuration"]["architecture"]["numerical_execution"] = "previous_kernel"
    old["configuration"]["resources"].update(admission_policy="memory_cap_only", admission_samples=1)
    old["code_revision"] = "old"
    write(source/"manifest.json", old)
    write(output/"manifest.json", manifest)
    write(source/"failures.json", dict(type="AssertionError", updates=0, target_predictions=0))
    write(source/"status.json", dict(state="failed"))
    write(source/"activity.json", dict(worker=None, spent_seconds=12., peak_group_mib=16848))
    write(source.with_suffix(".bootstrap.launch.json"),
          dict(pid=2147483647, start_ticks="0", started_at=123.))
    write(diagnostic/"status.json", dict(state="complete", diagnostic_only=True))
    write(diagnostic/"activity.json", dict(worker=None, spent_seconds=9., peak_group_mib=15000))
    state = dict(updates=0, pending=0, pending_tokens=0, pending_loss=0.,
        last_validated_update=0, bad_validations=0, best_score=None, early_reference=None,
        peak_memory={"preflight": {"allocated_bytes": 42}})
    saved = dict(binding=sha(source/"manifest.json"), parameter_version=0, state=state,
        optimizer={"state": {}, "param_groups": [{"lr": .0002}]}, adapter={"up": torch.arange(4.)},
        gradients={"up": None}, rng={"torch": torch.get_rng_state()}, progress={"updates": 0})
    torch.save(saved, source/"last.pt")
    return source, output, diagnostic, saved


def test_failed_preflight_exact_state_rebind_and_cumulative_activity(tmp_path):
    source, output, diagnostic, saved = failed_run(tmp_path)
    receipt = migrate_failed_preflight(source, output, [diagnostic])
    restored = torch.load(output/"last.pt", weights_only=False)
    assert restored["binding"] == sha(output/"manifest.json")
    assert restored["state"] == saved["state"] and restored["optimizer"] == saved["optimizer"]
    torch.testing.assert_close(restored["adapter"]["up"], saved["adapter"]["up"], rtol=0, atol=0)
    torch.testing.assert_close(restored["rng"]["torch"], saved["rng"]["torch"], rtol=0, atol=0)
    assert json.loads((output/"activity.json").read_text())["spent_seconds"] == 21.
    # A repeated migration must not roll an advanced checkpoint/ledger back to the old preflight.
    restored["state"]["updates"] = 10
    torch.save(restored, output/"last.pt")
    write(output/"activity.json", dict(spent_seconds=99.))
    assert migrate_failed_preflight(source, output, [diagnostic]) == receipt
    assert torch.load(output/"last.pt", weights_only=False)["state"]["updates"] == 10
    assert json.loads((output/"activity.json").read_text())["spent_seconds"] == 99.


@pytest.mark.parametrize("problem", ["training", "evaluation", "live_worker", "architecture", "duplicate_cost"])
def test_failed_preflight_refuses_unsafe_migration(tmp_path, problem):
    source, output, diagnostic, saved = failed_run(tmp_path)
    if problem == "training":
        saved["state"]["pending"] = 1
        torch.save(saved, source/"last.pt")
    elif problem == "evaluation":
        (source/"evaluations").mkdir()
        write(source/"evaluations/native.json", {"complete": False})
    elif problem == "live_worker":
        write(source/"activity.json", dict(worker={"pid": 123}, spent_seconds=12.))
    elif problem == "architecture":
        new = json.loads((output/"manifest.json").read_text())
        new["configuration"]["architecture"]["graph_width"] = 32
        write(output/"manifest.json", new)
    with pytest.raises(ValueError):
        migrate_failed_preflight(source, output,
            [diagnostic, diagnostic] if problem == "duplicate_cost" else [diagnostic])
    assert not (output/"last.pt").exists()


def real_proof(tmp_path):
    source, output, _, _ = failed_run(tmp_path)
    from gofa_f import allnode_recovery
    repository = Path(allnode_recovery.__file__).parent
    checked = tmp_path/"checked_snapshot"
    (checked/"src/gofa_f").mkdir(parents=True)
    (checked/"CODE_REVISION").write_text("old\n")
    old, new = (json.loads((p/"manifest.json").read_text()) for p in (source, output))
    names = ("allnode_model.py", "allnode_kernel.py", "allnode_stage.py",
             "allnode_launcher.py", "allnode_recovery.py")
    for name in names:
        (checked/"src/gofa_f"/name).write_bytes((repository/name).read_bytes())
    old["code"] = new["code"] = {name: sha(repository/name) for name in names}
    old["configuration"]["architecture"]["numerical_execution"] = \
        new["configuration"]["architecture"]["numerical_execution"]
    write(source/"manifest.json", old)
    write(output/"manifest.json", new)
    record = json.loads(source.with_suffix(".bootstrap.launch.json").read_text())
    write(source.with_suffix(".bootstrap.launch.json"), {**record, "snapshot": str(checked)})
    write(source/"cpu_preflight.json", dict(passed=True, real_weight_dtype="float32",
        manifest_sha256=sha(source/"manifest.json"),
        real_weights=dict(passed=True, tolerance=1e-6), parameters=dict(trainable=1060998)))
    return source, new, checked


def test_same_model_real_cpu_proof_reuse_for_admission_only(tmp_path):
    from gofa_f.allnode_recovery import reuse_real_cpu
    source, manifest, _ = real_proof(tmp_path)
    proof = reuse_real_cpu(source, manifest)
    assert proof["real_weights"]["passed"] and proof["real_weights"]["tolerance"] == 1e-6
    assert proof["source_cpu_proof_sha256"] == sha(source/"cpu_preflight.json")


@pytest.mark.parametrize("problem", ["model_module", "kernel", "incomplete", "source_tamper"])
def test_real_cpu_reuse_rejects_changed_or_incomplete_proof(tmp_path, problem):
    from gofa_f.allnode_recovery import reuse_real_cpu
    source, manifest, checked = real_proof(tmp_path)
    if problem == "model_module":
        manifest["code"]["allnode_model.py"] = "changed"
    elif problem == "kernel":
        manifest["configuration"]["architecture"]["numerical_execution"] = "different_execution"
    elif problem == "incomplete":
        proof = json.loads((source/"cpu_preflight.json").read_text())
        write(source/"cpu_preflight.json", {**proof, "passed": False})
    else:
        (checked/"src/gofa_f/allnode_model.py").write_text("modified\n")
    with pytest.raises(ValueError):
        reuse_real_cpu(source, manifest)
