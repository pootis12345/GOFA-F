import hashlib
import json
import random

import numpy as np
import pytest
import torch

from gofa_f.chattime_checks import assert_tree_equal
from gofa_f.chattime_continuation import copy_continuation, source_record
from gofa_f.chattime_stage import bind_window, deadline_extension, digest
from gofa_f.prefix_cache import atomic_json


@pytest.fixture
def continued(tmp_path):
    source, target = tmp_path/"old", tmp_path/"new"
    source.mkdir()
    target.mkdir()
    for path in (source, target):
        (path/"correlation_graph.npz").write_bytes(b"same graph")
    weights = tmp_path/"weights"
    weights.write_bytes(b"frozen")
    config = {"seed": 42, "microbatch": 1}
    window = bind_window(source, now=1000., shared_admission=True)
    inputs = {str(weights): digest(weights), str(source/"correlation_graph.npz"): digest(source/"correlation_graph.npz")}
    manifest = {"config": config, "window": window, "inputs": inputs}
    atomic_json(source/"manifest.json", manifest)
    binding = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    atomic_json(source/"launcher.json", {"state": "stopped_by_user"})
    atomic_json(source/"active_budget.json", {"spent_seconds": 8000., "worker": None})
    atomic_json(source/"budget.json", {"spent_seconds": 7990., "active_scope": None, "scopes": {"train": 7990.}})
    for suffix in (".launch.json", ".bootstrap.launch.json"):
        atomic_json(tmp_path/(source.name+suffix), {"pid": 123, "start_ticks": 45})
    folder = source/"gofa_chattime"
    folder.mkdir()
    saved = {"binding": binding, "model": {"head.weight": torch.arange(12.)},
        "optimizer": {"state": {0: {"step": torch.tensor(79.), "exp_avg": torch.arange(12.)/10}}},
        "gradients": {"head.weight": torch.arange(12.)/7},
        "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                "python": random.getstate(), "cuda": [torch.arange(64, dtype=torch.uint8)]},
        "state": {"updates": 79, "offset": 5105, "pending_windows": 49, "pending_count": 118985,
                  "pending_loss": 389512.1462402344, "stop_reason": "training_deadline_or_signal"}}
    torch.save(saved, folder/"last.pt")
    atomic_json(folder/"progress.json", {"checkpoint_sha256": digest(folder/"last.pt")})
    prepared = {"output": str(target), "inputs": {str(weights): digest(weights),
        str(target/"correlation_graph.npz"): digest(target/"correlation_graph.npz")}}
    return source, target, config, prepared, saved


def test_continuation_preserves_every_training_state_and_cost(continued):
    source, target, config, prepared, saved = continued
    record = source_record(source, config, prepared, identity=lambda _: None)
    result = copy_continuation(record, target, "new-source-binding")
    actual = torch.load(target/"gofa_chattime/last.pt", map_location="cpu", weights_only=False)
    assert_tree_equal({k: v for k, v in saved.items() if k != "binding"},
                      {k: v for k, v in actual.items() if k != "binding"})
    assert actual["binding"] == "new-source-binding" and result["state_restore"] == "exact"
    assert json.loads((target/"active_budget.json").read_text())["spent_seconds"] == 8000.
    actual["state"]["updates"] = 80
    torch.save(actual, target/"gofa_chattime/last.pt")
    copy_continuation(record, target, "new-source-binding")
    assert torch.load(target/"gofa_chattime/last.pt", weights_only=False)["state"]["updates"] == 80


def test_deadline_missed_without_gpu_or_predictions_can_continue_with_user_amendment(continued):
    source, target, config, prepared, _ = continued
    atomic_json(source/"launcher.json", {"state": "incomplete"})
    atomic_json(source/"status.json", {"reason": "mechanisms missed training deadline"})
    atomic_json(source/"final_selection.json", {"gofa": None, "updates": 79})
    record = source_record(source, config, prepared, identity=lambda _: None)
    amendment = deadline_extension(record, 70000.)
    bind_window(target, now=record["started_at"], hard_deadline=70000., adopted_from=record,
                deadline_amendment=amendment)
    migrated = copy_continuation(record, target, "new-binding")
    assert migrated["original_start_retained"] and not migrated["original_deadline_retained"]
    assert migrated["deadline_amendment"] == amendment
    assert not (target/"final_selection.json").exists()
    assert json.loads((target/"active_budget.json").read_text())["spent_seconds"] == 8000.


def test_arbitrary_incomplete_source_is_not_a_deadline_continuation(continued):
    source, _, config, prepared, _ = continued
    atomic_json(source/"launcher.json", {"state": "incomplete"})
    atomic_json(source/"status.json", {"reason": "other failure"})
    with pytest.raises(ValueError, match="stopped and fully saved"):
        source_record(source, config, prepared, identity=lambda _: None)


@pytest.mark.parametrize("bad", ["live", "config", "prediction", "active"])
def test_continuation_rejects_incompatible_or_active_source(continued, bad):
    source, _, config, prepared, _ = continued
    if bad == "config":
        config = {**config, "microbatch": 2}
    elif bad == "prediction":
        path = source/"native_chattime/predictions/test"
        path.mkdir(parents=True)
        (path/"000000.npz").write_bytes(b"test already evaluated")
    elif bad == "active":
        atomic_json(source/"active_budget.json", {"spent_seconds": 8000., "worker": {"pid": 321}})
    with pytest.raises(ValueError):
        source_record(source, config, prepared, identity=lambda _: "45" if bad == "live" else None)
