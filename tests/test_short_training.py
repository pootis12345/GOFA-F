from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import LlamaTokenizer

from gofa_f.allnode_graph import build_graph
from gofa_f.generative_data import Query, sha
from gofa_f.generative_model import GenerativePrefixLRU
from gofa_f.short_codec import ShortCodec, numeric_loss
from gofa_f.short_migration import INFERENCE_FILES, assert_equal, migrate, reusable_cpu_proof
from gofa_f.short_training import compact_forward, compact_loss, teacher_prefix
from test_allnode import tiny, source


@pytest.fixture(scope="module")
def codec():
    root = Path("/home/dcg/dcg_data/GOFA-F/data/weights/chattime-7b-20baa88")
    if not root.exists():
        pytest.skip("Official tokenizer unavailable")
    return ShortCodec(LlamaTokenizer.from_pretrained(root, local_files_only=True))


def inputs(codec):
    data = source(9)
    queries = [Query("pems04", 0, 500, 12, 12), Query("pems04", 1, 600, 48, 24)]
    records = []
    for q in queries:
        history = data.values[q.origin-q.length:q.origin, q.variable]
        target = data.values[q.origin:q.origin+q.horizon, q.variable].copy()
        target[::3] = np.nan
        records.append(codec.record(history, target))
    return records, [build_graph(data, q) for q in queries]


@pytest.mark.parametrize("size", [1, 2])
def test_compact_projection_matches_dense_loss_and_all_gradients(codec, size):
    model = tiny()
    with torch.no_grad():
        for layer in model.adapter.exchanges:
            layer.up.weight.normal_(std=.001)
    records, graphs = inputs(codec)
    records, graphs = records[:size], graphs[:size]
    expected, total = [], 0.
    for record, graph in zip(records, graphs):
        tokens = codec.batch([record])
        contexts = model.encode_origins([graph], codec)
        output = model(tokens, contexts=contexts).logits
        valid = tokens.labels[:, 1:] != -100
        expected.append(output[:, :-1][valid].index_select(-1, codec.alphabet("cpu")).detach())
        loss, _ = numeric_loss(output, tokens.labels, codec)
        total += float(loss.detach())
        loss.backward()
    gradients = {n: p.grad.clone() for n, p in model.adapter.named_parameters()}
    model.zero_grad(set_to_none=True)
    tokens = codec.batch(records)
    contexts = model.encode_origins(graphs, codec)
    hook = model.native.lm_head.register_forward_pre_hook(lambda *a:
        (_ for _ in ()).throw(AssertionError("Dense vocabulary projection executed")))
    try:
        result = compact_forward(model, codec, tokens, contexts)
    finally:
        hook.remove()
    loss, count = compact_loss(result)
    assert result[0].shape == (sum(sum(x != -100 for x in r["labels"]) for r in records), 10000)
    assert count == len(result[0])
    torch.testing.assert_close(result[0], torch.cat(expected), rtol=1e-6, atol=1e-6)
    assert float(loss) == pytest.approx(total, rel=1e-6, abs=1e-6)
    loss.backward()
    for name, p in model.adapter.named_parameters():
        torch.testing.assert_close(p.grad, gradients[name], rtol=1e-6, atol=1e-6)
    assert all(not p.requires_grad and p.grad is None for p in model.native.parameters())


def test_teacher_cache_reuse_and_other_query_cannot_change_first(codec):
    model = tiny()
    with torch.no_grad():
        for layer in model.adapter.exchanges:
            layer.up.weight.normal_(std=.001)
    records, graphs = inputs(codec)
    cache = GenerativePrefixLRU("numeric_training_test", max_bytes=2**26)
    tokens = codec.batch(records)
    a = teacher_prefix(model, tokens, cache)
    b = teacher_prefix(model, tokens, cache)
    torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
    assert cache.hits == 1
    contexts = model.encode_origins(graphs, codec)
    result = compact_forward(model, codec, tokens, contexts, a)
    solo = codec.batch(records[:1])
    own = compact_forward(model, codec, solo, contexts[:1])
    torch.testing.assert_close(result[0][result[2] == 0], own[0], rtol=1e-6, atol=1e-6)
    model.mark_updated()
    with pytest.raises(ValueError, match="Stale"):
        compact_forward(model, codec, tokens, contexts, a)
    # A frozen prefix remains valid after an adapter update.
    teacher_prefix(model, tokens, cache)
    assert cache.hits == 2


def migration_fixture(tmp_path):
    source, target = tmp_path/"source", tmp_path/"target"
    source.mkdir()
    target.mkdir()
    previous = dict(protocol="v2", training={"microbatch": 1})
    current = dict(protocol="v3", training={"microbatch": 2}, optimizations={"finite": True})
    manifest = dict(configuration=previous, code={n: n for n in INFERENCE_FILES})
    for key in ("weights_sha256", "raw_hashes", "query_sha256", "epsilon_sha256",
                "source_range_audit_sha256", "source_target_audit_sha256", "dependencies"):
        manifest[key] = "same"
    def write(root, name, value):
        p = root/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value))
    write(source, "manifest.json", manifest)
    payload = dict(binding=sha(source/"manifest.json"), adapter={"w": torch.ones(3)},
        optimizer={"state": {1: {"exp_avg": torch.ones(3)}}}, gradients={"w": torch.ones(3)*2},
        rng={"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "cuda": None},
        state=dict(updates=98, pending=48, pending_tokens=482, pending_loss=230.882,
                   last_validated_update=0, microbatch=1, best_score=None,
                   early_reference=None, bad_validations=0), parameter_version=98,
        progress=dict(measured_updates=98, train_seconds=1248., updates=98))
    torch.save(payload, source/"last.pt")
    current["continuation"] = dict(source_output=str(source), checkpoint_sha256=sha(source/"last.pt"))
    after = deepcopy(manifest)
    after["configuration"] = current
    write(target, "manifest.json", after)
    write(source, "launcher.json", {"state": "stopped_by_user"})
    write(source, "status.json", {"state": "resource_paused"})
    write(source, "activity.json", dict(worker=None, spent_seconds=1970., waiting_seconds=80., peak_group_mib=17246))
    write(source, "gpu_identity.json", {"uuid": "gpu"})
    write(source, "cpu_preflight.json", {"alphabet": {"alphabet_sha256": "alphabet"}})
    q = dict(source="pems04", variable=0, origin=500, length=12, horizon=12)
    write(target, "pools/queries.json", {"validation": [q]})
    old = sha(source/"manifest.json")+":native:alphabet"
    write(source, "evaluations/native/000000.json", dict(query=q, binding=old, mae=5., prediction=[1.]))
    write(source, "evaluations/native/complete.json", dict(binding=old, complete=True, queries=1, errors=[5.]))
    return current, source, target, payload


def test_migration_keeps_optimizer_partial_gradients_rng_baseline_and_activity(tmp_path):
    cfg, source, target, payload = migration_fixture(tmp_path)
    source_digest = sha(source/"last.pt")
    record = migrate(cfg, target)
    actual = torch.load(target/"last.pt", weights_only=False)
    for key in ("adapter", "optimizer", "gradients", "rng", "parameter_version"):
        assert_equal(payload[key], actual[key])
    assert actual["state"]["pending"] == 48 and actual["state"]["microbatch"] == 2
    assert record["activity_retained"]["spent_seconds"] == 1970.
    assert record["state_restore"] == "exact" and record["baseline_queries_preserved"] == 1
    assert sha(source/"last.pt") == source_digest
    assert migrate(cfg, target) == record
    actual["state"]["updates"] += 1
    torch.save(actual, target/"last.pt")
    with pytest.raises(ValueError, match="progressed"):
        migrate(cfg, target)


@pytest.mark.parametrize("change", ["inference", "live", "sha", "protocol"])
def test_migration_rejects_unsafe_sources(tmp_path, change):
    cfg, source, target, _ = migration_fixture(tmp_path)
    if change == "inference":
        p = target/"manifest.json"
        m = json.loads(p.read_text())
        m["code"][INFERENCE_FILES[0]] = "changed"
        p.write_text(json.dumps(m))
    elif change == "live":
        p = source/"activity.json"
        m = json.loads(p.read_text())
        m["worker"] = {"pid": 123}
        p.write_text(json.dumps(m))
    elif change == "sha":
        cfg["continuation"]["checkpoint_sha256"] = "wrong"
    else:
        cfg["training"]["learning_rate"] = .01
    with pytest.raises(ValueError):
        migrate(cfg, target)


def test_projection_only_selects_observed_positions_but_retains_native_width(codec, monkeypatch):
    import gofa_f.short_training as training
    model = tiny()
    records, graphs = inputs(codec)
    tokens = codec.batch(records)
    contexts = model.encode_origins(graphs, codec)
    original, widths = training.F.linear, []
    def linear(x, weight, bias=None):
        if weight.shape[0] >= 10000:
            widths.append((x.shape[0], weight.shape[0]))
        return original(x, weight, bias)
    monkeypatch.setattr(training.F, "linear", linear)
    result = compact_forward(model, codec, tokens, contexts)
    assert widths == [(128, model.native.lm_head.out_features)]
    assert result[0].shape[1] == 10000
    assert codec.nan_id not in codec.finite_set


def test_value_comparison_accepts_exact_bf16_upcast_but_not_real_errors():
    from gofa_f.short_training_checks import assert_value_close
    expected = torch.tensor([1., -2., 0.], dtype=torch.bfloat16)
    actual = expected.float()
    assert_value_close(actual, expected, .002, "same_values")
    with pytest.raises(AssertionError, match="real_error"):
        assert_value_close(actual+.01, expected, .002, "real_error")


def test_cpu_bf16_optimization_acceptance_checks_values_across_loss_dtypes(codec):
    from gofa_f.short_training_checks import optimization_checks
    model = tiny()
    model.native.to(dtype=torch.bfloat16)
    with torch.no_grad():
        for layer in model.adapter.exchanges:
            layer.up.weight.normal_(std=.001)
    data = source(9)
    queries = [dict(source="pems04", variable=v, origin=o, length=length, horizon=horizon, chunk=0)
               for v, o, length, horizon in [(0, 500, 10, 1), (0, 500, 48, 24), (1, 600, 48, 24)]]
    proof = optimization_checks(model, codec, {"pems04": data}, queries, .002)
    assert proof["passed"]
    assert all(v["violations"] == 0 for v in proof["diagnostics"].values())


def test_cpu_proof_reuse_rejects_changed_tensor_execution(tmp_path):
    cfg, source, target, _ = migration_fixture(tmp_path)
    before = json.loads((source/"manifest.json").read_text())
    before["configuration"] = deepcopy(cfg)
    before["code"]["short_training.py"] = "same_tensor_execution"
    for name in ("short_migration.py", "short_checks.py", "short_training_checks.py"):
        before["code"][name] = "old_diagnostics"
    after = deepcopy(before)
    cfg["protocol"] = "v4"
    after["configuration"] = cfg
    after["code"]["short_training_checks.py"] = "value_comparison_and_diagnostics_only"
    (source/"manifest.json").write_text(json.dumps(before))
    (target/"manifest.json").write_text(json.dumps(after))
    (source/"cpu_preflight.json").write_text(json.dumps(dict(passed=True,
        manifest_sha256=sha(source/"manifest.json"), alphabet={"alphabet_sha256": "same"},
        real_weights={"passed": True}, continued_training_optimizations={"passed": True})))
    (target/"migration.json").write_text('{"failed_acceptance_repair": true}')
    torch.save({"preserved": True}, target/"last.pt")
    proof = reusable_cpu_proof(cfg, target)
    assert proof and not proof["cpu_proof_reuse"]["gpu_acceptance_reused"]
    after["code"]["short_training.py"] = "changed_tensor_execution"
    (target/"manifest.json").write_text(json.dumps(after))
    assert reusable_cpu_proof(cfg, target) is None


@pytest.mark.parametrize("corrupt", [False, True])
def test_failed_acceptance_migration_retains_all_costs_and_checks_ancestor(tmp_path, corrupt):
    cfg, _, failed, _ = migration_fixture(tmp_path)
    migrate(cfg, failed)
    def write(root, name, value):
        (root/name).write_text(json.dumps(value))
    write(failed, "launcher.json", {"state": "failed"})
    write(failed, "status.json", {"state": "failed"})
    write(failed, "failures.json", {"phase": "gpu_mechanisms"})
    write(failed, "cpu_preflight.json", {"passed": True, "alphabet": {"alphabet_sha256": "alphabet"}})
    ledger = json.loads((failed/"activity.json").read_text())
    ledger["spent_seconds"] += 10.
    ledger["waiting_seconds"] += 110.
    write(failed, "activity.json", ledger)
    if corrupt:
        payload = torch.load(failed/"last.pt", weights_only=False)
        payload["gradients"]["w"] += 1
        torch.save(payload, failed/"last.pt")
    target = tmp_path/"repaired"
    target.mkdir()
    following = deepcopy(cfg)
    following["protocol"] = "v4"
    following["continuation"] = dict(source_output=str(failed), checkpoint_sha256=sha(failed/"last.pt"))
    manifest = json.loads((failed/"manifest.json").read_text())
    manifest["configuration"] = following
    write(target, "manifest.json", manifest)
    (target/"pools").mkdir()
    (target/"pools/queries.json").write_text((failed/"pools/queries.json").read_text())
    if corrupt:
        with pytest.raises(ValueError, match="tensor"):
            migrate(following, target)
    else:
        record = migrate(following, target)
        assert record["failed_acceptance_repair"]
        assert record["activity_retained"]["spent_seconds"] == 1980.
        assert record["activity_retained"]["waiting_seconds"] == 190.


def test_projection_diagnostic_costs_are_migrated_once_with_training_state_unchanged(tmp_path, monkeypatch):
    import gofa_f.short_migration as migration
    cfg, _, target, original = migration_fixture(tmp_path)
    diagnostic = {"activity": dict(spent_seconds=8., waiting_seconds=30., resource_pause_count=1,
                                   peak_group_mib=700), "output": "verified_probe"}
    monkeypatch.setattr(migration, "projection_diagnostic", lambda cfg: diagnostic)
    result = migrate(cfg, target)
    ledger = json.loads((target/"activity.json").read_text())
    assert ledger["spent_seconds"] == 1978. and ledger["waiting_seconds"] == 110.
    assert ledger["resource_pause_count"] == 1 and ledger["peak_group_mib"] == 17246
    assert result["prior_projection_diagnostic"] == diagnostic
    assert migrate(cfg, target) == result
    assert json.loads((target/"activity.json").read_text()) == ledger
    actual = torch.load(target/"last.pt", weights_only=False)
    for key in ("adapter", "optimizer", "gradients", "rng", "parameter_version"):
        assert_equal(original[key], actual[key])


@pytest.mark.parametrize("change", [None, "live", "head", "alphabet", "gpu", "result"])
def test_projection_diagnostic_requires_stopped_bound_rejection(tmp_path, monkeypatch, change):
    import gofa_f.short_migration as migration
    probe, source, weights = (tmp_path/name for name in ("probe", "source", "weights"))
    for root in (probe, source, weights):
        root.mkdir()
    def write(root, name, value):
        (root/name).write_text(json.dumps(value))
    probe.with_suffix(".launch.json").write_text(json.dumps(
        dict(pid=12345, start_ticks="100", code_revision="fixed_probe_code")))
    monkeypatch.setattr(migration, "process_identity", lambda pid: "100" if change == "live" else None)
    write(probe, "activity.json", dict(worker=None, spent_seconds=5., waiting_seconds=30.))
    write(probe, "launcher.json", dict(state="complete"))
    write(probe, "status.json", dict(state="complete"))
    write(probe, "launcher_manifest.json", dict(worker_mode="projection_probe", admission_policy="idle",
        resource_check_seconds=10, admission_samples=3, memory_cap_gib=40, torch_cap_gib=38))
    report = dict(candidate_passed=False, tolerance=.002, finite_tokens=10000, native_vocabulary=42001,
        candidate="finite_plus_one_zero_work_row", weight_shard_sha256="head", alphabet_sha256="alphabet",
        cases={str(n): {"tail_one_10001": {"logits": {"violations": 1}}} for n in (1, 25, 48)})
    if change in ("head", "alphabet"):
        report["weight_shard_sha256" if change == "head" else "alphabet_sha256"] = "wrong"
    if change == "result":
        for case in report["cases"].values():
            case["tail_one_10001"]["logits"]["violations"] = 0
    write(probe, "projection_probe.json", report)
    write(probe, "gpu_identity.json", dict(uuid="other" if change == "gpu" else "gpu"))
    write(source, "gpu_identity.json", dict(uuid="gpu"))
    write(source, "cpu_preflight.json", dict(alphabet=dict(alphabet_sha256="alphabet")))
    write(weights, "model.safetensors.index.json", dict(weight_map={"lm_head.weight": "head.safetensors"}))
    write(weights, "snapshot.json", dict(files={"head.safetensors": {"sha256": "head"}}))
    cfg = dict(continuation=dict(prior_projection_diagnostic=str(probe), source_output=str(source)),
               weights=dict(directory=str(weights)))
    if change:
        with pytest.raises(ValueError):
            migration.projection_diagnostic(cfg)
    else:
        result = migration.projection_diagnostic(cfg)
        assert result["activity"]["spent_seconds"] == 5.
        assert result["report_sha256"] == sha(probe/"projection_probe.json")
