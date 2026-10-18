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
from gofa_f.short_migration import INFERENCE_FILES, assert_equal, migrate
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
                   last_validated_update=0, microbatch=1), parameter_version=98,
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
