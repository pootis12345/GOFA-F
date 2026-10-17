from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

from gofa_f.allnode_codec import AllNodeCodec, OFFICIAL_PROMPT
from gofa_f.allnode_forecast import forecast_batch
from gofa_f.allnode_graph import AllNodeAdapter, AllNodeExchange, build_graph
from gofa_f.allnode_model import AllNodeGOFA
from gofa_f.generative_data import Query
from gofa_f.generative_graph import origin_graph
from gofa_f.generative_model import completion_loss


@pytest.fixture(scope="module")
def codec():
    root = Path("/home/dcg/dcg_data/GOFA-F/data/weights/chattime-7b-20baa88")
    if not root.exists():
        pytest.skip("Official tokenizer not installed on this CPU host")
    return AllNodeCodec(LlamaTokenizer.from_pretrained(root, local_files_only=True))


def tiny():
    torch.manual_seed(42)
    config = LlamaConfig(hidden_size=16, intermediate_size=24, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=2, vocab_size=42001,
        max_position_embeddings=2048, attention_dropout=0.)
    config._attn_implementation = "sdpa"
    return AllNodeGOFA(LlamaForCausalLM(config).eval())


def source(nodes=12):
    rng = np.random.default_rng(42)
    values = rng.normal(size=(1000, nodes)).astype(np.float32)
    if nodes > 1:
        values[:, 1] += values[:, 0]
    mask = np.ones_like(values, bool)
    if nodes > 2:
        mask[490:493, 2] = False
    roads = np.zeros((nodes, nodes), np.float32)
    roads[0, 1:] = 1
    ids = [f"v{i:03d}" for i in range(nodes)]
    result = SimpleNamespace(values=values, mask=mask, external=roads, ids=ids)
    result.graph = lambda q: origin_graph(values[q.origin-q.length:q.origin],
        mask[q.origin-q.length:q.origin], q.variable, roads, ids)
    return result


def query(length=12, chunk=0):
    return Query("pems04", 0, 500, length, 48, chunk)


@pytest.mark.parametrize("length", [10, 12, 14, 48, 192, 288])
def test_codec_official_prompt_positions_and_isolated_scaling(codec, length):
    row = np.linspace(10., 20., length)
    first = codec.record(row)
    codec.record(np.linspace(1e5, 1e6, length))
    pred, valid = codec.parse("###0.0000###", *first["scaling"], first["last"], 1)
    np.testing.assert_array_equal(pred, [15.])
    assert valid.all()
    assert len(first["value_positions"]) == length
    assert all(first["ids"][i] in codec.numeric_ids for i in first["value_positions"])
    prompt = codec.tokenizer.decode(first["ids"][:first["prompt_end"]+1], skip_special_tokens=False)
    assert "### Response:" in prompt
    assert "\n\n### Response:" not in OFFICIAL_PROMPT
    rolled = codec.record(np.r_[row, np.full(16, 1000.)])
    assert len(rolled["value_positions"]) == length+16
    assert any(first["ids"][a] != rolled["ids"][b]
               for a, b in zip(first["value_positions"], rolled["value_positions"]))
    assert first["scaling"] != rolled["scaling"]


def test_missing_zero_constant_parser_policy(codec):
    row = codec.record([3., 3., np.nan, 3.])
    assert row["scaling"][0] == 1.
    p, valid, why = codec.parse_with_reasons(
        "###0.0000### ###bad### ###NaN### ###inf###", 1., 0., 7., 6)
    np.testing.assert_array_equal(p, [.5, 7., 7., 7., 7., 7.])
    np.testing.assert_array_equal(valid, [True, False, False, False, False, False])
    assert why[-1] == "missing" and why[1] == "invalid"
    model = tiny()
    p, valid, why = forecast_batch(model, codec, [np.full(12, np.nan)], 24)
    assert (p == 0).all() and not valid.any() and set(why[0]) == {"empty_history"}


@pytest.mark.parametrize("nodes", [1, 3, 9, 12])
def test_sampling_incoming_deterministic_and_maximum(nodes):
    data, q = source(nodes), query()
    a, b = build_graph(data, q), build_graph(data, q)
    assert a.identity == b.identity
    assert len(a.variable_ids) == min(nodes, 9)
    assert a.variable_ids[0] == "v000"
    np.testing.assert_allclose(a.graph.base.sum(-1), 1., atol=1e-6)
    original = a.identity
    data.values[500:] = 1e8
    assert build_graph(data, q).identity == original


def test_empty_timestep_no_nan_forward_or_backward():
    data = source(3)
    data.mask[495] = False
    graph = build_graph(data, query()).graph
    exchange = AllNodeExchange(16)
    with torch.no_grad():
        exchange.up.weight.normal_(std=.01)
    hidden = torch.randn(3, 12, 16, requires_grad=True)
    delta = exchange(hidden, graph)
    assert torch.isfinite(delta).all()
    assert (delta[:, 7] == 0).all()
    delta.square().sum().backward()
    assert torch.isfinite(hidden.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in exchange.parameters())


def test_exchange_permutation_equivariance():
    data = source(4)
    g = build_graph(data, query()).graph
    exchange = AllNodeExchange(16)
    with torch.no_grad():
        exchange.up.weight.normal_(std=.01)
    hidden = torch.randn(4, 12, 16)
    original = exchange(hidden, g)
    p = torch.tensor([2, 0, 3, 1])
    changed = deepcopy(g)
    changed.history, changed.observed = g.history[:, p], g.observed[:, p]
    changed.prior, changed.base = g.prior[p][:, p], g.base[p][:, p]
    changed.edges = g.edges[p][:, p]
    torch.testing.assert_close(exchange(hidden[p], changed), original[p], rtol=1e-6, atol=1e-6)


def test_initial_full_node_execution_and_first_gradient(codec):
    data, model = source(), tiny()
    q = query()
    tokens = codec.batch([codec.record(data.values[488:500, 0], data.values[500:502, 0])])
    context = model.encode_origins([build_graph(data, q)], codec)
    assert context[0].nodes == 9 and model.origin_executions[-1]["layers"] == 4
    torch.testing.assert_close(model(tokens, contexts=context).logits, model(tokens).logits, rtol=0, atol=0)
    loss = completion_loss(model(tokens, contexts=context).logits, tokens.labels)[0]
    loss.backward()
    assert all(p.grad is None for p in model.native.parameters())
    assert all(module.up.weight.grad.abs().sum() > 0 for module in model.adapter.exchanges)
    assert all(module.down.weight.grad.abs().sum() == 0 for module in model.adapter.exchanges)


@pytest.mark.parametrize("offset", [0, 16, 32])
def test_nonzero_full_prefix_incremental_gradient_rolling_and_kv(codec, offset):
    data, model, q = source(3), tiny(), query()
    with torch.no_grad():
        for module in model.adapter.exchanges:
            module.up.weight.normal_(std=.01)
    graph = build_graph(data, q)
    history = np.r_[data.values[488:500, 0], np.full(offset, 1e4)]
    tokens = codec.batch([codec.record(history, data.values[500:502, 0])])
    context = model.encode_origins([graph], codec)
    output = model(tokens, contexts=context).logits
    completion_loss(output, tokens.labels)[0].backward()
    expected = {n: p.grad.clone() for n, p in model.adapter.named_parameters()}
    model.zero_grad()
    context = model.encode_origins([graph], codec)
    cached = model(tokens, contexts=context, prefix=model.prefix(tokens)).logits
    torch.testing.assert_close(cached, output, rtol=1e-6, atol=1e-6)
    completion_loss(cached, tokens.labels)[0].backward()
    for name, p in model.adapter.named_parameters():
        torch.testing.assert_close(p.grad, expected[name], rtol=1e-6, atol=1e-6)
    model.zero_grad()
    context = model.encode_origins([graph], codec)
    cut = int(tokens.prompt_ends[0])+1
    part = tokens.first(cut)
    out = model(part, contexts=context, use_cache=True)
    pieces, kv = [out.logits], out.past_key_values
    for index in range(cut, tokens.input_ids.shape[1]):
        part = part.append(tokens.input_ids[:, index:index+1])
        out = model(part, contexts=context, cache=kv, use_cache=True)
        kv = out.past_key_values
        pieces.append(out.logits)
    incremental = torch.cat(pieces, 1)
    torch.testing.assert_close(incremental, output, rtol=1e-6, atol=1e-6)
    completion_loss(incremental, tokens.labels)[0].backward()
    for name, p in model.adapter.named_parameters():
        torch.testing.assert_close(p.grad, expected[name], rtol=1e-6, atol=1e-6)
    bad = part.append(torch.tensor([[10]]))
    bad.full_ids = bad.full_ids.clone()
    bad.full_ids[0, 1] += 1
    with pytest.raises(ValueError, match="lineage"):
        model(bad, contexts=context, cache=kv, use_cache=True)
    bad = part.append(torch.tensor([[10]]))
    bad.value_positions = tuple(p+1 for p in bad.value_positions)
    with pytest.raises(ValueError, match="mapping"):
        model(bad, contexts=context, cache=kv, use_cache=True)
    model.mark_updated()
    with pytest.raises(ValueError, match="Stale"):
        model(part, contexts=context, cache=kv, use_cache=True)


def test_origin_isolation_and_incremental_greedy(codec):
    data, model = source(3), tiny()
    graph = build_graph(data, query())
    with torch.no_grad():
        for module in model.adapter.exchanges:
            module.up.weight.normal_(std=.01)
        context = model.encode_origins([graph], codec)
        history = data.values[488:500, 0]
        ta, tb = [], []
        a = forecast_batch(model, codec, [history], 24, context, cached=True, trace=ta.append)
        b = forecast_batch(model, codec, [history], 24, context, cached=False, trace=tb.append)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        assert ta == tb
        before = tuple(d.clone() for d in context[0].deltas)
        data.values[500:] *= -1000
        after = model.encode_origins([build_graph(data, query())], codec)
        for x, y in zip(before, after[0].deltas):
            torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_parameters_and_pinned_protocol():
    assert sum(p.numel() for p in AllNodeAdapter(4096).parameters()) == 1060998
    from gofa_f.allnode_stage import config_check
    config = yaml.safe_load(Path("configs/stage18_allnode_chattime.yaml").read_text())
    config_check(config)
    for section, key, value in [("architecture", "max_subgraph_nodes", 8),
        ("architecture", "origin_frozen", False), ("data", "target_evaluation_enabled", True),
        ("resources", "activity_limit_seconds", 28800), ("architecture", "graph_layers", [29, 30])]:
        bad = deepcopy(config)
        bad[section][key] = value
        with pytest.raises(ValueError):
            config_check(bad)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rmsnorm_fixed_rows_chunk_equivalence_and_native_bypass(dtype):
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
    from gofa_f.allnode_kernel import install_norm_tiles
    torch.manual_seed(42)
    norm = LlamaRMSNorm(4096).to(dtype=dtype)
    native = deepcopy(norm)
    enabled = [True]
    install_norm_tiles(norm, lambda: enabled[0], stable_fp32=True)
    x = torch.randn(1, 106, 4096, dtype=dtype, requires_grad=True)
    full = norm(x)
    incremental = torch.cat([norm(x[:, :103]), *[norm(x[:, i:i+1]) for i in range(103, 106)]], 1)
    torch.testing.assert_close(full, incremental, rtol=0, atol=0)
    full.float().square().mean().backward()
    expected = x.grad.clone()
    x.grad = None
    incremental.float().square().mean().backward()
    torch.testing.assert_close(x.grad, expected, rtol=0, atol=0)
    enabled[0] = False
    torch.testing.assert_close(norm(x), native(x), rtol=0, atol=0)


def test_best_reference_patience_and_idempotent_commit():
    from gofa_f.allnode_stage import record_validation
    state = dict(best_score=None, early_reference=None, bad_validations=0, last_validated_update=0)
    assert record_validation(state, 1., 250)
    assert record_validation(state, .9999, 500)  # save strict best; don't reset patience
    assert state["early_reference"] == 1. and state["bad_validations"] == 1
    saved = dict(state)
    assert not record_validation(state, .9999, 500) and state == saved
    for update in (750, 1000, 1250, 1500):
        record_validation(state, .9999, update)
    assert state["bad_validations"] == 5
    assert record_validation(state, .9989, 1750)
    assert state["bad_validations"] == 0 and state["early_reference"] == .9989


def test_native_analysis_bypass_unmodified(codec):
    model, history = tiny(), np.arange(12.)
    a = model.analyze(codec, history, "Is the sequence increasing?", max_new_tokens=2)
    with torch.no_grad():
        for exchange in model.adapter.exchanges:
            exchange.up.weight.normal_()
    model.mark_updated()
    assert model.analyze(codec, history, "Is the sequence increasing?", max_new_tokens=2) == a


def test_partial_checkpoint_state_and_next_materialized_update(tmp_path, codec):
    from gofa_f.allnode_stage import save_checkpoint, restore_checkpoint
    from gofa_f.allnode_progress import Progress
    data, model = source(3), tiny()
    optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=2e-4)
    tokens = codec.batch([codec.record(data.values[488:500, 0], data.values[500:502, 0])])
    context = model.encode_origins([build_graph(data, query())], codec)
    completion_loss(model(tokens, contexts=context).logits, tokens.labels)[0].backward()
    grads = {n: p.grad.clone() for n, p in model.adapter.named_parameters()}
    state = dict(updates=0, pending=1, pending_tokens=4, pending_loss=3.,
        early_reference=.99, best_score=.98, bad_validations=2, last_validated_update=500)
    save_checkpoint(tmp_path/"last.pt", model, optimizer, state, Progress(tmp_path), "bound")
    expected_draw = torch.rand(3)
    optimizer.step()
    expected = model.trainable_state()
    restored, _ = restore_checkpoint(tmp_path/"last.pt", model, optimizer, "bound")
    assert restored == state
    torch.testing.assert_close(torch.rand(3), expected_draw, rtol=0, atol=0)
    for name, p in model.adapter.named_parameters():
        torch.testing.assert_close(p.grad, grads[name], rtol=0, atol=0)
    optimizer.step()
    for name, value in model.trainable_state().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


@pytest.mark.parametrize("pause", [False, True, "unknown"])
@pytest.mark.parametrize("query_failure", [False, True])
def test_stage18_idle_admission_coexistence_pause_and_resume(tmp_path, monkeypatch, pause, query_failure):
    import json
    import gofa_f.allnode_launcher as supervisor
    config = tmp_path/"config.yaml"
    config.write_text(Path("configs/stage18_allnode_chattime.yaml").read_text())
    output = tmp_path/"run"
    clock, launches, signals = [0.], [], []
    gpu = dict(uuid="GPU-test", pids=[], utilization=0, compute_mode="Default", used_mib=0,
               process_used_mib={})
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(supervisor.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0]+seconds))
    monkeypatch.setattr(supervisor, "process_identity", lambda _: "ticks")
    monkeypatch.setattr(supervisor, "own_gpu_pids", lambda _: {800})
    def probe():
        if query_failure and clock[0] == 50:
            raise RuntimeError("GPU query failed")
        if launches and clock[0] <= launches[-1]+2:
            own = None if pause == "unknown" and len(launches) == 1 else (
                41000 if pause and len(launches) == 1 else 26000)
            return {**gpu, "pids": [800, 999], "used_mib": 70000, "utilization": 100,
                    "process_used_mib": {800: own, 999: 44000}}
        if clock[0] < 40:
            return {**gpu, "pids": [999], "used_mib": 48547, "utilization": 100,
                    "process_used_mib": {999: 48547}}
        return gpu
    monkeypatch.setattr(supervisor, "query_gpu", probe)
    monkeypatch.setattr(supervisor.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    class Child:
        pid, returncode = 800, None
        def __init__(self, command, cwd, env, start_new_session):
            assert "gofa_f.allnode_stage" in command and start_new_session
            assert env["GOFA_STAGE18_ADMITTED"] == "1" and env["CUDA_VISIBLE_DEVICES"] == "GPU-test"
            launches.append(clock[0])
            self.start = clock[0]
            self.resource = bool(pause and len(launches) == 1)
        def poll(self):
            if self.returncode is None and clock[0] > self.start+2:
                self.returncode = 75 if self.resource else 0
                supervisor.atomic_json(output/"status.json", {
                    "state": "resource_paused" if self.resource else "complete"})
            return self.returncode
    monkeypatch.setattr(supervisor.subprocess, "Popen", Child)
    supervisor.supervise(output, config, tmp_path)
    ledger = json.loads((output/"activity.json").read_text())
    assert launches[0] == (80 if query_failure else 60) and ledger["worker"] is None
    assert len(launches) == (2 if pause else 1)
    assert len(signals) == (1 if pause else 0)
    assert all(pid == 800 for pid, _ in signals)
    assert ledger["waiting_seconds"] >= launches[0]
    assert ledger["spent_seconds"] == 3*len(launches)
    if pause:
        assert launches[1] >= launches[0]+33 and ledger["resource_pause_count"] == 1
