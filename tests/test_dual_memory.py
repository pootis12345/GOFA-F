from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from gofa_f.allnode_codec import AllNodeTokens
from gofa_f.dual_memory_data import memory_input, sample_subgraph, task_text
from gofa_f.dual_memory_model import DualMemoryGOFA, MemoryExchange
from gofa_f.generative_data import Query


def series(nodes=31):
    t = np.arange(400)[:, None]; v = np.arange(nodes)[None]
    values = (10+v*2+np.sin(t/7+v)).astype(np.float32)
    a = np.zeros((nodes, nodes), np.float32)
    for i in range(1, min(11, nodes)):
        a[0, i] = 1
        if i+10 < nodes: a[i, i+10] = 1
        if i+20 < nodes: a[i+10, i+20] = 1
    return SimpleNamespace(values=values, mask=np.ones_like(values, bool), external=a,
        ids=[str(i) for i in range(nodes)], segments=np.zeros(400, int))


def tokens(nodes, length):
    x = torch.arange(nodes*length).reshape(nodes, length)%128+3
    mask = torch.ones_like(x, dtype=torch.bool)
    return AllNodeTokens(x, mask, torch.full_like(x, -100), torch.full((nodes,), length-1),
        tuple(torch.arange(length) for _ in range(nodes)), tuple(str(i) for i in range(nodes)), x.clone())


def tiny(depth):
    torch.manual_seed(42)
    config = LlamaConfig(hidden_size=16, intermediate_size=24, num_hidden_layers=8,
        num_attention_heads=2, num_key_value_heads=2, vocab_size=160,
        max_position_embeddings=2048, attention_dropout=0.)
    config._attn_implementation = 'sdpa'
    return DualMemoryGOFA(LlamaForCausalLM(config).eval(), depth=depth, width=8,
        weight_binding='synthetic-frozen', stable_fp32=True)


def activate(model):
    with torch.no_grad():
        for e in model.adapter.exchanges:
            e.gate.fill_(.15); e.ff_gate.fill_(.1)
        model.adapter.head[-1].weight.normal_(std=.03)


def test_dense_taglas_whole_hop_rng_and_induced_graph_reference():
    d = series(80); rng = np.random.default_rng(19)
    d.external = (rng.random((80, 80)) < .15).astype(np.float32)
    np.fill_diagonal(d.external, 0)
    before = torch.get_rng_state().clone()
    nodes, a, audit = sample_subgraph(d.external, 0, 52)
    # Independent literal single-root dense TAGLAS routine, isolated RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(52)
        receiver, sender = torch.from_numpy(d.external).nonzero(as_tuple=True)
        subsets = [torch.tensor([0])]
        for _ in range(3):
            mask = torch.zeros(80, dtype=torch.bool); mask[subsets[-1]] = True
            fringe = sender[mask[receiver]]
            if len(fringe) > 10: fringe = fringe[torch.randperm(len(fringe))[:10]]
            subsets.append(fringe)
        expected = torch.cat(subsets).unique().tolist()
    assert nodes == [0]+[v for v in expected if v != 0]
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(a, torch.from_numpy(d.external[nodes][:, nodes] > 0))
    assert max(audit['whole_fringe_counts']) <= 10 and len(nodes) <= 31


def test_real31_three_hops_direction_parent_paths_and_small_graphs():
    d = series(); nodes, _, audit = sample_subgraph(d.external, 0, 42)
    assert len(nodes) == 31 and audit['hop'].count(3) == 10
    for n, parent in zip(nodes, audit['parent_toward_root']):
        if parent is not None: assert d.external[parent, n] > 0
    assert len(sample_subgraph(series(6).external, 0, 42)[0]) == 6
    isolated = sample_subgraph(np.zeros((4, 4)), 0, 42)
    assert isolated[0] == [0] and isolated[2]['whole_fringe_counts'] == [0, 0, 0]
    assert set(sample_subgraph(d.external, 0, 42, allowed=list(range(8)))[0]) <= set(range(8))
    with pytest.raises(ValueError, match='three'): sample_subgraph(d.external, 0, 42, hops=2)


def test_future_independence_missing_negative_constant_and_metadata():
    d = series(); q = Query('pems04_flow', 0, 200, 48, 24)
    d.values[152:200, 1] = -3.; d.mask[152:200, 2] = False
    a = memory_input(d, q); d.values[200:] = 1e12; d.mask[200:] = False
    b = memory_input(d, q)
    assert a.identity == b.identity and torch.equal(a.edges, b.edges)
    assert a.extent[1] == 1 and a.last[1] == -3 and not a.present[2]
    assert not a.adjacency[2].any() and not a.adjacency[:, 2].any()
    text = task_text('speed', 'unknown (publisher unit unverified)', 300, 12, 12)
    assert 'unknown' in text and '12' in text


@pytest.mark.parametrize('depth', [3, 6])
@pytest.mark.parametrize('length,horizon', [(12, 12), (48, 24)])
def test_nonzero_full_cached_memory_output_and_all_trainable_gradients(depth, length, horizon):
    d = series(); q = Query('pems04_flow', 0, 200, length, horizon)
    item = memory_input(d, q); ts = tokens(31, length); model = tiny(depth); activate(model)
    metadata = torch.randn(16); prefix = model.frozen_history(ts)
    count = model.prefix_executions
    full, memories = model.enhanced(ts, item, metadata, prefix=prefix)
    full.square().mean().backward()
    expected = {k: None if p.grad is None else p.grad.clone() for k, p in model.adapter.named_parameters()}
    assert all(p.grad is None for p in model.native.parameters())
    assert all(v is not None and torch.isfinite(v).all() for v in expected.values())
    model.zero_grad(set_to_none=True)
    cached, cached_memories = model.enhanced(ts, item, metadata, prefix=prefix, optimized=True)
    torch.testing.assert_close(cached, full, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(cached_memories, memories, rtol=1e-6, atol=1e-6)
    cached.square().mean().backward()
    for k, p in model.adapter.named_parameters():
        torch.testing.assert_close(p.grad, expected[k], rtol=1e-6, atol=1e-6)
    assert model.prefix_executions == count
    assert all(p.grad.abs().sum() > 0 for n, p in model.adapter.named_parameters() if '.gate' in n)


def test_nonzero_checkpointed_tail_recompute_gradients_and_third_hop_reach():
    d = series(); item = memory_input(d, Query('pems04_flow', 0, 200, 12, 12))
    item.normalized.requires_grad_(); model = tiny(3); activate(model)
    ts = tokens(31, 12); meta = torch.ones(16); prefix = model.frozen_history(ts)
    out, _ = model.enhanced(ts, item, meta, prefix=prefix, optimized=True)
    out[0].sum().backward()
    assert item.normalized.grad[21:].abs().sum() > 0
    expected = {k: p.grad.clone() for k, p in model.adapter.named_parameters()}
    model.zero_grad(set_to_none=True)
    out2, _ = model.enhanced(ts, item, meta, prefix=prefix, optimized=True, checkpoint_tail=True)
    out2[0].sum().backward()
    torch.testing.assert_close(out2, out, rtol=0, atol=0)
    for k, p in model.adapter.named_parameters():
        torch.testing.assert_close(p.grad, expected[k], rtol=1e-6, atol=1e-6)


def test_native_isolation_no_hooks_rng_or_parameter_changes_and_cache_binding():
    model = tiny(3); ts = tokens(1, 12); before = torch.get_rng_state().clone()
    native = model.native_view(); reference = native(ts).logits.detach().clone()
    item = memory_input(series(), Query('pems04_flow', 0, 200, 12, 12))
    activate(model); graph_ts = tokens(31, 12)
    # activate consumes RNG; graph execution itself must not.
    rng = torch.get_rng_state().clone(); versions = [p._version for p in model.native.parameters()]
    value, _ = model.enhanced(graph_ts, item, torch.ones(16), optimized=True)
    value.mean().backward()
    assert torch.equal(rng, torch.get_rng_state()) and versions == [p._version for p in model.native.parameters()]
    assert torch.equal(native(ts).logits, reference)
    assert not any(layer._forward_pre_hooks for layer in model.native.model.layers)
    full = native(ts).logits; a = native(ts.first(5), use_cache=True)
    next_part = ts.first(5).append(ts.input_ids[:, 5:])
    b = native(next_part, cache=a.past_key_values, use_cache=True)
    torch.testing.assert_close(torch.cat((a.logits, b.logits), 1), full, rtol=1e-6, atol=1e-6)
    prefix = model.frozen_history(graph_ts); changed = deepcopy(graph_ts); changed.input_ids[0, 0] += 1
    with pytest.raises(ValueError, match='binding'): model.enhanced(changed, item, torch.ones(16), prefix=prefix)
    with pytest.raises(ValueError, match='contexts'): native(ts, contexts=['graph'])


def test_zero_graph_gates_identity_not_a_false_native_forecast_equivalence_claim():
    item = memory_input(series(), Query('pems04_flow', 0, 200, 12, 12))
    block = MemoryExchange(16, 8); m = torch.randn(31, 4, 16, requires_grad=True)
    assert torch.equal(block(m, item), m)
    block(m, item).square().mean().backward()
    assert block.gate.grad.abs() > 0 and block.ff_gate.grad.abs() > 0


def test_adapter_optimizer_reload_exact_next_update():
    model = tiny(3); activate(model); ts = tokens(31, 12)
    item = memory_input(series(), Query('pems04_flow', 0, 200, 12, 12)); prefix = model.frozen_history(ts)
    opt = torch.optim.AdamW(model.adapter.parameters(), lr=1e-4)
    def update():
        opt.zero_grad(); x, _ = model.enhanced(ts, item, torch.ones(16), prefix=prefix, optimized=True)
        loss = x.square().mean(); loss.backward(); opt.step(); return loss.detach().clone()
    update(); state = deepcopy(model.trainable_state()); optimizer = deepcopy(opt.state_dict())
    expected = update(); weights = model.trainable_state()
    model.load_trainable(state); opt.load_state_dict(optimizer)
    assert torch.equal(update(), expected)
    assert all(torch.equal(model.trainable_state()[k], v) for k, v in weights.items())
