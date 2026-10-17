import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from gofa_f.generative_data import relative_score, selection_score
from gofa_f.generative_graph import HistoryEncoder, origin_graph
from gofa_f.generative_model import GenerativeGOFA, QueryTokens, completion_loss
from gofa_f.generative_progress import Progress, render


def graph(nodes=3, length=12, target=0):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(length, nodes))
    return origin_graph(x, np.ones_like(x, bool), target)


def tiny():
    torch.manual_seed(42)
    cfg = LlamaConfig(hidden_size=32, intermediate_size=48, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=64, max_position_embeddings=1024,
        attention_dropout=0.)
    cfg._attn_implementation = "sdpa"
    return GenerativeGOFA(LlamaForCausalLM(cfg).eval())


def tokens():
    ids = torch.tensor([[1, 4, 5, 6, 7, 8], [1, 10, 11, 12, 13, 14]])
    return QueryTokens(ids, torch.ones_like(ids, dtype=torch.bool),
        torch.tensor([[-100]*3+[6, 7, 8], [-100]*3+[12, 13, 14]]), torch.tensor([2, 2]))


def test_origin_graph_correlation_direction_normalization_and_missing():
    g = graph(nodes=12)
    np.testing.assert_allclose(g.base.sum(1), 1, rtol=1e-6, atol=1e-6)
    assert ((g.base > 0).sum(1) <= 9).all()
    x = np.ones((10, 3))
    mask = np.ones_like(x, bool)
    mask[:, 2] = False
    g = origin_graph(x, mask, 0, np.ones((3, 3)))
    assert (g.prior[:, 2] == 0).all()
    assert torch.isfinite(HistoryEncoder()(g)).all()
    assert (HistoryEncoder()(g)[2] == 0).all()


@pytest.mark.parametrize("nodes,length", [(1, 10), (3, 12), (7, 48), (11, 192), (16, 288)])
def test_shared_encoder_variable_shapes(nodes, length):
    assert HistoryEncoder()(graph(nodes, length)).shape == (nodes, 64)


def test_initial_native_exact_prefix_gradient_freeze_and_branch_learning():
    model, batch = tiny(), tokens()
    graphs = [graph(), graph(5, target=2)]
    native = model.native(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
        position_ids=model.positions(batch.attention_mask, 6), use_cache=False).logits
    output = model(batch, graphs).logits
    torch.testing.assert_close(output, native, rtol=0, atol=0)
    prefix = model.prefix(batch)
    torch.testing.assert_close(model(batch, graphs, prefix=prefix).logits, output, rtol=1e-6, atol=1e-6)
    optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=2e-4)
    for _ in range(3):
        optimizer.zero_grad()
        completion_loss(model(batch, graphs).logits, batch.labels)[0].backward()
        optimizer.step()
        model.mark_updated()
    assert all(p.grad is None for p in model.native.parameters())
    assert all(any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
        for m in [model.adapter.encoder, *model.adapter.gnns, *model.adapter.interactions])
    optimizer.zero_grad()
    loss = completion_loss(model(batch, graphs).logits, batch.labels)[0]
    loss.backward()
    gradients = {n: p.grad.clone() for n, p in model.adapter.named_parameters()}
    optimizer.zero_grad()
    completion_loss(model(batch, graphs, prefix=prefix).logits, batch.labels)[0].backward()
    for n, p in model.adapter.named_parameters():
        torch.testing.assert_close(p.grad, gradients[n], rtol=1e-6, atol=1e-6)


def test_incremental_kv_logits_gradients_and_stale_rejection():
    model, batch = tiny(), tokens()
    graphs = [graph(), graph(5, target=2)]
    with torch.no_grad():
        for module in model.adapter.interactions:
            module.up.weight.normal_(std=.01)
    complete = model(batch, graphs).logits
    completion_loss(complete, batch.labels)[0].backward()
    gradients = {n: p.grad.clone() for n, p in model.adapter.named_parameters()}
    model.zero_grad()
    cache, outputs = None, []
    for i in range(6):
        part = QueryTokens(batch.input_ids[:, i:i+1], batch.attention_mask[:, :i+1],
                           batch.labels[:, i:i+1], batch.prompt_ends)
        result = model(part, graphs, cache=cache, use_cache=True)
        cache = result.past_key_values
        outputs.append(result.logits)
    incremental = torch.cat(outputs, 1)
    torch.testing.assert_close(incremental, complete, rtol=1e-6, atol=1e-6)
    completion_loss(incremental, batch.labels)[0].backward()
    for n, p in model.adapter.named_parameters():
        torch.testing.assert_close(p.grad, gradients[n], rtol=1e-6, atol=1e-6)
    model.mark_updated()
    with pytest.raises(ValueError, match="Stale"):
        model(part, graphs, cache=cache, use_cache=True)


def test_batch_isolation_causal_future_and_readout_roles():
    model, batch = tiny(), tokens()
    with torch.no_grad():
        for m in model.adapter.interactions:
            m.up.weight.normal_(std=.01)
        graphs = [graph(), graph(5, target=2)]
        original = model(batch, graphs).logits
        changed = model(batch, [graphs[0], graph(9, target=4)]).logits
        torch.testing.assert_close(original[0], changed[0], rtol=0, atol=0)
        future = QueryTokens(batch.input_ids.clone(), batch.attention_mask, batch.labels, batch.prompt_ends)
        future.input_ids[:, -1] = 20
        torch.testing.assert_close(model(future, graphs).logits[:, :-1], original[:, :-1], rtol=0, atol=0)
        native = model(batch).logits
        torch.testing.assert_close(original[:, :2], native[:, :2], rtol=0, atol=0)


def test_bf16_prefix_keeps_native_dtype_and_tolerance():
    model, batch = tiny(), tokens()
    model.native.to(dtype=torch.bfloat16)
    graphs = [graph(), graph(5, target=2)]
    full = model(batch, graphs).logits
    cached = model(batch, graphs, prefix=model.prefix(batch)).logits
    assert full.dtype == cached.dtype == torch.bfloat16
    torch.testing.assert_close(cached, full, rtol=2e-3, atol=2e-3)
    with torch.no_grad():
        for layer in model.adapter.interactions:
            layer.up.weight.normal_(std=.001)
        full = model(batch, graphs).logits
        pieces, cache = [], None
        for index in range(6):
            part = QueryTokens(batch.input_ids[:, index:index+1], batch.attention_mask[:, :index+1],
                               batch.labels[:, index:index+1], batch.prompt_ends)
            output = model(part, graphs, cache=cache, use_cache=True)
            cache = output.past_key_values
            pieces.append(output.logits)
        torch.testing.assert_close(torch.cat(pieces, 1), full, rtol=2e-3, atol=2e-3)


def test_hierarchical_metric_and_near_zero_rule():
    eps = {"a": [1e-6, 1e-6], "b": [1e-6]}
    baseline = {"variable_mae": {"a:0": 0., "a:1": 2., "b:0": 100.}}
    current = {"variable_mae": {"a:0": 0., "a:1": 8., "b:0": 25.}}
    result = relative_score(current, baseline, eps)
    assert result["variable_ratios"]["a:0"] == 1
    assert result["source_scores"]["a"] == 2
    assert result["score"] == pytest.approx(np.sqrt(.5))
    current["variable_mae"]["a:0"] = 1e-3
    assert relative_score(current, baseline, eps)["variable_ratios"]["a:0"] == pytest.approx(1000)
    with pytest.raises(ValueError):
        selection_score([1], [{"source": "a", "variable": 0}], eps)


def test_epoch_and_overall_eta_persist_and_wait_exclusion(tmp_path):
    now = [0.]
    progress = Progress(tmp_path, clock=lambda: now[0])
    progress.tick("train")
    now[0] = 10
    state = progress.tick("waiting_for_idle", force=True, updates=1, measured_updates=1)
    assert state["training_eta_seconds"] == 39990
    now[0] = 1000
    progress.tick(force=True)
    assert progress.state["train_seconds"] == 10
    restored = Progress(tmp_path, dict(progress.state), clock=lambda: now[0])
    assert restored.state["updates"] == 1
    assert "epoch 1" in render({"progress": restored.tick(force=True)})
