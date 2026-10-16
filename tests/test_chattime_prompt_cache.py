from dataclasses import replace
import json

import pytest
import torch

from gofa_f.chattime_prompt_cache import FixedPromptPrefix, check_fixed_prompt
from test_chattime import tiny as tiny_fixture


@pytest.fixture
def tiny():
    return tiny_fixture.__wrapped__()


@pytest.mark.parametrize("node_chunk", [1, 3])
@pytest.mark.parametrize("padded", [False, True])
def test_fixed_prompt_predictions_and_gradients(tiny, node_chunk, padded):
    model, encoded, features, _ = tiny
    if padded:
        mask = encoded.attention_mask.clone()
        mask[0, -3:] = False
        encoded = replace(encoded, attention_mask=mask)
    with torch.no_grad():
        model.head.weight.fill_(.001)
        for graph in model.graph.values():
            graph.gate.fill_(.1)
    ids = encoded.input_ids[0, :3].tolist()
    assert check_fixed_prompt(model, encoded, features, ids)["passed"]
    expected = model.prefix(encoded, node_chunk=node_chunk)
    model.prompt_prefix = FixedPromptPrefix(ids, "test")
    first = model.prefix(encoded, node_chunk=node_chunk)
    pairs = tuple((k.clone(), v.clone()) for k, v in model.prompt_prefix.pairs)
    second = model.prefix(encoded, node_chunk=node_chunk)
    torch.testing.assert_close(first[0], expected[0], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(second[0], first[0], rtol=0, atol=0)
    assert model.prompt_prefix.builds == (0 if padded else 1)
    assert model.prompt_prefix.reused_rows == (0 if padded else 6)
    for old, current in zip(pairs, model.prompt_prefix.pairs):
        for a, b in zip(old, current):
            assert torch.equal(a, b)


@pytest.mark.parametrize("mismatch", ["ids", "mask"])
def test_mismatched_static_prefix_falls_back_without_shortening(tiny, mismatch):
    model, encoded, features, _ = tiny
    ids = encoded.input_ids.clone()
    mask = encoded.attention_mask.clone()
    if mismatch == "ids":
        ids[0, 0] = (ids[0, 0]+1) % model.native.config.vocab_size
    else:
        mask[0, 0] = False
    changed = replace(encoded, input_ids=ids, attention_mask=mask)
    expected = model(changed, features)
    model.prompt_prefix = FixedPromptPrefix(encoded.input_ids[0, :3].tolist(), "test")
    actual = model(changed, features)
    assert torch.equal(actual, expected) and model.prompt_prefix.fallback_rows == 3
    assert model.prompt_prefix.builds == 0


def test_static_cache_rebuilds_on_native_weight_mutation_and_is_not_checkpointed(tiny):
    model, encoded, _, _ = tiny
    model.prompt_prefix = FixedPromptPrefix(encoded.input_ids[0, :3].tolist(), "test")
    keys = set(model.state_dict())
    model.prefix(encoded)
    assert set(model.state_dict()) == keys
    with torch.no_grad():
        model.native.model.embed_tokens.weight.add_(.001)
    actual = model.prefix(encoded)
    expected = model._full_prefix(encoded)
    torch.testing.assert_close(actual[0], expected[0], rtol=1e-6, atol=1e-6)
    assert model.prompt_prefix.builds == 2


def test_static_cache_rejects_unfrozen_backbone(tiny):
    model, encoded, _, _ = tiny
    model.prompt_prefix = FixedPromptPrefix(encoded.input_ids[0, :3].tolist(), "test")
    model.native.model.embed_tokens.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        model.prefix(encoded)


def test_warmup_history_never_changes_shared_text_states(tiny):
    model, encoded, _, _ = tiny
    cache = FixedPromptPrefix(encoded.input_ids[0, :3].tolist(), "test")
    model.prompt_prefix = cache
    model.prefix(encoded)
    ids = encoded.input_ids.clone()
    ids[:, 3:] = (ids[:, 3:]+7) % model.native.config.vocab_size
    changed = replace(encoded, input_ids=ids)
    reused = model.prefix(changed)
    expected = model._full_prefix(changed)
    torch.testing.assert_close(reused[0], expected[0], rtol=1e-6, atol=1e-6)
    assert cache.builds == 1


def test_cached_attention_retains_full_native_query_shape_and_causal_dispatch(tiny, monkeypatch):
    from gofa_f import chattime_prompt_cache as module
    model, encoded, _, _ = tiny
    encoded = replace(encoded, attention_mask=torch.ones_like(encoded.attention_mask))
    observations = []
    original = module.F.scaled_dot_product_attention
    def attention(query, key, value, *args, **kwargs):
        observations.append((query.shape, key.shape, query.stride(), kwargs.get("attn_mask"), kwargs.get("is_causal")))
        return original(query, key, value, *args, **kwargs)
    monkeypatch.setattr(module.F, "scaled_dot_product_attention", attention)
    expected = model._full_prefix(encoded)
    native = observations.copy()
    observations.clear()
    model.prompt_prefix = FixedPromptPrefix(encoded.input_ids[0, :3].tolist(), "test")
    actual = model.prefix(encoded)
    layers = model.graph_layers[0]+1
    assert len(observations) == layers*2  # one warm-up, one history computation
    for reference, cached in zip(native, observations[layers:]):
        assert reference[:3] == cached[:3]
        assert cached[3] is None and cached[4] is True
    torch.testing.assert_close(actual[0], expected[0], rtol=1e-6, atol=1e-6)


def test_failed_cache_check_records_diagnostics_without_relaxing_tolerance(tiny, monkeypatch, tmp_path):
    from gofa_f import chattime_prompt_cache as module
    model, encoded, features, _ = tiny
    encoded = replace(encoded, attention_mask=torch.ones_like(encoded.attention_mask))
    original = module.padded_causal_layer
    monkeypatch.setattr(module, "padded_causal_layer", lambda *args: original(*args)+1.)
    path = tmp_path/"failure.json"
    with pytest.raises(AssertionError):
        check_fixed_prompt(model, encoded, features, encoded.input_ids[0, :3].tolist(), diagnostics_path=path)
    diagnostic = json.loads(path.read_text())
    assert not diagnostic["passed"] and diagnostic["tolerance"] == {"rtol": 1e-6, "atol": 1e-6}
    assert diagnostic["prefix"]["max_abs"] > 0 and model.prompt_prefix is None
    assert all(p.grad is None for p in model.parameters())
