from dataclasses import replace

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
