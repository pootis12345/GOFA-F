from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from gofa_f.chattime import ChatTimeGOFA, HistoryCodec, PrefixLRU
from gofa_f.chattime_checks import check_model


class Tokenizer:
    eos_token, eos_token_id, padding_side = "</s>", 2, "right"
    pad_token = "</s>"

    def __call__(self, texts, padding=True, truncation=False, return_tensors="pt"):
        assert not truncation
        rows = [[1]+[3+(ord(c) % 55) for c in text[::7]] for text in texts]
        size = max(map(len, rows))
        masks, ids = [], []
        for row in rows:
            pad = [2]*(size-len(row))
            valid = [1]*len(row)
            if self.padding_side == "left":
                ids.append(pad+row)
                masks.append([0]*len(pad)+valid)
            else:
                ids.append(row+pad)
                masks.append(valid+[0]*len(pad))
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}

    def batch_decode(self, tokens, **kwargs):
        return [" ".join(["###0.1000###"]*12) for _ in tokens]


@pytest.fixture
def tiny():
    torch.manual_seed(42)
    config = LlamaConfig(hidden_size=32, intermediate_size=48, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
                         max_position_embeddings=1024, attention_dropout=0.)
    config._attn_implementation = "sdpa"
    native = LlamaForCausalLM(config).eval()
    physical = np.array([[1, .5, 0], [.5, 1, .2], [0, .2, 1]], np.float32)
    correlated = np.ones((3, 3), np.float32)
    model = ChatTimeGOFA(native, physical, correlated, rank=8, attention_rank=4)
    raw = np.arange(36, dtype=np.float32).reshape(1, 12, 3)
    mask = np.ones_like(raw, bool)
    mask[0, 0, 0] = False
    codec = HistoryCodec(Tokenizer())
    encoded = codec.encode(raw, mask, np.zeros(3))
    features = torch.randn(1, 3, 19)
    return model, encoded, features, codec


@pytest.mark.parametrize("all_valid_tokens", [False, True])
def test_manual_causal_path_cache_freeze_text_and_pending_resume(tiny, all_valid_tokens):
    model, encoded, features, _ = tiny
    if all_valid_tokens:
        encoded = replace(encoded, attention_mask=torch.ones_like(encoded.attention_mask))
    record = check_model(model, encoded, features)
    assert record["initial_persistence"] and record["text_bypass_exact"]
    if all_valid_tokens:
        assert record["native_prefill_shared_prediction_and_gradient"]


@pytest.mark.parametrize("values", [np.arange(12.)/7, np.full(12, 3.5),
    np.array([np.nan, 1., 3., np.nan, 2., 0., 5., 1., 0., 2., 3., 1.]),
    np.linspace(-10, 100, 12), np.array([1.]*11+[1.+1e-15])])
def test_official_scaler_and_bins(values):
    scaler_type = pytest.importorskip("sklearn.preprocessing").MinMaxScaler
    codec = HistoryCodec(Tokenizer())
    text, _, scale, present, inv_scale, inv_offset = codec.serialize(values, np.isfinite(values), 0)
    original = scaler_type()
    transformed = original.fit_transform(values[:, None]).reshape(-1)-.5
    quantized = codec.centers[np.digitize(transformed, codec.boundaries, right=True)]
    words = [f"###{q:.4f}###" if np.isfinite(v) else "###Nan###" for q, v in zip(quantized, values)]
    assert " ".join(words) in text and present
    assert inv_scale == original.scale_[0] and inv_offset == original.min_[0]
    assert scale == 1/original.scale_[0]


def test_empty_constant_and_invalid_generation():
    codec = HistoryCodec(Tokenizer())
    text, last, scale, present, inv_scale, inv_offset = codec.serialize(np.zeros(12), np.zeros(12, bool), 7.5)
    assert text.count("###Nan###") == 12 and last == 7.5 and scale == 1 and not present
    prediction, valid = codec.parse("###Nan### ###inf### ###broken### ###0.2###", inv_scale, inv_offset, last)
    assert np.isfinite(prediction).all() and valid.sum() == 1
    np.testing.assert_allclose(prediction[:3], last)


def test_official_prompt_blank_line_boundaries():
    codec = HistoryCodec(Tokenizer())
    text, *_ = codec.serialize(np.arange(12.), np.ones(12, bool), 0.)
    system, instruction, input_section, response = text.split("\n\n")
    assert system == ("You are a helpful assistant that performs time series prediction. "
                      "The user will provide a sequence and you will predict the sequence.")
    assert instruction == "### Instruction:\nPlease predict the following sequence carefully."
    assert input_section.startswith("### Input:\n###")
    assert response == "### Response:\n"


def test_raw_zero_is_an_observation_when_mask_says_so():
    codec = HistoryCodec(Tokenizer())
    text, _, _, present, _, _ = codec.serialize(np.zeros(12), np.ones(12, bool), 9.)
    assert "###Nan###" not in text and present


def test_left_right_padding_and_readout(tiny):
    model, encoded, features, codec = tiny
    ids = encoded.input_ids.clone()
    mask = encoded.attention_mask.clone()
    # Heterogeneous prompt lengths: a valid readout is not always the last column.
    mask[0, -4:] = False
    ids[0, -4:] = 2
    changed = replace(encoded, input_ids=ids, attention_mask=mask)
    with torch.no_grad():
        prefix = model.prefix(changed, node_chunk=3)
        result = model.last_hidden(prefix, changed, features)
        reference = model.native.model(input_ids=ids, attention_mask=mask).last_hidden_state
    torch.testing.assert_close(result[0], reference[0, -5], rtol=1e-6, atol=1e-6)
    assert codec.tokenizer.padding_side == "right"


def test_suffix_checkpointing_gradients(tiny):
    model, encoded, features, _ = tiny
    with torch.no_grad():
        model.head.weight.fill_(.001)
        for graph in model.graph.values():
            graph.gate.fill_(.1)
    prefix = model.prefix(encoded, node_chunk=3)
    results = []
    for enabled in (False, True):
        model.zero_grad(set_to_none=True)
        model.suffix_checkpointing = enabled
        prediction = model.from_prefix(prefix, encoded, features)
        prediction.square().mean().backward()
        results.append((prediction.detach(), {k: p.grad.clone() for k, p in model.named_parameters()
                                             if p.requires_grad and p.grad is not None}))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    for name in results[0][1]:
        torch.testing.assert_close(results[0][1][name], results[1][1][name], rtol=0, atol=0)


def test_registered_real_parameter_count():
    class ShapeOnlyNative(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4096, num_hidden_layers=32)

        def gradient_checkpointing_disable(self):
            pass

    model = ChatTimeGOFA(ShapeOnlyNative(), np.eye(2), np.eye(2))
    assert model.graph_layers == (29, 30)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 1120914


def test_bounded_lossless_cache(tiny):
    model, encoded, features, _ = tiny
    prefix = model.prefix(encoded, node_chunk=3)
    size = sum(v.numel()*v.element_size() for v in prefix)
    cache = PrefixLRU("revision+data+format+layer", size)
    key = cache.key(encoded)
    cache.put(key, prefix)
    actual = cache.get(key, "cpu")
    for a, b in zip(actual, prefix):
        assert torch.equal(a, b)
    altered = replace(encoded, attention_mask=encoded.attention_mask.clone())
    altered.attention_mask[0, 0] = False
    cache.put(cache.key(altered), prefix)
    assert cache.get(key, "cpu") is None and cache.used <= cache.max_bytes


def test_disk_prefix_rejects_different_binding(tiny, tmp_path):
    model, encoded, _, _ = tiny
    prefix = model.prefix(encoded, node_chunk=3)
    cache = PrefixLRU("one", max_bytes=10**7, disk_directory=tmp_path)
    key = cache.key(encoded)
    cache.put(key, prefix)
    resumed = PrefixLRU("one", max_bytes=10**7, disk_directory=tmp_path)
    for actual, expected in zip(resumed.get(key, "cpu"), prefix):
        assert torch.equal(actual, expected)
    wrong = PrefixLRU("two", max_bytes=10**7, disk_directory=tmp_path)
    with pytest.raises(ValueError):
        wrong.get(key, "cpu")


def test_affine_history_reuses_only_prefix_not_raw_scale(tiny):
    _, _, _, codec = tiny
    raw = np.arange(36, dtype=np.float32).reshape(1, 12, 3)
    first = codec.encode(raw, np.ones_like(raw, bool), np.zeros(3))
    second = codec.encode(raw*2+100, np.ones_like(raw, bool), np.zeros(3))
    cache = PrefixLRU("immutable")
    assert cache.key(first) == cache.key(second)
    torch.testing.assert_close(second.last, first.last*2+100)
    torch.testing.assert_close(second.scale, first.scale*2)


def test_graph_only_reads_history_and_is_batch_isolated(tiny):
    model, encoded, features, _ = tiny
    prefix = model.prefix(encoded, node_chunk=3)
    with torch.no_grad():
        model.head.weight.fill_(.001)
        for module in model.graph.values():
            module.gate.fill_(.2)
        a = model.from_prefix(prefix, encoded, features)
        # Target tensors have no entry point in the model/codec API.
        unrelated_target = torch.randn(1, 12, 3)*1000
        unrelated_target += 5
        b = model.from_prefix(prefix, encoded, features)
    assert torch.equal(a, b)


@pytest.mark.parametrize("bad", [np.array([[1., np.nan], [0, 1]]), np.array([[1., -1], [0, 1]])])
def test_reject_invalid_graph(tiny, bad):
    with pytest.raises(ValueError):
        ChatTimeGOFA(tiny[0].native, bad, np.eye(2))
