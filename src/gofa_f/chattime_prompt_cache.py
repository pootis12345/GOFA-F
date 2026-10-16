"""Read-only KV reuse for the frozen common prompt, never a training KV cache."""
import torch
from transformers.cache_utils import Cache, DynamicCache


class ReadOnlyPrefixKV(Cache):
    def __init__(self, pairs):
        super().__init__()
        self.pairs = pairs

    def get_seq_length(self, layer_idx=0):
        return self.pairs[layer_idx][0].shape[-2]

    def get_max_cache_shape(self):
        return None

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        key, value = self.pairs[layer_idx]
        batch = key_states.shape[0]
        return (torch.cat((key.expand(batch, -1, -1, -1), key_states), dim=-2),
                torch.cat((value.expand(batch, -1, -1, -1), value_states), dim=-2))


class FixedPromptPrefix:
    """One detached copy; every input must match its IDs, mask and positions."""
    def __init__(self, token_ids, binding):
        self.ids = tuple(int(v) for v in token_ids)
        if not self.ids:
            raise ValueError("A common prompt must contain tokens")
        self.binding = str(binding)
        self.signature, self.pairs, self.boundary = None, (), None
        self.builds = self.reused_rows = self.fallback_rows = 0

    def stats(self):
        return {"enabled": True, "format": "fixed_prompt_kv_v1", "binding": self.binding,
            "tokens": len(self.ids), "builds": self.builds, "reused_rows": self.reused_rows,
            "fallback_rows": self.fallback_rows,
            "single_copy_bytes": sum(t.numel()*t.element_size() for pair in self.pairs for t in pair)
                + (self.boundary.numel()*self.boundary.element_size() if self.boundary is not None else 0),
            "precision": "native dtype, lossless storage; no input truncation"}

    @torch.no_grad()
    def compute(self, model, encoded, node_chunk, check):
        length = len(self.ids)
        ids, mask = encoded.input_ids, encoded.attention_mask
        wanted = torch.tensor(self.ids, device=ids.device, dtype=ids.dtype)
        if ids.shape[1] <= length or not bool(mask.all()) \
                or not bool((ids[:, :length] == wanted).all()):
            self.fallback_rows += len(ids)
            return model._full_prefix(encoded, node_chunk, check)
        if model.native.training or any(p.requires_grad for p in model.native.parameters()):
            raise ValueError("Fixed prompt caching requires an eval-mode frozen native model")
        signature = (model.graph_layers, self.ids, tuple(ids.shape), node_chunk,
            tuple((p.data_ptr(), p._version, p.dtype, p.device) for p in model.native.parameters()))
        if signature != self.signature:
            check()
            # Warm with the original query/batch shape once. Only causal states
            # before the history are retained, independent of the warm-up tail.
            hidden = model.native.model.embed_tokens(ids[:node_chunk])
            context = model.context(hidden, mask[:node_chunk])
            cache = DynamicCache()
            for index in range(model.graph_layers[0]+1):
                hidden = model.layer(index, hidden, context, past=cache)
            # Dynamic tails must never append into this shared cache.
            self.pairs = tuple((key[:1, :, :length].detach().clone(), value[:1, :, :length].detach().clone())
                               for key, value in zip(cache.key_cache, cache.value_cache))
            self.boundary = hidden[:1, :length].detach().clone()
            del cache, hidden
            self.signature = signature
            self.builds += 1
        readonly = ReadOnlyPrefixKV(self.pairs)
        chunks = []
        for start in range(0, len(ids), node_chunk):
            check()
            dynamic = model.native.model.embed_tokens(ids[start:start+node_chunk, length:])
            context = model.context(dynamic, mask[start:start+node_chunk], past=readonly, offset=length)
            for index in range(model.graph_layers[0]+1):
                dynamic = model.layer(index, dynamic, context, past=readonly)
            chunks.append(torch.cat((self.boundary.expand(len(dynamic), -1, -1), dynamic), dim=1))
        self.reused_rows += len(ids)
        return torch.cat(chunks), mask


def prompt_token_ids(tokenizer):
    from gofa_f.chattime import PROMPT
    return tokenizer(PROMPT.split("{}")[0], add_special_tokens=True)["input_ids"]


def check_fixed_prompt(model, encoded, features, token_ids, check=lambda: None):
    """Same active weights; smooth loss tests suffix and graph gradients."""
    from gofa_f.chattime_task import gradients
    original = model.prompt_prefix
    tolerance = {"rtol": 2e-3, "atol": 2e-3} if model.native.dtype == torch.bfloat16 else {
        "rtol": 1e-6, "atol": 1e-6}
    predictions, derivative = [], []
    try:
        for cached in (False, True):
            check()
            model.prompt_prefix = FixedPromptPrefix(token_ids, "mechanism-check") if cached else None
            model.zero_grad(set_to_none=True)
            prediction = model(encoded, features, node_chunk=len(encoded.input_ids), check=check)
            prediction.square().mean().backward()
            predictions.append(prediction.detach().clone())
            derivative.append(gradients(model))
        torch.testing.assert_close(predictions[0], predictions[1], **tolerance)
        assert derivative[0].keys() == derivative[1].keys()
        maximum = 0.
        for name in derivative[0]:
            assert torch.isfinite(derivative[1][name]).all(), name
            torch.testing.assert_close(derivative[0][name], derivative[1][name], **tolerance)
            maximum = max(maximum, float((derivative[0][name]-derivative[1][name]).abs().max()))
        stats = model.prompt_prefix.stats()
        assert (stats["reused_rows"] > 0 and stats["builds"] == 1) or stats["fallback_rows"] == len(encoded.input_ids)
        assert all(p.grad is None for p in model.native.parameters())
        return {"passed": True, "prediction_max_abs": float((predictions[0]-predictions[1]).abs().max()),
            "gradient_max_abs": maximum, "tolerance": tolerance, "cache": stats,
            "optimized_exercised": stats["reused_rows"] > 0}
    finally:
        model.zero_grad(set_to_none=True)
        model.prompt_prefix = original
