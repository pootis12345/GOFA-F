"""Frozen causal ChatTime with two late, summary-only graph exchanges.

No hooks, vocabulary resizing, native-head edits, or teacher-forced future tokens.
The native CausalLM remains usable directly for text and greedy forecasting.
"""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from gofa_f.temporal_context import SupportedRoutedExchange

REPO = "ChengsenWang/ChatTime-1-7B-Chat"
REVISION = "20baa88d6ab838b34abbd1d2b1f4cea57c8ff870"
SOURCE_REVISION = "8c2d4c209302d2b2bd6cc3c154586842341e3247"
PROMPT = ("You are a helpful assistant that performs time series prediction. "
          "The user will provide a sequence and you will predict the sequence.\n"
          "### Instruction:\nPlease predict the following sequence carefully.\n\n"
          "### Input:\n{}\n\n### Response:\n")


@dataclass
class EncodedHistory:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    last: torch.Tensor
    scale: torch.Tensor
    present: torch.Tensor
    inverse_scale: np.ndarray
    inverse_offset: np.ndarray
    shape: tuple

    def to(self, device):
        return EncodedHistory(*(v.to(device) for v in (self.input_ids, self.attention_mask,
            self.last, self.scale, self.present)), self.inverse_scale, self.inverse_offset, self.shape)


class HistoryCodec:
    """Official float64 MinMaxScaler/digitize arithmetic, plus empty-history policy."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.boundaries = np.linspace(-1, 1, 10001)
        centers = (self.boundaries[1:]+self.boundaries[:-1])/2
        self.centers = np.concatenate((centers[:1], centers, centers[-1:]))
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def serialize(self, values, observed, fallback):
        values = np.asarray(values, dtype=np.float64)
        observed = np.asarray(observed, bool) & np.isfinite(values)
        if observed.any():
            minimum, maximum = values[observed].min(), values[observed].max()
            extent = maximum-minimum
            safe = 1.0 if extent < 10*np.finfo(np.float64).eps else extent
            inverse_scale = 1.0/safe
            inverse_offset = -minimum*inverse_scale
            scaled = values*inverse_scale+inverse_offset-0.5
            quantized = self.centers[np.digitize(scaled, self.boundaries, right=True)]
            last = values[np.flatnonzero(observed)[-1]]
        else:
            inverse_scale, inverse_offset, safe, last = 1.0, -float(fallback), 1.0, float(fallback)
            quantized = np.zeros_like(values)
        words = [f"###{q:.4f}###" if valid else "###Nan###" for q, valid in zip(quantized, observed)]
        return PROMPT.format(" ".join(words)), last, safe, bool(observed.any()), inverse_scale, inverse_offset

    def encode(self, raw, observed, train_mean, padding_side="right"):
        raw, observed = np.asarray(raw), np.asarray(observed)
        if raw.ndim != 3 or raw.shape != observed.shape or raw.shape[1] != 12:
            raise ValueError("Expected raw histories and masks [B,12,N]")
        b, _, n = raw.shape
        means = np.broadcast_to(np.asarray(train_mean), (b, n)).reshape(-1)
        rows = raw.transpose(0, 2, 1).reshape(b*n, 12)
        masks = observed.transpose(0, 2, 1).reshape(b*n, 12)
        records = [self.serialize(x, mask, mean) for x, mask, mean in zip(rows, masks, means)]
        old = self.tokenizer.padding_side
        try:
            self.tokenizer.padding_side = padding_side
            tokens = self.tokenizer([r[0] for r in records], padding=True, truncation=False,
                                    return_tensors="pt")
        finally:
            self.tokenizer.padding_side = old
        # No sequence truncation or shortening is used to meet resource limits.
        return EncodedHistory(tokens["input_ids"], tokens["attention_mask"].bool(),
            torch.tensor([r[1] for r in records], dtype=torch.float32).reshape(b, n),
            torch.tensor([r[2] for r in records], dtype=torch.float32).reshape(b, n),
            torch.tensor([r[3] for r in records], dtype=torch.bool).reshape(b, n),
            np.array([r[4] for r in records]), np.array([r[5] for r in records]), (b, n))

    @staticmethod
    def parse(text, inverse_scale, inverse_offset, last, horizon=12):
        tokens = re.findall(r"###(.*?)###", text)
        prediction = np.full(horizon, last, dtype=np.float32)
        valid = np.zeros(horizon, dtype=bool)
        for i, token in enumerate(tokens[:horizon]):
            try:
                value = (float(token)+0.5-inverse_offset)/inverse_scale
                if np.isfinite(value) and abs(value) <= np.finfo(np.float32).max:
                    prediction[i], valid[i] = value, True
            except (ValueError, OverflowError):
                pass
        return prediction, valid


class PrefixLRU:
    """Lossless CPU tensors; cache identity binds token IDs, masks and source binding."""
    def __init__(self, binding, max_bytes=64*2**30, disk_directory=None, disk_limit=128*2**30):
        self.binding, self.max_bytes = str(binding), int(max_bytes)
        self.entries, self.used = OrderedDict(), 0
        self.hits, self.misses = 0, 0
        self.peak = 0
        self.disk_directory = Path(disk_directory) if disk_directory is not None else None
        self.disk_limit, self.disk_used = int(disk_limit), 0
        if self.disk_directory is not None:
            self.disk_directory.mkdir(parents=True, exist_ok=True)
            self.disk_used = sum(p.stat().st_size for p in self.disk_directory.glob("*.pt"))

    def key(self, encoded):
        digest = hashlib.sha256(self.binding.encode())
        for tensor in (encoded.input_ids, encoded.attention_mask):
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def get(self, key, device):
        if key not in self.entries:
            path = self.disk_directory/(key+".pt") if self.disk_directory is not None else None
            if path is not None and path.exists():
                saved = torch.load(path, map_location="cpu", weights_only=True)
                if saved["binding"] != self.binding or saved["key"] != key:
                    raise ValueError("Frozen prefix disk binding differs")
                self.hits += 1
                self.put(key, saved["prefix"], write_disk=False)
                return tuple(v.to(device) for v in saved["prefix"])
            self.misses += 1
            return None
        self.hits += 1
        value, size = self.entries.pop(key)
        self.entries[key] = value, size
        return tuple(v.to(device) for v in value)

    def put(self, key, value, write_disk=True):
        tensors = tuple(v.detach().cpu().clone() for v in value)
        size = sum(v.numel()*v.element_size() for v in tensors)
        if size > self.max_bytes:
            return
        if key in self.entries:
            self.used -= self.entries.pop(key)[1]
        while self.entries and self.used+size > self.max_bytes:
            self.used -= self.entries.popitem(last=False)[1][1]
        self.entries[key] = tensors, size
        self.used += size
        self.peak = max(self.peak, self.used)
        if write_disk and self.disk_directory is not None:
            path = self.disk_directory/(key+".pt")
            if not path.exists() and self.disk_used+size+16384 <= self.disk_limit:
                temporary = path.with_suffix(".tmp")
                torch.save({"binding": self.binding, "key": key, "prefix": tensors}, temporary)
                actual = temporary.stat().st_size
                if self.disk_used+actual > self.disk_limit:
                    temporary.unlink()
                else:
                    temporary.replace(path)
                    self.disk_used += actual


class ChatTimeGOFA(nn.Module):
    def __init__(self, native, physical, correlation, horizon=12, rank=64, feature_dim=19,
                 attention_rank=16, prior_floor=.5):
        super().__init__()
        self.native = native.requires_grad_(False).eval()
        self.native.gradient_checkpointing_disable()
        self.graph_layers = tuple(range(native.config.num_hidden_layers-3, native.config.num_hidden_layers-1))
        if len(self.graph_layers) != 2 or self.graph_layers[0] < 0:
            raise ValueError("Two graph exchanges require at least three native layers")
        width = native.config.hidden_size
        physical, correlation = np.asarray(physical), np.asarray(correlation)
        if physical.shape != correlation.shape or physical.ndim != 2 or physical.shape[0] != physical.shape[1] \
                or not np.isfinite(physical).all() or not np.isfinite(correlation).all() \
                or (physical < 0).any() or (correlation < 0).any():
            raise ValueError("Invalid graph support")
        self.register_buffer("physical", torch.tensor(physical, dtype=torch.float32))
        self.graph = nn.ModuleDict({str(i): SupportedRoutedExchange(width, rank, feature_dim,
            attention_rank, prior_floor, np.maximum(physical, correlation)) for i in self.graph_layers})
        self.head = nn.Linear(width, horizon)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.horizon, self.suffix_checkpointing = horizon, False

    def train(self, mode=True):
        super().train(mode)
        self.native.eval()
        return self

    def trainable_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                if k.startswith(("graph.", "head."))}

    def load_trainable(self, state):
        expected = self.trainable_state()
        if expected.keys() != state.keys():
            raise ValueError("Trainable checkpoint names changed")
        self.load_state_dict(state, strict=False)

    def context(self, hidden, mask):
        model = self.native.model
        positions = torch.arange(hidden.shape[1], device=hidden.device)
        position_ids = positions[None]
        causal_mask = model._update_causal_mask(mask, hidden, positions, None, False)
        rotary = model.rotary_emb(hidden, position_ids)
        return causal_mask, positions, position_ids, rotary

    def layer(self, index, hidden, context):
        causal, positions, ids, rotary = context
        return self.native.model.layers[index](hidden, attention_mask=causal,
            position_ids=ids, past_key_value=None, output_attentions=False, use_cache=False,
            cache_position=positions, position_embeddings=rotary)[0]

    @torch.no_grad()
    def prefix(self, encoded, node_chunk=64, check=lambda: None):
        hidden_chunks = []
        for start in range(0, len(encoded.input_ids), node_chunk):
            check()
            ids = encoded.input_ids[start:start+node_chunk]
            mask = encoded.attention_mask[start:start+node_chunk]
            hidden = self.native.model.embed_tokens(ids)
            context = self.context(hidden, mask)
            for i in range(self.graph_layers[0]+1):
                hidden = self.layer(i, hidden, context)
            hidden_chunks.append(hidden)
        return torch.cat(hidden_chunks), encoded.attention_mask

    def exchange(self, index, hidden, mask, present, features):
        b, n = present.shape
        rows = torch.arange(b*n, device=hidden.device)
        positions = torch.arange(hidden.shape[1], device=hidden.device)[None].expand_as(mask)
        last = positions.masked_fill(~mask, -1).max(1).values
        if (last < 0).any():
            raise ValueError("An encoded prompt cannot be empty")
        summary = hidden[rows, last].reshape(b, n, 1, -1)
        exchanged = self.graph[str(index)](summary, self.physical, present[..., None], features)
        result = hidden.clone()
        result[rows, last] = exchanged.reshape(b*n, -1)
        return result

    def last_hidden(self, prefix, encoded, features):
        hidden, mask = prefix
        context = self.context(hidden, mask)
        hidden = self.exchange(self.graph_layers[0], hidden, mask, encoded.present, features)
        for i in range(self.graph_layers[0]+1, len(self.native.model.layers)):
            if self.suffix_checkpointing and torch.is_grad_enabled():
                hidden = checkpoint(lambda value, index=i: self.layer(index, value, context),
                                    hidden, use_reentrant=False)
            else:
                hidden = self.layer(i, hidden, context)
            if str(i) in self.graph:
                hidden = self.exchange(i, hidden, mask, encoded.present, features)
        hidden = self.native.model.norm(hidden)
        rows = torch.arange(len(mask), device=hidden.device)
        last = torch.arange(mask.shape[1], device=hidden.device)[None].expand_as(mask).masked_fill(~mask, -1).max(1).values
        return hidden[rows, last].float()

    def from_prefix(self, prefix, encoded, features):
        summary = self.last_hidden(prefix, encoded, features)
        with torch.autocast(summary.device.type, enabled=False):
            delta = self.head(summary).reshape(*encoded.shape, self.horizon)
        delta = delta.masked_fill(~encoded.present[..., None], 0)
        raw = encoded.last[..., None]+encoded.scale[..., None]*delta
        return raw.transpose(1, 2)

    def forward(self, encoded, features, cache=None, node_chunk=64, check=lambda: None):
        key = cache.key(encoded) if cache is not None else None
        prefix = cache.get(key, encoded.input_ids.device) if cache is not None else None
        if prefix is None:
            prefix = self.prefix(encoded, node_chunk, check)
            if cache is not None:
                cache.put(key, prefix)
        return self.from_prefix(prefix, encoded, features)
