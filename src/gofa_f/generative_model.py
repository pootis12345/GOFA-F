"""Native causal language head with graph pre-hooks at the final two blocks.

Graph-conditioned KV is scoped to one parameter version and immutable origin.
Only the unconditioned first N-2 blocks may be cached across optimizer steps.
"""
from dataclasses import dataclass
import hashlib

import numpy as np
import torch
from torch import nn
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from gofa_f.chattime import HistoryCodec
from gofa_f.generative_graph import GraphModules
from gofa_f.generative_kernel import install_linear_tiles, stable_attention
from gofa_f.chattime import PrefixLRU


@dataclass
class GraphKV:
    native: object
    version: int
    graph_ids: tuple
    memories: object
    prompt_ends: torch.Tensor


@dataclass
class QueryTokens:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    prompt_ends: torch.Tensor

    def to(self, device):
        return QueryTokens(*(value.to(device) for value in
            (self.input_ids, self.attention_mask, self.labels, self.prompt_ends)))


class GenerativeCodec(HistoryCodec):
    def record(self, history, response=None):
        history = np.asarray(history, np.float64)
        prompt, last, _, present, inverse_scale, inverse_offset = self.serialize(
            history, np.isfinite(history), 0.)
        ids = list(self.tokenizer(prompt, add_special_tokens=True)["input_ids"])
        end = len(ids)-1
        labels = [-100]*len(ids)
        if response is not None:
            response = np.asarray(response, np.float64)
            scaled = response*inverse_scale+inverse_offset-.5
            bins = self.centers[np.digitize(scaled, self.boundaries, right=True)]
            for index, value in enumerate(response):
                word = f"###{bins[index]:.4f}###" if np.isfinite(value) else "###Nan###"
                part = list(self.tokenizer((" " if index else "")+word,
                                           add_special_tokens=False)["input_ids"])
                ids.extend(part)
                labels.extend(part if np.isfinite(value) else [-100]*len(part))
            ids.append(self.tokenizer.eos_token_id)
            labels.append(self.tokenizer.eos_token_id)
        return {"ids": ids, "labels": labels, "prompt_end": end, "last": last,
                "present": present, "inverse_scale": inverse_scale, "inverse_offset": inverse_offset}

    def batch(self, records):
        size = max(len(r["ids"]) for r in records)
        return QueryTokens(torch.tensor([r["ids"]+[self.tokenizer.eos_token_id]*(size-len(r["ids"]))
            for r in records]), torch.tensor([[1]*len(r["ids"])+[0]*(size-len(r["ids"]))
            for r in records], dtype=torch.bool), torch.tensor([r["labels"]+[-100]*(size-len(r["ids"]))
            for r in records]), torch.tensor([r["prompt_end"] for r in records]))


class GenerativeGOFA(nn.Module):
    def __init__(self, native):
        super().__init__()
        self.native = native.requires_grad_(False).eval()
        self.native.gradient_checkpointing_disable()
        self.prefix_layers = len(native.model.layers)-2
        if self.prefix_layers < 1:
            raise ValueError("At least three Transformer layers required")
        self.graph_layers = (self.prefix_layers, self.prefix_layers+1)
        self.adapter = GraphModules(native.config.hidden_size).float()
        self.parameter_version = 0
        self._active = None
        self._kernel_active = False
        install_linear_tiles(self.native, lambda: self._kernel_active)
        self._handles = [native.model.layers[index].register_forward_pre_hook(
            self._hook(slot), with_kwargs=True) for slot, index in enumerate(self.graph_layers)]

    def train(self, mode=True):
        super().train(mode)
        self.native.eval()
        return self

    def _hook(self, slot):
        def inject(module, args, kwargs):
            if self._active is None:
                return args, kwargs
            state = self._active
            hidden = args[0] if args else kwargs["hidden_states"]
            positions = state["positions"]
            selected = (positions[None] >= state["ends"][:, None]) & state["mask"][:, positions]
            changed = self.adapter.interactions[slot](hidden, state["memories"][slot],
                                                      state["graphs"], selected)
            if args:
                return (changed, *args[1:]), kwargs
            return args, dict(kwargs, hidden_states=changed)
        return inject

    def trainable_state(self):
        return {name: value.detach().cpu().clone() for name, value in self.adapter.state_dict().items()}

    def load_trainable(self, state):
        self.adapter.load_state_dict(state, strict=True)
        self.parameter_version += 1

    def mark_updated(self):
        self.parameter_version += 1

    def positions(self, mask, length):
        ids = mask.long().cumsum(-1)-1
        return ids.masked_fill(~mask, 0)[:, -length:]

    @torch.no_grad()
    def prefix(self, tokens):
        model = self.native.model
        hidden = model.embed_tokens(tokens.input_ids)
        positions = torch.arange(hidden.shape[1], device=hidden.device)
        ids = self.positions(tokens.attention_mask, hidden.shape[1])
        causal = model._update_causal_mask(tokens.attention_mask, hidden, positions, None, False)
        rotary = model.rotary_emb(hidden, ids)
        with stable_attention(self, positions, tokens.attention_mask, tokens.prompt_ends):
            for layer in model.layers[:self.prefix_layers]:
                hidden = layer(hidden, attention_mask=causal, position_ids=ids, cache_position=positions,
                    position_embeddings=rotary, use_cache=False, output_attentions=False)[0]
        return hidden.detach(), tokens.attention_mask

    def forward(self, tokens, graphs=None, cache=None, use_cache=False, prefix=None, memories=None):
        if self._active is not None:
            raise RuntimeError("Graph forwards are not reentrant")
        graph_ids = tuple(graph.identity for graph in graphs) if graphs is not None else ()
        if cache is not None:
            if not isinstance(cache, GraphKV) or cache.version != self.parameter_version \
                    or cache.graph_ids != graph_ids or not torch.equal(cache.prompt_ends, tokens.prompt_ends):
                raise ValueError("Stale or differently conditioned graph KV")
            memories = cache.memories
        native_past = cache.native if cache is not None else None
        if use_cache and native_past is None:
            native_past = DynamicCache()
        offset = native_past.get_seq_length() if native_past is not None else 0
        positions = torch.arange(offset, offset+tokens.input_ids.shape[1], device=tokens.input_ids.device)
        if graphs is not None:
            if len(graphs) != len(tokens.input_ids):
                raise ValueError("One isolated origin graph per target query required")
            memories = self.adapter.memories(graphs) if memories is None else memories
            self._active = {"graphs": graphs, "memories": memories, "ends": tokens.prompt_ends,
                            "mask": tokens.attention_mask, "positions": positions}
        try:
            ids = self.positions(tokens.attention_mask, tokens.input_ids.shape[1])
            with stable_attention(self, positions, tokens.attention_mask, tokens.prompt_ends):
                if prefix is None:
                    output = self.native(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask,
                        position_ids=ids, past_key_values=native_past, use_cache=use_cache, return_dict=True)
                else:
                    if cache is not None or use_cache:
                        raise ValueError("Frozen hidden prefix is a training/full-forward cache, not suffix KV")
                    hidden, saved_mask = prefix
                    if not torch.equal(saved_mask, tokens.attention_mask):
                        raise ValueError("Prefix mask differs")
                    native = self.native.model
                    causal = native._update_causal_mask(tokens.attention_mask, hidden, positions, None, False)
                    rotary = native.rotary_emb(hidden, ids)
                    for layer in native.layers[self.prefix_layers:]:
                        hidden = layer(hidden, attention_mask=causal, position_ids=ids, cache_position=positions,
                            position_embeddings=rotary, use_cache=False, output_attentions=False)[0]
                    output = CausalLMOutputWithPast(logits=self.native.lm_head(native.norm(hidden)))
        finally:
            self._active = None
        if use_cache:
            output.past_key_values = GraphKV(output.past_key_values, self.parameter_version,
                                            graph_ids, memories, tokens.prompt_ends.detach().clone())
        return output

    def parameter_counts(self):
        modules = {name: sum(p.numel() for p in module.parameters()) for name, module in
            [("encoder", self.adapter.encoder), *[(f"gnn_{i+1}", m) for i, m in enumerate(self.adapter.gnns)],
             *[(f"interaction_{i+1}", m) for i, m in enumerate(self.adapter.interactions)]]}
        return {"modules": modules, "trainable": sum(modules.values()),
                "registered": sum(p.numel() for p in self.parameters()),
                "forward_parameters": sum(p.numel() for p in self.parameters())}


def completion_loss(logits, labels):
    shifted = labels[:, 1:]
    valid = shifted != -100
    counts = valid.sum(-1)
    if (counts == 0).any():
        raise ValueError("Every training query requires observed completion tokens")
    losses = torch.nn.functional.cross_entropy(logits[:, :-1].float().transpose(1, 2),
                                               shifted, ignore_index=-100, reduction="none")
    return (losses.sum(-1)/counts).sum(), int(counts.sum())


def frozen_cache_binding(model, weight_hash):
    return hashlib.sha256(str((weight_hash, model.prefix_layers, "before", model.native.dtype,
        "position_ids=mask.cumsum-1", "right_padding", "bf16_rows128_sdpa_fixed_prompt_capacity_v1")).encode()).hexdigest()


class GenerativePrefixLRU(PrefixLRU):
    def key(self, encoded):
        # Numerical attention capacity also depends on the original prompt boundary.
        digest = hashlib.sha256(super().key(encoded).encode())
        digest.update(encoded.prompt_ends.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()
