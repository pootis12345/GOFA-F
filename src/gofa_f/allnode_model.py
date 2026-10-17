"""All sampled nodes traverse a shared frozen LLM; original numeric slots exchange graph messages."""
from dataclasses import dataclass
import hashlib

import numpy as np
import torch
from torch import nn
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from gofa_f.allnode_graph import AllNodeAdapter
from gofa_f.allnode_codec import PROMPT_SHA
from gofa_f.generative_kernel import install_linear_tiles, stable_attention
from gofa_f.generative_model import GenerativeGOFA


@dataclass
class OriginContext:
    identity: str
    version: int
    generation: int
    deltas: tuple
    length: int
    nodes: int


@dataclass
class AllNodeKV:
    native: object
    version: int
    generation: int
    origins: tuple
    prompt_hashes: tuple
    consumed_ids: torch.Tensor
    consumed_mask: torch.Tensor
    value_positions: tuple
    prompt_ends: torch.Tensor
    dtype: torch.dtype


class AllNodeGOFA(GenerativeGOFA):
    def __init__(self, native, stable_fp32=False):
        nn.Module.__init__(self)
        self.native = native.requires_grad_(False).eval()
        self.native.gradient_checkpointing_disable()
        self.prefix_layers = len(native.model.layers)-2
        self.graph_layers = (self.prefix_layers, self.prefix_layers+1)
        self.adapter = AllNodeAdapter(native.config.hidden_size).float()
        self.parameter_version = 0
        self.cache_generation = 0
        self._active = None
        self._kernel_active = False
        self.stable_fp32 = stable_fp32
        self.origin_executions = []
        install_linear_tiles(native, lambda: self._kernel_active, stable_fp32)
        self._handles = [native.model.layers[index].register_forward_pre_hook(
            self._hook(slot), with_kwargs=True) for slot, index in enumerate(self.graph_layers)]

    def load_trainable(self, state):
        super().load_trainable(state)
        self.cache_generation += 1

    def mark_updated(self):
        super().mark_updated()
        self.cache_generation += 1

    def _hook(self, slot):
        def inject(module, args, kwargs):
            if self._active is None:
                return args, kwargs
            hidden = args[0] if args else kwargs["hidden_states"]
            tokens, contexts, offset = self._active
            changed = []
            for row, context in enumerate(contexts):
                positions = tokens.value_positions[row][:context.length]
                local = positions-offset
                valid = (local >= 0) & (local < hidden.shape[1])
                if valid.any():
                    where = local[valid]
                    value = hidden[row].index_select(0, where)
                    delta = context.deltas[slot][valid].to(hidden.dtype)
                    changed.append(hidden[row].index_copy(0, where, value+delta))
                else:
                    changed.append(hidden[row])
            result = torch.stack(changed)
            return ((result, *args[1:]), kwargs) if args else (args, dict(kwargs, hidden_states=result))
        return inject

    def cached_prefix(self, tokens, cache=None, role="origin_graph"):
        key = hashlib.sha256((role+cache.key(tokens)).encode()).hexdigest() if cache is not None else None
        result = cache.get(key, tokens.input_ids.device) if cache is not None else None
        if result is None:
            result = self.prefix(tokens)
            if cache is not None:
                cache.put(key, result)
        return result

    def encode_origins(self, graphs, codec, cache=None, check=lambda: None):
        result = []
        device = next(self.native.parameters()).device
        for item in graphs:
            check()
            graph = item.graph
            histories = np.where(graph.observed.cpu().numpy(), graph.history.cpu().numpy(), np.nan).T
            tokens = codec.batch([codec.record(row) for row in histories]).to(device)
            graph = graph.to(device)
            hidden, _ = self.cached_prefix(tokens, cache)
            model = self.native.model
            positions = torch.arange(hidden.shape[1], device=device)
            ids = self.positions(tokens.attention_mask, hidden.shape[1])
            causal = model._update_causal_mask(tokens.attention_mask, hidden, positions, None, False)
            rotary = model.rotary_emb(hidden, ids)
            deltas = []
            with stable_attention(self, positions, tokens.attention_mask, tokens.prompt_ends):
                for slot, layer_index in enumerate(self.graph_layers):
                    check()
                    selected = torch.stack([hidden[i].index_select(0, p)
                                            for i, p in enumerate(tokens.value_positions)])
                    delta = self.adapter.exchanges[slot](selected, graph)
                    deltas.append(delta[0])
                    hidden = torch.stack([hidden[i].index_copy(0, p,
                        hidden[i].index_select(0, p)+delta[i].to(hidden.dtype))
                        for i, p in enumerate(tokens.value_positions)])
                    # Layer31 must retain paths from every neighbor through exchange1.
                    # Layer32 origin outputs are not decoder inputs; still execute all nodes.
                    if slot == 0:
                        hidden = model.layers[layer_index](hidden, attention_mask=causal,
                            position_ids=ids, cache_position=positions, position_embeddings=rotary,
                            use_cache=False, output_attentions=False)[0]
                    else:
                        with torch.no_grad():
                            end = model.layers[layer_index](hidden, attention_mask=causal,
                                position_ids=ids, cache_position=positions, position_embeddings=rotary,
                                use_cache=False, output_attentions=False)[0]
                            model.norm(end)
            self.origin_executions.append({"nodes": len(histories), "layers": len(model.layers),
                                           "history": len(graph.history)})
            self.origin_executions = self.origin_executions[-32:]
            result.append(OriginContext(item.identity, self.parameter_version, self.cache_generation,
                                       tuple(deltas), len(graph.history), len(histories)))
        return result

    def forward(self, tokens, contexts=None, cache=None, use_cache=False, prefix=None):
        if self._active is not None:
            raise RuntimeError("All-node forwards are not reentrant")
        contexts = list(contexts) if contexts is not None else None
        origins = tuple(c.identity for c in contexts) if contexts is not None else ()
        if contexts is not None:
            if len(contexts) != len(tokens.input_ids):
                raise ValueError("One isolated origin context per target row required")
            for c, p in zip(contexts, tokens.value_positions):
                if c.version != self.parameter_version or c.generation != self.cache_generation or len(p) < c.length:
                    raise ValueError("Stale origin messages or missing original-time coordinates")
        native_past = None
        if cache is not None:
            if not isinstance(cache, AllNodeKV) or cache.version != self.parameter_version \
                    or cache.generation != self.cache_generation or cache.origins != origins \
                    or cache.prompt_hashes != tokens.prompt_hashes or cache.dtype != self.native.dtype \
                    or not torch.equal(cache.prompt_ends, tokens.prompt_ends):
                raise ValueError("Stale or differently conditioned KV")
            if len(cache.value_positions) != len(tokens.value_positions) or any(
                    not torch.equal(a, b) for a, b in zip(cache.value_positions, tokens.value_positions)):
                raise ValueError("KV numeric position mapping changed")
            length = cache.consumed_ids.shape[1]
            if tokens.full_ids.shape[1] != length+tokens.input_ids.shape[1] \
                    or not torch.equal(tokens.full_ids[:, :length], cache.consumed_ids) \
                    or not torch.equal(tokens.attention_mask[:, :length], cache.consumed_mask) \
                    or not torch.equal(tokens.full_ids[:, length:], tokens.input_ids):
                raise ValueError("KV token/mask lineage differs; rolling prompts require fresh prefill")
            native_past = cache.native
        if use_cache and native_past is None:
            native_past = DynamicCache()
        offset = native_past.get_seq_length() if native_past is not None else 0
        positions = torch.arange(offset, offset+tokens.input_ids.shape[1], device=tokens.input_ids.device)
        if contexts is not None:
            self._active = (tokens, contexts, offset)
        try:
            ids = self.positions(tokens.attention_mask, tokens.input_ids.shape[1])
            with stable_attention(self, positions, tokens.attention_mask, tokens.prompt_ends):
                if prefix is None:
                    output = self.native(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask,
                        position_ids=ids, past_key_values=native_past, use_cache=use_cache, return_dict=True)
                else:
                    if cache is not None or use_cache:
                        raise ValueError("Hidden prefix is not a per-layer KV cache")
                    hidden, saved_mask = prefix
                    if not torch.equal(saved_mask, tokens.attention_mask):
                        raise ValueError("Prefix mask differs")
                    model = self.native.model
                    causal = model._update_causal_mask(tokens.attention_mask, hidden, positions, None, False)
                    rotary = model.rotary_emb(hidden, ids)
                    for layer in model.layers[self.prefix_layers:]:
                        hidden = layer(hidden, attention_mask=causal, position_ids=ids, cache_position=positions,
                            position_embeddings=rotary, use_cache=False, output_attentions=False)[0]
                    output = CausalLMOutputWithPast(logits=self.native.lm_head(model.norm(hidden)))
        finally:
            self._active = None
        if use_cache:
            output.past_key_values = AllNodeKV(output.past_key_values, self.parameter_version,
                self.cache_generation, origins, tokens.prompt_hashes,
                tokens.full_ids.detach().clone(), tokens.attention_mask.detach().clone(),
                tuple(p.detach().clone() for p in tokens.value_positions),
                tokens.prompt_ends.detach().clone(), self.native.dtype)
        return output

    def parameter_counts(self):
        modules = {f"exchange_{i+1}": sum(p.numel() for p in module.parameters())
                   for i, module in enumerate(self.adapter.exchanges)}
        return {"modules": modules, "trainable": sum(modules.values()),
                "registered": sum(p.numel() for p in self.parameters()),
                "forward_parameters": getattr(self, "_verified_forward_parameters", None),
                "forward_definition": "full tensors in executed modules, including embedding tables"}

    @torch.no_grad()
    def analyze(self, codec, history, question, **generation):
        if self._active is not None or self._kernel_active:
            raise RuntimeError("Native analysis cannot overlap graph execution")
        inputs = codec.analysis_inputs(history, question)
        device = next(self.native.parameters()).device
        inputs = {name: value.to(device) for name, value in inputs.items()}
        options = dict(do_sample=False, max_new_tokens=128, pad_token_id=codec.tokenizer.eos_token_id)
        options.update(generation)
        output = self.native.generate(**inputs, **options)
        return codec.tokenizer.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def frozen_cache_binding(model, weights_sha):
    return hashlib.sha256(str((weights_sha, PROMPT_SHA, "allnode_v1", model.prefix_layers,
        model.native.dtype, "before", "bf16_rows128_sdpa_fixed_prompt_capacity_v1")).encode()).hexdigest()
