"""Protected Native stream plus appended soft Memory GNN/LLM interleaving.

Native has no graph hooks. Only appended Memory is written. Ordinary prompt
positions precede Memory causally; their frozen layer outputs remain cacheable.
The reference tail runs all positions. The optimized tail runs Memory queries
against separately recomputed frozen historical KV, retaining Memory autograd.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import weakref

import torch
from torch import nn
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache

from gofa_f.allnode_kernel import install_norm_tiles
from gofa_f.generative_kernel import install_linear_tiles, stable_attention

EXECUTION = 'dual_memory_v1:fixed128_linear_norm:math_sdpa:origin_capacity'


def tensor_hash(*values):
    h = hashlib.sha256()
    for v in values:
        x = v.detach().cpu().contiguous()
        h.update(str((str(x.dtype), tuple(x.shape))).encode())
        h.update(x.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


@dataclass
class FrozenHistory:
    hidden: torch.Tensor
    mask: torch.Tensor
    ends: torch.Tensor
    binding: str

    def to(self, device):
        return FrozenHistory(self.hidden.to(device), self.mask.to(device), self.ends.to(device), self.binding)


@dataclass
class ProtectedKV:
    native: object
    ids: torch.Tensor
    mask: torch.Tensor
    ends: torch.Tensor


class MemoryExchange(nn.Module):
    """One-hop, same-slot cross-node attention; all trainable arithmetic FP32."""
    def __init__(self, hidden, width=128):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.q = nn.Linear(hidden, width, bias=False)
        self.k = nn.Linear(hidden, width, bias=False)
        self.v = nn.Linear(hidden, width, bias=False)
        self.edge = nn.Sequential(nn.Linear(4, 32), nn.SiLU(), nn.Linear(32, 1))
        self.up = nn.Linear(width, hidden, bias=False)
        self.ff = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, width), nn.SiLU(), nn.Linear(width, hidden))
        self.gate = nn.Parameter(torch.zeros(())); self.ff_gate = nn.Parameter(torch.zeros(()))
        self.width = width

    def forward(self, memory, item):
        z = self.norm(memory.float()); q, k, v = self.q(z), self.k(z), self.v(z)
        score = torch.einsum('nkw,mkw->knm', q, k)/(self.width**.5)
        score = score+self.edge(item.edges.float()).squeeze(-1)[None]
        allowed = item.adjacency.bool()
        # All-unavailable auxiliary rows stay zero; no empty-softmax NaNs.
        score = score.masked_fill(~allowed[None], torch.finfo(score.dtype).min)
        attention = score.softmax(-1)*allowed[None]
        attention = attention/attention.sum(-1, keepdim=True).clamp_min(1e-12)
        message = torch.einsum('knm,mkw->nkw', attention, v)
        changed = memory.float()+self.gate.tanh()*self.up(message)
        changed = changed+self.ff_gate.tanh()*self.ff(changed)
        return torch.where(item.present[:, None, None], changed, memory.float())


class MemoryInterface(nn.Module):
    def __init__(self, hidden, depth, slots=4, width=128):
        super().__init__()
        self.slots = slots
        self.temporal = nn.GRU(3, width, batch_first=True)
        self.history = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, width), nn.SiLU())
        self.metadata = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, width), nn.SiLU())
        self.slot_seed = nn.Parameter(torch.randn(slots, width)*.02)
        self.write = nn.Linear(width, hidden)
        self.exchanges = nn.ModuleList(MemoryExchange(hidden, width) for _ in range(depth))
        self.read = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, width), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(slots*width+width+4, width), nn.SiLU(), nn.Linear(width, 1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def initialize(self, prefix, item, metadata):
        n, length = item.normalized.shape
        clock = torch.linspace(-1., 0., length, device=prefix.hidden.device).expand(n, length)
        raw = torch.stack((item.normalized.float(), item.observed.float(), clock), -1)
        temporal, _ = self.temporal(raw)
        temporal = temporal[:, -1]
        positions = prefix.mask.long().sum(-1)-1
        last = prefix.hidden[torch.arange(n, device=prefix.hidden.device), positions].float()
        # Metadata are separately encoded by the same frozen backbone, no prompt edits.
        context = self.history(last)+temporal+self.metadata(metadata.float())[None]
        memory = self.write(context[:, None]+self.slot_seed[None])
        return memory, temporal

    def predict(self, memory, temporal, item):
        n, h = len(memory), item.horizon
        read = self.read(memory.float()).flatten(1)
        u = torch.arange(1, h+1, device=memory.device, dtype=torch.float32)/h
        shape = u[:, None].expand(h, 1)
        # Shape-relative/absolute step features shared across12/24horizons.
        time = torch.cat((shape, shape.square(), torch.full_like(shape, h/24),
                          torch.full_like(shape, item.history.shape[1]/48)), -1)
        features = torch.cat((read[:, None].expand(n, h, -1),
            temporal[:, None].expand(n, h, -1), time[None].expand(n, h, -1)), -1)
        residual = self.head(features).squeeze(-1).double()
        return item.last[:, None].double()+item.extent[:, None]*residual


class NativeView(nn.Module):
    def __init__(self, owner):
        super().__init__(); object.__setattr__(self, '_owner', weakref.ref(owner))

    @property
    def native(self):
        return self._owner().native

    def forward(self, tokens, contexts=None, cache=None, use_cache=False):
        if contexts is not None:
            raise ValueError('Protected Native cannot consume graph contexts')
        return self._owner().protected_forward(tokens, cache, use_cache)


class DualMemoryGOFA(nn.Module):
    def __init__(self, native, depth=6, slots=4, width=128, weight_binding='', stable_fp32=False):
        super().__init__()
        if depth not in (3, 6) or len(native.model.layers) <= depth:
            raise ValueError('Only three or six interleaved tail blocks registered')
        self.native = native.requires_grad_(False).eval()
        self.native.gradient_checkpointing_disable()
        self.depth = depth; self.prefix_layers = len(native.model.layers)-depth
        self.adapter = MemoryInterface(native.config.hidden_size, depth, slots, width).float()
        self.weight_binding = weight_binding
        self.stable_fp32 = stable_fp32; self._kernel_active = False
        self.version = 0; self.prefix_executions = 0
        install_linear_tiles(native, lambda: self._kernel_active, stable_fp32)
        install_norm_tiles(native, lambda: self._kernel_active, stable_fp32)

    def train(self, mode=True):
        super().train(mode); self.native.eval(); return self

    def native_view(self):
        return NativeView(self)

    def input_binding(self, tokens):
        fields = (self.weight_binding, self.prefix_layers, str(self.native.dtype), EXECUTION,
                  tensor_hash(tokens.input_ids, tokens.attention_mask, tokens.prompt_ends))
        return hashlib.sha256(str(fields).encode()).hexdigest()

    @torch.no_grad()
    def frozen_history(self, tokens):
        model = self.native.model; mask = tokens.attention_mask
        hidden = model.embed_tokens(tokens.input_ids)
        positions = torch.arange(hidden.shape[1], device=hidden.device)
        ids = (mask.long().cumsum(-1)-1).clamp_min(0)
        causal = model._update_causal_mask(mask, hidden, positions, None, False)
        rotary = model.rotary_emb(hidden, ids)
        with stable_attention(self, positions, mask, tokens.prompt_ends):
            for layer in model.layers[:self.prefix_layers]:
                hidden = layer(hidden, attention_mask=causal, position_ids=ids,
                    cache_position=positions, position_embeddings=rotary, use_cache=False)[0]
        self.prefix_executions += 1
        return FrozenHistory(hidden.detach(), mask.detach().clone(), tokens.prompt_ends.detach().clone(), self.input_binding(tokens))

    @torch.no_grad()
    def metadata_vector(self, tokens):
        value = self.frozen_history(tokens)
        # A content summary before the graph tail, not an untrained token lookup.
        return (value.hidden.float()*value.mask[..., None]).sum((0, 1))/value.mask.sum().clamp_min(1)

    def protected_forward(self, tokens, cache=None, use_cache=False):
        if self._kernel_active:
            raise RuntimeError('Native and enhanced execution must be serial and isolated')
        past = None
        if cache is not None:
            if not isinstance(cache, ProtectedKV) or not torch.equal(cache.ends, tokens.prompt_ends):
                raise ValueError('Protected Native KV binding changed')
            size = cache.ids.shape[1]
            if not torch.equal(cache.ids, tokens.full_ids[:, :size]) or \
                    not torch.equal(cache.mask, tokens.attention_mask[:, :size]) or \
                    tokens.full_ids.shape[1] != size+tokens.input_ids.shape[1]:
                raise ValueError('Protected Native KV lineage changed')
            past = cache.native
        elif use_cache:
            past = DynamicCache()
        offset = past.get_seq_length() if past is not None else 0
        positions = torch.arange(offset, offset+tokens.input_ids.shape[1], device=tokens.input_ids.device)
        ids = ((tokens.attention_mask.long().cumsum(-1)-1).clamp_min(0))[:, -len(positions):]
        with stable_attention(self, positions, tokens.attention_mask, tokens.prompt_ends):
            out = self.native(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask,
                position_ids=ids, past_key_values=past, use_cache=use_cache, return_dict=True)
        if use_cache:
            out.past_key_values = ProtectedKV(out.past_key_values, tokens.full_ids.detach().clone(),
                tokens.attention_mask.detach().clone(), tokens.prompt_ends.detach().clone())
        return out

    def _tail(self, prefix, memory, item, optimized, checkpoint_tail=False):
        from torch.utils.checkpoint import checkpoint
        model = self.native.model; history = prefix.hidden.detach()
        n, length, _ = history.shape; slots = memory.shape[1]
        valid = torch.cat((prefix.mask, torch.ones(n, slots, dtype=torch.bool, device=history.device)), 1)
        ids = (valid.long().cumsum(-1)-1).clamp_min(0)
        positions = torch.arange(length+slots, device=history.device)
        rotary = model.rotary_emb(history, ids)
        causal = model._update_causal_mask(valid, torch.cat((history, memory.to(history.dtype)), 1), positions, None, False)
        for index, layer in enumerate(model.layers[self.prefix_layers:]):
            memory = self.adapter.exchanges[index](memory, item).to(history.dtype)
            if not optimized:
                full = torch.cat((history, memory), 1)
                with stable_attention(self, positions, valid, prefix.ends):
                    result = layer(full, attention_mask=causal, position_ids=ids,
                        cache_position=positions, position_embeddings=rotary, use_cache=False)[0]
                history, memory = result[:, :length].detach(), result[:, length:]
            else:
                # Historical states cannot see appended Memory. Recompute their
                # frozen KV, then execute only Memory queries with autograd.
                with torch.no_grad():
                    cache = DynamicCache()
                    hpos = positions[:length]; hids = ids[:, :length]
                    hrot = tuple(v[:, :length] for v in rotary)
                    hmask = model._update_causal_mask(prefix.mask, history, hpos, None, False)
                    with stable_attention(self, hpos, prefix.mask, prefix.ends):
                        next_history = layer(history, attention_mask=hmask, position_ids=hids,
                            cache_position=hpos, position_embeddings=hrot, past_key_value=cache,
                            use_cache=True)[0]
                    # Keep frozen KV immutable across checkpoint recomputation.
                    key, value = cache.key_cache[layer.self_attn.layer_idx], cache.value_cache[layer.self_attn.layer_idx]
                def execute(m, layer=layer, key=key, value=value):
                    local = DynamicCache()
                    local.update(key, value, layer.self_attn.layer_idx)
                    with stable_attention(self, positions[length:], valid, prefix.ends):
                        return layer(m, attention_mask=None if causal is None else causal[:, :, length:], position_ids=ids[:, length:],
                            cache_position=positions[length:], position_embeddings=tuple(v[:, length:] for v in rotary),
                            past_key_value=local, use_cache=True)[0]
                memory = checkpoint(execute, memory, use_reentrant=False) if checkpoint_tail and torch.is_grad_enabled() else execute(memory)
                history = next_history.detach()
        return model.norm(memory).float()

    def enhanced(self, tokens, item, metadata, prefix=None, optimized=False, checkpoint_tail=False):
        if self._kernel_active:
            raise RuntimeError('Enhanced forward cannot overlap Native execution')
        if len(tokens.input_ids) != len(item.node_ids):
            raise ValueError('Every real sampled node must traverse shared frozen ChatTime')
        prefix = self.frozen_history(tokens) if prefix is None else prefix
        if prefix.binding != self.input_binding(tokens) or not torch.equal(prefix.mask, tokens.attention_mask):
            raise ValueError('Frozen history binding changed; no stale or graph-conditioned reuse')
        memory, temporal = self.adapter.initialize(prefix, item, metadata)
        memory = self._tail(prefix, memory, item, optimized, checkpoint_tail)
        return self.adapter.predict(memory, temporal, item), memory

    def trainable_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.adapter.state_dict().items()}

    def load_trainable(self, state):
        self.adapter.load_state_dict(state, strict=True); self.version += 1
