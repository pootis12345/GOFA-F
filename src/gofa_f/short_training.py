"""Training-only compact vocabulary projection and isolated two-query batching.

The public all-node forward and all generation paths remain unchanged. Per-row
attention keeps each query's original capacity and one-row math-SDPA execution;
linear/MLP work and the loss are batched without mixing graph contexts.
"""
from contextlib import contextmanager
import hashlib
import math

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from gofa_f.allnode_graph import build_graph


@contextmanager
def teacher_attention(owner, tokens):
    if owner.native.dtype not in (torch.bfloat16, torch.float16) and not owner.stable_fp32:
        yield
        return
    if owner._kernel_active:
        raise RuntimeError("Training attention cannot nest graph execution")
    owner._kernel_active = True
    original = F.scaled_dot_product_attention
    length = tokens.input_ids.shape[1]
    positions = torch.arange(length, device=tokens.input_ids.device)
    masks = []
    for row, end in enumerate(tokens.prompt_ends.tolist()):
        capacity = math.ceil((end+1+128)/32)*32
        if length > capacity:
            raise ValueError("Teacher batch exceeds a row's original attention capacity")
        valid = F.pad(tokens.attention_mask[row:row+1].bool(), (0, capacity-length))
        causal = torch.ones((capacity, capacity), dtype=torch.bool, device=valid.device).tril()
        masks.append((capacity, causal[None, None] & valid[:, None, None]))
    def attention(query, key, value, attn_mask=None, dropout_p=0., is_causal=False, scale=None, **kwargs):
        if dropout_p or query.shape[2] != length or key.shape[2] != length:
            raise ValueError("Only uncached, zero-dropout teacher attention is supported")
        parts = []
        for row, (capacity, allowed) in enumerate(masks):
            q = query[row:row+1]
            full_q = q.new_zeros((*q.shape[:2], capacity, q.shape[-1]))
            full_q = full_q.index_copy(2, positions, q).contiguous()
            k = F.pad(key[row:row+1], (0, 0, 0, capacity-length)).contiguous()
            v = F.pad(value[row:row+1], (0, 0, 0, capacity-length)).contiguous()
            result = original(full_q, k, v, attn_mask=allowed, dropout_p=0.,
                              is_causal=False, scale=scale, **kwargs)
            parts.append(result.index_select(2, positions))
        return torch.cat(parts, 0)
    F.scaled_dot_product_attention = attention
    try:
        with sdpa_kernel(SDPBackend.MATH):
            yield
    finally:
        F.scaled_dot_product_attention = original
        owner._kernel_active = False


@torch.no_grad()
def teacher_prefix(model, tokens, cache=None):
    key = hashlib.sha256(("stage19_two_query_teacher_v1"+cache.key(tokens)).encode()).hexdigest() if cache else None
    saved = cache.get(key, tokens.input_ids.device) if cache else None
    if saved is not None:
        return saved
    native = model.native.model
    hidden = native.embed_tokens(tokens.input_ids)
    positions = torch.arange(hidden.shape[1], device=hidden.device)
    ids = model.positions(tokens.attention_mask, hidden.shape[1])
    causal = native._update_causal_mask(tokens.attention_mask, hidden, positions, None, False)
    rotary = native.rotary_emb(hidden, ids)
    with teacher_attention(model, tokens):
        for layer in native.layers[:model.prefix_layers]:
            hidden = layer(hidden, attention_mask=causal, position_ids=ids,
                cache_position=positions, position_embeddings=rotary,
                use_cache=False, output_attentions=False)[0]
    result = hidden.detach(), tokens.attention_mask
    if cache:
        cache.put(key, result)
    return result


def compact_forward(model, codec, tokens, contexts, prefix=None):
    """Return [observed numeric slots, 10000], never [batch, sequence, 42001]."""
    if model._active is not None or len(contexts) != len(tokens.input_ids):
        raise ValueError("One isolated graph context per teacher query required")
    for context, positions in zip(contexts, tokens.value_positions):
        if context.version != model.parameter_version or context.generation != model.cache_generation \
                or len(positions) < context.length:
            raise ValueError("Stale graph context in numeric training")
    if prefix is None:
        prefix = teacher_prefix(model, tokens)
    hidden, mask = prefix
    if not torch.equal(mask, tokens.attention_mask):
        raise ValueError("Teacher prefix mask changed")
    native = model.native.model
    positions = torch.arange(hidden.shape[1], device=hidden.device)
    ids = model.positions(mask, hidden.shape[1])
    causal = native._update_causal_mask(mask, hidden, positions, None, False)
    rotary = native.rotary_emb(hidden, ids)
    model._active = (tokens, list(contexts), 0)
    try:
        with teacher_attention(model, tokens):
            for layer in native.layers[model.prefix_layers:]:
                hidden = layer(hidden, attention_mask=causal, position_ids=ids,
                    cache_position=positions, position_embeddings=rotary,
                    use_cache=False, output_attentions=False)[0]
            hidden = native.norm(hidden)
    finally:
        model._active = None
    labels = tokens.labels[:, 1:]
    rows, steps = torch.nonzero(labels != -100, as_tuple=True)
    counts = (labels != -100).sum(-1)
    if (counts == 0).any():
        raise ValueError("Each query needs an observed numeric target")
    alphabet = codec.alphabet(hidden.device)
    head = model.native.lm_head
    weight = head.weight.index_select(0, alphabet)
    bias = head.bias.index_select(0, alphabet) if head.bias is not None else None
    selected = hidden[rows, steps]
    # Retain the registered 128-row projection execution, including a short tail.
    parts = []
    for start in range(0, len(selected), 128):
        part = selected[start:start+128]
        parts.append(F.linear(F.pad(part, (0, 0, 0, 128-len(part))), weight, bias)[:len(part)])
    scores = torch.cat(parts, 0).float()
    targets = codec.lookup(head.out_features, hidden.device)[labels[rows, steps]]
    if (targets < 0).any():
        raise ValueError("Supervised token is outside the finite numeric alphabet")
    return scores, targets, rows, counts


def compact_loss(result):
    scores, targets, rows, counts = result
    losses = F.cross_entropy(scores, targets, reduction="none")
    per_query = scores.new_zeros(len(counts)).scatter_add(0, rows, losses)/counts
    return per_query.sum(), int(counts.sum())


def training_batch(model, codec, series, queries, cache=None, check=lambda: None):
    records, graphs = [], []
    for q in queries:
        check()
        data = series[q.source]
        history = np.where(data.mask[q.origin-q.length:q.origin, q.variable],
                           data.values[q.origin-q.length:q.origin, q.variable], np.nan)
        target = np.where(data.mask[q.origin:q.origin+q.horizon, q.variable],
                          data.values[q.origin:q.origin+q.horizon, q.variable], np.nan)
        records.append(codec.record(history, target))
        graphs.append(build_graph(data, q))
    device = next(model.native.parameters()).device
    tokens = codec.batch(records).to(device)
    prefix = teacher_prefix(model, tokens, cache)
    contexts = model.encode_origins(graphs, codec, cache, check)
    return compact_loss(compact_forward(model, codec, tokens, contexts, prefix))
