"""Shape-stable BF16 execution for full versus one-token causal forwards.

No parameter/dtype/causal-meaning changes. Projection rows are tiled identically
and SDPA uses a fixed origin-prompt capacity, including in incremental decoding.
The native text bypass does not enter this context. KV remains native HF KV.
"""
from contextlib import contextmanager
import math

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def install_linear_tiles(native, active, stable_fp32=False):
    for layer in native.modules():
        if not isinstance(layer, torch.nn.Linear):
            continue
        original = layer.forward
        def tiled(x, layer=layer, original=original):
            if not active() or (x.dtype not in (torch.bfloat16, torch.float16) and not stable_fp32):
                return original(x)
            shape = x.shape
            rows = x.reshape(-1, shape[-1])
            result = []
            for start in range(0, len(rows), 128):
                part = rows[start:start+128]
                count = len(part)
                part = F.pad(part, (0, 0, 0, 128-count))
                result.append(F.linear(part, layer.weight, layer.bias)[:count])
            return torch.cat(result, 0).reshape(*shape[:-1], layer.out_features)
        layer.forward = tiled


@contextmanager
def stable_attention(owner, positions, mask, prompt_ends):
    if owner.native.dtype not in (torch.bfloat16, torch.float16) and not owner.stable_fp32:
        yield
        return
    capacity = math.ceil((int(prompt_ends.max())+1+128)/32)*32
    if mask.shape[1] > capacity:
        raise ValueError("Completion exceeds the fixed numerical execution capacity; no truncation")
    if owner._kernel_active:
        raise RuntimeError("Shape-stable model execution is not reentrant")
    owner._kernel_active = True
    original = F.scaled_dot_product_attention
    valid = F.pad(mask.bool(), (0, capacity-mask.shape[1]))
    causal = torch.ones((capacity, capacity), dtype=torch.bool, device=mask.device).tril()
    allowed = causal[None, None] & valid[:, None, None]
    def attention(query, key, value, attn_mask=None, dropout_p=0., is_causal=False, scale=None, **kwargs):
        if dropout_p:
            raise ValueError("Frozen Stage17 backbone requires zero attention dropout")
        if query.shape[2] != len(positions) or key.shape[2] > capacity:
            raise ValueError("Unexpected native KV/query positions")
        full_query = query.new_zeros((*query.shape[:2], capacity, query.shape[-1]))
        full_query = full_query.index_copy(2, positions, query).contiguous()
        full_key = F.pad(key, (0, 0, 0, capacity-key.shape[2])).contiguous()
        full_value = F.pad(value, (0, 0, 0, capacity-value.shape[2])).contiguous()
        output = original(full_query, full_key, full_value, attn_mask=allowed,
                          dropout_p=0., is_causal=False, scale=scale, **kwargs)
        return output.index_select(2, positions)
    F.scaled_dot_product_attention = attention
    try:
        with sdpa_kernel(SDPBackend.MATH):
            yield
    finally:
        F.scaled_dot_product_attention = original
        owner._kernel_active = False
