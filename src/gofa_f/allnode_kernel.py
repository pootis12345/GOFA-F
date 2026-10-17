"""Stage18 shape-stable RMSNorm; native text bypass and old stages are unchanged."""
import torch
from torch.nn import functional as F


EXECUTION = "bf16_rows128_rms128_sdpa_fixed_prompt_capacity_v2"


def install_norm_tiles(native, active, stable_fp32=False):
    for layer in native.modules():
        if type(layer).__name__ != "LlamaRMSNorm":
            continue
        original = layer.forward
        def tiled(x, original=original):
            if not active() or (x.dtype not in (torch.bfloat16, torch.float16) and not stable_fp32):
                return original(x)
            shape, parts = x.shape, []
            rows = x.reshape(-1, shape[-1])
            for start in range(0, len(rows), 128):
                part = rows[start:start+128]
                count = len(part)
                part = F.pad(part, (0, 0, 0, 128-count)).contiguous()
                parts.append(original(part)[:count])
            return torch.cat(parts, 0).reshape(shape)
        layer.forward = tiled
