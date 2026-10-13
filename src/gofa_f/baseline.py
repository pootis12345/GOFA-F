"""Adapter for the official IJCAI 2025 DST-SGNN implementation.

Download pinned upstream sources ON THE SERVER with scripts/fetch_baseline.py.
No substitute model is silently reported as DST-SGNN.
"""
import importlib
import json
from pathlib import Path
import sys

import torch
from torch import nn

from gofa_f.data import file_sha256


class DSTSGNNAdapter(nn.Module):
    def __init__(self, context, horizon, nodes, root="external/dst_sgnn", width=64, hidden_size=1024):
        super().__init__()
        root = Path(root).resolve()
        manifest = json.loads((root / "manifest.json").read_text())
        for name, expected in manifest["adapted_sha256"].items():
            if file_sha256(root / name) != expected:
                raise ValueError(f"Baseline source hash mismatch: {name}")
        sys.path.insert(0, str(root))
        module = importlib.import_module("dst_sgnn_upstream.MODEL")
        self.provenance = manifest
        self.inner = module.DSTSGNN(stride=24, pre_length=horizon, embed_size=width,
                                   feature_size=nodes, seq_length=context, hidden_size=hidden_size,
                                   patch_len=min(24, context), d_model=width).float()

    def forward(self, x, x_mask):
        return self.inner(torch.where(x_mask, x, 0)).transpose(1, 2)
