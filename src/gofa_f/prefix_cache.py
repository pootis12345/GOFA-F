"""Disk-backed BF16 frozen-prefix cache, scoped to one dataset and layout.

No validation/test labels or trainable hidden states are cached. Cache keys include
the prepared data, preprocessing, weights, split and exact window origins.
"""
import json
from pathlib import Path

import numpy as np
import torch


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class PrefixCache:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / "metadata.json").read_text())
        if not self.metadata["complete"]:
            raise ValueError("Incomplete prefix cache")
        self.hidden = np.load(self.directory / "hidden.npy", mmap_mode="r+")
        self.mask = np.load(self.directory / "mask.npy", mmap_mode="r+")
        self.loc = np.load(self.directory / "loc.npy", mmap_mode="r+")
        self.scale = np.load(self.directory / "scale.npy", mmap_mode="r+")

    @classmethod
    def build(cls, directory, key, model, batches, count, check_budget):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        metadata = directory / "metadata.json"
        if metadata.exists():
            old = json.loads(metadata.read_text())
            compatible = dict(key, prefix_layers=old.get("prefix_layers", 1))
            if (old.get("complete") and old["key"] == compatible
                    and old["prefix_layers"] <= min(model.graph_layers) + 1):
                return cls(directory)
        atomic_json(metadata, {"complete": False, "key": key, "prefix_layers": 1})
        offset, arrays = 0, None
        for batch in batches:
            check_budget()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                prefix = model.prefix(batch["x"], batch["x_mask"], 1)
            hidden, mask, loc, scale = (v.detach().cpu() for v in prefix)
            if arrays is None:
                specs = {"hidden": (hidden.shape[1:], np.uint16), "mask": (mask.shape[1:], bool),
                         "loc": (loc.shape[1:], np.float32), "scale": (scale.shape[1:], np.float32)}
                arrays = {name: np.lib.format.open_memmap(directory / f"{name}.npy", mode="w+",
                                                         dtype=dtype, shape=(count, *shape))
                          for name, (shape, dtype) in specs.items()}
            end = offset + len(hidden)
            arrays["hidden"][offset:end] = hidden.to(torch.bfloat16).view(torch.uint16).numpy()
            for name, value in (("mask", mask), ("loc", loc), ("scale", scale)):
                arrays[name][offset:end] = value.numpy()
            offset = end
        if offset != count:
            raise ValueError("Prefix cache count mismatch")
        for array in arrays.values():
            array.flush()
        atomic_json(metadata, {"complete": True, "key": key, "prefix_layers": 1})
        return cls(directory)

    def take(self, indices, device):
        # Advanced indexing creates a writable, contiguous batch; never retain a GPU copy of the whole cache.
        hidden = torch.from_numpy(self.hidden[indices]).view(torch.bfloat16).to(device)
        other = [torch.from_numpy(array[indices]).to(device) for array in (self.mask, self.loc, self.scale)]
        return hidden, *other

    @torch.no_grad()
    def upgrade(self, model, microbatch, check_budget):
        if self.metadata["prefix_layers"] == 2:
            return
        if min(model.graph_layers) < 1:
            raise ValueError("Cannot upgrade past a graph module")
        # Mark incomplete before in-place conversion: a crash forces a rebuild, never a mixed cache.
        self.metadata["complete"] = False
        atomic_json(self.directory / "metadata.json", self.metadata)
        for start in range(0, len(self.hidden), microbatch):
            check_budget()
            indices = np.arange(start, min(start + microbatch, len(self.hidden)))
            hidden, mask, _, _ = self.take(indices, next(model.parameters()).device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = model.encode_range(hidden, mask, 1, 2, graphs=False)
            self.hidden[indices] = hidden.cpu().to(torch.bfloat16).view(torch.uint16).numpy()
        self.hidden.flush()
        self.metadata.update(complete=True, prefix_layers=2)
        self.metadata["key"]["prefix_layers"] = 2
        atomic_json(self.directory / "metadata.json", self.metadata)

    def close(self, delete=False):
        for name in ("hidden", "mask", "loc", "scale"):
            array = getattr(self, name)
            array._mmap.close()
            if delete:
                (self.directory / f"{name}.npy").unlink()
        if delete:
            (self.directory / "metadata.json").unlink()
