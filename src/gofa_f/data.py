"""Chronological forecast windows with train-only statistics and graph fitting."""
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class Series:
    values: np.ndarray
    mask: np.ndarray
    segments: np.ndarray
    train_end: int
    val_end: int
    fingerprint: str

    @classmethod
    def load(cls, path, train_ratio=0.7, val_ratio=0.1):
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            values = data["values"].astype(np.float32)
            mask = data["mask"].astype(bool) & np.isfinite(values)
            segments = data["segments"] if "segments" in data else np.zeros(len(values), dtype=np.int64)
            train_end = int(data["train_end"]) if "train_end" in data else int(len(values) * train_ratio)
            val_end = int(data["val_end"]) if "val_end" in data else int(len(values) * (train_ratio + val_ratio))
        if values.ndim != 2 or mask.shape != values.shape or segments.shape != (len(values),):
            raise ValueError("Expected values/mask [time,nodes] and segments [time]")
        if not 0 < train_end < val_end < len(values):
            raise ValueError("Invalid chronological split boundaries")
        if mask[:train_end].sum() < 2:
            raise ValueError("At least two observed training values are required")
        return cls(np.where(mask, values, 0), mask, segments, train_end, val_end, file_sha256(path))

    def standardize(self):
        x, m = self.values[:self.train_end], self.mask[:self.train_end]
        count = m.sum(0)
        global_mean = x[m].mean(dtype=np.float64)
        global_var = x[m].var(dtype=np.float64)
        mean = (x * m).sum(0, dtype=np.float64) / np.maximum(count, 1)
        var = (((x - mean) ** 2) * m).sum(0) / np.maximum(count, 1)
        # Cold-start stations use same-unit pooled TRAINING statistics, never future data.
        mean = np.where(count >= 2, mean, global_mean)
        var = np.where(count >= 2, var, global_var)
        std = np.sqrt(var).clip(1e-5).astype(np.float32)
        mean = mean.astype(np.float32)
        z = np.where(self.mask, (self.values - mean) / std, 0).astype(np.float32)
        return z, mean, std


class Windows(Dataset):
    def __init__(self, series, normalized, split, context, horizon, stride=1):
        if context < 1 or horizon < 1 or stride < 1:
            raise ValueError("context, horizon and stride must be positive")
        lo, hi = {"train": (0, series.train_end), "val": (series.train_end, series.val_end),
                  "test": (series.val_end, len(series.values))}[split]
        # Targets lie wholly inside the split. Histories may use earlier observed splits.
        cuts = np.flatnonzero(series.segments[1:] != series.segments[:-1]) + 1
        self.origins = []
        for origin in range(max(context, lo), hi - horizon + 1, stride):
            left = np.searchsorted(cuts, origin - context, side="right")
            right = np.searchsorted(cuts, origin + horizon - 1, side="right")
            if left == right and series.mask[origin:origin + horizon].any():
                self.origins.append(origin)
        if not self.origins:
            raise ValueError(f"No valid {split} windows; check splits, context and segment lengths")
        self.series, self.z, self.context, self.horizon = series, normalized, context, horizon

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        t, c, h = self.origins[index], self.context, self.horizon
        return {"x": torch.from_numpy(self.z[t-c:t]),
                "x_mask": torch.from_numpy(self.series.mask[t-c:t]),
                "y": torch.from_numpy(self.z[t:t+h]),
                "y_raw": torch.from_numpy(self.series.values[t:t+h]),
                "y_mask": torch.from_numpy(self.series.mask[t:t+h])}


def correlation_graph(values, mask, k=8, min_overlap=24):
    """Pairwise-complete Pearson correlations. Call with the training prefix only.

    adjacency[target, source] is nonnegative, symmetric and has self loops.
    """
    x = np.where(mask, values, 0).astype(np.float64)
    m = mask.astype(np.float64)
    overlap = m.T @ m
    safe = np.maximum(overlap, 1)
    sums = x.T @ m
    squared_sums = (x * x).T @ m
    cov = x.T @ x - sums * sums.T / safe
    var = np.maximum(squared_sums - sums * sums / safe, 0)
    corr = np.abs(cov / np.sqrt(np.maximum(var * var.T, 1e-12)))
    corr[overlap < min_overlap] = 0
    np.fill_diagonal(corr, 0)
    n = len(corr)
    k = min(max(int(k), 0), n - 1)
    keep = np.zeros_like(corr)
    if k:
        idx = np.argsort(corr, axis=1)[:, -k:]
        np.put_along_axis(keep, idx, np.take_along_axis(corr, idx, axis=1), axis=1)
    return (np.maximum(keep, keep.T) + np.eye(n)).astype(np.float32)
