"""Causal, unit-aligned node statistics and explicitly available clock features."""
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from gofa_f.data import file_sha256
from gofa_f.models import GraphExchange


FEATURE_NAMES = ("level", "log_scale", "last", "trend", "observed_fraction",
                 "sin_day", "cos_day", "sin_week", "cos_week", "day_known", "week_known")
FEATURE_DIM = len(FEATURE_NAMES)


def clock_features(calendar, known):
    """calendar[...,2] = source-clock day fraction and weekday; unknown fields have no phase."""
    day, weekday = calendar.unbind(-1)
    day_known = known[..., 0]
    week_known = known[..., 0] & known[..., 1]
    daily = day * (2 * math.pi)
    weekly = (weekday + day) * (2 * math.pi / 7)
    return torch.stack((daily.sin() * day_known, daily.cos() * day_known,
                        weekly.sin() * week_known, weekly.cos() * week_known,
                        day_known.to(day.dtype), week_known.to(day.dtype)), -1)


def node_features(x, observed, train_mean, train_std, pooled_mean, pooled_std, clock):
    """x[B,L,N] is train-standardized; no target values are accepted by this API.

    Convert back to common physical units BEFORE aligning nodes with pooled TRAIN
    statistics. Passing each node's separately normalized loc alone is insufficient.
    """
    raw = x.float() * train_std + train_mean
    count = observed.sum(1).clamp_min(1)
    safe = max(float(pooled_std), 1e-5)
    values = torch.where(observed, (raw - pooled_mean) / safe, 0)
    level = values.sum(1) / count
    variance = (torch.where(observed, values - level[:, None], 0).square()).sum(1) / count
    scale = variance.clamp_min(0).sqrt().log1p()
    steps = torch.arange(x.shape[1], device=x.device, dtype=torch.float32)[None, :, None]
    last_index = steps.long().expand_as(x).masked_fill(~observed, -1).max(1).values
    last = values.gather(1, last_index.clamp_min(0)[:, None]).squeeze(1)
    last = last.masked_fill(last_index < 0, 0)
    time_mean = (steps * observed).sum(1) / count
    centered_time = torch.where(observed, steps - time_mean[:, None], 0)
    trend = (centered_time * values).sum(1) / centered_time.square().sum(1).clamp_min(1)
    trend = trend * max(x.shape[1] - 1, 1)
    fraction = observed.float().mean(1)
    stats = torch.stack((level, scale, last, trend, fraction), -1)
    stats = stats.masked_fill(~observed.any(1)[..., None], 0)
    time = clock[:, None].expand(-1, x.shape[2], -1)
    result = torch.cat((stats, time), -1)
    if result.shape[-1] != FEATURE_DIM or not torch.isfinite(result).all():
        raise ValueError("Invalid causal node context")
    return result


class ContextGraphExchange(GraphExchange):
    def __init__(self, width, rank, feature_dim):
        super().__init__(width, rank, 0.0)
        # Adding a zero projection must not perturb the NEXT graph layer's paired initialization.
        with torch.random.fork_rng(devices=[]):
            self.context_projection = nn.Linear(feature_dim, rank, bias=False)
        # Preserve the legacy graph model when adding this branch to an existing checkpoint.
        nn.init.zeros_(self.context_projection.weight)

    def forward(self, hidden, adjacency, token_mask, features):
        if features is None or features.shape != (*hidden.shape[:2], self.context_projection.in_features):
            raise ValueError("Context graph requires explicit per-node statistics and clock features")
        z = self.down(self.norm(hidden)) + self.context_projection(features.float())[:, :, None]
        z = z * token_mask[..., None]
        weights = adjacency.to(z.dtype)
        message = torch.einsum("ij,bjpr->bipr", weights, z)
        mass = torch.einsum("ij,bjp->bip", weights, token_mask.to(z.dtype)).clamp_min(1e-6)
        message = self.up(torch.nn.functional.gelu(message / mass[..., None]))
        return hidden + self.gate.tanh() * message


def load_calendar(path, fingerprint, length):
    path = Path(path)
    metadata = json.loads(path.with_suffix(".metadata.json").read_text())
    if metadata["prepared_sha256"] != fingerprint or metadata["sidecar_sha256"] != file_sha256(path):
        raise ValueError("Calendar sidecar does not match prepared data or its audit hash")
    with np.load(path, allow_pickle=False) as source:
        calendar, known = source["calendar"].astype(np.float32), source["known"].astype(bool)
    if calendar.shape != (length, 2) or known.shape != calendar.shape or not np.isfinite(calendar).all():
        raise ValueError("Invalid calendar shape or nonfinite values")
    if ((calendar[known[:, 0], 0] < 0) | (calendar[known[:, 0], 0] >= 1)).any():
        raise ValueError("Day fraction must be in [0,1)")
    weekdays = calendar[known[:, 1], 1]
    if ((weekdays < 0) | (weekdays > 6) | (weekdays != np.floor(weekdays))).any():
        raise ValueError("Invalid known weekday")
    return calendar, known, metadata
