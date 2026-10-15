"""Stage11 train-prefix lead/lag association support, without causal claims."""
import numpy as np

from gofa_f.temporal_context import SketchData, TemporalSketchBolt


def lagged_correlation_graph(values, mask, segments, train_end, lags=(1, 2, 3), k=8, min_overlap=24):
    """adjacency[target,source] uses source[t-lag] paired with target[t]."""
    if values.shape != mask.shape or values.ndim != 2 or len(segments) != len(values):
        raise ValueError("Invalid values, masks or segment labels")
    if not 0 < train_end <= len(values) or not lags or any(lag <= 0 or lag >= train_end for lag in lags):
        raise ValueError("Invalid train prefix or lags")
    values = values[:train_end]
    mask = mask[:train_end] & np.isfinite(values)
    segments = np.asarray(segments[:train_end])
    runs = np.cumsum(np.r_[0, segments[1:] != segments[:-1]])
    nodes = values.shape[1]
    score, best_lag = np.zeros((nodes, nodes), np.float64), np.zeros((nodes, nodes), np.int16)
    for lag in lags:
        same_run = runs[lag:] == runs[:-lag]
        mt = mask[lag:][same_run].astype(np.float64)
        ms = mask[:-lag][same_run].astype(np.float64)
        yt = np.where(mt, values[lag:][same_run], 0).astype(np.float64)
        xs = np.where(ms, values[:-lag][same_run], 0).astype(np.float64)
        overlap = mt.T @ ms
        safe = np.maximum(overlap, 1)
        sy, sx = yt.T @ ms, mt.T @ xs
        covariance = yt.T @ xs - sy*sx/safe
        vy = np.maximum((yt*yt).T @ ms - sy*sy/safe, 0)
        vx = np.maximum(mt.T @ (xs*xs) - sx*sx/safe, 0)
        corr = np.abs(covariance / np.sqrt(np.maximum(vy*vx, 1e-12))).clip(0, 1)
        corr[overlap < min_overlap] = 0
        np.fill_diagonal(corr, 0)
        better = corr > score
        best_lag[better] = lag
        score = np.maximum(score, corr)
    keep = np.zeros_like(score)
    count = min(max(int(k), 0), nodes-1)
    if count:
        indices = np.argsort(score, axis=1)[:, -count:]
        np.put_along_axis(keep, indices, np.take_along_axis(score, indices, axis=1), axis=1)
    best_lag[keep <= 0] = 0
    return (keep+np.eye(nodes)).astype(np.float32), best_lag


class LaggedSupportBolt(TemporalSketchBolt):
    def __init__(self, native, physical, correlation, lagged, horizon, variant, **architecture):
        if variant != "l1_lagged" or lagged.shape != correlation.shape or \
                not np.isfinite(lagged).all() or (lagged < 0).any():
            raise ValueError("Invalid Stage11 variant or lagged support")
        super().__init__(native, physical, np.maximum(correlation, lagged), horizon,
                         "t2_correlated", **architecture)
        self.graph_architecture.pop("stage8_variant")
        self.graph_architecture.update({"stage11_variant": variant,
            "dynamic_support": "physical plus locked zero-lag correlation plus directed max-lag-1-2-3 top8"})
