import numpy as np
import torch


def masked_loss(prediction, target, mask, kind="mae"):
    count = mask.sum()
    if count == 0:
        raise ValueError("No observed targets")
    error = (prediction - target).masked_select(mask)
    if not torch.isfinite(error).all():
        raise FloatingPointError("Nonfinite forecast or target")
    return error.square().mean() if kind == "mse" else error.abs().mean()


class Metrics:
    def __init__(self, horizon, mape_threshold=1e-5):
        self.abs = np.zeros(horizon, dtype=np.float64)
        self.square = np.zeros(horizon, dtype=np.float64)
        self.count = np.zeros(horizon, dtype=np.int64)
        self.ape = np.zeros(horizon, dtype=np.float64)
        self.ape_count = np.zeros(horizon, dtype=np.int64)
        self.threshold = mape_threshold

    def update(self, prediction, target, mask):
        p, y, m = (v.detach().cpu().numpy() for v in (prediction, target, mask))
        if not np.isfinite(p[m]).all():
            raise FloatingPointError("Nonfinite predictions on observed targets")
        error = np.where(m, p-y, 0).astype(np.float64)
        self.abs += np.abs(error).sum((0, 2))
        self.square += np.square(error).sum((0, 2))
        self.count += m.sum((0, 2))
        pm = m & (np.abs(y) > self.threshold)
        self.ape += np.where(pm, np.abs(error) / np.maximum(np.abs(y), self.threshold), 0).sum((0, 2))
        self.ape_count += pm.sum((0, 2))

    def result(self):
        if np.any(self.count == 0):
            raise ValueError("At least one horizon has no observed evaluation targets")
        count, ape_count = self.count.sum(), self.ape_count.sum()
        return {"mae": float(self.abs.sum()/count), "mse": float(self.square.sum()/count),
                "rmse": float(np.sqrt(self.square.sum()/count)),
                "mape_pct": float(100*self.ape.sum()/ape_count) if ape_count else None,
                "observations": int(count),
                "by_horizon": [{"step": i+1, "mae": float(a/c), "rmse": float(np.sqrt(s/c)),
                                "observations": int(c)}
                               for i, (a, s, c) in enumerate(zip(self.abs, self.square, self.count))]}
