"""Stage9 adds lossless masked within-segment order to the Stage8 graph context."""
import numpy as np
import torch

from gofa_f.native import NativeBolt
from gofa_f.temporal_context import SketchData, SupportedRoutedExchange


def ordered_history(x, observed, mean, std, pooled_mean, pooled_std):
    if x.ndim != 3 or x.shape[1] != 12 or observed.shape != x.shape:
        raise ValueError("Expected twelve ordered history values and masks")
    raw = (x.float()*std + mean - pooled_mean) / pooled_std
    grouped = raw.reshape(x.shape[0], 4, 3, x.shape[2])
    valid = observed.reshape_as(grouped)
    center = grouped.masked_fill(~valid, 0).sum(2, keepdim=True) / valid.sum(2, keepdim=True).clamp_min(1)
    residual = (grouped-center).masked_fill(~valid, 0).reshape_as(x)
    # Adjacent value/mask pairs retain the position of every observed step.
    return torch.stack((residual, observed.to(raw.dtype)), -1).permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[2], 24)


class OrderedData(SketchData):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.context_provenance["feature_names"] += [f"history_{i}_{kind}" for i in range(12)
                                                     for kind in ("segment_residual", "observed")]

    def batch(self, split, indices):
        batch = super().batch(split, indices)
        detail = ordered_history(batch["x"], batch["x_mask"], self.mean, self.std,
                                 self.pooled_mean, self.pooled_std)
        batch["graph_features"] = torch.cat((batch["graph_features"], detail), -1)
        return batch


class OrderedHistoryBolt(NativeBolt):
    def __init__(self, native, physical, correlation, horizon, variant, **architecture):
        if variant != "o1_ordered" or architecture.get("graph_feature_dim") != 43:
            raise ValueError("Unregistered Stage9 architecture")
        if physical.shape != correlation.shape or not np.isfinite(correlation).all() or (correlation < 0).any():
            raise ValueError("Invalid train-prefix correlation graph")
        super().__init__(native, physical, horizon, **architecture)
        support = np.maximum(physical, correlation)
        for index, old in tuple(self.graph.items()):
            with torch.random.fork_rng(devices=[]):
                module = SupportedRoutedExchange(native.config.d_model, architecture["rank"], 43,
                    architecture["graph_attention_rank"], architecture["graph_prior_floor"], support)
            module.load_state_dict(old.state_dict(), strict=False)
            self.graph[index] = module
        self.graph_architecture.update({"stage9_variant": variant,
            "history": "19 Stage8 features plus 12 ordered within-segment deviations and 12 masks",
            "dynamic_support": "physical_plus_train_correlation"})
