"""Stage8 history sketches and dynamic-edge support for interleaved GOFA-F."""

import numpy as np
import torch

from gofa_f.native import NativeBolt
from gofa_f.native_stage import Data
from gofa_f.routed_graph import PriorRoutedGraphExchange


def history_sketch(x, observed, mean, std, pooled_mean, pooled_std):
    """Four ordered, masked three-step means and coverages; never reads targets."""
    if x.ndim != 3 or x.shape[1] != 12 or observed.shape != x.shape:
        raise ValueError("Stage8 requires twelve history values and matching masks")
    values = (x * std[None, None, :] + mean[None, None, :] - pooled_mean) / pooled_std
    values = values.reshape(x.shape[0], 4, 3, x.shape[2])
    valid = observed.reshape_as(values)
    count = valid.sum(2)
    average = values.masked_fill(~valid, 0).sum(2) / count.clamp_min(1)
    coverage = count.to(values.dtype) / 3
    result = torch.stack((average, coverage), dim=-1).permute(0, 2, 1, 3)
    return result.reshape(x.shape[0], x.shape[2], 8)


class SketchData(Data):
    def __init__(self, config, device):
        super().__init__(config, device)
        if self.context != 12 or not self.context_enabled or self.context_mode != "values_and_clock":
            raise ValueError("Stage8 requires 12-step history and the 11 clock/value features")
        self.context_provenance["feature_names"] = [*self.context_provenance["feature_names"],
            *(f"segment_{i}_{kind}" for i in range(4) for kind in ("masked_mean", "coverage"))]

    def batch(self, split, indices):
        result = super().batch(split, indices)
        sketch = history_sketch(result["x"], result["x_mask"], self.mean, self.std,
                                self.pooled_mean, self.pooled_std)
        result["graph_features"] = torch.cat((result["graph_features"], sketch), dim=-1)
        return result


class SupportedRoutedExchange(PriorRoutedGraphExchange):
    """Correlation edges affect learned routing only; inherited prior uses physical graph."""

    def __init__(self, width, rank, feature_dim, attention_rank, prior_floor, support):
        super().__init__(width, rank, feature_dim, attention_rank, prior_floor)
        self.register_buffer("dynamic_support", torch.as_tensor(support, dtype=torch.float32))

    def routing(self, z, adjacency, token_mask):
        return super().routing(z, self.dynamic_support, token_mask)


class TemporalSketchBolt(NativeBolt):
    def __init__(self, native, physical, correlation, horizon, variant, **architecture):
        if variant not in ("t1_sketch", "t2_correlated"):
            raise ValueError("Unknown Stage8 variant")
        if architecture.get("graph_feature_dim") != 19 or architecture.get("graph_prior_floor", 0) < .5:
            raise ValueError("Stage8 history features or physical prior changed")
        if physical.shape != correlation.shape or not np.isfinite(correlation).all() or (correlation < 0).any():
            raise ValueError("Invalid correlation support")
        super().__init__(native, physical, horizon, **architecture)
        support = np.maximum(physical, correlation) if variant == "t2_correlated" else physical
        for index, old in tuple(self.graph.items()):
            with torch.random.fork_rng(devices=[]):
                replacement = SupportedRoutedExchange(native.config.d_model, architecture["rank"], 19,
                    architecture["graph_attention_rank"], architecture["graph_prior_floor"], support)
            replacement.load_state_dict(old.state_dict(), strict=False)
            self.graph[index] = replacement
        self.graph_architecture.update({"stage8_variant": variant,
                                        "dynamic_support": "physical_plus_train_correlation" if variant == "t2_correlated"
                                        else "physical_only"})
