"""Stage10 nonlinear local history encoding before interleaved graph aggregation."""
import numpy as np
import torch
from torch import nn

from gofa_f.native import NativeBolt
from gofa_f.ordered_context import OrderedData
from gofa_f.temporal_context import SupportedRoutedExchange


class HistoryProjection(nn.Module):
    """Keep common context linear; encode ordered deviations nonlinearly per node."""
    in_features = 43

    def __init__(self, rank, hidden_width=32):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.base = nn.Linear(19, rank, bias=False)
            self.temporal = nn.Sequential(nn.Linear(24, hidden_width), nn.GELU(),
                                          nn.Linear(hidden_width, rank))
        nn.init.zeros_(self.base.weight)
        nn.init.zeros_(self.temporal[-1].weight)
        nn.init.zeros_(self.temporal[-1].bias)

    def forward(self, features):
        return self.base(features[..., :19]) + self.temporal(features[..., 19:])


class EncodedHistoryBolt(NativeBolt):
    def __init__(self, native, physical, correlation, horizon, variant, history_encoder_width=32, **architecture):
        if variant != "e1_encoded" or architecture.get("graph_feature_dim") != 43 or history_encoder_width != 32:
            raise ValueError("Unregistered Stage10 architecture")
        if physical.shape != correlation.shape or not np.isfinite(correlation).all() or (correlation < 0).any():
            raise ValueError("Invalid train-prefix correlation support")
        super().__init__(native, physical, horizon, **architecture)
        support = np.maximum(physical, correlation)
        for index, old in tuple(self.graph.items()):
            with torch.random.fork_rng(devices=[]):
                module = SupportedRoutedExchange(native.config.d_model, architecture["rank"], 43,
                    architecture["graph_attention_rank"], architecture["graph_prior_floor"], support)
            module.load_state_dict(old.state_dict(), strict=False)
            module.context_projection = HistoryProjection(architecture["rank"], history_encoder_width)
            self.graph[index] = module
        self.graph_architecture.update({"stage10_variant": variant, "history_encoder_width": 32,
            "history": "19 linear context features plus nonlinear 24-to-32-to-rank ordered-history encoding",
            "dynamic_support": "physical_plus_train_correlation"})
