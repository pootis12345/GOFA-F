"""Stage12 interleaved graph increments encode neighbor contrast."""
import numpy as np
import torch

from gofa_f.native import NativeBolt
from gofa_f.temporal_context import SketchData, SupportedRoutedExchange


class ContrastExchange(SupportedRoutedExchange):
    def forward(self, hidden, adjacency, token_mask, features):
        if features is None or features.shape != (*hidden.shape[:2], self.context_projection.in_features):
            raise ValueError("Contrast graph requires the registered node context")
        if adjacency.shape != (hidden.shape[1], hidden.shape[1]) or token_mask.shape != hidden.shape[:3]:
            raise ValueError("Graph or history mask dimensions changed")
        with torch.autocast(hidden.device.type, enabled=False):
            safe = hidden.float().masked_fill(~token_mask[..., None], 0)
            z = self.down(self.norm(safe)) + self.context_projection(features.float())[:, :, None]
            z = z.masked_fill(~token_mask[..., None], 0)
            attention, fraction, amplitude = self.routing(z, adjacency, token_mask)
            prior = self.aggregate(adjacency.float(), z, token_mask)
            dynamic = self.aggregate(attention, z, token_mask)
            # The native residual already contains self state; graph increments carry contrast.
            contrast = prior + fraction[:, :, None]*(dynamic-prior) - z
            delta = amplitude[:, :, None]*self.up(torch.nn.functional.gelu(contrast))
            delta = delta.masked_fill(~token_mask[..., None], 0)
        return hidden + delta.to(hidden.dtype)


class ContrastGraphBolt(NativeBolt):
    def __init__(self, native, physical, correlation, horizon, variant, **architecture):
        if variant != "d1_contrast" or architecture.get("graph_feature_dim") != 19:
            raise ValueError("Unregistered Stage12 contrast architecture")
        if physical.shape != correlation.shape or not np.isfinite(correlation).all() or (correlation < 0).any():
            raise ValueError("Invalid train-prefix correlation graph")
        super().__init__(native, physical, horizon, **architecture)
        support = np.maximum(physical, correlation)
        for index, old in tuple(self.graph.items()):
            with torch.random.fork_rng(devices=[]):
                module = ContrastExchange(native.config.d_model, architecture["rank"], 19,
                    architecture["graph_attention_rank"], architecture["graph_prior_floor"], support)
            module.load_state_dict(old.state_dict(), strict=False)
            self.graph[index] = module
        self.graph_architecture.update({"stage12_variant": variant,
            "message": "physical_dynamic_mixture_minus_self_latent", "dynamic_support": "physical_plus_train_correlation"})
