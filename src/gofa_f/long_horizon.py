"""Pre-registered METR-LA clock and shared-node residual candidates."""
import torch
from torch import nn

from gofa_f.native import NativeBolt


class TemporalResidual(nn.Module):
    """Causal 12-step history correction in the native standardized scale."""
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.first = nn.Linear(30, 64)
            self.last = nn.Linear(64, 12)
        nn.init.zeros_(self.last.weight)
        nn.init.zeros_(self.last.bias)

    def forward(self, history, observed, clock, loc, scale):
        if history.shape[1] != 12 or clock.shape != (history.shape[0], 6):
            raise ValueError("Residual head requires 12 observed history steps and six origin-clock values")
        safe_scale = scale.float().clamp_min(1e-5)
        values = history.transpose(1, 2).float()
        mask = observed.transpose(1, 2)
        values = torch.where(mask, (values - loc.float()) / safe_scale, 0)
        features = torch.cat((values, mask.float(),
                              clock[:, None].expand(-1, values.shape[1], -1).float()), -1)
        with torch.autocast(history.device.type, enabled=False):
            delta = self.last(torch.nn.functional.gelu(self.first(features))) * safe_scale
        return delta.masked_fill(~mask.any(-1)[..., None], 0)


class LongHorizonBolt(NativeBolt):
    requires_history = True

    def __init__(self, *args, temporal_residual=False, **kwargs):
        super().__init__(*args, **kwargs)
        if self.horizon != 12 or self.graph_feature_dim != 11:
            raise ValueError("Stage6 is fixed to 12 horizons and eleven causal graph features")
        self.temporal_residual = TemporalResidual() if temporal_residual else None
        self.graph_architecture["temporal_residual"] = bool(temporal_residual)

    def from_prefix(self, prefix, layers, return_quantiles=False, graph_features=None,
                    history=None, history_mask=None):
        quantiles = super().from_prefix(prefix, layers, True, graph_features)
        if self.temporal_residual is not None:
            if history is None or history_mask is None or graph_features is None:
                raise ValueError("Causal residual inputs are required")
            loc, scale = prefix[2:]
            delta = self.temporal_residual(history, history_mask, graph_features[:, 0, 5:], loc, scale)
            quantiles = quantiles + delta.transpose(1, 2)[:, :, None, :].to(quantiles.dtype)
        return quantiles if return_quantiles else quantiles[:, :, self.median_index].transpose(1, 2)

    def forward(self, x, mask, graph_features=None, return_quantiles=False):
        layers = min(self.graph_layers) + 1
        return self.from_prefix(self.prefix(x, mask, layers), layers, return_quantiles,
                                graph_features, x, mask)
