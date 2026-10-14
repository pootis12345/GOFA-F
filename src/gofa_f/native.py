"""Native Bolt quantile decoding and the pinned official HSDGNN baseline.

The full pretrained encoder, decoder, scaling and output head are frozen.
Only GraphExchange modules are trained. Transformers is pinned to 4.46.3.
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from gofa_f.data import file_sha256
from gofa_f.models import GraphExchange


class NativeBolt(nn.Module):
    def __init__(self, native, adjacency, horizon, graph_layers=(), rank=64):
        super().__init__()
        self.native = native.requires_grad_(False).eval()
        self.native.gradient_checkpointing_disable()
        self.horizon = horizon
        self.graph_layers = tuple(graph_layers)
        layers = len(native.encoder.block)
        if len(set(graph_layers)) != len(graph_layers) or any(i < 0 or i >= layers - 1 for i in graph_layers):
            raise ValueError("Graphs must be distinct inter-layer positions")
        if horizon > native.chronos_config.prediction_length:
            raise ValueError("This protocol does not roll forecasts beyond the native output head")
        self.register_buffer("adjacency", torch.as_tensor(adjacency, dtype=torch.float32))
        self.graph = nn.ModuleDict({str(i): GraphExchange(native.config.d_model, rank, 0.0)
                                    for i in graph_layers})
        self.median_index = native.chronos_config.quantiles.index(.5)

    def train(self, mode=True):
        super().train(mode)
        self.native.eval()  # Freeze dropout as well as weights, including the decoder.
        return self

    def embed(self, x, observed):
        b, _, n = x.shape
        context = x.transpose(1, 2).reshape(b*n, -1).masked_fill(
            ~observed.transpose(1, 2).reshape(b*n, -1), torch.nan)
        # Match chronos_bolt.py encode(), including left patch padding and REG.
        scaled, (loc, scale) = self.native.instance_norm(context)
        scaled = scaled.to(self.native.dtype)
        mask = torch.isfinite(context).to(self.native.dtype)
        patched_mask = torch.nan_to_num(self.native.patch(mask), nan=0.0)
        patched = torch.where(patched_mask > 0, self.native.patch(scaled), 0)
        hidden = self.native.input_patch_embedding(torch.cat((patched, patched_mask), -1))
        attention = patched_mask.sum(-1) > 0
        if self.native.chronos_config.use_reg_token:
            ids = torch.full((b*n, 1), self.native.config.reg_token_id, device=x.device)
            hidden = torch.cat((hidden, self.native.shared(ids)), -2)
            attention = torch.cat((attention, torch.ones_like(ids, dtype=torch.bool)), -1)
        return hidden.reshape(b, n, *hidden.shape[1:]), attention.reshape(b, n, -1), \
            loc.reshape(b, n, 1), scale.reshape(b, n, 1)

    def encode_range(self, hidden, mask, start, stop, graphs):
        b, n, p, d = hidden.shape
        stack = self.native.encoder
        flat_mask = mask.reshape(b*n, p)
        additive = (~flat_mask)[:, None, None, :].to(hidden.dtype) * torch.finfo(hidden.dtype).min
        positions = torch.arange(p, device=hidden.device)
        bias = stack.block[0].layer[0].SelfAttention.compute_bias(p, p, device=hidden.device,
                                                               cache_position=positions) + additive
        hidden = hidden.reshape(b*n, p, d)
        for i in range(start, stop):
            result = stack.block[i](hidden, attention_mask=additive, position_bias=bias,
                                    use_cache=False, output_attentions=False, cache_position=positions)
            hidden, bias = result[0], result[1]
            if graphs and str(i) in self.graph:
                graph_mask = mask.clone()
                if self.native.chronos_config.use_reg_token:
                    graph_mask[..., -1] = mask[..., :-1].any(-1)
                hidden = self.graph[str(i)](hidden.reshape(b, n, p, d), self.adjacency, graph_mask)
                hidden = hidden.reshape(b*n, p, d)
        return hidden.reshape(b, n, p, d)

    @torch.no_grad()
    def prefix(self, x, mask, layers):
        if self.graph_layers and layers > min(self.graph_layers) + 1:
            raise ValueError("Cannot cache a trainable graph or the layers after it")
        hidden, attention, loc, scale = self.embed(x, mask)
        hidden = self.encode_range(hidden, attention, 0, layers, graphs=False)
        return hidden, attention, loc, scale

    def from_prefix(self, prefix, layers, return_quantiles=False):
        hidden, attention, loc, scale = prefix
        # The cache ends BEFORE the graph following its last frozen block.
        if str(layers - 1) in self.graph:
            graph_mask = attention.clone()
            if self.native.chronos_config.use_reg_token:
                graph_mask[..., -1] = attention[..., :-1].any(-1)
            hidden = self.graph[str(layers - 1)](hidden, self.adjacency, graph_mask)
        hidden = self.encode_range(hidden, attention, layers, len(self.native.encoder.block), graphs=True)
        b, n, p, d = hidden.shape
        hidden = self.native.encoder.final_layer_norm(hidden).reshape(b*n, p, d)
        mask = attention.reshape(b*n, p)
        decoded = self.native.decode(hidden, mask, hidden)
        quantiles = self.native.output_patch_embedding(decoded).reshape(
            b*n, self.native.num_quantiles, self.native.chronos_config.prediction_length)
        quantiles = self.native.instance_norm.inverse(quantiles.flatten(1),
                                                     (loc.reshape(b*n, 1), scale.reshape(b*n, 1)))
        quantiles = quantiles.reshape(b, n, self.native.num_quantiles, -1)[..., :self.horizon]
        if return_quantiles:
            return quantiles.permute(0, 2, 3, 1)
        return quantiles[:, :, self.median_index].transpose(1, 2)

    def forward(self, x, mask):
        layers = min(self.graph_layers) + 1 if self.graph_layers else len(self.native.encoder.block)
        return self.from_prefix(self.prefix(x, mask, layers), layers)


class HSDGNNAdapter(nn.Module):
    def __init__(self, root, context, horizon, nodes, steps_per_day, width=408, embed_dim=32):
        super().__init__()
        source = Path(root) / "model/HSDGNN.py"
        if file_sha256(source) != "ac1d8c3bf0c22679f769adc78bb2f63f1e76ad2c9593074aaeba1e7527359cfd":
            raise ValueError("HSDGNN source differs from the audited pinned revision")
        spec = importlib.util.spec_from_file_location("official_hsdgnn", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args = SimpleNamespace(batch_size=64, rnn_units=width, num_nodes=nodes, input_dim=1,
                               output_dim=1, lag=context, horizon=horizon, embed_dim=embed_dim,
                               steps_per_day=steps_per_day)
        self.model = module.HSDGNN(args)
        # Exactly the initialization loop in official main.py; bare Parameters are otherwise uninitialized.
        for parameter in self.model.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
            else:
                nn.init.uniform_(parameter)
        self.steps_per_day = steps_per_day
        self.provenance = {"source_sha256": file_sha256(source),
                           "revision": "3b83c6a8abb2229bc176c97ac71f7676bbf584d1",
                           "initialization": "official main.py xavier_uniform/uniform loop",
                           "missing_input": "train-standardized zero; no extra mask channel",
                           "time_features": "sample-index cyclic phase, not claimed calendar weekday",
                           "graph": "official learned dynamic graph; no physical graph augmentation"}

    def forward(self, x, mask, origins):
        b, length, n = x.shape
        t = origins[:, None] + torch.arange(-length, 0, device=x.device)
        phase = (t % self.steps_per_day).float() / self.steps_per_day
        weekday = (t // self.steps_per_day % 7).float()
        inputs = torch.stack((x.masked_fill(~mask, 0), phase[..., None].expand(b, length, n),
                              weekday[..., None].expand(b, length, n)), -1)
        return self.model(inputs).squeeze(-1)
