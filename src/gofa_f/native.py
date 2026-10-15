"""Native Bolt quantile decoding and the pinned official HSDGNN baseline.

The full pretrained encoder, decoder, scaling and output head are frozen.
Graph modules and an optional private low-rank head delta are trained.
Transformers is pinned to 4.46.3. The shared native module is never patched.
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from gofa_f.data import file_sha256
from gofa_f.models import GraphExchange
from gofa_f.graph_context import ContextGraphExchange
from gofa_f.routed_graph import PriorRoutedGraphExchange


def late_contiguous_layers(block_count, graph_count=2):
    """Keep the last Transformer block after the final graph exchange."""
    if graph_count < 1 or block_count < graph_count + 1:
        raise ValueError("Not enough Transformer blocks for interleaved late graphs")
    return tuple(range(block_count - graph_count - 1, block_count - 1))


class OutputAdapter(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        if rank < 1 or rank > min(in_features, out_features):
            raise ValueError("Invalid output adapter rank")
        # A private CPU generator preserves the paired graph/global RNG state.
        generator = torch.Generator(device="cpu").manual_seed(42)
        self.a = nn.Parameter(torch.randn(rank, in_features, generator=generator) / in_features**.5)
        self.b = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, hidden):
        # Train the small delta in FP32, then preserve the native head output dtype.
        with torch.autocast(hidden.device.type, enabled=False):
            return (hidden.float() @ self.a.t()) @ self.b.t()


class NativeBolt(nn.Module):
    def __init__(self, native, adjacency, horizon, graph_layers=(), rank=64, graph_feature_dim=0,
                 output_adapter_rank=0, graph_exchange="static", graph_attention_rank=16,
                 graph_prior_floor=.5):
        super().__init__()
        self.native = native.requires_grad_(False).eval()
        self.native.gradient_checkpointing_disable()
        self.horizon = horizon
        self.graph_layers = tuple(graph_layers)
        layers = len(native.encoder.block)
        if len(set(graph_layers)) != len(graph_layers) or any(i < 0 or i >= layers - 1 for i in graph_layers):
            raise ValueError("Graphs must be distinct inter-layer positions")
        if graph_exchange not in ("static", "prior_routed"):
            raise ValueError("Unknown graph exchange architecture")
        if graph_exchange == "prior_routed" and (
            not graph_feature_dim or tuple(graph_layers) != late_contiguous_layers(layers)
        ):
            raise ValueError("Prior-routed GOFA requires context and late_contiguous graph layers")
        if horizon > native.chronos_config.prediction_length:
            raise ValueError("This protocol does not roll forecasts beyond the native output head")
        self.register_buffer("adjacency", torch.as_tensor(adjacency, dtype=torch.float32))
        if graph_exchange == "prior_routed" and (
            self.adjacency.ndim != 2 or self.adjacency.shape[0] != self.adjacency.shape[1]
            or not torch.isfinite(self.adjacency).all() or (self.adjacency < 0).any()
        ):
            raise ValueError("Prior adjacency must be finite, square and nonnegative")
        self.graph_exchange = graph_exchange
        self.graph_architecture = {
            "exchange": graph_exchange, "layers": list(graph_layers), "rank": rank,
            "feature_dim": graph_feature_dim, "attention_rank": graph_attention_rank,
            "prior_floor": graph_prior_floor, "output_adapter_rank": output_adapter_rank,
        }
        self.graph_feature_dim = graph_feature_dim
        def exchange_module():
            if graph_exchange == "prior_routed":
                return PriorRoutedGraphExchange(native.config.d_model, rank, graph_feature_dim,
                                                graph_attention_rank, graph_prior_floor)
            return ContextGraphExchange(native.config.d_model, rank, graph_feature_dim) \
                if graph_feature_dim else GraphExchange(native.config.d_model, rank, 0.0)
        self.graph = nn.ModuleDict({str(i): exchange_module() for i in graph_layers})
        self.median_index = native.chronos_config.quantiles.index(.5)
        head = native.output_patch_embedding.output_layer
        self.output_adapter = OutputAdapter(head.in_features, head.out_features, output_adapter_rank) \
            if output_adapter_rank else None

    def output_head(self, decoded):
        head = self.native.output_patch_embedding
        if self.output_adapter is None:
            return head(decoded)
        hidden = head.act(head.hidden_layer(decoded))
        projected = head.output_layer(hidden)
        projected = projected + self.output_adapter(hidden).to(projected.dtype)
        output = head.dropout(projected) + head.residual_layer(decoded)
        return head.layer_norm(output) if head.use_layer_norm else output

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

    def exchange(self, index, hidden, mask, features):
        graph_mask = mask.clone()
        if self.native.chronos_config.use_reg_token:
            graph_mask[..., -1] = mask[..., :-1].any(-1)
        module = self.graph[str(index)]
        if isinstance(module, ContextGraphExchange):
            return module(hidden, self.adjacency, graph_mask, features)
        return module(hidden, self.adjacency, graph_mask)

    def encode_range(self, hidden, mask, start, stop, graphs, graph_features=None):
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
                hidden = self.exchange(i, hidden.reshape(b, n, p, d), mask, graph_features)
                hidden = hidden.reshape(b*n, p, d)
        return hidden.reshape(b, n, p, d)

    @torch.no_grad()
    def prefix(self, x, mask, layers):
        if self.graph_layers and layers > min(self.graph_layers) + 1:
            raise ValueError("Cannot cache a trainable graph or the layers after it")
        hidden, attention, loc, scale = self.embed(x, mask)
        hidden = self.encode_range(hidden, attention, 0, layers, graphs=False)
        return hidden, attention, loc, scale

    def from_prefix(self, prefix, layers, return_quantiles=False, graph_features=None):
        hidden, attention, loc, scale = prefix
        # The cache ends BEFORE the graph following its last frozen block.
        if str(layers - 1) in self.graph:
            hidden = self.exchange(layers - 1, hidden, attention, graph_features)
        hidden = self.encode_range(hidden, attention, layers, len(self.native.encoder.block), graphs=True,
                                   graph_features=graph_features)
        b, n, p, d = hidden.shape
        hidden = self.native.encoder.final_layer_norm(hidden).reshape(b*n, p, d)
        mask = attention.reshape(b*n, p)
        decoded = self.native.decode(hidden, mask, hidden)
        quantiles = self.output_head(decoded).reshape(
            b*n, self.native.num_quantiles, self.native.chronos_config.prediction_length)
        quantiles = self.native.instance_norm.inverse(quantiles.flatten(1),
                                                     (loc.reshape(b*n, 1), scale.reshape(b*n, 1)))
        quantiles = quantiles.reshape(b, n, self.native.num_quantiles, -1)[..., :self.horizon]
        if return_quantiles:
            return quantiles.permute(0, 2, 3, 1)
        return quantiles[:, :, self.median_index].transpose(1, 2)

    def forward(self, x, mask, graph_features=None):
        layers = min(self.graph_layers) + 1 if self.graph_layers else len(self.native.encoder.block)
        return self.from_prefix(self.prefix(x, mask, layers), layers, graph_features=graph_features)


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

    def forward(self, x, mask, origins, calendar=None):
        b, length, n = x.shape
        t = origins[:, None] + torch.arange(-length, 0, device=x.device)
        phase = (t % self.steps_per_day).float() / self.steps_per_day
        weekday = (t // self.steps_per_day % 7).float()
        if calendar is not None:
            values, known = calendar
            phase = torch.where(known[..., 0], values[..., 0], phase)
            weekday = torch.where(known[..., 1], values[..., 1], weekday)
        inputs = torch.stack((x.masked_fill(~mask, 0), phase[..., None].expand(b, length, n),
                              weekday[..., None].expand(b, length, n)), -1)
        return self.model(inputs).squeeze(-1)
