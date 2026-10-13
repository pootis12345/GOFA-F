"""Pretrained T5 encoders with graph modules BETWEEN encoder blocks.

Both backbone pairs use the same adapter/head and input protocol within the pair.
Chronos uses its native quantized input vocabulary; T5 uses masked patches + text.
"""
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer, T5Config, T5EncoderModel


class GraphExchange(nn.Module):
    def __init__(self, width, rank, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        self.gate = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, adjacency, token_mask):
        # hidden [batch,nodes,tokens,width]; graph exchanges aligned history tokens.
        z = self.down(self.norm(hidden)) * token_mask[..., None]
        weights = adjacency.to(z.dtype)
        message = torch.einsum("ij,bjpr->bipr", weights, z)
        mass = torch.einsum("ij,bjp->bip", weights, token_mask.to(z.dtype)).clamp_min(1e-6)
        message = self.up(torch.nn.functional.gelu(message / mass[..., None]))
        return hidden + self.gate.tanh() * self.dropout(message)


class ForecastEncoder(nn.Module):
    def __init__(self, context, horizon, adjacency, backbone="google-t5/t5-small", revision=None,
                 graph_layers=(), patch_size=12, graph_rank=64, dropout=0.1,
                 freeze_backbone=False, gradient_checkpointing=True,
                 prompt="Forecast future sensor observations from the observed history.",
                 random_config=None):
        super().__init__()
        self.context, self.horizon, self.patch_size = context, horizon, patch_size
        self.gradient_checkpointing = gradient_checkpointing
        if random_config is None:
            self.backbone = T5EncoderModel.from_pretrained(backbone, revision=revision)
        else:
            self.backbone = T5EncoderModel(T5Config(**random_config))
        config = self.backbone.config
        self.resolved_revision = getattr(config, "_commit_hash", None)
        self.chronos = getattr(config, "chronos_config", None)
        self.is_chronos = self.chronos is not None
        self.register_buffer("adjacency", torch.as_tensor(adjacency, dtype=torch.float32))
        self.graph_layers = tuple(graph_layers)
        if any(i < 0 or i >= len(self.backbone.encoder.block) - 1 for i in self.graph_layers):
            raise ValueError("Graph layer indices must precede another Transformer block")
        if len(set(self.graph_layers)) != len(self.graph_layers):
            raise ValueError("Duplicate graph layer indices")
        self.graph = nn.ModuleDict({str(i): GraphExchange(config.d_model, graph_rank, dropout)
                                    for i in self.graph_layers})
        if self.is_chronos:
            if self.chronos["tokenizer_class"] != "MeanScaleUniformBins":
                raise ValueError("Only original Chronos-T5 MeanScaleUniformBins is supported")
            if context > self.chronos["context_length"]:
                raise ValueError("Context exceeds pretrained Chronos context length")
            limits = self.chronos["tokenizer_kwargs"]
            centers = torch.linspace(limits["low_limit"], limits["high_limit"],
                                     self.chronos["n_tokens"] - self.chronos["n_special_tokens"] - 1)
            self.register_buffer("bin_boundaries", torch.cat((torch.tensor([-1e20]),
                                 (centers[1:] + centers[:-1]) / 2, torch.tensor([1e20]))))
            self.adapter = nn.Linear(config.d_model, config.d_model)
            self.register_buffer("prompt_ids", torch.empty(0, dtype=torch.long))
        else:
            self.adapter = nn.Linear(patch_size * 2, config.d_model)
            if random_config is None:
                ids = AutoTokenizer.from_pretrained(backbone, revision=revision)(prompt)["input_ids"]
            else:
                ids = [1]
            self.register_buffer("prompt_ids", torch.tensor(ids, dtype=torch.long))
        self.head = nn.Sequential(nn.LayerNorm(2 * config.d_model), nn.Dropout(dropout),
                                  nn.Linear(2 * config.d_model, horizon))
        self.skip = nn.Linear(context, horizon)
        if freeze_backbone:
            self.backbone.requires_grad_(False)
        # Do not detach frozen Transformer blocks: graph/patch adapters still need input gradients.

    def tokenize(self, z, observed):
        b, n, length = z.shape
        if self.is_chronos:
            scale = (z.abs() * observed).sum(-1) / observed.sum(-1).clamp_min(1)
            scale = torch.where(scale > 0, scale, 1)
            ids = torch.bucketize((z / scale[..., None]).contiguous(), self.bin_boundaries, right=True)
            ids = (ids + self.chronos["n_special_tokens"]).clamp_max(self.chronos["n_tokens"] - 1)
            ids = ids.masked_fill(~observed, self.chronos["pad_token_id"])
            if self.chronos["use_eos_token"]:
                ids = torch.cat((ids, ids.new_full((b, n, 1), self.chronos["eos_token_id"])), -1)
                mask = torch.cat((observed, torch.ones_like(observed[..., :1])), -1)
            else:
                mask = observed
            hidden = self.backbone.shared(ids)
            return hidden + self.adapter(hidden), mask
        pad = math.ceil(length / self.patch_size) * self.patch_size - length
        values = torch.nn.functional.pad(z, (0, pad)).unfold(-1, self.patch_size, self.patch_size)
        masks = torch.nn.functional.pad(observed, (0, pad)).unfold(-1, self.patch_size, self.patch_size)
        hidden = self.adapter(torch.cat((values, masks.to(z.dtype)), -1))
        mask = masks.any(-1)
        prompt = self.backbone.shared(self.prompt_ids).expand(b, n, -1, -1)
        hidden = torch.cat((prompt, hidden), -2)
        mask = torch.cat((torch.ones((b, n, len(self.prompt_ids)), device=z.device, dtype=torch.bool), mask), -1)
        return hidden, mask

    def forward(self, x, x_mask):
        x, observed = x.transpose(1, 2), x_mask.transpose(1, 2)
        count = observed.sum(-1, keepdim=True).clamp_min(1)
        mean = (x * observed).sum(-1, keepdim=True) / count
        std = (((x - mean).square() * observed).sum(-1, keepdim=True) / count + 1e-5).sqrt()
        z = torch.where(observed, (x - mean) / std, 0)
        hidden, mask = self.tokenize(z, observed)
        b, n, p, d = hidden.shape
        stack = self.backbone.encoder
        hidden = stack.dropout(hidden.reshape(b * n, p, d))
        additive_mask = (~mask.reshape(b * n, p))[:, None, None, :].to(hidden.dtype)
        additive_mask = additive_mask * torch.finfo(hidden.dtype).min
        cache_position = torch.arange(p, device=hidden.device)
        bias = None
        for i, block in enumerate(stack.block):
            # T5 relative position bias is shared after block 0 (transformers 4.46.3).
            def run(h, position_bias, layer=block):
                result = layer(h, attention_mask=additive_mask, position_bias=position_bias,
                               use_cache=False, output_attentions=False, cache_position=cache_position)
                return result[0], result[1]
            if self.training and self.gradient_checkpointing:
                hidden, bias = checkpoint(run, hidden, bias, use_reentrant=False)
            else:
                hidden, bias = run(hidden, bias)
            if str(i) in self.graph:
                hidden = self.graph[str(i)](hidden.reshape(b, n, p, d), self.adjacency, mask)
                hidden = hidden.reshape(b * n, p, d)
        hidden = stack.dropout(stack.final_layer_norm(hidden)).reshape(b, n, p, d)
        # Pool numeric history; T5 prompt provides semantic context but is not itself pooled.
        numeric_mask = mask.clone()
        if not self.is_chronos:
            numeric_mask[..., :len(self.prompt_ids)] = False
        else:
            if self.chronos["use_eos_token"]:
                numeric_mask[..., -1] = False
        pooled = (hidden * numeric_mask[..., None]).sum(-2) / numeric_mask.sum(-1, keepdim=True).clamp_min(1)
        positions = torch.arange(p, device=x.device).expand(b, n, p)
        last = positions.masked_fill(~numeric_mask, 0).max(-1).values
        recent = hidden.gather(-2, last[..., None, None].expand(b, n, 1, d)).squeeze(-2)
        forecast = self.head(torch.cat((pooled, recent), -1)) + self.skip(z)
        return (forecast * std + mean).transpose(1, 2)


class Persistence(nn.Module):
    def __init__(self, horizon):
        super().__init__()
        self.horizon = horizon

    def forward(self, x, x_mask):
        index = torch.arange(x.shape[1], device=x.device)[None, :, None].expand_as(x)
        index = index.masked_fill(~x_mask, -1).max(1).values
        last = x.gather(1, index.clamp_min(0)[:, None, :])
        last = last.masked_fill(index[:, None, :] < 0, 0)
        return last.expand(-1, self.horizon, -1)


def parameter_counts(model):
    return {"total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
