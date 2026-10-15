"""TITAN-inspired input routing, restricted to an always-retained prior graph.

This is a GOFA-F exchange module, not a reproduction of TITAN's five experts.
All learned parameters are shared across nodes and independent of node count.
"""
import math

import torch
from torch import nn

from gofa_f.graph_context import ContextGraphExchange


class PriorRoutedGraphExchange(ContextGraphExchange):
    def __init__(self, width, rank, feature_dim, attention_rank=16, prior_floor=.5):
        if not isinstance(attention_rank, int) or not 1 <= attention_rank <= rank:
            raise ValueError("Attention rank must be between 1 and graph rank")
        if not .5 <= prior_floor < 1:
            raise ValueError("Retain at least half of the prior aggregation")
        super().__init__(width, rank, feature_dim)
        self.attention_rank = attention_rank
        self.prior_floor = float(prior_floor)
        # Extra modules must not shift the next paired graph layer's initialization.
        with torch.random.fork_rng(devices=[]):
            self.query = nn.Linear(rank, attention_rank, bias=False)
            self.key = nn.Linear(rank, attention_rank, bias=False)
            self.router = nn.Linear(rank, 2)
        # Initial attention equals the weighted prior. Node gates initially equal
        # the legacy scalar gate, whose zero initialization preserves native output.
        nn.init.zeros_(self.key.weight)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

    def routing(self, z, adjacency, token_mask):
        """Return history-only attention, dynamic fraction and residual amplitude.

        z is [B,N,P,R]. The [B,N,N] relation is shared across history patches;
        per-patch missing-source normalization happens during aggregation.
        """
        summary = z.sum(2) / token_mask.sum(2).clamp_min(1)[..., None]
        query, key = self.query(summary).tanh(), self.key(summary).tanh()
        logits = query @ key.transpose(-1, -2) / math.sqrt(self.attention_rank)
        prior = adjacency.float()
        support = (prior > 0)[None] & token_mask.any(2)[:, None, :]
        logits = logits + prior.clamp_min(torch.finfo(prior.dtype).tiny).log()[None]
        has_source = support.any(-1, keepdim=True)
        logits = logits.masked_fill(~support, -torch.inf)
        # Empty neighborhoods get zero mass, never softmax(-inf,...,-inf).
        logits = torch.where(has_source, logits, torch.zeros_like(logits))
        attention = logits.softmax(-1).masked_fill(~support, 0)
        routes = self.router(summary)
        dynamic_fraction = (1 - self.prior_floor) * routes[..., :1].sigmoid()
        amplitude = (self.gate + routes[..., 1:]).tanh()
        return attention, dynamic_fraction, amplitude

    @staticmethod
    def aggregate(weights, z, token_mask):
        b, n, p, r = z.shape
        numerator = torch.matmul(weights, z.reshape(b, n, p*r)).reshape(b, n, p, r)
        mass = torch.matmul(weights, token_mask.float()).clamp_min(1e-6)
        return numerator / mass[..., None]

    def forward(self, hidden, adjacency, token_mask, features):
        if features is None or features.shape != (*hidden.shape[:2], self.context_projection.in_features):
            raise ValueError("Routed graph requires explicit per-node history features")
        if adjacency.shape != (hidden.shape[1], hidden.shape[1]):
            raise ValueError("Prior adjacency must match the node dimension")
        if token_mask.shape != hidden.shape[:3]:
            raise ValueError("Token mask must match history tokens")
        # Keep routing/normalization in FP32 even with a BF16 frozen backbone.
        with torch.autocast(hidden.device.type, enabled=False):
            safe_hidden = hidden.float().masked_fill(~token_mask[..., None], 0)
            z = self.down(self.norm(safe_hidden)) + self.context_projection(features.float())[:, :, None]
            z = z.masked_fill(~token_mask[..., None], 0)
            attention, fraction, amplitude = self.routing(z, adjacency, token_mask)
            prior_message = self.aggregate(adjacency.float(), z, token_mask)
            dynamic_message = self.aggregate(attention, z, token_mask)
            message = prior_message + fraction[:, :, None] * (dynamic_message - prior_message)
            delta = amplitude[:, :, None] * self.up(torch.nn.functional.gelu(message))
            delta = delta.masked_fill(~token_mask[..., None], 0)
        return hidden + delta.to(hidden.dtype)
