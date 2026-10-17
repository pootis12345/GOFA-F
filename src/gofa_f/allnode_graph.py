"""Deterministic nine-node origin subgraphs and two token-aligned graph exchanges."""
from dataclasses import dataclass
import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from gofa_f.generative_graph import OriginGraph


@dataclass
class AllNodeGraph:
    graph: OriginGraph
    variable_ids: tuple
    query_key: str

    def to(self, device):
        return AllNodeGraph(self.graph.to(device), self.variable_ids, self.query_key)

    @property
    def identity(self):
        return self.graph.identity


def build_graph(series, query):
    full = series.graph(query)
    present = full.observed.any(0)
    external = series.external
    selected, frontier = [query.variable], [query.variable]
    # Incoming senders, one hop before two hops; never reverse a road edge.
    for _ in range(2):
        candidates = {}
        for receiver in frontier:
            for sender in torch.nonzero(full.base[receiver] > 0).flatten().tolist():
                if sender in selected or not present[sender]:
                    continue
                physical = external is not None and external[receiver, sender] > 0
                signed, overlap = full.edges[receiver, sender].tolist()
                rank = (not physical, -abs(signed)*overlap, str(series.ids[sender]))
                candidates[sender] = min(rank, candidates.get(sender, rank))
        frontier = sorted(candidates, key=candidates.get)[:9-len(selected)]
        selected.extend(frontier)
        if len(selected) == 9 or not frontier:
            break
    selected = [query.variable]+sorted(selected[1:], key=lambda i: str(series.ids[i]))
    ix = torch.tensor(selected)
    def matrix(value):
        value = value[ix][:, ix].clone()
        denominator = value.sum(-1, keepdim=True)
        return torch.where(denominator > 0, value/denominator.clamp_min(1e-30), 0.)
    ids = tuple(str(series.ids[i]) for i in selected)
    identity = hashlib.sha256(str((full.identity, query.key, ids, "allnode_v1")).encode()).hexdigest()
    graph = OriginGraph(full.history[:, ix].clone(), full.observed[:, ix].clone(),
        matrix(full.prior), matrix(full.base), full.edges[ix][:, ix].clone(), 0, identity)
    return AllNodeGraph(graph, ids, query.key)


def time_weights(weights, observed):
    value = weights[None].float()*observed[:, None, :].float()
    total = value.sum(-1, keepdim=True)
    return torch.where(total > 0, value/total.clamp_min(1e-30), 0.)


class AllNodeExchange(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.down = nn.Linear(width, 64, bias=False)
        self.query = nn.Linear(64, 16, bias=False)
        self.key = nn.Linear(64, 16, bias=False)
        self.value = nn.Linear(64, 64, bias=False)
        self.edge_score = nn.Linear(2, 1, bias=False)
        self.gate = nn.Linear(64, 1)
        self.up = nn.Linear(64, width, bias=False)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, hidden, graph):
        # hidden [N,L,D]; graph supplies only observations at the immutable origin.
        with torch.autocast(hidden.device.type, enabled=False):
            z = self.down(self.norm(hidden.float())).transpose(0, 1)
            prior = time_weights(graph.prior, graph.observed)
            base = time_weights(graph.base, graph.observed)
            support = base > 0
            logits = self.query(z) @ self.key(z).transpose(-1, -2)/4
            logits = logits+base.clamp_min(torch.finfo(torch.float32).tiny).log() \
                +self.edge_score(graph.edges.float()).squeeze(-1)[None]
            empty = ~support.any(-1, keepdim=True)
            logits = logits.masked_fill(~support, -torch.inf)
            # Do not evaluate an all-minus-infinity softmax, including in backward.
            logits = torch.where(empty, torch.zeros_like(logits), logits)
            attention = logits.softmax(-1)*support.float()
            values = self.value(z)
            message = .5*(prior @ values)+.5*(attention @ values)
            delta = self.gate(z).sigmoid()*self.up(F.gelu(message))
            delta = delta.transpose(0, 1)
            return delta.masked_fill(~graph.observed.any(0)[:, None, None], 0.)


class AllNodeAdapter(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.exchanges = nn.ModuleList([AllNodeExchange(width), AllNodeExchange(width)])
