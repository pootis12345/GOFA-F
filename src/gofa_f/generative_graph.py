"""Stage17: origin-frozen history graphs and two independent graph interactions."""
from dataclasses import dataclass
import hashlib
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class OriginGraph:
    history: torch.Tensor  # [L,N], already masked; never extended during rollout
    observed: torch.Tensor
    prior: torch.Tensor
    base: torch.Tensor
    edges: torch.Tensor
    target: int
    identity: str

    def to(self, device):
        return OriginGraph(*(x.to(device) for x in (self.history, self.observed,
            self.prior, self.base, self.edges)), self.target, self.identity)


def row_normalize(weights, present):
    weights = np.maximum(np.asarray(weights, np.float64), 0)*present[None, :]
    return np.divide(weights, weights.sum(1, keepdims=True), out=np.zeros_like(weights),
                     where=weights.sum(1, keepdims=True) > 0).astype(np.float32)


def origin_graph(history, observed, target, external=None, variable_ids=None):
    x = np.asarray(history, np.float64)
    mask = np.asarray(observed, bool) & np.isfinite(x)
    if x.ndim != 2 or x.shape != mask.shape or not 0 <= target < x.shape[1]:
        raise ValueError("Expected synchronized history [L,N] and a valid target")
    length, nodes = x.shape
    if length < 4:
        raise ValueError("Four history bins require at least four time steps")
    ids = list(map(str, variable_ids if variable_ids is not None else range(nodes)))
    if len(ids) != nodes or len(set(ids)) != nodes:
        raise ValueError("Unique canonical variable IDs required")
    present = mask.any(0)
    # Pairwise overlap statistics use only the immutable origin window.
    valid = mask.astype(np.float64)
    values = np.where(mask, x, 0)
    count = valid.T @ valid
    sums = values.T @ valid
    square = (values*values).T @ valid
    covariance = values.T @ values - sums*sums.T/np.maximum(count, 1)
    variance = np.maximum(square-sums*sums/np.maximum(count, 1), 0)
    denominator = np.sqrt(variance*variance.T)
    rho = np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 0)
    rho = np.clip(rho, -1, 1)
    eligible = (count >= max(8, math.ceil(.5*length))) & (denominator > 0)
    correlation = np.zeros((nodes, nodes), np.float64)
    for i in range(nodes):
        candidates = [j for j in range(nodes) if j != i and eligible[i, j] and present[j]]
        candidates.sort(key=lambda j: (-abs(rho[i, j]), ids[j]))
        for j in candidates[:8]:
            correlation[i, j] = abs(rho[i, j])*count[i, j]/length
    adjacency = np.zeros_like(correlation) if external is None else (np.asarray(external) > 0).astype(float)
    if adjacency.shape != correlation.shape:
        raise ValueError("External topology and sensor order must match")
    np.fill_diagonal(adjacency, 0)
    identity = np.eye(nodes)
    prior = row_normalize((correlation if external is None else adjacency)+identity, present)
    base = row_normalize(adjacency+correlation+identity, present)
    edges = np.stack((np.where(eligible, rho, 0), count/length), -1).astype(np.float32)
    edges[np.arange(nodes), np.arange(nodes), 0] = 1
    digest = hashlib.sha256()
    for array in (values, mask, prior, base, edges):
        digest.update(str(array.shape).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(str((target, ids)).encode())
    return OriginGraph(torch.from_numpy(values.astype(np.float32)), torch.from_numpy(mask),
        torch.from_numpy(prior), torch.from_numpy(base), torch.from_numpy(edges), int(target), digest.hexdigest())


class HistoryEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 32, 5)
        self.conv2 = nn.Conv1d(32, 64, 3, dilation=2)
        self.projection = nn.Linear(260, 64)
        self.norm = nn.LayerNorm(64)

    def forward(self, graph):
        x, mask = graph.history.float(), graph.observed
        present = mask.any(0)
        low = x.masked_fill(~mask, torch.inf).amin(0).masked_fill(~present, 0)
        high = x.masked_fill(~mask, -torch.inf).amax(0).masked_fill(~present, 0)
        extent = high-low
        extent = torch.where(extent < 10*np.finfo(np.float64).eps, 1., extent)
        scaled = ((x-low)/extent-.5).masked_fill(~mask, 0)
        difference = torch.zeros_like(scaled)
        difference[1:] = (scaled[1:]-scaled[:-1]).masked_fill(~(mask[1:] & mask[:-1]), 0)
        position = torch.linspace(-1, 0, len(x), device=x.device)[:, None].expand_as(x)
        inputs = torch.stack((scaled, mask.float(), difference, position), 0).permute(2, 0, 1)
        encoded = F.gelu(self.conv1(F.pad(inputs, (4, 0))))
        encoded = F.gelu(self.conv2(F.pad(encoded, (4, 0))))
        summaries, coverage = [], []
        for k in range(4):
            start, end = k*len(x)//4, (k+1)*len(x)//4
            valid = mask[start:end].T
            summaries.append((encoded[:, :, start:end]*valid[:, None]).sum(-1)/valid.sum(-1).clamp_min(1)[:, None])
            coverage.append(valid.float().mean(-1, keepdim=True))
        value = self.norm(self.projection(torch.cat(summaries+coverage, -1)))
        return value.masked_fill(~present[:, None], 0)


class HistoryGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_projection = nn.Linear(64, 64)
        self.message = nn.Linear(64, 64)
        self.edge = nn.Linear(2, 64, bias=False)
        self.norm = nn.LayerNorm(64)

    def forward(self, z, graph):
        weight = .5*(graph.prior+graph.base)
        # Linear edge projection commutes with aggregation; avoids NxNx64 tensors.
        edge_message = self.edge((weight[..., None]*graph.edges).sum(1))
        value = self.norm(z+F.gelu(self.self_projection(z)+weight @ self.message(z)+edge_message))
        return value.masked_fill(~graph.observed.any(0)[:, None], 0)


class GraphInteraction(nn.Module):
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

    def forward(self, hidden, memories, graphs, selected):
        results = []
        with torch.autocast(hidden.device.type, enabled=False):
            for row, (z, graph) in enumerate(zip(memories, graphs)):
                base, prior = graph.base[graph.target], graph.prior[graph.target]
                support = base > 0
                positions = torch.flatnonzero(selected[row])
                keys, values = self.key(z), self.value(z)
                prior_message = (prior @ values)[None]
                score_bias = base.clamp_min(torch.finfo(torch.float32).tiny).log() \
                    +self.edge_score(graph.edges[graph.target]).squeeze(-1)
                changed = []
                # A query is token-local. Keep the same one-row FP32 arithmetic
                # for full teacher forcing and single-token KV generation.
                for position in positions.unbind():
                    value = hidden[row, position:position+1]
                    q = self.down(self.norm(value.float()))+z[graph.target]
                    logits = self.query(q) @ keys.T/4+score_bias
                    attention = logits.masked_fill(~support[None], -torch.inf).softmax(-1) \
                        if support.any() else torch.zeros_like(logits)
                    message = .5*prior_message+.5*(attention @ values)
                    delta = self.gate(q).sigmoid()*self.up(F.gelu(message))
                    changed.append(value+delta.to(hidden.dtype))
                results.append(hidden[row].index_copy(0, positions, torch.cat(changed, 0))
                               if changed else hidden[row])
        return torch.stack(results)


class GraphModules(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.encoder = HistoryEncoder()
        self.gnns = nn.ModuleList([HistoryGNN(), HistoryGNN()])
        self.interactions = nn.ModuleList([GraphInteraction(width), GraphInteraction(width)])

    def memories(self, graphs):
        with torch.autocast(graphs[0].history.device.type, enabled=False):
            states = [self.encoder(graph) for graph in graphs]
            result = []
            for layer in self.gnns:
                states = [layer(z, graph) for z, graph in zip(states, graphs)]
                result.append(states)
            return result
