"""Small source-only mechanism probe, not a ChatTime numerical prediction head."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class LagExchange(nn.Module):
    """Cross-node, history-causal messages; no cross edge means exact identity."""
    def __init__(self, width=32, rank=8, lags=(0, 1, 2, 3)):
        super().__init__()
        self.lags = tuple(lags)
        self.rank = rank
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.query = nn.Linear(width, rank, bias=False)
        self.key = nn.Linear(width, rank, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.edge_score = nn.Linear(2, 1, bias=False)
        self.edge_value = nn.Linear(2, width, bias=False)
        self.lag_bias = nn.Parameter(torch.zeros(len(lags)))
        self.gate = nn.Linear(width, 1)
        self.up = nn.Linear(width, width, bias=False)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, hidden, observed, adjacency, edge_features):
        # B,N,T,D. A[receiver,sender]; only immutable source histories enter.
        batch, nodes, length, _ = hidden.shape
        z = self.norm(hidden)
        histories, masks = [], []
        for lag in self.lags:
            if lag >= length:
                histories.append(torch.zeros_like(z))
                masks.append(torch.zeros_like(observed))
            elif lag:
                histories.append(F.pad(z[:, :, :-lag], (0, 0, lag, 0)))
                masks.append(F.pad(observed[:, :, :-lag], (lag, 0), value=False))
            else:
                histories.append(z)
                masks.append(observed)
        history = torch.stack(histories, -2)
        valid_history = torch.stack(masks, -1).permute(0, 2, 1, 3)
        q = self.query(z).permute(0, 2, 1, 3)
        keys = self.key(history).permute(0, 2, 1, 3, 4)
        values = self.value(history).permute(0, 2, 1, 3, 4)
        cross = adjacency.bool() & ~torch.eye(nodes, device=hidden.device, dtype=torch.bool)[None]
        present = observed.any(-1)
        cross = cross & present[:, :, None] & present[:, None, :]
        support = cross[:, None, :, :, None] & valid_history[:, :, None]
        score = torch.einsum('btir,btjkr->btijk', q, keys)/math.sqrt(self.rank)
        score = score+self.edge_score(edge_features).squeeze(-1)[:, None, :, :, None] \
            +self.lag_bias[None, None, None, None]
        empty = ~support.flatten(-2).any(-1, keepdim=True)
        score = score.masked_fill(~support, -torch.inf).flatten(-2)
        score = torch.where(empty, torch.zeros_like(score), score)
        attention = (score.softmax(-1)*support.flatten(-2)).reshape(
            batch, length, nodes, nodes, len(self.lags))
        context = torch.einsum('btijk,btjkd->btid', attention, values)
        context = context+torch.einsum('btij,bijd->btid',
            attention.sum(-1), self.edge_value(edge_features))
        delta = self.up(F.gelu(context))*self.gate(z.permute(0, 2, 1, 3)).sigmoid()
        delta = delta.masked_fill(empty, 0.).permute(0, 2, 1, 3)
        return (hidden+delta)*present[:, :, None, None]


class LagGraphProbe(nn.Module):
    """Shared temporal convolutions, two lag exchanges, horizon-conditioned readout.

    Continuous source-GNN output is used only to test spatial information.
    The proposed downstream ChatTime system retains its language generation head.
    """
    def __init__(self, width=32, rank=8, lags=(0, 1, 2, 3)):
        super().__init__()
        self.conv1 = nn.Conv1d(2, width, 3)
        self.conv2 = nn.Conv1d(width, width, 3, dilation=2)
        self.exchanges = nn.ModuleList([LagExchange(width, rank, lags) for _ in range(2)])
        self.head = nn.Sequential(nn.Linear(2*width+2, width), nn.GELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, adjacency, edge_features, horizon=12, return_residual=False):
        batch, nodes, length, _ = x.shape
        observed = x[..., 1].bool()
        present = observed.any(-1)
        z = F.gelu(self.conv1(F.pad(x.permute(0, 1, 3, 2).reshape(batch*nodes, 2, length), (2, 0))))
        z = z.reshape(batch, nodes, -1, length).permute(0, 1, 3, 2)*present[:, :, None, None]
        z = self.exchanges[0](z, observed, adjacency, edge_features)
        d = z.shape[-1]
        z = F.gelu(self.conv2(F.pad(z.permute(0, 1, 3, 2).reshape(batch*nodes, d, length), (4, 0))))
        z = z.reshape(batch, nodes, d, length).permute(0, 1, 3, 2)*present[:, :, None, None]
        z = self.exchanges[1](z, observed, adjacency, edge_features)
        mean = (z*observed[..., None]).sum(2)/observed.sum(2).clamp_min(1)[..., None]
        features = torch.cat((z[:, :, -1], mean), -1)
        h = torch.arange(horizon, dtype=x.dtype, device=x.device)
        position = torch.stack((h/max(horizon-1, 1), (h+1)/length), -1)
        context = features[:, :, None].expand(-1, -1, horizon, -1)
        position = position[None, None].expand(batch, nodes, -1, -1)
        residual = self.head(torch.cat((context, position), -1)).squeeze(-1)
        if return_residual:
            return residual*present[:, :, None]
        indices = torch.arange(length, device=x.device)[None, None].expand(batch, nodes, -1)
        last_index = indices.masked_fill(~observed, -1).amax(-1).clamp_min(0)
        last = x[..., 0].gather(-1, last_index[..., None]).squeeze(-1)
        return (last[:, :, None]+residual)*present[:, :, None]


def state_digest(model):
    import hashlib
    value = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value.update(name.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()
