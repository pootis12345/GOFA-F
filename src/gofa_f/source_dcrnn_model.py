"""Inductive PyTorch DCRNN adaptation; variable graphs, no node-ID parameters.

Architecture reference: Li et al., ICLR2018; official MIT-licensed repository
liyaguang/DCRNN, revision602afd9d767d3aa1c9b3eac51710d6aeee12c227.
This is an independent implementation of the diffusion-GRU equations.
"""
import torch
from torch import nn


class Diffusion(nn.Module):
    def __init__(self, inputs, outputs, bias=0.):
        super().__init__()
        self.linear = nn.Linear(5*inputs, outputs)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, bias)

    def forward(self, x, supports):
        terms = [x]
        for i in range(2):
            a = supports[:, i]
            first = torch.bmm(a, x)
            second = 2*torch.bmm(a, first)-x
            terms.extend((first, second))
        return self.linear(torch.cat(terms, -1))


class DCGRU(nn.Module):
    def __init__(self, inputs, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.gates = Diffusion(inputs+hidden, 2*hidden, 1.)
        self.candidate = Diffusion(inputs+hidden, hidden)

    def forward(self, x, state, supports, present):
        r, u = self.gates(torch.cat((x, state), -1), supports).sigmoid().chunk(2, -1)
        candidate = self.candidate(torch.cat((x, r*state), -1), supports).tanh()
        result = u*state+(1-u)*candidate
        return result*present[:, :, None]


class DCRNN(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.encoder = nn.ModuleList((DCGRU(2, hidden), DCGRU(hidden, hidden)))
        self.decoder = nn.ModuleList((DCGRU(1, hidden), DCGRU(hidden, hidden)))
        self.output = nn.Linear(hidden, 1)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, supports, present, horizon, teacher=None, probability=0., rng=None):
        batch, _, nodes, _ = x.shape
        states = [x.new_zeros((batch, nodes, self.hidden)) for _ in range(2)]
        for step in x.unbind(1):
            current = step
            for i, cell in enumerate(self.encoder):
                states[i] = cell(current, states[i], supports, present)
                current = states[i]
        previous = x.new_zeros((batch, nodes, 1))
        outputs = []
        for h in range(horizon):
            current = previous
            for i, cell in enumerate(self.decoder):
                states[i] = cell(current, states[i], supports, present)
                current = states[i]
            prediction = self.output(current).squeeze(-1)*present
            outputs.append(prediction)
            # y[h] can affect only predictions AFTER h, never its own forecast.
            if teacher is not None and torch.rand((), generator=rng).item() < probability:
                previous = teacher[:, h, :, None]
            else:
                previous = prediction[:, :, None]
        return torch.stack(outputs, 1)


def masked_query_loss(prediction, truth, observed):
    error = (prediction-truth).abs()*observed
    per_query = error.flatten(1).sum(1)/observed.flatten(1).sum(1).clamp_min(1)
    if not observed.flatten(1).any(1).all():
        raise ValueError('Empty source supervision query')
    return per_query.mean()
