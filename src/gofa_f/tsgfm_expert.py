"""Source-transfer expert using TSGFM's parallel space/time message equations.

Independent implementation from the published equations, not a bundled copy of
the upstream repository. Changes: causal local scales/missing masks, inherited
physical rosters/edges and a shared direct multi-horizon residual readout.
There are no ChatTime, Memory, dataset-ID or target-adaptation parameters here.
"""
import torch
from torch import nn


UPSTREAM_COMMIT = '01356fbe6488105dcaf21e182f9a438254bffe93'


def edge_mean(messages, scores, receiver, count):
    """Receiver-normalized attention; empty neighborhoods return exactly zero."""
    if not len(receiver):
        return messages.new_zeros(count, messages.shape[-1])
    maximum = scores.new_full((count,), -torch.inf)
    maximum.scatter_reduce_(0, receiver, scores.detach(), reduce='amax', include_self=True)
    weights = (scores-maximum[receiver]).exp()
    denominator = weights.new_zeros(count).index_add(0, receiver, weights)
    weights = weights/denominator[receiver].clamp_min(1e-12)
    return messages.new_zeros(count, messages.shape[-1]).index_add(
        0, receiver, messages*weights[:, None])


def pair_scores(network, source, destination, edge=None):
    """Factor only the first affine map; exactly the concatenation MLP."""
    first, activation, last = network
    width = source.shape[-1]
    x = torch.nn.functional.linear(source, first.weight[:, :width], first.bias)
    if edge is not None:
        x = x+torch.nn.functional.linear(edge, first.weight[:, width:width+edge.shape[-1]])
    x = x+torch.nn.functional.linear(destination, first.weight[:, -width:])
    return last(activation(x)).squeeze(-1)


class TSGFMExpert(nn.Module):
    """One shared parameter set for all real nodes and both registered shapes."""
    def __init__(self, width=128, maximum_history=48):
        super().__init__()
        self.width, self.maximum_history = width, maximum_history
        self.node_embedding = nn.Linear(2, width)
        self.spatial_attention = nn.Sequential(nn.Linear(2*width, width),
            nn.LeakyReLU(), nn.Linear(width, 1))
        self.temporal_embedding = nn.Embedding(maximum_history, maximum_history)
        self.temporal_attention = nn.Sequential(nn.Linear(2*width+maximum_history, width),
            nn.LeakyReLU(), nn.Linear(width, 1))
        self.combine_aggrs = nn.Linear(2*width, width)
        self.node_update = nn.GRUCell(width, width)
        self.readout = nn.Sequential(nn.Linear(width+4, width), nn.ReLU(), nn.Linear(width, 1))

    def encode(self, item):
        n, length = item.normalized.shape
        if length > self.maximum_history or len(item.node_ids) != n or n > 31:
            raise ValueError('Registered real-node roster/history contract violated')
        observed = item.observed.bool() & item.present[:, None].bool()
        x = torch.stack((item.normalized.float().masked_fill(~observed, 0), observed.float()), -1)
        hidden = self.node_embedding(x).reshape(n*length, self.width)
        # One node at one historical timestamp is a space-time vertex. Message
        # sources are observed history only; no forward target values exist.
        receiver, sender = item.adjacency.bool().nonzero(as_tuple=True)
        clock = torch.arange(length, device=hidden.device)
        src = (sender[:, None]*length+clock).flatten()
        dst = (receiver[:, None]*length+clock).flatten()
        lawful = observed.flatten()[src]
        src, dst = src[lawful], dst[lawful]
        spatial = edge_mean(hidden[src], pair_scores(self.spatial_attention, hidden[src], hidden[dst]),
            dst, n*length)
        past, later = torch.triu_indices(length, length, 1, device=hidden.device)
        base = torch.arange(n, device=hidden.device)[:, None]*length
        src, dst = (base+past).flatten(), (base+later).flatten()
        delta = (later-past)[None].expand(n, -1).flatten()
        lawful = observed.flatten()[src] & item.present[:, None].expand(n, len(past)).flatten()
        src, dst, delta = src[lawful], dst[lawful], delta[lawful]
        temporal = edge_mean(hidden[src], pair_scores(self.temporal_attention, hidden[src], hidden[dst],
            self.temporal_embedding(delta)), dst, n*length)
        messages = self.combine_aggrs(torch.cat((temporal, spatial), -1))
        # T=1: the spatial and temporal messages read the SAME prior state.
        encoded = self.node_update(messages, hidden).reshape(n, length, self.width)
        return encoded[:, -1].masked_fill(~item.present[:, None].bool(), 0)

    def forecast(self, item):
        encoded = self.encode(item)
        n, horizon = len(encoded), item.horizon
        if (item.history.shape[1], horizon) not in ((12,12), (48,24)):
            raise ValueError('Only the two frozen forecasting tasks are allowed')
        u = torch.arange(1, horizon+1, device=encoded.device, dtype=torch.float32)[:, None]/horizon
        time = torch.cat((u, u.square(), torch.full_like(u, horizon/24),
            torch.full_like(u, item.history.shape[1]/48)), -1)
        features = torch.cat((encoded[:, None].expand(n,horizon,-1), time[None].expand(n,-1,-1)), -1)
        residual = self.readout(features).squeeze(-1).double()
        return item.last[:, None].double()+item.extent[:, None].double()*residual


class TSGFMEngine:
    def __init__(self, manifest, store):
        self.model = self.state_model = TSGFMExpert()
        self.manifest, self.store = manifest, store
        self.microbatch, self.cuda = 1, False

    def warmup(self, enabled):
        pass

    def loss(self, rows):
        from gofa_f.dual_memory_policy import query_loss
        return sum(query_loss(self.model.forecast(item), target, mask, item.present,
            self.manifest['training_unit_normalizers'][row['source']])
            for row in rows for item,target,mask in [self.store.get(row, 'train')])

    @torch.no_grad()
    def predict(self, row, split, role='fit'):
        item, target, mask = self.store.get(row, split, role)
        p = self.model.forecast(item)
        return p[0].numpy(), target[0].numpy(), mask[0].numpy(), item.identity

    def close(self):
        self.model = self.state_model = None
