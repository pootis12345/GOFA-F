"""Single-root TAGLAS dense sampling and history-only inputs for dual Memory.

The entire fringe is truncated, BEFORE the final union/deduplication. This
matches TAGLAS k_hop_subgraph's dense branch, including repeated fringe entries;
it is deliberately not per-parent fanout or a unique-new-node BFS sampler.
"""
from dataclasses import dataclass, replace
import hashlib
import json

import numpy as np
import torch

SAMPLER = 'TAGLAS.k_hop_subgraph:dense:single_root:whole_fringe:3x10:v1'
SHAPES = ((12, 12), (48, 24))
GROUPS = {'pems04_flow': 'traffic', 'pems04_speed': 'traffic',
          'boston2017': 'weather', 'penmanshiel2017': 'wind'}


def query_seed(query, seed=42):
    return int.from_bytes(hashlib.sha256(f'{seed}:{query.key}:{SAMPLER}'.encode()).digest()[:8], 'big') % (2**63-1)


def sample_subgraph(adjacency, root, seed, allowed=None, hops=3, maximum=10):
    """Receiver rows / sender columns; expand incoming message ancestors.

    Dense edge order is receiver-major/sender-minor, explicitly pinned. An
    independent upstream-reference test must validate this RNG/order contract.
    The private CPU generator never changes training/Native random state.
    """
    a = torch.as_tensor(np.asarray(adjacency) > 0, dtype=torch.bool).clone()
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not 0 <= root < len(a):
        raise ValueError('Square receiver/sender topology and lawful root required')
    if (hops, maximum) != (3, 10):
        raise ValueError('The preregistered sampler is three whole-fringe hops of ten')
    keep = torch.ones(len(a), dtype=torch.bool)
    if allowed is not None:
        keep.zero_(); keep[torch.as_tensor(allowed, dtype=torch.long)] = True
    if not keep[root]:
        raise ValueError('Root outside the source partition')
    a &= keep[:, None] & keep[None, :]
    a.fill_diagonal_(False)
    receiver, sender = a.nonzero(as_tuple=True)
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    frontier = torch.tensor([root], dtype=torch.long)
    subsets = [frontier]
    first_hop = {int(root): 0}; parents = {}
    for hop in range(1, hops+1):
        node_mask = torch.zeros(len(a), dtype=torch.bool); node_mask[frontier] = True
        fringe = sender[node_mask[receiver]]
        if len(fringe) > maximum:
            fringe = fringe[torch.randperm(len(fringe), generator=generator)[:maximum]]
        for node in fringe.tolist():
            if node not in first_hop:
                # A sampled path toward the target is retained in the union.
                candidates = frontier[a[frontier, node]]
                parents[node] = int(candidates.min())
                first_hop[node] = hop
        subsets.append(fringe); frontier = fringe
    union = torch.cat(subsets).unique(sorted=True).tolist()
    # Target role0, then stable global IDs; no training-dependent ranking.
    nodes = [root]+[int(n) for n in union if n != root]
    induced = a[nodes][:, nodes]
    if len(nodes) > 31:
        raise AssertionError('Whole-hop cap violated')
    audit = dict(sampler=SAMPLER, seed=int(seed), real_nodes=len(nodes),
        whole_fringe_counts=[len(v) for v in subsets[1:]],
        hop=[first_hop[n] for n in nodes],
        parent_toward_root=[None if n == root else parents[n] for n in nodes],
        direction='receiver_rows_sender_columns; incoming ancestors; induced edges',
        artificial_real_nodes=0)
    return nodes, induced, audit


@dataclass
class MemoryInput:
    history: torch.Tensor             # N,L raw units; never future
    observed: torch.Tensor
    normalized: torch.Tensor
    minimum: torch.Tensor
    extent: torch.Tensor
    adjacency: torch.Tensor           # N,N receiver/sender, plus self for present nodes
    edges: torch.Tensor               # receiver,sender,features
    present: torch.Tensor
    last: torch.Tensor                # raw last observation or0 for unavailable auxiliary
    node_ids: tuple
    identity: str
    audit: dict
    horizon: int

    def to(self, device):
        return replace(self, **{n: getattr(self, n).to(device) for n in (
            'history', 'observed', 'normalized', 'minimum', 'extent',
            'adjacency', 'edges', 'present', 'last')})


def memory_input(data, query, allowed=None, seed=42):
    if (query.length, query.horizon) not in SHAPES or data.external is None:
        raise ValueError('Two registered shapes and an audited external graph required')
    nodes, topology, audit = sample_subgraph(data.external, query.variable, query_seed(query, seed), allowed)
    start, end = query.origin-query.length, query.origin
    if start < 0 or end > len(data.values):
        raise ValueError('History origin out of range')
    if hasattr(data, 'segments') and data.segments[start] != data.segments[end-1]:
        raise ValueError('History crosses a time discontinuity')
    raw = np.asarray(data.values[start:end, nodes], np.float64).T.copy()
    mask = np.asarray(data.mask[start:end, nodes], bool).T.copy() & np.isfinite(raw)
    present = mask.any(1)
    if not present[0]:
        raise ValueError('Target requires an observed history')
    low = np.where(present, np.where(mask, raw, np.inf).min(1), 0.)
    high = np.where(present, np.where(mask, raw, -np.inf).max(1), 0.)
    scale = high-low
    scale = np.where(scale < 10*np.finfo(np.float64).eps, 1., scale)
    normalized = np.where(mask, (raw-low[:, None])/scale[:, None]-.5, 0.)
    last_index = np.where(mask, np.arange(query.length)[None], -1).max(1)
    last = np.where(present, raw[np.arange(len(nodes)), last_index.clip(0)], 0.)
    adjacency = topology.numpy().copy() | np.eye(len(nodes), dtype=bool)
    adjacency &= present[:, None] & present[None, :]
    hop = np.array(audit['hop'], np.float64)
    # No clipping: preserve scale and offset information across physical units.
    edges = np.stack(np.broadcast_arrays(np.log(scale[None, :]/scale[:, None]),
        np.arcsinh((low[None, :]-low[:, None])/scale[:, None]),
        hop[:, None]/3, hop[None, :]/3), -1).astype(np.float32)
    fields = dict(history=torch.from_numpy(np.where(mask, raw, 0.).astype(np.float32)),
        observed=torch.from_numpy(mask), normalized=torch.from_numpy(normalized.astype(np.float32)),
        minimum=torch.from_numpy(low), extent=torch.from_numpy(scale),
        adjacency=torch.from_numpy(adjacency), edges=torch.from_numpy(edges),
        present=torch.from_numpy(present), last=torch.from_numpy(last))
    digest = hashlib.sha256()
    for value in fields.values():
        digest.update(value.contiguous().numpy().tobytes())
    ids = tuple(str(data.ids[n]) for n in nodes)
    digest.update(json.dumps([query.key, ids, audit], sort_keys=True).encode())
    audit.update(global_indices=nodes, node_ids=ids, length=query.length, horizon=query.horizon)
    return MemoryInput(**fields, node_ids=ids, identity=digest.hexdigest(), audit=audit, horizon=query.horizon)


def all_node_labels(data, query, item):
    """Separate supervised output accessor. Never called by input construction."""
    nodes = item.audit['global_indices']; o = query.origin; h = query.horizon
    if hasattr(data, 'segments') and data.segments[o-1] != data.segments[o+h-1]:
        raise ValueError('Labels cross a time discontinuity')
    raw = np.asarray(data.values[o:o+h, nodes], np.float64).T.copy()
    mask = np.asarray(data.mask[o:o+h, nodes], bool).T.copy() & np.isfinite(raw)
    return torch.from_numpy(np.where(mask, raw, 0.)), torch.from_numpy(mask)


def task_text(quantity, unit, interval_seconds, length, horizon):
    if not quantity or not unit or interval_seconds <= 0 or (length, horizon) not in SHAPES:
        raise ValueError('Verified metadata or explicit unknown unit required')
    return (f'Task: multi-step time-series forecasting. Quantity: {quantity}. '
        f'Raw observation unit: {unit}. Sampling interval: {interval_seconds:g} seconds. '
        f'Observed history: {length} steps. Forecast: {horizon} future steps. '
        'Numerical model inputs use each node\'s observed-origin MinMax scale. '
        'Graph messages describe other nodes observed before the same forecast origin.')
