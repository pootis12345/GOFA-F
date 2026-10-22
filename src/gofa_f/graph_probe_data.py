"""Source-road components, immutable queries, causal features and matched controls."""
from collections import deque
from dataclasses import asdict
import hashlib

import numpy as np

from gofa_f.generative_data import Query

VIEWS = ('pems04_flow', 'pems04_speed')
CONDITIONS = ('physical', 'self', 'rewired')


def road_adjacency(adjacency, contract='csv_directed'):
    value=np.asarray(adjacency)>0
    if value.ndim!=2 or value.shape[0]!=value.shape[1]:
        raise ValueError('Square audited road adjacency required')
    if contract=='csv_directed':return value.astype(np.float32)
    if contract=='publisher_bidirectional':return (value|value.T).astype(np.float32)
    raise ValueError('Unregistered road graph convention')


def component_partition(adjacency, ids):
    a = np.asarray(adjacency) > 0
    if a.shape != (len(ids), len(ids)) or len(set(ids)) != len(ids):
        raise ValueError('Unique source node mapping required')
    undirected = a | a.T
    remaining = set(range(len(ids)))
    components = []
    while remaining:
        root = min(remaining, key=lambda i: str(ids[i]))
        visited, todo = {root}, deque([root])
        while todo:
            for j in np.flatnonzero(undirected[todo.popleft()]):
                j = int(j)
                if j not in visited:
                    visited.add(j)
                    todo.append(j)
        remaining -= visited
        components.append(sorted(visited))
    components.sort(key=lambda c: (-len(c), tuple(str(ids[i]) for i in c)))
    if len(components) < 4 or len(components[1]) < 16 or any(len(c) < 4 for c in components[2:4]):
        raise ValueError('Fixed disconnected source-region design not feasible')
    confirmation = components[1]
    validation = sorted(components[2]+components[3])
    train = sorted(set(range(len(ids)))-set(validation)-set(confirmation))
    result = dict(train=train, validation=validation, confirmation=confirmation,
        component_sizes=[len(c) for c in components], method='topology_only_disconnected_components')
    for left, right in (('train','validation'),('train','confirmation'),('validation','confirmation')):
        if set(result[left]) & set(result[right]) or a[np.ix_(result[left], result[right])].any() \
                or a[np.ix_(result[right], result[left])].any():
            raise ValueError('Road components must be disjoint with no cross edge')
    return result


def select_training(pool, variables, count, view_index):
    rng = np.random.default_rng(np.random.SeedSequence([42, 36, view_index]))
    variables = rng.permutation(variables).tolist()
    allocation = [variables[i % len(variables)] for i in range(count)]
    result = []
    for v in variables:
        candidates = sorted([q for q in pool if q['variable'] == v], key=lambda q: q['origin'])
        own = allocation.count(v)
        if len(candidates) < own:
            raise ValueError('Source training pool lacks fixed per-variable capacity')
        for group in np.array_split(np.arange(len(candidates)), own):
            q = dict(candidates[int(rng.choice(group))])
            if q['length'] != 12 or q['horizon'] != 12 or q.get('chunk', 0) != 0:
                raise ValueError('Traffic probe requires the audited12→12 template')
            result.append(dict(query=q, block=-1))
    if len(result) != count or len({Query(**r['query']).key for r in result}) != count:
        raise ValueError('Balanced source training budget changed')
    return sorted(result, key=lambda r: (r['query']['origin'], r['query']['variable']))


def evaluation_queries(series, variables, split, excluded):
    """Same sensor/origin pairs for flow and speed;4 disjoint temporal strata."""
    if split not in ('validation', 'confirmation'):
        raise ValueError('Source validation/confirmation only')
    result = {v: [] for v in VIEWS}
    reference = series[VIEWS[0]]
    n = len(reference.values)
    start, end = (int(.6*n), int(.8*n)) if split == 'validation' else (int(.8*n), n)
    cuts = [start+(end-start)*i//4 for i in range(5)]
    for variable in sorted(variables):
        points = None
        for name in VIEWS:
            candidates = series[name].candidates(variable, 12, 12,
                'validation' if split == 'validation' else 'test')
            points = candidates if points is None else np.intersect1d(points, candidates)
        for old in excluded:
            if old['source'] in VIEWS and old['variable'] == variable:
                points = points[(points+12 <= old['origin']) | (points >= old['origin']+old['horizon'])]
        for block in range(4):
            legal = points[(points >= cuts[block]) & (points+12 <= cuts[block+1])]
            if not len(legal):
                raise ValueError('No fresh lawful source time stratum; no replacement region')
            rng = np.random.default_rng(np.random.SeedSequence(
                [42, 36, int(split == 'confirmation'), variable, block]))
            origin = int(rng.choice(legal))
            for name in VIEWS:
                result[name].append(dict(query=asdict(Query(name, variable, origin, 12, 12)), block=block))
    for rows in result.values():
        rows.sort(key=lambda r: (r['block'], r['query']['variable']))
    return result


def node_roster(adjacency, target, allowed, ids, maximum=9):
    allowed = set(allowed)
    if target not in allowed:
        raise ValueError('Target outside the designated source-node partition')
    undirected = (adjacency > 0) | (adjacency.T > 0)
    selected, todo = [target], deque([target])
    # Undirected traversal ONLY chooses a fixed roster. Message direction is not changed.
    while todo and len(selected) < maximum:
        receiver = todo.popleft()
        neighbors = sorted((int(i) for i in np.flatnonzero(undirected[receiver])
                            if int(i) in allowed and int(i) not in selected), key=lambda i: str(ids[i]))
        for i in neighbors:
            if len(selected) == maximum:
                break
            selected.append(i)
            todo.append(i)
    return [target]+sorted(selected[1:], key=lambda i: str(ids[i]))


def pack(data, record, allowed, labels=True, split='train'):
    q = Query(**record['query'])
    if q.source not in VIEWS or q.length != 12 or q.horizon != 12 or q.chunk:
        raise ValueError('Only fixed traffic source queries are admitted')
    if split not in ('train','validation','confirmation'):
        raise ValueError('No target-data access in the source probe')
    n = len(data.values)
    a, b = {'train': (0, int(.6*n)), 'validation': (int(.6*n), int(.8*n)),
            'confirmation': (int(.8*n), n)}[split]
    if q.origin < max(a, q.length) or q.origin+q.horizon > b \
            or data.segments[q.origin-q.length] != data.segments[q.origin+q.horizon-1]:
        raise ValueError('Source time split or continuity crossed')
    indices = node_roster(data.external, q.variable, allowed, data.ids)
    values = data.values[q.origin-q.length:q.origin][:, indices].astype(np.float64).T
    mask = data.mask[q.origin-q.length:q.origin][:, indices].T.astype(bool) & np.isfinite(values)
    present = mask.any(1)
    if not present[0]:
        raise ValueError('Empty source target history')
    low = np.where(present, np.where(mask, values, np.inf).min(1), 0.)
    high = np.where(present, np.where(mask, values, -np.inf).max(1), 0.)
    scale = np.where(present & (high-low > 10*np.finfo(np.float64).eps), high-low, 1.)
    x = np.zeros((9, q.length, 2), np.float32)
    count = len(indices)
    x[:count, :, 0] = np.where(mask, (values-low[:, None])/scale[:, None]-.5, 0)
    x[:count, :, 1] = mask
    adjacency = np.zeros((9,9), bool)
    adjacency[:count, :count] = data.external[np.ix_(indices, indices)] > 0
    np.fill_diagonal(adjacency, False)
    edge = np.zeros((9,9,2), np.float32)
    edge[:count, :count, 0] = np.clip(np.log(scale[None]/scale[:, None]), -5, 5)
    edge[:count, :count, 1] = np.clip((low[None]-low[:, None])/scale[:, None], -5, 5)
    padded_ids = np.full(9, -1, np.int64)
    padded_ids[:count] = indices
    last = float(values[0, np.flatnonzero(mask[0])[-1]])
    result = dict(x=x, adjacency=adjacency, edge_features=edge, nodes=padded_ids,
        low=np.float64(low[0]), scale=np.float64(scale[0]), persistence=np.float64(last))
    if labels:
        if split == 'confirmation':
            raise ValueError('Confirmation labels require the separate locked-confirmation interface')
        target = data.values[q.origin:q.origin+q.horizon, q.variable].astype(np.float64)
        observed = data.mask[q.origin:q.origin+q.horizon, q.variable].astype(bool) & np.isfinite(target)
        if not observed.any():
            raise ValueError('No valid source labels')
        result.update(y=np.where(observed, (target-low[0])/scale[0]-.5, 0).astype(np.float32),
            observed=observed, raw_target=np.where(observed, target, 0))
    return result


def controls(adjacency, nodes, records, condition):
    result = np.asarray(adjacency).copy()
    if condition == 'physical':
        return result
    if condition == 'self':
        return np.zeros_like(result)
    if condition != 'rewired':
        raise ValueError('Unknown graph control')
    for i, (ids, record) in enumerate(zip(nodes, records)):
        count = int(np.count_nonzero(ids >= 0))
        others = np.arange(1, count)
        permutation = np.arange(len(ids))
        if len(others) > 1:
            key = Query(**record['query']).key
            seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little')
            rng = np.random.default_rng(np.random.SeedSequence([42,36,seed]))
            order = rng.permutation(others)
            permutation[order] = np.roll(order, 1)
        result[i] = result[i, permutation][:, permutation]
    return result
