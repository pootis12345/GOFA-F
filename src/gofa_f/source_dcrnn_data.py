"""Same causal9-node histories as GOFA-F; labels restricted to source pools."""
import numpy as np

from gofa_f.allnode_graph import build_graph
from gofa_f.generative_data import Query


def sample(data, query, labels=False, split=None):
    q = Query(**query)
    if split:
        a, b = ((0, int(.6*len(data.values))) if split == 'train'
                else (int(.6*len(data.values)), int(.8*len(data.values))))
        if not a <= q.origin or q.origin+q.horizon > b:
            raise ValueError('Source label split crossed')
    if data.segments[q.origin-q.length] != data.segments[q.origin+q.horizon-1]:
        raise ValueError('Discontinuous query')
    own = build_graph(data, q)
    graph = own.graph
    history = graph.history.numpy().astype(np.float64)
    observed = graph.observed.numpy()
    count = history.shape[1]
    present = observed.any(0)
    low = np.where(observed, history, np.inf).min(0)
    high = np.where(observed, history, -np.inf).max(0)
    low = np.where(present, low, 0.)
    scale = np.where(present & (high-low > 10*np.finfo(np.float64).eps), high-low, 1.)
    x = np.zeros((q.length, 9, 2), np.float32)
    x[:, :count, 0] = np.where(observed, (history-low)/scale-.5, 0)
    x[:, :count, 1] = observed
    support = np.zeros((2, 9, 9), np.float32)
    adjacency = .5*(graph.prior.numpy()+graph.base.numpy())
    reverse = adjacency.T.copy()
    reverse = np.divide(reverse, reverse.sum(1, keepdims=True), out=np.zeros_like(reverse),
                        where=reverse.sum(1, keepdims=True) > 0)
    support[0, :count, :count] = adjacency
    support[1, :count, :count] = reverse
    padded_present = np.zeros(9, np.float32)
    padded_present[:count] = present
    result = dict(x=x, supports=support, present=padded_present,
                  low=np.float64(low[0]), scale=np.float64(scale[0]),
                  empty_target=np.bool_(not present[0]))
    if labels:
        mapping = {str(v): i for i, v in enumerate(data.ids)}
        indices = [mapping[v] for v in own.variable_ids]
        values = data.values[q.origin:q.origin+q.horizon][:, indices]
        valid = data.mask[q.origin:q.origin+q.horizon][:, indices].astype(bool)&np.isfinite(values)
        y = np.zeros((q.horizon, 9), np.float32)
        mask = np.zeros_like(y, bool)
        y[:, :count] = np.where(valid, (values-low)/scale-.5, 0)
        mask[:, :count] = valid & present[None]
        teacher = np.zeros_like(y)
        carry = np.zeros(count, np.float32)
        for i in range(count):
            ix = np.flatnonzero(observed[:, i])
            carry[i] = (history[ix[-1], i]-low[i])/scale[i]-.5 if len(ix) else 0.
        for h in range(q.horizon):
            carry = np.where(valid[h], y[h, :count], carry)
            teacher[h, :count] = carry
        result.update(y=y, observed=mask, teacher=teacher, raw_target=values[:, 0].astype(np.float64))
    return result


def stack_samples(samples):
    return {name: np.stack([row[name] for row in samples]) for name in samples[0]}
