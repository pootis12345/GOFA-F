"""Stage47: fixed inverse-hop attention prior, matched shuffled geometry control."""
from collections import deque
from contextlib import contextmanager
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from gofa_f import source_graph_probe as p
from gofa_f import source_neighborhood_probe as n
from gofa_f import source_normalized_probe as bounded
from gofa_f.generative_data import sha

PROTOCOL='stage47-source-distance-prior-probe-v1'
PARENT_REPORT='b9508f08f6e65c9e5b6f71c06fccdfe4ebeb7e3cec533fc12f92adef96a04602'
SEMANTICS=dict(physical='physical_roster_dense_support_correct_inverse_road_hops',
    self='same_physical_roster_no_cross_messages',
    rewired='same_physical_roster_target_fixed_permuted_road_distances')


def shortest_paths(external,allowed):
    """Published receiver/sender direction; no paths outside source node partition."""
    allowed=set(allowed);size=len(external);distance=np.full((size,size),np.inf)
    neighbors={i:[int(j) for j in np.flatnonzero(external[i]>0) if int(j) in allowed] for i in allowed}
    for receiver in allowed:
        distance[receiver,receiver]=0.;todo=deque([receiver])
        while todo:
            current=todo.popleft()
            for sender in neighbors[current]:
                if not np.isfinite(distance[receiver,sender]):
                    distance[receiver,sender]=distance[receiver,current]+1;todo.append(sender)
    return distance


def road_bias(distance,nodes,permuted=False):
    nodes=np.asarray(nodes);count=int((nodes>=0).sum());selected=distance[np.ix_(nodes[:count],nodes[:count])]
    if not np.isfinite(selected).all():raise ValueError('Physical roster must have finite published road distances')
    prior=-np.log(np.maximum(selected,1.))
    if permuted and count>2:
        rng=np.random.default_rng(np.random.SeedSequence([42,47,*nodes[:count].tolist()]))
        others=rng.permutation(np.arange(1,count));order=np.arange(count);order[others]=np.roll(others,1)
        prior=prior[order][:,order]
    result=np.zeros((len(nodes),len(nodes)),np.float32);result[:count,:count]=prior
    return result


class DistanceScore(torch.nn.Module):
    def __init__(self,original):
        super().__init__();self.weight=original.weight

    def forward(self,edge):
        if edge.shape[-1] not in (2,3):raise ValueError('Two original features plus optional fixed bias required')
        # Two-channel inputs explicitly mean neutral geometry (legacy/profile).
        score=F.linear(edge[...,:2],self.weight)
        return score+edge[...,2:3] if edge.shape[-1]==3 else score


def original_value_features(module,inputs):return (inputs[0][...,:2],)


def add_distance_prior(model):
    if getattr(model,'_stage47_distance',False):raise ValueError('Distance prior already installed')
    for block in model.exchanges:
        block.edge_score=DistanceScore(block.edge_score)
        block.edge_value.register_forward_pre_hook(original_value_features)
    model._stage47_distance=True
    return model


def reused_caches(output,m,charge):
    parent=Path(m['configuration']['parent_output']);old=p.read(parent/'manifest.json')
    if sha(parent/'manifest.json')!=m['parent_manifest_sha256'] or old['queries']!=m['queries']:
        raise ValueError('Exact Stage40 parent required')
    root=Path(old['configuration']['parent_output'])
    if sha(root/'manifest.json')!=p.read(parent/'cache_reuse.json')['parent_manifest_sha256']:
        raise ValueError('Immutable cache ancestry changed')
    proxy=dict(m,parent_manifest_sha256=sha(root/'manifest.json'),configuration=dict(m['configuration'],parent_output=str(root)))
    original=bounded.reuse_parent_caches(output,proxy,charge);data=p.source_series(m)
    result={c:{s:{} for s in ('train','validation','confirmation')} for c in p.CONDITIONS};proof={}
    for split in result['physical']:
        distance=shortest_paths(data[p.VIEWS[0]].external,m['partition'][split])
        for view in p.VIEWS:
            np.testing.assert_array_equal(data[view].external,data[p.VIEWS[0]].external)
            a=original['physical'][split][view]
            for c in ('physical','rewired'):
                bias=np.stack([road_bias(distance,nodes,permuted=c=='rewired') for nodes in a['nodes']])
                edge=np.concatenate((a['edge_features'],bias[...,None]),-1)
                result[c][split][view]=dict(a,edge_features=edge)
                proof[f'{c}-{split}-{view}']=hashlib.sha256(edge.tobytes()).hexdigest()
            charge()
    result['self']=result['physical']
    p.immutable(output/'distance_prior.json',dict(manifest_sha256=sha(output/'manifest.json'),
        formula='attention_logits += -log(max(published_shortest_hops,1))',new_parameters=0,
        same_rosters_and_observations=True,confirmation_labels_read=False,derived_float32_sha256=proof))
    return result


@contextmanager
def distance_prior_runtime():
    with bounded.normalized_runtime():
        old=(p.model_for,n.PROTOCOL,n.PARENT_REPORT,n.code_hashes,n.caches,n.SEMANTICS)
        factory,_,_,hashes,_,_=old
        p.model_for=lambda cfg,seed:add_distance_prior(factory(cfg,seed))
        n.PROTOCOL=PROTOCOL;n.PARENT_REPORT=PARENT_REPORT;n.SEMANTICS=SEMANTICS
        n.code_hashes=lambda:dict(hashes(),source_distance_prior_probe=sha(Path(__file__)))
        n.caches=reused_caches
        try:yield
        finally:p.model_for,n.PROTOCOL,n.PARENT_REPORT,n.code_hashes,n.caches,n.SEMANTICS=old


if __name__=='__main__':
    with distance_prior_runtime():n.main()
