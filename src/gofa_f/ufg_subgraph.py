"""Audited fixed-roster input adapter for the official UFGTime forward.

This module contains no query selector, training loop, checkpoint selection or
target-label access. It is a transfer-protocol building block, not a new score.
"""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from gofa_f.ufg import UFGTime


@dataclass
class SubgraphInput:
    normalized: torch.Tensor
    observed: torch.Tensor
    minimum: torch.Tensor
    extent: torch.Tensor
    present: torch.Tensor
    real_nodes: int


def pack_history(history,observed,maximum_nodes=9):
    """Only [batch,history,node] arrays; there is no future/label argument.

    MinMax uses each node's observed origin history in float64, the accepted
    ChatTime constant-range rule, and zero fill in the normalized UFG input.
    Real physical-roster nodes come first, target at0; absent trailing slots
    are explicitly masked zero, retaining the unchanged fixed-N official model.
    """
    raw=torch.as_tensor(history)
    mask=torch.as_tensor(observed,device=raw.device,dtype=torch.bool)
    if raw.ndim!=3 or raw.shape!=mask.shape or not 1<=raw.shape[2]<=maximum_nodes:
        raise ValueError('Explicit B,L,N history/mask with one to nine real nodes required')
    raw=raw.double();mask=mask&torch.isfinite(raw);present=mask.any(1)
    if not present[:,0].all():raise ValueError('Target origin history must have an observation')
    low=raw.masked_fill(~mask,torch.inf).amin(1).masked_fill(~present,0.)
    high=raw.masked_fill(~mask,-torch.inf).amax(1).masked_fill(~present,0.)
    extent=high-low;extent=torch.where(extent<10*np.finfo(np.float64).eps,torch.ones_like(extent),extent)
    normalized=((raw-low[:,None])/extent[:,None]-.5).masked_fill(~mask,0.).float()
    count=raw.shape[2];padding=maximum_nodes-count
    return SubgraphInput(F.pad(normalized,(0,padding)),F.pad(mask,(0,padding)),
        F.pad(low,(0,padding)),F.pad(extent,(0,padding),value=1.),F.pad(present,(0,padding)),count)


class SubgraphUFGTime(nn.Module):
    """Original effective UFGTime, fixed nine roster roles and shape-specific weights."""
    def __init__(self,root,length,horizon,device='cpu',width=128):
        super().__init__()
        if (length,horizon) not in ((12,12),(48,24)) or width!=128:
            raise ValueError('Only the two fixed task shapes and official default width128 are registered')
        self.length,self.horizon=length,horizon
        self.model=UFGTime(root,length,horizon,9,device=device,width=width)
        self.provenance=dict(self.model.provenance,scale='official_default_width128',history_encoding='node-wise origin MinMax minus0.5; missing zero fill',
            original_effective_forward_unchanged=True,fixed_roster_roles=9,target_role=0,
            padded_nodes_are_virtual=True,padded_nodes_count_as_observed=False,
            physical_roster_but_native_learned_frequency_graph=True,extra_future_covariates=False,
            cross_shape_single_checkpoint=False)

    def forward(self,packed,all_nodes=False):
        if packed.normalized.ndim!=3 or packed.normalized.shape[1:]!=(self.length,9):
            raise ValueError('UFG checkpoint task shape changed')
        normalized=self.model(packed.normalized,packed.observed)
        if not torch.isfinite(normalized).all():raise FloatingPointError('Nonfinite official UFG prediction')
        raw=(normalized.double()+.5)*packed.extent[:,None]+packed.minimum[:,None]
        return raw[:,:,:packed.real_nodes] if all_nodes else raw[:,:,0]
