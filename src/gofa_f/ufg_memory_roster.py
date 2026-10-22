"""31-role adaptation of the unchanged, pinned official UFGTime effective model.

Shape-specific weights are explicit. No target-specific weights or input losses.
Absent trailing roles are zero padded, never counted as real observations/labels.
"""
import torch
from torch import nn

from gofa_f.ufg import UFGTime
from gofa_f.ufg_subgraph import pack_history


class MemoryRosterUFG(nn.Module):
    def __init__(self, root, length, horizon, device='cpu'):
        super().__init__()
        if (length, horizon) not in ((12, 12), (48, 24)):
            raise ValueError('Only the registered two shapes are permitted')
        self.length, self.horizon = length, horizon
        self.model = UFGTime(root, length, horizon, 31, device=device, width=128)
        self.provenance = dict(self.model.provenance, real_node_cap=31, padding='trailing masked-zero roles',
            target_role=0, physical_node_ids_learned=False, source_shape_specific_weights=True,
            external_topology_not_an_upstream_input=True, maximum_nodes_not_reduced=True)

    def forward(self, packed, all_nodes=False):
        if packed.normalized.shape[1:] != (self.length, 31):
            raise ValueError('UFG task shape or node cap changed')
        normalized = self.model(packed.normalized, packed.observed)
        raw = (normalized.double()+.5)*packed.extent[:, None]+packed.minimum[:, None]
        if not torch.isfinite(raw).all():
            raise FloatingPointError('UFG produced nonfinite forecasts')
        return raw[:, :, :packed.real_nodes] if all_nodes else raw[:, :, 0]


def pack_roster(item):
    return pack_history(item.history.T[None], item.observed.T[None], maximum_nodes=31)
