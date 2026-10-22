from pathlib import Path
import pytest
import torch

from gofa_f.ufg_subgraph import pack_history
from gofa_f.ufg_memory_roster import MemoryRosterUFG

ROOT = Path('/home/dcg/dcg_data/GOFA-F/external/ufgtime')


@pytest.mark.parametrize('length,horizon,nodes', [(12,12,1),(12,12,16),(12,12,31),(48,24,6),(48,24,31)])
def test_official31_forward_inverse_missing_padding_gradients_reload(length, horizon, nodes):
    torch.manual_seed(42); model = MemoryRosterUFG(ROOT, length, horizon)
    history = torch.randn(1, length, nodes, requires_grad=True)
    observed = torch.ones_like(history, dtype=torch.bool)
    if nodes > 1: observed[:, 1::3, 1] = False
    packed = pack_history(history, observed, maximum_nodes=31)
    actual = model(packed, all_nodes=True)
    expected = (model.model(packed.normalized, packed.observed).double()+.5)*packed.extent[:, None]+packed.minimum[:, None]
    torch.testing.assert_close(actual, expected[:, :, :nodes], rtol=0, atol=0)
    assert actual.shape == (1, horizon, nodes) and packed.real_nodes == nodes
    actual.square().mean().backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters() if p.requires_grad)
    assert history.grad[~observed].eq(0).all()
    if nodes > 1: assert history.grad[:, :, 1:].abs().sum() > 0
    with torch.no_grad():
        copy = MemoryRosterUFG(ROOT, length, horizon); copy.load_state_dict(model.state_dict())
        torch.testing.assert_close(copy(packed, all_nodes=True), actual, rtol=0, atol=0)
    assert model.model.width == 128 and model.provenance['real_node_cap'] == 31
