import numpy as np
import pytest
import torch

from gofa_f.models import ForecastEncoder, Persistence


def tiny(graph=(), frozen=False, chronos=False, checkpointing=False):
    config = dict(d_model=16, d_ff=32, d_kv=8, num_heads=2, num_layers=4,
                  vocab_size=64, dropout_rate=0.0, decoder_start_token_id=0)
    if chronos:
        config["chronos_config"] = {"tokenizer_class": "MeanScaleUniformBins", "n_tokens": 64,
            "n_special_tokens": 2, "pad_token_id": 0, "eos_token_id": 1, "use_eos_token": True,
            "context_length": 512, "tokenizer_kwargs": {"low_limit": -15., "high_limit": 15.}}
    return ForecastEncoder(12, 3, np.ones((3, 3), np.float32), graph_layers=graph,
                           patch_size=4, graph_rank=4, dropout=0.0, freeze_backbone=frozen,
                           random_config=config, gradient_checkpointing=checkpointing)


@pytest.mark.parametrize("chronos", [False, True])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_frozen_backbone_keeps_adapter_gradients(chronos, checkpointing):
    model = tiny((0, 2), True, chronos, checkpointing)
    x = torch.randn(2, 12, 3)
    mask = torch.ones_like(x, dtype=torch.bool)
    output = model(x, mask)
    assert output.shape == (2, 3, 3)
    output.square().mean().backward()
    assert model.adapter.weight.grad is not None
    assert model.adapter.weight.grad.abs().sum() > 0
    assert model.graph["0"].gate.grad is not None
    assert model.graph["0"].gate.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.backbone.parameters())


def test_zero_gates_reproduce_no_graph_control():
    torch.manual_seed(7)
    control = tiny()
    graph = tiny((0, 2))
    graph.load_state_dict(control.state_dict(), strict=False)
    x = torch.randn(2, 12, 3)
    mask = torch.ones_like(x, dtype=torch.bool)
    control.eval()
    graph.eval()
    torch.testing.assert_close(control(x, mask), graph(x, mask))


def test_graph_exchanges_nodes():
    model = tiny((0, 2))
    model.eval()
    for block in model.graph.values():
        block.gate.data.fill_(1)
    x = torch.randn(1, 12, 3)
    mask = torch.ones_like(x, dtype=torch.bool)
    initial = model(x, mask)
    x[:, :, 1] = torch.randn(1, 12) * 3
    changed = model(x, mask)
    assert not torch.allclose(initial[:, :, 0], changed[:, :, 0])


def test_no_graph_does_not_exchange_nodes():
    model = tiny()
    model.eval()
    x = torch.randn(1, 12, 3)
    mask = torch.ones_like(x, dtype=torch.bool)
    initial = model(x, mask)
    x[:, :, 1] = torch.randn(1, 12) * 3
    changed = model(x, mask)
    torch.testing.assert_close(initial[:, :, 0], changed[:, :, 0])


@pytest.mark.parametrize("chronos", [False, True])
def test_missing_inputs_are_ignored(chronos):
    model = tiny((0, 2), chronos=chronos)
    model.eval()
    x = torch.randn(1, 12, 3)
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[:, :4, 0] = False
    mask[:, :, 2] = False
    initial = model(x, mask)
    x[~mask] = 1e5
    changed = model(x, mask)
    torch.testing.assert_close(initial, changed)
    assert torch.isfinite(changed).all()


def test_persistence_uses_last_observed_value():
    model = Persistence(2)
    x = torch.tensor([[[1., 4.], [2., 5.], [999., 6.]]])
    mask = torch.tensor([[[True, False], [True, False], [False, False]]])
    torch.testing.assert_close(model(x, mask), torch.tensor([[[2., 0.], [2., 0.]]]))
