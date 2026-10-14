import numpy as np
import pytest
import torch

from gofa_f.native import NativeBolt, late_contiguous_layers
from test_native import random_bolt


def test_late_contiguous_keeps_last_transformer():
    assert late_contiguous_layers(4) == (1, 2)
    assert late_contiguous_layers(6) == (3, 4)
    with pytest.raises(ValueError):
        late_contiguous_layers(2)


@pytest.mark.parametrize("layers", [(), (1, 2)])
def test_adapter_zero_identity_private_weights_and_gradients(layers):
    torch.manual_seed(42)
    native = random_bolt()
    reference = NativeBolt(native, np.eye(3), 3, layers, rank=4)
    torch.manual_seed(99)
    adapted = NativeBolt(native, np.eye(3), 3, layers, rank=4, output_adapter_rank=4)
    # Constructing the adapter itself must not consume global RNG.
    from gofa_f.native import OutputAdapter

    before = torch.get_rng_state().clone()
    delta = OutputAdapter(32, 48, 4)
    assert torch.equal(before, torch.get_rng_state())
    assert delta.b.count_nonzero() == 0
    original_weights = {k: v.detach().clone() for k, v in native.state_dict().items()}
    x, mask = torch.randn(2, 12, 3), torch.ones(2, 12, 3, dtype=torch.bool)
    torch.testing.assert_close(reference(x, mask), adapted(x, mask), rtol=0, atol=0)
    adapted(x, mask).square().mean().backward()
    assert adapted.output_adapter.b.grad.abs().sum() > 0
    assert all(p.grad is None for p in native.parameters())
    with torch.no_grad():
        adapted.output_adapter.b.add_(0.1 * adapted.output_adapter.b.grad)
    assert not torch.equal(reference(x, mask), adapted(x, mask))
    assert all(torch.equal(v, native.state_dict()[k]) for k, v in original_weights.items())


def test_adapter_cached_predictions_and_gradients():
    model = NativeBolt(
        random_bolt(), np.ones((3, 3)), 3, (1, 2), rank=4, graph_feature_dim=11, output_adapter_rank=4
    )
    for layer in model.graph.values():
        layer.gate.data.fill_(0.1)
    model.output_adapter.b.data.normal_(0, 0.01)
    x, mask, features = torch.randn(2, 12, 3), torch.ones(2, 12, 3, dtype=torch.bool), torch.randn(2, 3, 11)
    outputs, gradients = [], []
    for cached in (False, True):
        model.zero_grad(set_to_none=True)
        y = (
            model.from_prefix(model.prefix(x, mask, 2), 2, graph_features=features)
            if cached
            else model(x, mask, graph_features=features)
        )
        y.square().mean().backward()
        outputs.append(y.detach())
        gradients.append({k: p.grad.clone() for k, p in model.named_parameters() if p.requires_grad})
    torch.testing.assert_close(*outputs)
    for k in gradients[0]:
        torch.testing.assert_close(gradients[0][k], gradients[1][k])


def test_raw_targets_do_not_roundtrip_through_scaler(tmp_path):
    from gofa_f.native_stage import Data

    values = np.array([[123.456, 0.0]] * 32 + [[100.0, 2.0]] * 32, np.float32)
    path = tmp_path / "data.npz"
    np.savez(path, values=values, mask=np.ones_like(values, bool))
    data = Data({"data": str(path), "context": 8, "horizons": [3]}, "cpu")
    batch = data.batch("train", [0])
    assert torch.equal(batch["y_raw"], torch.from_numpy(values[8:11]).unsqueeze(0))
