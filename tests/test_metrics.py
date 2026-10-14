import pytest
import torch

from gofa_f.metrics import Metrics, masked_loss


def test_metrics_weight_observations_not_batches():
    metrics = Metrics(1)
    metrics.update(torch.tensor([[[2.], [0.]]]).transpose(1, 2),
                   torch.zeros(1, 1, 2), torch.tensor([[[True, False]]]))
    metrics.update(torch.ones(3, 1, 1), torch.zeros(3, 1, 1), torch.ones(3, 1, 1, dtype=torch.bool))
    result = metrics.result()
    assert result["mae"] == 1.25
    assert result["mse"] == 1.75
    assert result["mape_pct"] is None


def test_loss_ignores_masked_nonfinite_targets():
    pred = torch.tensor([1., 2.], requires_grad=True)
    loss = masked_loss(pred, torch.tensor([0., float("nan")]), torch.tensor([True, False]))
    assert loss.item() == 1


def test_nonfinite_forecast_raises():
    with pytest.raises(FloatingPointError):
        masked_loss(torch.tensor([float("nan")]), torch.zeros(1), torch.ones(1, dtype=torch.bool))


def test_raw_zero_exclusion_and_wape_use_original_targets():
    metrics = Metrics(1)
    metrics.update(torch.tensor([[[1., 3., 99.]]]), torch.tensor([[[0., 2., float("nan")]]]),
                   torch.tensor([[[True, True, False]]]))
    result = metrics.result()
    assert result["mae"] == 1
    assert result["mape_pct"] == 50
    assert result["mape_observations"] == 1
    assert result["mape_valid_fraction"] == .5
    assert result["wape_pct"] == 100
