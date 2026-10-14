from pathlib import Path

import pytest
import torch

from gofa_f.ufg import UFGTime, active_name, audit_parameters, matching_width, real_dof

ROOT = Path("external/ufgtime")
requires_source = pytest.mark.skipif(not (ROOT / "manifest.json").exists(), reason="Pinned UFG source absent")


def test_complex_real_dof_and_unused_names_are_explicit():
    assert real_dof(torch.zeros(3, dtype=torch.cfloat)) == 6
    assert real_dof(torch.zeros(3)) == 3
    assert not active_name("conv2_list.0.linear.weight")
    assert not active_name("embeddings")
    assert active_name("filters.0")
    with pytest.raises(ValueError):
        active_name("unclassified.weight")


@requires_source
@pytest.mark.parametrize("length,horizon,nodes", [(12, 12, 207), (12, 3, 325), (24, 6, 437),
                                                 (12, 12, 36), (8, 8, 321), (24, 3, 137)])
def test_actual_forward_width_match_counts_complex_dof(length, horizon, nodes):
    pytest.importorskip("dgl")
    width, expected = matching_width(ROOT, length, horizon, nodes)
    model = UFGTime(ROOT, length, horizon, nodes, scale="A")
    counts = audit_parameters(model)
    assert width == model.width
    assert counts["active_real_dof"] == expected == counts["trainable_real_dof"]
    assert abs(expected / 108930 - 1) < .10
    assert counts["registered_real_dof"] > counts["active_real_dof"]
    assert all(not p.requires_grad for n, p in model.official.named_parameters() if not active_name(n))


@requires_source
@pytest.mark.parametrize("length", [8, 12, 24])
def test_official_effective_forward_gradient_and_batch_isolation(length):
    pytest.importorskip("dgl")
    torch.manual_seed(42)
    model = UFGTime(ROOT, length, 3, 3, width=8)
    x = torch.randn(2, length, 3)
    observed = torch.rand_like(x) > .1
    output = model(x, observed)
    with torch.no_grad():
        expected = model.official(x.masked_fill(~observed, 0)).transpose(1, 2)
        separate = torch.cat([model(x[i:i+1], observed[i:i+1]) for i in range(2)])
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(output, separate, rtol=1e-5, atol=1e-5)
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad)
    changed = x.clone()
    changed[1] += 100
    with torch.no_grad():
        torch.testing.assert_close(model(changed, observed)[0], output[0], rtol=1e-5, atol=1e-5)


@requires_source
def test_short_tail_batch_and_fully_missing_history_are_finite():
    pytest.importorskip("dgl")
    model = UFGTime(ROOT, 8, 3, 3, width=8)
    output = model(torch.zeros(1, 8, 3), torch.zeros(1, 8, 3, dtype=torch.bool))
    assert output.shape == (1, 3, 3) and torch.isfinite(output).all()
