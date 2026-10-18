import torch
from torch.nn import functional as F

from gofa_f.short_projection_probe import differences, probe, project


def test_tail_work_row_is_never_a_supervised_value_and_preserves_gradient():
    generator = torch.Generator().manual_seed(18)
    weight = torch.randn(27, 8, generator=generator)
    alphabet = torch.tensor([1, 5, 7, 21, 24])
    hidden = torch.randn(7, 8, generator=generator, requires_grad=True)
    expected = F.linear(hidden, weight).index_select(-1, alphabet)
    actual = project(hidden, weight, alphabet, extra_rows=1)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert actual.shape[-1] == len(alphabet)
    first, = torch.autograd.grad(actual.square().sum(), hidden, retain_graph=True)
    second, = torch.autograd.grad(expected.square().sum(), hidden)
    torch.testing.assert_close(first, second, rtol=1e-6, atol=1e-6)
    assert weight.grad is None


def test_probe_compares_real_numeric_logits_loss_and_input_gradients():
    weight = torch.randn(10005, 8, generator=torch.Generator().manual_seed(42))
    alphabet = torch.arange(5, 10005)
    report = probe(weight, alphabet, 1e-6)
    assert report["candidate_passed"]
    assert set(report["cases"]) == {"1", "25", "48"}
    assert all(set(case["tail_one_10001"]) == {"logits", "loss", "hidden_gradient"}
               for case in report["cases"].values())


def test_projection_difference_does_not_mask_real_bf16_errors():
    expected = torch.tensor([1., 8.], dtype=torch.bfloat16)
    assert differences(expected.float(), expected, .002)["violations"] == 0
    assert differences(expected.float()+.0625, expected, .002)["violations"] == 2
