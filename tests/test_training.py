import json

import numpy as np
import pytest
import torch

from gofa_f.train import clip_gradients, run


def tiny_config(tmp_path, **overrides):
    data = tmp_path / "tiny.npz"
    values = np.random.default_rng(4).normal(size=(60, 3)).astype(np.float32)
    np.savez(data, values=values, mask=np.ones_like(values, bool), train_end=36, val_end=48)
    config = dict(data=str(data), context=12, horizon=3, device="cpu", epochs=1,
                  batch_size=2, model={"random_config": dict(d_model=16, d_ff=32, d_kv=8, num_heads=2,
                         num_layers=4, vocab_size=64, dropout_rate=0.0, decoder_start_token_id=0),
                         "graph_layers": [0, 2], "patch_size": 4, "graph_rank": 4})
    config.update(overrides)
    return config


def test_finite_large_gradients_are_clipped_without_norm_overflow():
    parameter = torch.nn.Parameter(torch.ones(2))
    parameter.grad = torch.full((2,), 1e25)
    norm = clip_gradients([parameter], 1.0)
    assert torch.isfinite(norm)
    assert parameter.grad.norm().item() == pytest.approx(1., abs=1e-5)


def test_end_to_end_training_and_provenance(tmp_path):
    config = tiny_config(tmp_path, max_steps=2)
    output = tmp_path / "run"
    result = run(config, output)
    assert result["status"] == "smoke"
    assert (output / "best.pt").exists()
    assert len(result["environment"]["data_sha256"]) == 64
    assert result["test"]["raw"]["observations"] > 0
    assert json.loads((output / "result.json").read_text())["best_epoch"] == 0


@pytest.mark.parametrize("model", [None, {"family": "persistence"}])
def test_validation_only_never_constructs_test_windows(tmp_path, monkeypatch, model):
    from gofa_f import train

    config = tiny_config(tmp_path, evaluation_split="validation")
    if model:
        config["model"] = model
    original = train.Windows
    constructed = []

    def guarded_windows(series, normalized, split, *args, **kwargs):
        assert split != "test", "Tuning must not access test targets"
        constructed.append(split)
        return original(series, normalized, split, *args, **kwargs)

    monkeypatch.setattr(train, "Windows", guarded_windows)
    result = run(config, tmp_path / "validation")
    assert constructed == ["train", "val"]
    assert result["status"] == "validated"
    assert result["validation"]["raw"]["observations"] > 0
    assert "test" not in result and "test_seconds" not in result
    assert "test" not in result["environment"]["windows"]
    if model is None:
        history = json.loads((tmp_path / "validation" / "history.jsonl").read_text())
        assert result["validation"] == history["validation"]


def test_validation_only_smoke_stays_smoke(tmp_path):
    result = run(tiny_config(tmp_path, evaluation_split="validation", max_steps=1), tmp_path / "smoke")
    assert result["status"] == "smoke"
    assert "test" not in result


def test_invalid_evaluation_split_does_not_create_output(tmp_path):
    with pytest.raises(ValueError, match="evaluation_split"):
        run({"evaluation_split": "typo"}, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
