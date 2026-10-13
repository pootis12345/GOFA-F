import json

import numpy as np

from gofa_f.train import run


def test_end_to_end_training_and_provenance(tmp_path):
    data = tmp_path / "tiny.npz"
    values = np.random.default_rng(4).normal(size=(60, 3)).astype(np.float32)
    np.savez(data, values=values, mask=np.ones_like(values, bool), train_end=36, val_end=48)
    config = dict(data=str(data), context=12, horizon=3, device="cpu", epochs=1, max_steps=2,
                  batch_size=2, model={"random_config": dict(d_model=16, d_ff=32, d_kv=8, num_heads=2,
                         num_layers=4, vocab_size=64, dropout_rate=0.0, decoder_start_token_id=0),
                         "graph_layers": [0, 2], "patch_size": 4, "graph_rank": 4})
    output = tmp_path / "run"
    result = run(config, output)
    assert result["status"] == "smoke"
    assert (output / "best.pt").exists()
    assert len(result["environment"]["data_sha256"]) == 64
    assert result["test"]["raw"]["observations"] > 0
    assert json.loads((output / "result.json").read_text())["best_epoch"] == 0
