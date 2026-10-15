"""Real pretrained METR-LA Stage6 CPU preflight; never creates a CUDA context."""
import json
import os
from pathlib import Path

import numpy as np
import torch

from gofa_f.long_stage import config_file
from gofa_f.long_task import data_for, new_model
from gofa_f.prefix_cache import atomic_json
from gofa_f.pure_graph_task import model_counts


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise RuntimeError("CPU verification requires CUDA_VISIBLE_DEVICES empty")
    from chronos.chronos_bolt import ChronosBoltModelForForecasting
    config = config_file("configs/stage6_metr_long.yaml")
    native = ChronosBoltModelForForecasting.from_pretrained(config["native_weights"],
        local_files_only=True, torch_dtype=torch.float32).requires_grad_(False).eval()
    data, settings = data_for(config, torch.device("cpu"))
    assert "test" not in data.windows
    indices = [0, len(data.windows["train"])-1]
    batch = data.batch("train", indices)
    assert batch["x"].shape == (2, 12, 207)
    assert batch["graph_features"].shape == (2, 207, 11)
    assert (batch["graph_features"][:, :, -2:] == 1).all()
    changed_targets = dict(batch, y=batch["y"]+1000, y_raw=batch["y_raw"]+1000)
    assert torch.equal(batch["graph_features"], changed_targets["graph_features"])
    context = batch["x"].transpose(1, 2).reshape(-1, 12).masked_fill(
        ~batch["x_mask"].transpose(1, 2).reshape(-1, 12), torch.nan)
    with torch.no_grad():
        native_prediction = native(context=context).quantile_preds[:,
            native.chronos_config.quantiles.index(.5), :12].reshape(2, 207, 12).transpose(1, 2)
    records, paired = {}, None
    for variant in ("c1_clock", "c2_residual"):
        model = new_model(config, native, data, settings, variant)
        common = {n: p.detach().clone() for n, p in model.named_parameters()
                  if p.requires_grad and not n.startswith("temporal_residual.")}
        if paired is None:
            paired = common
        else:
            for name in paired:
                assert torch.equal(paired[name], common[name]), name
        with torch.no_grad():
            direct = model(batch["x"], batch["x_mask"], batch["graph_features"])
            prefix = model.prefix(batch["x"], batch["x_mask"], 2)
            cached = model.from_prefix(prefix, 2, graph_features=batch["graph_features"],
                history=batch["x"], history_mask=batch["x_mask"])
        torch.testing.assert_close(direct, native_prediction, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(direct, cached, rtol=1e-5, atol=1e-5)
        for module in model.graph.values():
            module.gate.data.fill_(.1)
        outputs, gradients = [], []
        for direct_path in (True, False):
            model.zero_grad(set_to_none=True)
            output = model(batch["x"], batch["x_mask"], batch["graph_features"]) if direct_path \
                else model.from_prefix(prefix, 2, graph_features=batch["graph_features"],
                                       history=batch["x"], history_mask=batch["x_mask"])
            output.float().square().mean().backward()
            outputs.append(output.detach())
            gradients.append({n: p.grad.detach().clone() for n, p in model.named_parameters()
                              if p.requires_grad and p.grad is not None})
        torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-5, atol=1e-5)
        for name in gradients[0]:
            assert torch.isfinite(gradients[0][name]).all()
            torch.testing.assert_close(gradients[0][name], gradients[1][name], rtol=1e-4, atol=1e-5)
        assert all(p.grad is None for p in native.parameters())
        missing_mask = batch["x_mask"].clone()
        missing_mask[:, :, 0] = False
        with torch.no_grad():
            assert torch.isfinite(model(batch["x"], missing_mask, batch["graph_features"])).all()
        records[variant] = {"initial_max_difference": float((direct-native_prediction).abs().max()),
            "cache_max_difference": float((direct-cached).abs().max()),
            "paired_common_parameters": True, "cached_gradients_match": True,
            "frozen_native": True, "missing_node_finite": True,
            "trainable_parameters": model_counts(model)["trainable_real_dof"]}
        del model
    result = {"passed": True, "cuda_initialized": torch.cuda.is_initialized(),
        "dataset": "metr_la", "seed": 42, "context": 12, "horizon": 12,
        "source_clock_known": True, "test_targets_loaded": False, "variants": records}
    target = Path("outputs/stage6-metr-long-v1/cpu_verification.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(target, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
