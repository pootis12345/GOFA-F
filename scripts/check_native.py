"""Necessary real-weight mechanisms, not throughput or accuracy benchmarking.

Only training-prefix data are read. CUDA seconds are charged to the new stage.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
import yaml
from chronos.chronos_bolt import ChronosBoltModelForForecasting

from gofa_f.data import file_sha256
from gofa_f.models import parameter_counts
from gofa_f.native import HSDGNNAdapter, NativeBolt
from gofa_f.graph_context import FEATURE_DIM
from gofa_f.native_stage import Data, GPU, dataset_settings, fit, seed, stage_config, trained_state
from gofa_f.prefix_cache import PrefixCache, atomic_json
from gofa_f.stage_budget import Budget


CODE_FILES = ["src/gofa_f/native.py", "src/gofa_f/native_stage.py", "src/gofa_f/prefix_cache.py",
              "src/gofa_f/stage_budget.py", "src/gofa_f/graph_context.py", "configs/next_stage.yaml",
              "scripts/check_native.py"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/next_stage.yaml")
    args = parser.parse_args()
    prior_seconds = 0.0
    if Path(args.output).exists():
        prior_seconds = json.loads(Path(args.output).read_text()).get("gpu_seconds", 0.0)
    start = time.monotonic()
    config = stage_config(args.config)
    enabled = config.get("graph_context", {}).get("enabled", False)
    graph_options = {"graph_feature_dim": FEATURE_DIM} if enabled else {}
    spec = yaml.safe_load(Path("configs/benchmark.yaml").read_text())
    gpu = GPU(config["resources"])
    with tempfile.TemporaryDirectory(prefix="gofa-native-mechanisms-") as temporary:
        budget = Budget(Path(temporary) / "budget.json", 900)
        budget.activate("mechanism", 900)
        native = ChronosBoltModelForForecasting.from_pretrained(
            config["models"]["native_bolt"]["local_weights"], local_files_only=True,
            torch_dtype=torch.bfloat16).to(gpu.device).eval()
        data = Data(dataset_settings(spec["datasets"]["metr_la"], config, "metr_la"), gpu.device)
        seed(42)
        graph = np.ones((207, 207), dtype=np.float32) / 207
        model = NativeBolt(native, graph, data.horizon, (1, 2), **graph_options).to(gpu.device).train()
        batch = data.batch("train", np.arange(2))
        context = batch["x"].transpose(1, 2).reshape(414, 12).masked_fill(
            ~batch["x_mask"].transpose(1, 2).reshape(414, 12), torch.nan)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            official = native(context=context).quantile_preds[:, 4, :12].reshape(2, 207, 12).transpose(1, 2)
            actual = model(batch["x"], batch["x_mask"], batch.get("graph_features"))
        torch.testing.assert_close(actual, official, rtol=2e-3, atol=2e-3)
        for layer in model.graph.values():
            layer.gate.data.fill_(.05)
            if enabled:
                layer.context_projection.weight.data.normal_(0, .01)
        cache = PrefixCache.build(Path(temporary) / "cache", {"prefix_layers": 1}, model,
                                  [batch], 2, lambda: gpu.check(budget))
        predictions, gradients = [], []
        for cached in (False, True):
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model.from_prefix(cache.take(np.arange(2), gpu.device), 1,
                    graph_features=batch.get("graph_features")) if cached else model(
                    batch["x"], batch["x_mask"], batch.get("graph_features"))
                loss = prediction.square().mean()
            loss.backward()
            predictions.append(prediction.detach())
            gradients.append({name: p.grad.clone() for name, p in model.named_parameters() if p.requires_grad})
        torch.testing.assert_close(predictions[0], predictions[1], rtol=2e-3, atol=2e-3)
        for name in gradients[0]:
            torch.testing.assert_close(gradients[0][name], gradients[1][name], rtol=2e-3, atol=2e-3)
            assert torch.isfinite(gradients[0][name]).all()
        assert all(p.grad is None for p in native.parameters())
        cache.upgrade(model, 2, lambda: gpu.check(budget))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            upgraded = model.from_prefix(cache.take(np.arange(2), gpu.device), 2,
                                         graph_features=batch.get("graph_features"))
        torch.testing.assert_close(upgraded, predictions[1], rtol=2e-3, atol=2e-3)
        cache.close(delete=True)
        seed(42)
        hsd = HSDGNNAdapter("external/hsdgnn", 12, 12, 207, 288).to(gpu.device)
        calendar = tuple(v[:1] for v in batch["history_calendar"]) if enabled else None
        output = hsd(batch["x"][:1], batch["x_mask"][:1], batch["origins"][:1], calendar=calendar)
        output.square().mean().backward()
        assert torch.isfinite(output).all()
        assert all(torch.isfinite(p.grad).all() for p in hsd.parameters() if p.grad is not None)
        counts = {"native_bolt": parameter_counts(native), "gofa_bolt": parameter_counts(model),
                  "hsdgnn": parameter_counts(hsd)}
        assert abs(counts["hsdgnn"]["total"] / counts["gofa_bolt"]["total"] - 1) <= .1
        # Check the ACTUAL training loop and optimizer/RNG/sampler restoration without testing accuracy.
        synthetic_path = Path(temporary) / "synthetic.npz"
        synthetic_values = np.random.default_rng(9).normal(size=(320, 3)).astype(np.float32)
        np.savez(synthetic_path, values=synthetic_values, mask=np.ones_like(synthetic_values, dtype=bool))
        synthetic = Data({"data": str(synthetic_path), "context": 12, "horizons": [3],
                          "graph_context": {"enabled": enabled}}, gpu.device)
        seed(42)
        resumed = NativeBolt(native, np.eye(3), 3, (1, 2), **graph_options).to(gpu.device)
        initial = trained_state(resumed)
        fit(resumed, synthetic, Path(temporary) / "resumed", gpu, budget, .001, max_updates=2)
        state = fit(resumed, synthetic, Path(temporary) / "resumed", gpu, budget, .001, max_updates=3)
        seed(42)
        continuous = NativeBolt(native, np.eye(3), 3, (1, 2), **graph_options).to(gpu.device)
        assert all(torch.equal(initial[name], value) for name, value in trained_state(continuous).items())
        reference_state = fit(continuous, synthetic, Path(temporary) / "continuous", gpu, budget, .001, max_updates=3)
        assert state["updates"] == reference_state["updates"] == 3
        for name, value in trained_state(continuous).items():
            torch.testing.assert_close(trained_state(resumed)[name], value, rtol=1e-6, atol=1e-7)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        report = {"passed": True, "gpu_seconds": prior_seconds + elapsed,
                  "last_check_gpu_seconds": elapsed, "parameters": counts,
                  "code_sha256": {name: file_sha256(name) for name in set(CODE_FILES + [args.config])},
                  "checks": ["official zero-gate median parity", "disk BF16 cache prediction/gradient parity",
                             "E1-to-E2 cache upgrade parity", "frozen native parameters have no gradients",
                             "full-width official HSDGNN real-shape backward finite",
                             "actual fused optimizer 2+1 resumed updates equal 3 uninterrupted updates"],
                  "training_data_sha256": data.series.fingerprint,
                  "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()}
        atomic_json(args.output, report)
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
