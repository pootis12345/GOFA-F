"""C-only, CUDA-disabled real-weight checks; reads training histories only."""
import argparse
import gc
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from chronos.chronos_bolt import ChronosBoltModelForForecasting

from gofa_f.data import file_sha256
from gofa_f.graphs import build_graph
from gofa_f.models import parameter_counts
from gofa_f.native import NativeBolt
from gofa_f.native_stage import Data
from gofa_f.prefix_cache import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/titan_inspired.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("Run on C with CUDA_VISIBLE_DEVICES='' and no CUDA context")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    config = yaml.safe_load(Path(args.config).read_text())
    for name, expected in config["native_file_sha256"].items():
        if file_sha256(Path(config["native_weights"]) / name) != expected:
            raise ValueError("Pretrained weights differ from the pinned revision")
    native = ChronosBoltModelForForecasting.from_pretrained(
        config["native_weights"], local_files_only=True, torch_dtype=torch.float32).eval()
    frozen = {n: p.detach().clone() for n, p in native.named_parameters()}
    benchmark = yaml.safe_load(Path("configs/benchmark.yaml").read_text())
    records = []
    for dataset in ("metr_la", "pems_bay", "aqi", "aqi_36", "electricity", "solar"):
        settings = dict(benchmark["datasets"][dataset], validation_only=True,
                        graph_context={"enabled": True, "feature_mode": "values_only"})
        # Match the pinned paired history/horizon tasks, not the old benchmark grid.
        settings["context"], horizon = {"metr_la": (12, 12), "pems_bay": (12, 3), "aqi": (24, 6),
                                        "aqi_36": (12, 12), "electricity": (8, 8), "solar": (24, 3)}[dataset]
        settings["horizons"] = [horizon]
        data = Data(settings, "cpu")
        graph, _ = build_graph(data.series, settings)
        batch = data.batch("train", np.arange(1))
        torch.manual_seed(42)
        options = dict(config["architecture"])
        options["graph_exchange"] = "static"
        baseline = NativeBolt(native, graph, horizon, **options)
        torch.manual_seed(42)
        model = NativeBolt(native, graph, horizon, **config["architecture"])
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
        with torch.no_grad():
            before = baseline(batch["x"], batch["x_mask"], batch["graph_features"])
            initial = model(batch["x"], batch["x_mask"], batch["graph_features"])
        torch.testing.assert_close(before, initial, rtol=0, atol=0)
        for module in model.graph.values():
            module.gate.data.fill_(.1)
            module.key.weight.data.normal_(0, .01)
            module.router.weight.data.normal_(0, .01)
        prefix = model.prefix(batch["x"], batch["x_mask"], 2)
        gradients, outputs = [], []
        for cached in (False, True):
            model.zero_grad(set_to_none=True)
            output = model.from_prefix(prefix, 2, graph_features=batch["graph_features"]) if cached else model(
                batch["x"], batch["x_mask"], batch["graph_features"])
            output.square().mean().backward()
            assert torch.isfinite(output).all()
            outputs.append(output.detach())
            gradients.append({n: p.grad.detach().clone() for n, p in model.named_parameters() if p.requires_grad})
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        for name in gradients[0]:
            assert torch.isfinite(gradients[0][name]).all()
            torch.testing.assert_close(gradients[0][name], gradients[1][name], rtol=0, atol=0)
        assert all(p.grad is None and torch.equal(p, frozen[n]) for n, p in native.named_parameters())
        counts = parameter_counts(model)
        counts["point_loss_active_trainable_upper_bound"] = counts["trainable"] - (
            model.output_adapter.b.numel() - horizon * model.output_adapter.b.shape[1])
        records.append({"dataset": dataset, "context": data.context, "horizon": horizon,
                        "nodes": data.values.shape[1], "initial_max_difference": (initial-before).abs().max().item(),
                        "cache_predictions_and_gradients_equal": True, "native_unchanged_and_frozen": True,
                        "parameters": counts})
        print(dataset + ": passed", flush=True)
        del model, baseline, data, batch, prefix, outputs, gradients
        gc.collect()
    assert not torch.cuda.is_initialized()
    atomic_json(args.output, {"protocol": config["protocol"], "device": "cpu", "cuda_initialized": False,
                             "torch_version": torch.__version__, "threads": 2, "records": records,
                             "code_sha256": {str(p): file_sha256(p) for p in (
                                 Path("src/gofa_f/routed_graph.py"), Path("src/gofa_f/native.py"),
                                 Path("src/gofa_f/native_stage.py"), Path(args.config), Path(__file__))}})


if __name__ == "__main__":
    main()
