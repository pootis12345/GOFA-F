"""Audit actual six-dataset stage wiring on C without a CUDA context or launch window."""
import argparse
import gc
import hashlib
import os
from pathlib import Path

import torch
import yaml
from chronos.chronos_bolt import ChronosBoltModelForForecasting

from gofa_f.data import file_sha256
from gofa_f.prefix_cache import atomic_json
from gofa_f.routed_stage import DATASETS, config_file, manifest, write_report
from gofa_f.routed_task import compatible_cache, data_for, new_model
from gofa_f.adaptation_stage import prefix_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("CPU audit requires CUDA_VISIBLE_DEVICES=''")
    torch.set_num_threads(2)
    config = config_file("configs/stage5_routed.yaml")
    # This is not a persisted training window and does not authorize a GPU task.
    binding = manifest(config, {"cpu_audit_only": True})
    spec = yaml.safe_load(Path("configs/benchmark.yaml").read_text())
    native = ChronosBoltModelForForecasting.from_pretrained(config["native_weights"], local_files_only=True,
                                                          torch_dtype=torch.float32).eval()
    records = []
    for dataset in DATASETS:
        data, settings = data_for(spec, dataset, "cpu")
        model = new_model(config, native, data, settings)
        batch = data.batch("train", [0])
        with torch.no_grad():
            output = model(batch["x"], batch["x_mask"], batch["graph_features"])
        assert torch.isfinite(output).all() and output.shape == batch["y"].shape
        assert set(data.windows) == {"train", "val"} and batch["graph_features"].shape[-1] == 5
        cache = compatible_cache(Path(config["stage5"]["reference_output"])/"cache"/dataset,
            prefix_key(data, config["native_revision"]), len(data.windows["train"]), data.values.shape[1],
            native.config.d_model)
        records.append({"dataset": dataset, "history": data.context, "horizon": data.horizon,
                        "nodes": data.values.shape[1], "train_windows": len(data.windows["train"]),
                        "forward_finite": True, "test_targets_excluded": True,
                        "old_readonly_cache_metadata_compatible": cache is not None})
        if cache:
            cache.close()
        del model, data, batch, output
        gc.collect()
    assert not torch.cuda.is_initialized()
    directory = Path(args.output).parent
    directory.mkdir(parents=True, exist_ok=True)
    write_report(directory, config)
    atomic_json(args.output, {"passed": True, "cuda_initialized": False, "records": records,
        "bound_input_hashes": binding["inputs"], "historical_checkpoint_sha256": binding["historical_checkpoint_sha256"],
        "versions": binding["versions"], "code_lf_sha256": {str(p): hashlib.sha256(
            p.read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in (
                Path("src/gofa_f/routed_stage.py"), Path("src/gofa_f/routed_task.py"),
                Path("src/gofa_f/prefix_cache.py"), Path("configs/stage5_routed.yaml"), Path(__file__))},
        "reference_summary_sha256": file_sha256(config["stage5"]["reference_summary"])})


if __name__ == "__main__":
    main()
