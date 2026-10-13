"""Expand and execute the registered matrix; completed results are resumable."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from gofa_f.models import parameter_counts
from gofa_f.train import build_model, run


def match_baseline(config, reference, settings, nodes):
    # Instantiate on CPU; counts, not test/validation scores, determine width.
    reference_config = copy.deepcopy(config)
    reference_config["model"] = reference
    target_model = build_model(reference_config, np.eye(nodes, dtype=np.float32), nodes)
    target = parameter_counts(target_model)[settings["basis"]]
    del target_model
    best = None
    for width in settings["widths"]:
        for hidden_size in settings["hidden_sizes"]:
            candidate = copy.deepcopy(config)
            candidate["model"].update(width=width, hidden_size=hidden_size)
            with torch.device("meta"):
                model = build_model(candidate, np.eye(nodes, dtype=np.float32), nodes)
            count = parameter_counts(model)[settings["basis"]]
            error = abs(count/target-1)
            if best is None or error < best[0]:
                best = (error, width, hidden_size, count)
            del model
    error, width, hidden_size, count = best
    if error > settings["tolerance"]:
        raise ValueError(f"No baseline within parameter tolerance: target={target}, closest={count}")
    config["model"].update(width=width, hidden_size=hidden_size)
    config["parameter_match"] = {"basis": settings["basis"], "target": target,
                                 "tolerance": settings["tolerance"], "actual": count}


def expand(spec):
    for dataset, info in spec["datasets"].items():
        for horizon in info["horizons"]:
            for seed in spec["seeds"]:
                for name, model in spec["models"].items():
                    references = spec.get("parameter_matching", {}).get("references", []) if name == "dst_sgnn" else [None]
                    for reference in references or [None]:
                        label = f"{name}_matched_{reference}" if reference else name
                        config = copy.deepcopy(spec["defaults"])
                        config.update(dataset=dataset, domain=info["domain"], data=info["data"],
                                      context=info["context"], horizon=horizon, seed=seed,
                                      model=copy.deepcopy(model), model_name=label)
                        if reference:
                            config["match_reference"] = reference
                        yield f"{dataset}/{label}/h{horizon}/seed{seed}", config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--output", default="outputs/benchmark-v1")
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--seeds", nargs="*", type=int, help="Run only registered seeds")
    parser.add_argument("--horizons", nargs="*", type=int, help="Run only registered forecast lengths")
    parser.add_argument("--plan", action="store_true", help="Print jobs without loading data/models")
    args = parser.parse_args()
    spec = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for name, config in expand(spec):
        if args.seeds and config["seed"] not in args.seeds:
            continue
        if args.horizons and config["horizon"] not in args.horizons:
            continue
        if args.datasets and config["dataset"] not in args.datasets:
            continue
        if args.models and config["model_name"] not in args.models:
            continue
        if args.plan:
            print(name)
            continue
        output = Path(args.output) / name
        if (output / "result.json").exists():
            existing = json.loads((output / "result.json").read_text())
            if existing["status"] != "complete":
                raise ValueError(f"Smoke result cannot substitute for full run: {output}")
            recorded = dict(existing["config"])
            recorded.pop("parameter_match", None)
            if config.get("match_reference"):
                recorded["model"] = config["model"]
            if recorded != config:
                raise ValueError(f"Existing result uses a different config; select a new output directory: {output}")
            from gofa_f.data import file_sha256
            if file_sha256(config["data"]) != existing["environment"]["data_sha256"]:
                raise ValueError(f"Prepared data changed since previous run: {output}")
            print(f"Completed: {name}", flush=True)
            continue
        if output.exists():
            raise FileExistsError(f"Incomplete run: {output}; preserve it and choose a new output root")
        if "match_reference" in config:
            with np.load(config["data"], allow_pickle=False) as data:
                nodes = data["values"].shape[1]
            match_baseline(config, spec["models"][config["match_reference"]], spec["parameter_matching"], nodes)
        print(f"Starting: {name}", flush=True)
        run(config, output)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
