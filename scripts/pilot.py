"""Real-data short training checks. Outputs are always SMOKE and excluded from scores."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from gofa_f.benchmark import expand, match_baseline
from gofa_f.train import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=["aqi_36"])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", default="outputs/real-data-smoke-v1")
    args = parser.parse_args()
    spec = yaml.safe_load(Path("configs/benchmark.yaml").read_text())
    spec["seeds"] = [42]
    for dataset in spec["datasets"].values():
        dataset["horizons"] = [12]
    failures = []
    for name, config in expand(spec):
        if args.datasets and config["dataset"] not in args.datasets:
            continue
        config.update(epochs=1, max_steps=args.steps, max_eval_windows=16, accumulation=1)
        output = Path(args.output) / name
        if (output / "result.json").exists():
            continue
        try:
            if "match_reference" in config:
                with np.load(config["data"], allow_pickle=False) as data:
                    nodes = data["values"].shape[1]
                match_baseline(config, spec["models"][config["match_reference"]], spec["parameter_matching"], nodes)
            result = run(config, output)
            assert result["status"] == "smoke"
            print(json.dumps({"run": name, "status": "passed", "parameters": result["environment"]["parameters"]}), flush=True)
        except Exception as error:
            # Retain each failed run and check independent model/dataset combinations.
            failures.append({"run": name, "error": repr(error)})
            print(json.dumps(failures[-1]), flush=True)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    Path(args.output, "failures.json").write_text(json.dumps(failures, indent=2))
    if failures:
        raise SystemExit(f"{len(failures)} pilot checks failed; see failures.json")


if __name__ == "__main__":
    main()
