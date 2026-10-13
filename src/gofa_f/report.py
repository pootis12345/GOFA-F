"""Aggregate completed runs; never promote smoke runs into experimental claims."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--output", default="outputs/report.md")
    args = parser.parse_args()
    grouped = defaultdict(list)
    seen = set()
    for path in Path(args.root).rglob("result.json"):
        result = json.loads(path.read_text())
        if result["status"] != "complete":
            continue
        config = result["config"]
        key = (config["dataset"], config["model_name"], config["horizon"])
        identity = key + (config["seed"],)
        if identity in seen:
            raise ValueError(f"Duplicate seed result: {identity}")
        seen.add(identity)
        grouped[key].append(result)
    rows = ["# GOFA-F measured results", "", "Only completed runs are included. ± denotes sample standard deviation.",
            "Raw metrics are dataset-specific; standardized metrics use train-only normalization.", "",
            "|Dataset|Model|Horizon|Seeds|Raw MAE|Raw RMSE|Standardized MSE|Total params|Trainable params|",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for (dataset, model, horizon), results in sorted(grouped.items()):
        hashes = {r["environment"]["data_sha256"] for r in results}
        if len(hashes) != 1:
            raise ValueError("Cannot aggregate results using different prepared data")
        def summary(scale, metric):
            values = [r["test"][scale][metric] for r in results]
            deviation = f"{np.std(values, ddof=1):.4f}" if len(values) > 1 else "n/a"
            return f"{np.mean(values):.4f} ± {deviation}"
        counts = results[0]["environment"]["parameters"]
        rows.append(f"|{dataset}|{model}|{horizon}|{len(results)}|{summary('raw', 'mae')}|"
                    f"{summary('raw', 'rmse')}|{summary('standardized', 'mse')}|{counts['total']}|{counts['trainable']}|")
    if not grouped:
        rows.extend(["", "No completed benchmark runs found. No performance claim is available."])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
