"""Equal-trial-budget learning-rate search using validation targets only."""
import argparse
from collections import defaultdict
import copy
import json
import math
from pathlib import Path

import yaml

from gofa_f.benchmark import execute, expand


def trials(spec, search, datasets=None, models=None):
    for key in ("seeds", "horizons"):
        if not search[key] or len(set(search[key])) != len(search[key]):
            raise ValueError(f"Search {key} must be nonempty and distinct")
    rates = search["learning_rates"]
    if not rates or len(set(rates)) != len(rates) or any(not math.isfinite(r) or r <= 0 for r in rates):
        raise ValueError("Learning rates must be distinct, positive and finite")
    jobs = []
    for name, config in expand(spec):
        if config["seed"] not in search["seeds"] or config["horizon"] not in search["horizons"]:
            continue
        if config["model"].get("family") == "persistence":
            continue
        if datasets and config["dataset"] not in datasets:
            continue
        if models and config["model_name"] not in models:
            continue
        if config.get("max_steps") or config.get("max_eval_windows"):
            raise ValueError("A truncated run cannot be used for full validation search")
        for index, rate in enumerate(rates):
            candidate = copy.deepcopy(config)
            candidate.update(lr=rate, evaluation_split="validation",
                             selection_scale=search["selection_scale"], selection_metric=search["selection_metric"])
            jobs.append((f"lr{index}/{name}", candidate))
    if not jobs:
        raise ValueError("No registered jobs matched the search filters")
    if not set(search["seeds"]).issubset(spec["seeds"]):
        raise ValueError("Search seeds must be registered in the benchmark")
    for dataset in {config["dataset"] for _, config in jobs}:
        if not set(search["horizons"]).issubset(spec["datasets"][dataset]["horizons"]):
            raise ValueError(f"Search horizons must be registered for {dataset}")
    return jobs


def select(root, jobs):
    """Require the entire registered grid, including every seed, before selecting."""
    groups = defaultdict(lambda: defaultdict(list))
    for name, expected in jobs:
        path = Path(root) / name / "result.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["status"] != "validated" or "test" in result:
            raise ValueError(f"Selection requires full validation-only results: {path}")
        if result["environment"].get("git_dirty"):
            raise ValueError(f"Selection requires a clean recorded code snapshot: {path}")
        recorded = dict(result["config"])
        recorded.pop("parameter_match", None)
        if expected.get("match_reference"):
            recorded["model"] = expected["model"]
        if recorded != expected:
            raise ValueError(f"Search result configuration mismatch: {path}")
        scale, metric = expected["selection_scale"], expected["selection_metric"]
        score = result["validation"][scale][metric]
        if not math.isfinite(score):
            raise ValueError(f"Nonfinite validation score: {path}")
        key = (expected["dataset"], expected["model_name"], expected["horizon"])
        groups[key][expected["lr"]].append({"seed": expected["seed"], "score": score,
            "path": str(path), "data_sha256": result["environment"]["data_sha256"],
            "graph_sha256": result["environment"].get("graph", {}).get("graph_sha256"),
            "git_revision": result["environment"]["git_revision"]})
    selections = []
    for (dataset, model, horizon), rates in sorted(groups.items()):
        entries = [entry for results in rates.values() for entry in results]
        if len({entry["data_sha256"] for entry in entries}) != 1:
            raise ValueError("Cannot select across different prepared data versions")
        if len({entry["git_revision"] for entry in entries}) != 1:
            raise ValueError("Cannot select across different code revisions")
        if len({entry["graph_sha256"] for entry in entries}) != 1:
            raise ValueError("Cannot select across different spatial graph versions")
        candidates = [{"lr": rate, "mean_validation": sum(e["score"] for e in results) / len(results),
                       "runs": results} for rate, results in sorted(rates.items())]
        # Deterministic tie break by smaller learning rate, never by test metrics.
        best = min(candidates, key=lambda c: (c["mean_validation"], c["lr"]))
        selections.append({"dataset": dataset, "model_name": model, "horizon": horizon,
                           "selected_lr": best["lr"], "candidates": candidates})
    return {"selection": "mean validation score; ties use smaller learning rate", "groups": selections}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--search", default="configs/tuning.yaml")
    parser.add_argument("--output", default="outputs/lr-search-v1")
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--select-only", action="store_true", help="Select from the complete saved grid; no training")
    args = parser.parse_args()
    spec = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    search = yaml.safe_load(Path(args.search).read_text(encoding="utf-8"))
    jobs = trials(spec, search, args.datasets, args.models)
    if args.plan:
        for name, _ in jobs:
            print(name)
        return
    root = Path(args.output)
    manifest = {"search": search, "jobs": [{"name": n, "config": c} for n, c in jobs]}
    manifest_path = root / "search_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Search grid changed; preserve this search and choose a new output root")
    elif args.select_only:
        raise FileNotFoundError(f"Search manifest not found: {manifest_path}")
    else:
        root.mkdir(parents=True, exist_ok=False)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not args.select_only:
        for name, config in jobs:
            execute(name, config, root / name, spec)
    selection = select(root, jobs)
    (root / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(root / "selection.json")


if __name__ == "__main__":
    main()
