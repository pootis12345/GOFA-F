"""Read run progress without loading models, checkpoints, arrays or test scores."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path


def inspect_run(config_path):
    directory = config_path.parent
    summary = {"path": str(directory), "status": "incomplete", "epochs": 0}
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        for key in ("dataset", "model_name", "horizon", "seed", "evaluation_split"):
            if key in config:
                summary[key] = config[key]
        environment = directory / "environment.json"
        if environment.exists():
            env = json.loads(environment.read_text(encoding="utf-8"))
            for key in ("pid", "git_revision", "git_dirty"):
                summary[key] = env.get(key)
        history = directory / "history.jsonl"
        if history.exists():
            # A concurrent append may leave the final line incomplete. Ignore that line only.
            lines = history.read_text(encoding="utf-8").splitlines()
            latest = None
            for index, line in enumerate(lines):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    if index == len(lines) - 1:
                        summary["partial_history_line"] = True
                        break
                    raise
                summary["epochs"] += 1
                latest = record
            if latest:
                summary["last_epoch"] = latest["epoch"]
                summary["steps"] = latest["steps"]
                scale = config.get("selection_scale", "standardized")
                metric = config.get("selection_metric", "mae")
                summary["last_validation"] = {"scale": scale, "metric": metric,
                                              "value": latest["validation"][scale][metric]}
            summary["history_updated_utc"] = datetime.fromtimestamp(
                history.stat().st_mtime, timezone.utc).isoformat()
        result_path = directory / "result.json"
        if result_path.exists():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                # result.json is also written by the running process; retry on the next inspection.
                summary["partial_result"] = True
            else:
                status = result["status"]
                if status not in ("complete", "validated", "smoke"):
                    raise ValueError(f"Unknown result status: {status}")
                summary["status"] = status
                summary["best_epoch"] = result["best_epoch"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        summary.update(status="unreadable", error=str(error))
    return summary


def inspect(root, expected=None):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Run directory not found: {root}")
    runs = [inspect_run(path) for path in sorted(root.rglob("config.json"))]
    counts = dict(Counter(run["status"] for run in runs))
    snapshot = {"root": str(root), "observed_runs": len(runs), "counts": counts, "runs": runs,
                "note": "Stage status and recorded PIDs are advisory; verify processes before launching work."}
    stage_status = Path(f"{root}.status")
    if stage_status.exists():
        snapshot["stage_status"] = stage_status.read_text(encoding="utf-8").strip()
    if expected is not None:
        if expected < len(runs):
            raise ValueError("Expected count is smaller than observed runs; check the selected output root")
        snapshot.update(expected_runs=expected, not_started=expected - len(runs))
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--expected", type=int, help="Expected number of runs in this output root")
    args = parser.parse_args()
    print(json.dumps(inspect(args.root, args.expected), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
