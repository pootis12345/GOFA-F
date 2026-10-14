"""Read native stage status without loading models, data, or test scores."""
import argparse
import json
from pathlib import Path


def inspect(directory):
    root = Path(directory)
    status = json.loads((root / "status.json").read_text())
    budget = json.loads((root / "budget.json").read_text())
    progress = []
    for path in sorted(root.glob("*/*/*/progress.json")):
        record = json.loads(path.read_text())
        progress.append({"job": str(path.parent.relative_to(root)),
                         **{key: record[key] for key in ("epoch", "offset", "updates", "microbatch")}})
    return {"status": status, "budget_spent_seconds": budget["spent_seconds"],
            "budget_remaining_seconds": max(0, 28800-budget["spent_seconds"]),
            "completed_results": len(list(root.glob("*/*/result.json"))),
            "expected_results": 24, "training": progress,
            "note": "PIDs are advisory; verify processes before starting or resuming."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    args = parser.parse_args()
    print(json.dumps(inspect(args.root), indent=2))


if __name__ == "__main__":
    main()
