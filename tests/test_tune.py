import copy
import json
from pathlib import Path

import pytest
import yaml

from gofa_f.tune import select, trials


def settings():
    spec = yaml.safe_load(Path("configs/benchmark.yaml").read_text(encoding="utf-8"))
    search = yaml.safe_load(Path("configs/tuning.yaml").read_text(encoding="utf-8"))
    return spec, search


def test_registered_search_has_equal_trial_budgets_and_no_test_evaluation():
    spec, search = settings()
    jobs = trials(spec, search)
    assert len(jobs) == 108  # Six datasets, six trained model settings, three learning rates.
    assert len({name for name, _ in jobs}) == 108
    assert {config["evaluation_split"] for _, config in jobs} == {"validation"}
    for _, config in jobs:
        for key in ("epochs", "patience", "batch_size", "accumulation", "train_ratio", "val_ratio"):
            assert config[key] == spec["defaults"][key]
        assert config["model"]["family"] != "persistence"


def test_search_rejects_truncated_and_unregistered_grids():
    spec, search = settings()
    spec["defaults"]["max_steps"] = 2
    with pytest.raises(ValueError, match="truncated"):
        trials(spec, search)
    spec["defaults"].pop("max_steps")
    for field, invalid in (("seeds", [42, 42]), ("seeds", [42, 999]), ("horizons", [12, 999]),
                           ("learning_rates", [1e-4, 1e-4]), ("learning_rates", [float("nan")])):
        changed = copy.deepcopy(search)
        changed[field] = invalid
        with pytest.raises(ValueError):
            trials(spec, changed)


def saved_trials(tmp_path):
    spec, search = settings()
    # Exercise future multi-seed completeness independently of the current one-seed plan.
    spec["seeds"] = [42, 43]
    search["seeds"] = [42, 43]
    jobs = trials(spec, search, datasets=["aqi_36"], models=["t5"])
    for name, config in jobs:
        directory = tmp_path / name
        directory.mkdir(parents=True)
        result = {"status": "validated", "config": config,
                  "environment": {"data_sha256": "same-data", "git_revision": "same-code", "git_dirty": False},
                  "validation": {"standardized": {"mae": 1 if config["lr"] == 1e-4 else 2}}}
        (directory / "result.json").write_text(json.dumps(result))
    return jobs


def test_current_benchmark_is_restricted_to_seed42():
    from gofa_f.benchmark import expand

    spec, _ = settings()
    jobs = list(expand(spec))
    assert spec["seeds"] == [42]
    assert len(jobs) == 42
    assert {config["seed"] for _, config in jobs} == {42}


def test_dataset_default_horizons_keep_all_datasets_in_validation_search():
    spec, search = settings()
    jobs = trials(spec, search)
    assert {config["dataset"] for _, config in jobs} == set(spec["datasets"])
    for _, config in jobs:
        assert config["horizon"] in spec["datasets"][config["dataset"]]["horizons"]
    explicit = dict(search, horizons=[12])
    assert {config["dataset"] for _, config in trials(spec, explicit)} == {"metr_la", "aqi_36"}
    with pytest.raises(ValueError, match="horizons"):
        trials(spec, dict(search, horizons="all"))


def test_selection_requires_every_candidate_and_seed(tmp_path):
    jobs = saved_trials(tmp_path)
    result = select(tmp_path, jobs)
    assert result["groups"][0]["selected_lr"] == 1e-4
    assert all(len(c["runs"]) == 2 for c in result["groups"][0]["candidates"])
    (tmp_path / jobs[-1][0] / "result.json").unlink()
    with pytest.raises(FileNotFoundError):
        select(tmp_path, jobs)


@pytest.mark.parametrize("case", ["smoke", "test", "data", "revision", "dirty", "config", "nonfinite"])
def test_selection_rejects_invalid_comparisons(tmp_path, case):
    jobs = saved_trials(tmp_path)
    path = tmp_path / jobs[-1][0] / "result.json"
    result = json.loads(path.read_text())
    if case == "smoke":
        result["status"] = "smoke"
    elif case == "test":
        result["test"] = {"raw": {"mae": 0.1}}
    elif case == "data":
        result["environment"]["data_sha256"] = "other-data"
    elif case == "revision":
        result["environment"]["git_revision"] = "other-code"
    elif case == "dirty":
        result["environment"]["git_dirty"] = True
    elif case == "config":
        result["config"]["lr"] = 999
    else:
        result["validation"]["standardized"]["mae"] = float("nan")
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        select(tmp_path, jobs)
