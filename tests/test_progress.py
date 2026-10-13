import json

import pytest

from gofa_f.progress import inspect


def write_run(root, name, result=None):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps({"dataset": "tiny", "model_name": name}))
    if result:
        (directory / "result.json").write_text(json.dumps(result))
    return directory


def test_progress_reports_partial_history_without_test_metrics(tmp_path):
    complete = write_run(tmp_path, "complete", {"status": "complete", "best_epoch": 2,
                                               "test": {"secret_score": 123456.789}})
    active = write_run(tmp_path, "incomplete")
    (active / "history.jsonl").write_text(json.dumps({"epoch": 0, "steps": 10,
        "validation": {"standardized": {"mae": 1.25}}}) + '\n{"epoch":')
    (complete / "environment.json").write_text(json.dumps({"pid": 123, "git_revision": "abc"}))
    result = inspect(tmp_path, expected=42)
    assert result["counts"] == {"complete": 1, "incomplete": 1}
    assert result["not_started"] == 40
    assert result["runs"][1]["epochs"] == 1
    assert result["runs"][1]["partial_history_line"] is True
    assert "secret_score" not in json.dumps(result) and "123456.789" not in json.dumps(result)


def test_progress_keeps_validation_smoke_and_unreadable_separate(tmp_path):
    write_run(tmp_path, "validation", {"status": "validated", "best_epoch": 0})
    write_run(tmp_path, "smoke", {"status": "smoke", "best_epoch": 0})
    bad = write_run(tmp_path, "bad")
    (bad / "config.json").write_text("{")
    assert inspect(tmp_path)["counts"] == {"unreadable": 1, "smoke": 1, "validated": 1}
    with pytest.raises(ValueError, match="Expected count"):
        inspect(tmp_path, expected=1)


def test_progress_tolerates_result_being_written(tmp_path):
    directory = write_run(tmp_path, "active")
    (directory / "result.json").write_text("{")
    result = inspect(tmp_path)
    assert result["counts"] == {"incomplete": 1}
    assert result["runs"][0]["partial_result"] is True


def test_progress_missing_root_is_not_reported_as_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        inspect(tmp_path / "absent")
