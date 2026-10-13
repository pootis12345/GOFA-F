import json

import pytest

from gofa_f.benchmark import execute
from gofa_f.data import file_sha256


@pytest.mark.parametrize("split,status", [("validation", "validated"), ("test", "complete")])
def test_completed_run_resume_checks_evaluation_mode_and_data(tmp_path, split, status):
    data = tmp_path / "data.npz"
    data.write_bytes(b"prepared data fingerprint fixture")
    config = {"data": str(data), "evaluation_split": split, "model": {"family": "persistence"}}
    output = tmp_path / "run"
    output.mkdir()
    path = output / "result.json"
    recorded = {"config": config, "status": status, "environment": {"data_sha256": file_sha256(data)}}
    path.write_text(json.dumps(recorded))
    assert execute("fixture", config, output, {}) == recorded

    recorded["status"] = "complete" if split == "validation" else "validated"
    path.write_text(json.dumps(recorded))
    with pytest.raises(ValueError, match="cannot substitute"):
        execute("fixture", config, output, {})
    recorded["status"] = "smoke"
    path.write_text(json.dumps(recorded))
    with pytest.raises(ValueError, match="cannot substitute"):
        execute("fixture", config, output, {})
    recorded["status"] = status
    path.write_text(json.dumps(recorded))
    data.write_bytes(b"changed data")
    with pytest.raises(ValueError, match="Prepared data changed"):
        execute("fixture", config, output, {})


def test_incomplete_output_is_preserved(tmp_path):
    output = tmp_path / "interrupted"
    output.mkdir()
    marker = output / "history.jsonl"
    marker.write_text("retained diagnostic")
    with pytest.raises(FileExistsError, match="Incomplete run"):
        execute("fixture", {}, output, {})
    assert marker.read_text() == "retained diagnostic"
