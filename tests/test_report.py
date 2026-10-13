import json
import sys

from gofa_f.report import main


def test_report_does_not_promote_validation_or_smoke_results(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    for status in ("validated", "smoke"):
        directory = root / status
        directory.mkdir(parents=True)
        (directory / "result.json").write_text(json.dumps({"status": status}))
    output = tmp_path / "report.md"
    monkeypatch.setattr(sys, "argv", ["gofa-report", str(root), "--output", str(output)])
    main()
    assert "No completed benchmark runs found" in output.read_text()
