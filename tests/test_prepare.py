import sys

import numpy as np
import pandas as pd
import pytest

from gofa_f.prepare import main


def test_timestamp_gaps_become_segments(tmp_path, monkeypatch):
    source, output = tmp_path / "raw.csv", tmp_path / "prepared.npz"
    pd.DataFrame({"time": ["2024-01-01 00:00", "2024-01-01 00:05", "2024-01-01 00:20"],
                  "sensor": [0., 1., 2.], "label": [1, 0, 0]}).to_csv(source, index=False)
    monkeypatch.setattr(sys, "argv", ["prepare", "--input", str(source), "--output", str(output),
        "--columns", "sensor", "--time-column", "time", "--frequency", "5min"])
    main()
    with np.load(output) as data:
        assert data["values"].shape == (3, 1)
        assert data["mask"].all()  # True zero is observed by default.
        np.testing.assert_array_equal(data["segments"], [0, 0, 1])


def test_unordered_timestamps_rejected(tmp_path, monkeypatch):
    source = tmp_path / "raw.csv"
    pd.DataFrame({"t": ["2024-01-02", "2024-01-01"], "x": [1, 2]}).to_csv(source, index=False)
    monkeypatch.setattr(sys, "argv", ["prepare", "--input", str(source), "--output", str(tmp_path / "x.npz"),
                                     "--columns", "x", "--time-column", "t", "--frequency", "1d"])
    with pytest.raises(ValueError, match="ordered"):
        main()
