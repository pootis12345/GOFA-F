"""Prepare separate clock sidecars; never rewrite prepared data or running inputs."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gofa_f.data import file_sha256


SOURCES = {"metr_la": ("metr_la.h5", "data"), "pems_bay": ("pems_bay.h5", "data"),
           "aqi": ("full437.h5", "pm25"), "aqi_36": ("small36.h5", "pm25")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/features/context-v1")
    args = parser.parse_args()
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("metr_la", "pems_bay", "aqi", "aqi_36", "electricity", "solar"):
        prepared = Path("data/processed") / f"{name}.npz"
        info = json.loads(prepared.with_suffix(".metadata.json").read_text())
        fingerprint = file_sha256(prepared)
        if fingerprint != info["prepared_sha256"]:
            raise ValueError("Prepared data changed")
        with np.load(prepared, allow_pickle=False) as series:
            values, mask = series["values"], series["mask"]
        calendar = np.zeros((len(values), 2), dtype=np.float32)
        known = np.zeros_like(calendar, dtype=bool)
        source_hash, timezone = None, None
        if name in SOURCES:
            filename, key = SOURCES[name]
            source = Path("data/raw/tsl/extracted") / filename
            source_hash = file_sha256(source)
            if source_hash != info["hdf_sha256"]:
                raise ValueError("Original HDF version differs from prepared data")
            frame = pd.read_hdf(source, key=key)
            raw = frame.to_numpy(dtype=np.float32)
            if not np.array_equal(np.where(mask, raw, 0), values):
                raise ValueError("Timestamp rows do not match prepared values")
            index = frame.index
            if not isinstance(index, pd.DatetimeIndex) or index.hasnans or index.has_duplicates:
                raise ValueError("Invalid original timestamps")
            calendar[:, 0] = (index.hour * 3600 + index.minute * 60 + index.second) / 86400
            calendar[:, 1] = index.dayofweek
            known[:] = True
            timezone = str(index.tz) if index.tz else "source naive clock; no timezone inferred"
        path = directory / f"{name}.npz"
        if path.exists():
            old = json.loads(path.with_suffix(".metadata.json").read_text())
            if old["prepared_sha256"] != fingerprint or old["sidecar_sha256"] != file_sha256(path):
                raise ValueError("Existing clock sidecar differs; do not overwrite")
            print("Verified", path)
            continue
        np.savez(path, calendar=calendar, known=known)
        metadata = {"version": 1, "dataset": name, "prepared_sha256": fingerprint,
                    "sidecar_sha256": file_sha256(path), "original_hdf_sha256": source_hash,
                    "clock": timezone, "known_calendar": bool(known.any()),
                    "note": "Energy source matrices have no timestamps; unknown clocks stay zero with unavailable flags."}
        path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2))
        print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
