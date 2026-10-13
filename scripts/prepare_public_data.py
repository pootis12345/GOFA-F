"""Prepare the four public datasets from the pinned TSL source archives.

Run download_public_data.py first. Original HDF values are preserved with NaN masks.
Synthetic imputation evaluation masks are NOT forecasting observation masks.
"""
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from gofa_f.data import file_sha256

SOURCES = {
    "metr_la": ("traffic/metr_la.zip", "metr_la.h5", "data", "5min", True),
    "pems_bay": ("traffic/pems_bay.zip", "pems_bay.h5", "data", "5min", True),
    "aqi": ("air/air_quality.zip", "full437.h5", "pm25", "1h", False),
    "aqi_36": ("air/air_quality.zip", "small36.h5", "pm25", "1h", False),
}


def main():
    root = Path("data/raw/tsl")
    manifest = json.loads((root / "download_manifest.json").read_text())
    output = Path("data/processed")
    output.mkdir(parents=True, exist_ok=True)
    for name, (archive, member, key, frequency, zero_missing) in SOURCES.items():
        path = root / archive
        if file_sha256(path) != manifest["files"][archive]["sha256"]:
            raise ValueError(f"Downloaded archive changed: {path}")
        extracted = root / "extracted" / member
        extracted.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as stream:
            # Fixed expected member name prevents archive path traversal.
            raw = stream.read(member)
        if extracted.exists() and extracted.read_bytes() != raw:
            raise ValueError(f"Extracted source differs from archive: {extracted}")
        if not extracted.exists():
            extracted.write_bytes(raw)
        frame = pd.read_hdf(extracted, key=key)
        time = frame.index
        if not isinstance(time, pd.DatetimeIndex) or time.hasnans or not time.is_monotonic_increasing or time.has_duplicates:
            raise ValueError(f"Invalid timestamps: {name}")
        interval = pd.to_timedelta(frequency)
        cuts = time.to_series().diff().fillna(interval).ne(interval).to_numpy()
        values = frame.to_numpy(dtype=np.float32)
        mask = np.isfinite(values)
        if zero_missing:
            mask &= values != 0
        target = output / f"{name}.npz"
        if target.exists():
            previous = json.loads(target.with_suffix(".metadata.json").read_text())
            if (previous["hdf_sha256"] != file_sha256(extracted)
                    or previous["prepared_sha256"] != file_sha256(target)):
                raise ValueError(f"Existing data differs from source: {target}")
            print(f"Verified existing: {target}", flush=True)
            continue
        np.savez_compressed(target, values=np.where(mask, values, 0), mask=mask, segments=np.cumsum(cuts))
        metadata = {"dataset": name, "shape": list(values.shape), "frequency": frequency,
                    "start": str(time[0]), "end": str(time[-1]), "zero_is_missing": zero_missing,
                    "missing_ratio": float(1-mask.mean()), "source": manifest,
                    "hdf_sha256": file_sha256(extracted), "prepared_sha256": file_sha256(target),
                    "segment_count": int(cuts.sum()+1), "hdf_key": key,
                    "note": "Air datasets contain PM2.5 concentrations, not aggregate AQI index values."}
        target.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
