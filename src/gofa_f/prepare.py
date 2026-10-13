"""Explicitly prepare raw arrays; never infer timestamp/label columns as sensors."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gofa_f.data import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--key", default="data", help="NPZ key or HDF key")
    parser.add_argument("--columns", help="Comma separated sensor columns for CSV/Excel; required")
    parser.add_argument("--time-column", help="Optional timestamp column; rejects unordered/duplicate timestamps")
    parser.add_argument("--frequency", help="Expected interval, e.g. 5min or 1h; splits at timestamp gaps")
    parser.add_argument("--sheet", default="0")
    parser.add_argument("--skiprows", type=int, default=0)
    parser.add_argument("--channel", type=int, default=0, help="Select feature from raw [time,node,feature]")
    parser.add_argument("--zero-missing", action="store_true", help="Only when dataset defines zero as missing")
    parser.add_argument("--train-end", type=int)
    parser.add_argument("--val-end", type=int)
    parser.add_argument("--segments", help="Optional NPY segment ID vector; windows cannot cross boundaries")
    args = parser.parse_args()
    source = Path(args.input)
    suffix = source.suffix.lower()
    time = None
    if suffix == ".npy":
        values = np.load(source, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(source, allow_pickle=False) as data:
            values = data[args.key]
    elif suffix in (".h5", ".hdf", ".hdf5"):
        frame = pd.read_hdf(source, key=args.key)
        if isinstance(frame.index, pd.DatetimeIndex):
            time = pd.Series(frame.index)
        values = frame.to_numpy()
    elif suffix in (".csv", ".xlsx", ".xls"):
        if not args.columns:
            parser.error("CSV/Excel input requires --columns listing sensor columns only")
        if suffix == ".csv":
            frame = pd.read_csv(source, skiprows=args.skiprows)
        else:
            sheet = int(args.sheet) if args.sheet.isdigit() else args.sheet
            frame = pd.read_excel(source, sheet_name=sheet, skiprows=args.skiprows)
        frame.columns = frame.columns.str.strip()
        if args.time_column:
            time = pd.to_datetime(frame[args.time_column], errors="raise")
        values = frame[[c.strip() for c in args.columns.split(",")]].apply(pd.to_numeric, errors="raise").to_numpy()
    else:
        raise ValueError(f"Unsupported raw format: {suffix}")
    if values.ndim == 3:
        values = values[:, :, args.channel]
    if values.ndim != 2:
        raise ValueError("Input must resolve to [time,nodes]")
    values = values.astype(np.float32)
    mask = np.isfinite(values)
    if args.zero_missing:
        mask &= values != 0
    segments = np.load(args.segments, allow_pickle=False) if args.segments else np.zeros(len(values), np.int64)
    if time is not None:
        if time.isna().any() or not time.is_monotonic_increasing or time.duplicated().any():
            raise ValueError("Raw timestamps must be complete, ordered and unique")
        if not args.frequency:
            parser.error("Timestamped input requires --frequency to detect collection gaps")
        interval = pd.to_timedelta(args.frequency)
        if interval <= pd.Timedelta(0):
            parser.error("--frequency must be positive")
        gaps = time.diff().fillna(interval).ne(interval).to_numpy()
        explicit_cuts = np.r_[False, segments[1:] != segments[:-1]]
        segments = np.cumsum(gaps | explicit_cuts)
    if segments.shape != (len(values),):
        raise ValueError("Invalid segment vector shape")
    extra = {"segments": segments}
    if (args.train_end is None) != (args.val_end is None):
        parser.error("Specify both --train-end and --val-end, or neither")
    if args.train_end is not None:
        if not 0 < args.train_end < args.val_end < len(values):
            raise ValueError("Invalid split boundaries")
        extra.update(train_end=args.train_end, val_end=args.val_end)
    output = Path(args.output)
    if output.suffix != ".npz":
        parser.error("Output must end in .npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    np.savez_compressed(output, values=np.where(mask, values, 0), mask=mask, **extra)
    metadata = {"source_sha256": file_sha256(source), "prepared_sha256": file_sha256(output),
                "shape": list(values.shape), "missing_ratio": float(1-mask.mean()), "arguments": vars(args)}
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
