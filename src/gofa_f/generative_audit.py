"""Target DATA audit only. No language models, forecasts or target fitting."""
import json
import io
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from gofa_f.generative_data import DRIVE, Query, Series, save_series, sha
from gofa_f.prefix_cache import atomic_json

TARGETS = {name: {"length": info[1], "horizon": info[2], "variables": info[3]} for name, info in DRIVE.items()}
TARGETS.update(nasdaq100={"length": 10, "horizon": 1, "variables": 105, "target_variables": [104]},
               pems08={"length": 12, "horizon": 12, "variables": 170},
               noaa2025={"length": 48, "horizon": 24, "variables": 16})


def align_nasdaq(stocks, index):
    """Align actual timestamps; never shift an already-aligned provider matrix twice."""
    index = index.sort_index()
    if index.index.has_duplicates or not index.index.is_monotonic_increasing:
        raise ValueError("NASDAQ index timestamps must be unique")
    result = pd.DataFrame(index=index.index)
    days = index.index.normalize().unique().sort_values()
    for ticker, frame in stocks.items():
        if frame.index.has_duplicates:
            raise ValueError("Duplicate original stock timestamp")
        value = frame.reindex(index.index).astype(float)
        if ticker == "LLTC" and len(days) > 157:
            value = value.mask((value.index.normalize() >= days[157]) & value.eq(0))
        result[ticker] = value
    result["NDX"] = index.astype(float)
    segments = pd.factorize(index.index.normalize())[0]
    # Timestamp matching drops the unmatched first stock point without reading a future minute.
    return result, segments


def noaa_hourly(frame):
    """Official field10 T_HR_AVG, UTC end-of-hour; field9 T_CALC is different."""
    if frame.shape[1] < 10:
        raise ValueError("Not an official hourly02 record")
    times = pd.to_datetime(frame.iloc[:, 1].astype(str)+frame.iloc[:, 2].astype(str).str.zfill(4),
                           format="%Y%m%d%H%M", utc=True)
    values = pd.to_numeric(frame.iloc[:, 9], errors="raise").astype(float)
    values = values.mask(values.eq(-9999.0))
    return pd.Series(values.to_numpy(), index=times), {"wban": str(frame.iloc[0, 0]),
        "longitude": float(frame.iloc[0, 6]), "latitude": float(frame.iloc[0, 7]),
        "field": "T_HR_AVG", "field_number": 10, "unit": "Celsius", "timestamp": "UTC hour ending"}


def geographic_stations(stations, count=16):
    candidates = sorted([s for s in stations if 24 <= s["latitude"] <= 50 and
                         -125 <= s["longitude"] <= -66], key=lambda s: int(s["wban"]))
    if len(candidates) < count:
        raise ValueError("Insufficient continental USCRN metadata")
    positions = np.radians([[s["latitude"], s["longitude"]] for s in candidates])
    latitude, longitude = positions[:, 0], positions[:, 1]
    a = np.sin((latitude[:, None]-latitude[None])/2)**2+np.cos(latitude[:, None])*np.cos(latitude[None]) \
        *np.sin((longitude[:, None]-longitude[None])/2)**2
    distance = 2*np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    selected = [0]
    while len(selected) < count:
        nearest = distance[:, selected].min(1)
        nearest[selected] = -1
        selected.append(int(np.argmax(nearest)))
    return [candidates[i] for i in selected]


def target_queries(data, name, descriptor, count=2048, seed=42):
    variables = descriptor.get("target_variables", list(range(data.values.shape[1])))
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(variables))
    queries = []
    for rank, variable in enumerate(order):
        quota = count//len(order)+int(rank < count % len(order))
        origins = data.candidates(int(variable), descriptor["length"], descriptor["horizon"], "test",
                                  require_history=False)
        quota = min(quota, len(origins))
        for group in np.array_split(origins, quota) if quota else []:
            if len(group):
                queries.append(Query(name, int(variable), int(rng.choice(group)),
                                     descriptor["length"], descriptor["horizon"]).__dict__)
    return queries


def nasdaq_day(frame, day, stock):
    """Author archive: column0 relative minute, column1 close; first offset is blank.

    The author documents stock points 2..391 corresponding to index 1..390.
    These are documented session-relative times, not independently observed UTC.
    """
    if list(frame.columns[1:3]) != ["0", "1"]:
        raise ValueError("Unverified NASDAQ original column schema")
    minute = pd.to_numeric(frame["0"], errors="raise")
    if len(minute) and pd.isna(minute.iloc[0]):
        minute.iloc[0] = 0
    if minute.isna().any() or (minute != np.floor(minute)).any() or minute.duplicated().any():
        raise ValueError("NASDAQ minute offsets not unique integers")
    minute = minute.astype(int)-(1 if stock else 0)
    index = pd.Timestamp(day)+pd.to_timedelta(minute.to_numpy(), unit="min")
    return pd.Series(pd.to_numeric(frame["1"], errors="raise").to_numpy(), index=index)


def nasdaq_sessions(frame):
    minute = pd.to_numeric(frame["0"], errors="raise")
    starts = np.flatnonzero(minute.isna() | minute.diff().lt(0))
    starts = np.unique(np.r_[0, starts])
    return [frame.iloc[a:b].reset_index(drop=True) for a, b in
            zip(starts, np.r_[starts[1:], len(frame)])]


def verified_nasdaq_session(frame, reference):
    parts = nasdaq_sessions(frame)
    expected = np.asarray(reference, float)
    expected = expected[np.isfinite(expected)]
    matches = []
    for i, part in enumerate(parts):
        values = pd.to_numeric(part["1"], errors="raise").to_numpy()
        values = values[np.isfinite(values)]
        if len(values) == len(expected) and np.allclose(values, expected, rtol=1e-8, atol=1e-8):
            matches.append(i)
    if len(matches) != 1:
        raise ValueError("No unique NASDAQ raw session matching the author matrix")
    return parts[matches[0]], matches[0], len(parts)


def provenance_exclusion(source_audit):
    # Identity and observation-period separation is explicit; byte equality alone is insufficient.
    evidence = {
        "solar": "137 Alabama solar-generation series in 2006; no ETT transformer/electric-meter signal",
        "nn5": "111 ATM withdrawals on the original 1996-1998 grid; not FX or stock/index prices",
        "pems04": "January-February 2018 flow; no shared observation period with Traffic 2015-2016 or PEMS08 2016",
        "aq_shunyi": "Beijing Shunyi 2013-2017; not German weather 2020 or USCRN 2025"}
    if set(source_audit) != set(evidence):
        raise ValueError("Source allowlist changed")
    return {name: {"observation_overlap": "excluded_by_signal_identity_and_period_provenance",
        "evidence": evidence[name], "raw_sha256": record["provenance"]["sha256"],
        "entity_overlap": "not_claimed_unseen; road sensor identity overlap remains unverified"}
        for name, record in source_audit.items()}


def prepare_extras(root):
    root = Path(root)
    raw, prepared = root/"raw/targets", root/"targets"
    pems = raw/"PEMS08.npz"
    if pems.exists() and (raw/"PEMS08.csv").exists():
        with np.load(pems) as data:
            values = data["data"][:, :, 0]
        if values.shape[1] != 170:
            raise ValueError("Expected original 170-node PEMS08 flow")
        roads = pd.read_csv(raw/"PEMS08.csv")
        adjacency = np.zeros((170, 170), np.float32)
        for a, b in roads.iloc[:, :2].itertuples(index=False, name=None):
            adjacency[int(a), int(b)] = 1
        save_series(prepared, "pems08", values, pd.date_range("2016-07-01", periods=len(values), freq="5min"),
            list(map(str, range(170))), {"sha256": sha(pems), "road_sha256": sha(raw/"PEMS08.csv"),
                "channel": "flow", "source": "official ASTGNN pinned raw data; July-August 2016"}, adjacency)
    metadata = raw/"noaa_station_metadata.json"
    if metadata.exists():
        selected = geographic_stations(json.loads(metadata.read_text()))
        atomic_json(raw/"noaa_selected_stations.json", selected)
        if all((raw/"noaa"/s["filename"]).exists() for s in selected):
            columns, identities, hashes = [], [], {}
            for station in selected:
                path = raw/"noaa"/station["filename"]
                frame = pd.read_csv(path, sep=r"\s+", header=None, dtype=str)
                values, description = noaa_hourly(frame)
                if int(description["wban"]) != int(station["wban"]):
                    raise ValueError("NOAA station identity differs")
                columns.append(values)
                identities.append(str(station["wban"]))
                hashes[station["filename"]] = sha(path)
            index = pd.date_range("2025-01-01", "2025-12-31 23:00", freq="h", tz="UTC")
            matrix = pd.concat(columns, axis=1).reindex(index)
            coords = np.radians([[s["latitude"], s["longitude"]] for s in selected])
            lat, lon = coords[:, 0], coords[:, 1]
            distance = np.sin((lat[:, None]-lat[None])/2)**2+np.cos(lat[:, None])*np.cos(lat[None]) \
                *np.sin((lon[:, None]-lon[None])/2)**2
            adjacency = np.zeros((16, 16), np.float32)
            for i in range(16):
                neighbors = sorted([j for j in range(16) if j != i],
                    key=lambda j: (distance[i, j], int(identities[j])))[:4]
                adjacency[i, neighbors] = 1
            adjacency = np.maximum(adjacency, adjacency.T)
            save_series(prepared, "noaa2025", matrix.to_numpy(), index, identities,
                {"sha256": sha(metadata), "raw_file_sha256": hashes, "selection": selected,
                 "field_number": 10, "field": "T_HR_AVG", "unit": "Celsius",
                 "hours": "2025 UTC end-labeled hours", "selection_uses_errors": False}, adjacency)
    archive = raw/"nasdaq100.zip"
    if archive.exists():
        with zipfile.ZipFile(archive) as stream:
            names = stream.namelist()
            symbol_file = next(n for n in names if n.endswith("full/stock_name.txt"))
            symbols = stream.read(symbol_file).decode().split()
            matrix_name = next(n for n in names if n.endswith("full/full_non_padding.csv"))
            matrix = pd.read_csv(io.BytesIO(stream.read(matrix_name)))
            days = sorted(Path(n).stem.removeprefix("NDX_") for n in names if
                n.startswith("nasdaq100/full/stock_data_GOOGLE/NDX_") and n.endswith(".csv"))
            blocks, offset = {}, 0
            for day in days:
                size = 211 if day == "2016-11-25" else 391
                blocks[day] = matrix.iloc[offset:offset+size]
                offset += size
            if offset != len(matrix) or list(matrix.columns) != symbols or len(days) != 191:
                raise ValueError("NASDAQ full author day/variable coverage differs")
            frames, repeated = {}, []
            for symbol in symbols:
                members = [n for n in names if "/full/stock_data_GOOGLE/" in "/"+n and n.endswith(".csv")
                           and not n.startswith("__MACOSX/")
                           and (symbol in Path(n).parts or Path(n).stem.startswith(symbol+"_")
                                or Path(n).stem == symbol)]
                if not members:
                    raise ValueError("Original NASDAQ per-timestamp member unavailable: "+symbol)
                parts = []
                for member in sorted(members):
                    frame = pd.read_csv(io.BytesIO(stream.read(member)))
                    day = Path(member).stem.removeprefix(symbol+"_")
                    session, selected, count = verified_nasdaq_session(frame, blocks[day][symbol])
                    if count > 1:
                        repeated.append({"member": member, "selected_part": selected, "parts": count,
                            "evidence": "unique exact close sequence match to author matrix daily block"})
                    parts.append(nasdaq_day(session, day, stock=symbol != "NDX"))
                frames[symbol] = pd.concat(parts).sort_index()
            if symbols[-1] != "NDX" or len(symbols) != 105:
                raise ValueError("NASDAQ full universe differs")
            # Keep missing index minutes as masked targets. Never compress their time.
            days = sorted(frames["NDX"].index.normalize().unique())
            grid = pd.DatetimeIndex(np.concatenate([pd.date_range(day, periods=210 if
                day == pd.Timestamp("2016-11-25") else 390, freq="min").to_numpy() for day in days]))
            index = frames["NDX"].reindex(grid)
            aligned, sessions = align_nasdaq({s: frames[s] for s in symbols[:-1]}, index)
            save_series(prepared, "nasdaq100", aligned.to_numpy(), aligned.index, symbols,
                {"sha256": sha(archive), "source": "author original full archive",
                 "alignment": "join documented relative-minute offsets: stock minute minus1 vs index minute",
                 "timestamps": "session-relative offsets and filename dates; no independently verified UTC times",
                 "provider_matrix_shape": list(matrix.shape), "reconstructed_shape": list(aligned.shape),
                 "repeated_sessions_verified_and_removed": repeated,
                 "short_session": "2016-11-25:210 index minutes", "missing_index_minutes": int(index.isna().sum()),
                 "provider_matrix_shifted_again": False,
                 "matrix_policy": "reconstruct raw minute join; do not shift provider matrix",
                 "lltc": "post-day157 pseudo-zero missing", "interpolation": "none"}, segments=sessions)


def audit_targets(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source_audit = json.loads((root/"source_audit.json").read_text())
    prepare_extras(root)
    exclusion = provenance_exclusion(source_audit)
    records = {}
    raw = root/"raw/targets"
    prepared = root/"targets"
    for name, descriptor in TARGETS.items():
        record = dict(descriptor, forecasts_run=0, pretraining_exposure="not_independently_verifiable",
                      sample_manifest_complete=False)
        if name in DRIVE:
            path = raw/(name+".csv")
            record["official_drive_id"] = DRIVE[name][0]
            if path.exists():
                frame = pd.read_csv(path, encoding="cp1252")
                if "date" not in frame:
                    raise ValueError("Expected original synchronized multivariate CSV, not Hist/Pred demo")
                values = frame.drop(columns="date")
                if values.shape[1] != descriptor["variables"]:
                    raise ValueError(f"Official target variable coverage differs: {name}")
                source_hashes = {r["provenance"]["sha256"] for r in source_audit.values()}
                if sha(path) in source_hashes:
                    raise ValueError("Exact source/target raw overlap")
                dates = pd.to_datetime(frame["date"], dayfirst="." in str(frame["date"].iloc[0]))
                record["data"] = save_series(prepared, name, values.to_numpy(), dates,
                    list(values.columns), {"official_drive_id": DRIVE[name][0], "sha256": sha(path),
                                           "mask": "original_nonfinite_only; sentinel_protocol_unverified"})
            else:
                record["preparation_issue"] = "original_file_not_downloaded"
            record["original_chat_time_exclusion"] = "author_claim; corpus lacks independent row provenance"
        path = prepared/(name+".npz")
        if path.exists():
            data = Series(path)
            queries = target_queries(data, name, descriptor)
            atomic_json(output/(name+"-queries.json"), queries)
            record.update(sample_manifest_complete=True, queries=len(queries),
                          sample_sha256=sha(output/(name+"-queries.json")),
                          variable_coverage=len({q["variable"] for q in queries}))
        elif "preparation_issue" not in record:
            record["preparation_issue"] = "raw_original_timestamps_not_yet_prepared"
        records[name] = record
    audit = {"targets": records, "source_exclusion": exclusion, "target_predictions": 0,
        "strict_table4_reproduction": False,
        "reason": "No released complete origin/variable/metric-scale/random-seed evaluation manifest verified",
        "evaluation_kind": "independent_stratified_subset_when_separately_authorized",
        "future_decode": {"samples": 8, "aggregate": "pointwise_nanmedian", "chunk": 16,
                          "top_k": 100, "top_p": 1., "temperature": 1., "native_dtype": "float16"}}
    atomic_json(output/"target_audit.json", audit)
    return audit
