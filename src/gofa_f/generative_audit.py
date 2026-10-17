"""Target DATA audit only. No language models, forecasts or target fitting."""
import json
from pathlib import Path

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
        origins = data.candidates(int(variable), descriptor["length"], descriptor["horizon"], "test")
        quota = min(quota, len(origins))
        for group in np.array_split(origins, quota) if quota else []:
            if len(group):
                queries.append(Query(name, int(variable), int(rng.choice(group)),
                                     descriptor["length"], descriptor["horizon"]).__dict__)
    return queries


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


def audit_targets(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source_audit = json.loads((root/"source_audit.json").read_text())
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
