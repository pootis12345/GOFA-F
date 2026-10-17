"""Raw synchronized Stage17 data, immutable query pools and source-only selection."""
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from gofa_f.prefix_cache import atomic_json

TEMPLATES = ((10, 1), (12, 12), (14, 7), (48, 24), (192, 96), (288, 144))
SOURCES = {"solar": 137, "nn5": 111, "pems04": 307, "aq_shunyi": 11}
ASTGNN_REVISION = "9c2e19b98c4cedf1f35214d8789685b6381b3aad"
TFB_REVISION = "889459188ec89c9cb3c7e5ec40f30b8c0c5c2529"
DRIVE = {
    "etth1": ("19ul8ID0_HBhl4Aj4nlklSixytCPuIW47", 48, 24, 7),
    "etth2": ("17K39Ubj1rZBddaadFBVJnuESmkCW9-4j", 48, 24, 7),
    "ettm1": ("1tu8e8xUqiuA64O3FTwQimCTZWmltakuN", 192, 96, 7),
    "ettm2": ("1SDSmK4aVGC4fNLl_qlbvIqa2aiRtmZtY", 192, 96, 7),
    "electric": ("1D3nEnreVsjt9J0hqGGLtVX5bWisWYw6K", 48, 24, 321),
    "exchange": ("1bpT6Er8NEw4hKWRNCW3UhP4vF-Ksvn-K", 14, 7, 8),
    "traffic": ("1MHPsrX5C_tqcNOF2zOYj6mRBspRVOqpY", 48, 24, 862),
    "weather": ("1pQ19JYIo5wTS4uqzEJyW8x8I44aVinDs", 288, 144, 21),
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url, path, md5=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        request = urllib.request.Request(url, headers={"User-Agent": "GOFA-F-public-research/Stage17"})
        temporary = path.with_suffix(path.suffix+".part")
        with opener.open(request, timeout=90) as response, temporary.open("wb") as output:
            while chunk := response.read(2**20):
                output.write(chunk)
        temporary.replace(path)
    if md5 and hashlib.md5(path.read_bytes()).hexdigest() != md5:
        raise ValueError(f"Official checksum differs: {path}")
    return path


def parse_tsf(path):
    with zipfile.ZipFile(path) as archive:
        members = [n for n in archive.namelist() if n.endswith(".tsf")]
        if len(members) != 1:
            raise ValueError("Expected one original TSF")
        text = archive.read(members[0]).decode("cp1252")
    rows, ids, starts = [], [], []
    data = False
    for line in text.splitlines():
        line = line.strip()
        if line.lower() == "@data":
            data = True
        elif data and line and not line.startswith("#"):
            identifier, start, raw = line.split(":", 2)
            ids.append(identifier)
            starts.append(datetime.strptime(start, "%Y-%m-%d %H-%M-%S"))
            rows.append(np.array([float(v) if v != "?" else np.nan for v in raw.split(",")], np.float32))
    if not rows or len(set(starts)) != 1 or len({len(row) for row in rows}) != 1:
        raise ValueError("TSF does not establish a shared start and synchronized grid")
    return np.stack(rows, 1), ids, starts[0]


@dataclass(frozen=True)
class Query:
    source: str
    variable: int
    origin: int
    length: int
    horizon: int
    chunk: int = 0

    @property
    def key(self):
        return f"{self.source}:{self.variable}:{self.origin}:{self.length}:{self.horizon}:{self.chunk}"


class Series:
    def __init__(self, path):
        self.path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            self.values, self.mask = data["values"], data["mask"]
            self.times, self.segments = data["times"], data["segments"]
            self.external = data["external"] if "external" in data else None
        self.meta = json.loads(self.path.with_suffix(".json").read_text())
        self.ids = self.meta["variables"]
        self.prefix_counts = np.concatenate((np.zeros((1, self.mask.shape[1]), np.int64),
                                            self.mask.cumsum(0, dtype=np.int64)), 0)
        self._eligible = OrderedDict()

    def candidates(self, variable, length, horizon, split, chunk=0):
        key = (variable, length, horizon, split, chunk)
        if key in self._eligible:
            return self._eligible[key]
        total = len(self.values)
        boundaries = (0, int(.6*total), int(.8*total), total)
        start, end = {"train": boundaries[:2], "validation": boundaries[1:3],
                      "test": boundaries[2:]}[split]
        origins = np.arange(max(length, start), end-horizon+1)
        finish = np.minimum(horizon, chunk+16) if split == "train" else horizon
        begin = chunk if split == "train" else 0
        counts = self.prefix_counts[:, variable]
        valid = (self.segments[origins-length] == self.segments[origins+horizon-1]) \
            & (counts[origins]-counts[origins-length] > 0) \
            & (counts[origins+finish]-counts[origins+begin] > 0)
        result = origins[valid]
        self._eligible[key] = result
        while len(self._eligible) > 256:
            self._eligible.popitem(last=False)
        return result

    def training_record(self, query):
        o, length, v, chunk = query.origin, query.length, query.variable, query.chunk
        history = np.where(self.mask[o-length:o+chunk, v], self.values[o-length:o+chunk, v], np.nan)
        end = o+min(query.horizon, chunk+16)
        response = np.where(self.mask[o+chunk:end, v], self.values[o+chunk:end, v], np.nan)
        return history, response

    def graph(self, query):
        from gofa_f.generative_graph import origin_graph
        o, length = query.origin, query.length
        return origin_graph(self.values[o-length:o], self.mask[o-length:o], query.variable,
                            self.external, self.ids)


def save_series(root, name, values, times, ids, provenance, external=None, segments=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    values = np.asarray(values, np.float32)
    times = np.asarray(times).astype("datetime64[ns]").astype(np.int64)
    if values.shape != (len(times), len(ids)) or len(set(ids)) != len(ids):
        raise ValueError(f"Invalid synchronized dimensions: {name}")
    if np.any(np.diff(times) <= 0):
        raise ValueError(f"Non-increasing timestamps: {name}")
    if segments is None:
        interval = int(np.median(np.diff(times)))
        segments = np.cumsum(np.r_[False, np.diff(times) != interval]).astype(np.int64)
    mask = np.isfinite(values)
    arrays = dict(values=np.where(mask, values, 0), mask=mask, times=times, segments=segments)
    if external is not None:
        arrays["external"] = np.asarray(external, np.float32)
    path = root/(name+".npz")
    if path.exists():
        old = json.loads(path.with_suffix(".json").read_text())
        if old["sha256"] != sha(path) or old["provenance"] != provenance:
            raise ValueError(f"Existing prepared data binding changed: {name}")
        return old
    np.savez_compressed(path, **arrays)
    result = {"name": name, "shape": list(values.shape), "variables": list(map(str, ids)),
        "start": str(pd.Timestamp(times[0])), "end": str(pd.Timestamp(times[-1])),
        "missing_fraction": float(1-mask.mean()), "provenance": provenance,
        "sha256": sha(path), "segment_count": int(segments.max()+1)}
    atomic_json(path.with_suffix(".json"), result)
    return result


def prepare_sources(root, raw):
    root, raw = Path(root), Path(raw)
    records = {}
    for name, record, filename, checksum, frequency in (
        ("solar", "4656144", "solar_10_minutes_dataset.zip", "84c0de18383c911091a3cd274661b029", "10min"),
        ("nn5", "4656110", "nn5_daily_dataset_with_missing_values.zip", "2529489455b2d9f730fbb3c65d39df88", "1D")):
        url = f"https://zenodo.org/records/{record}/files/{filename}?download=1"
        path = fetch(url, raw/filename, checksum)
        values, ids, start = parse_tsf(path)
        times = pd.date_range(start, periods=len(values), freq=frequency)
        records[name] = save_series(root, name, values, times, ids,
            {"url": url, "sha256": sha(path), "official_md5": checksum, "aligned_tsf_start": str(start)})
    base = f"https://raw.githubusercontent.com/guoshnBJTU/ASTGNN/{ASTGNN_REVISION}/data"
    for name in ("pems04",):
        code = name.upper()
        signals = fetch(f"{base}/{code}/{code}.npz", raw/(code+".npz"))
        roads = fetch(f"{base}/{code}/{code}.csv", raw/(code+".csv"))
        with np.load(signals) as data:
            values = data["data"][:, :, 0]  # original flow, not speed or occupancy
        adjacency = np.zeros((values.shape[1], values.shape[1]), np.float32)
        frame = pd.read_csv(roads)
        for a, b in frame.iloc[:, :2].itertuples(index=False, name=None):
            adjacency[int(a), int(b)] = 1
        records[name] = save_series(root, name, values,
            pd.date_range("2018-01-01", periods=len(values), freq="5min"), list(map(str, range(values.shape[1]))),
            {"url": f"{base}/{code}", "sha256": sha(signals), "road_sha256": sha(roads),
             "source_revision": ASTGNN_REVISION, "tfb_revision": TFB_REVISION, "channel": "flow"}, adjacency)
    url = "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip"
    archive = fetch(url, raw/"beijing_multisite.zip")
    with zipfile.ZipFile(archive) as stream:
        members = [n for n in stream.namelist() if "Shunyi" in n and n.endswith(".csv")]
        if len(members) != 1:
            raise ValueError("One original Shunyi station CSV required")
        frame = pd.read_csv(io.BytesIO(stream.read(members[0])))
    ids = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
    records["aq_shunyi"] = save_series(root, "aq_shunyi", frame[ids].to_numpy(),
        pd.to_datetime(frame[["year", "month", "day", "hour"]]), ids,
        {"url": url, "sha256": sha(archive), "member": members[0], "tfb_revision": TFB_REVISION,
         "interpolation": "none", "station": "Shunyi"})
    for name, nodes in SOURCES.items():
        if records[name]["shape"][1] != nodes:
            raise ValueError(f"Preregistered source width differs: {name}")
    return records


def build_pools(series, output, per_source=8192, seed=42):
    output = Path(output)
    rng = np.random.default_rng(seed)
    train, validation, epsilons = {}, [], {}
    for source, data in series.items():
        queries = []
        nodes = data.values.shape[1]
        for index in range(per_source):
            variable = index % nodes
            length, horizon = TEMPLATES[(index//nodes+variable) % len(TEMPLATES)]
            chunk = 16*int(rng.integers((horizon+15)//16))
            origins = data.candidates(variable, length, horizon, "train", chunk)
            if not len(origins):
                raise ValueError(f"No legal training query: {source}/{variable}/{length}/{horizon}/{chunk}")
            queries.append(asdict(Query(source, variable, int(rng.choice(origins)), length, horizon, chunk)))
        train[source] = queries
        stop = int(.6*len(data.values))
        epsilons[source] = []
        for variable in range(nodes):
            observed = data.values[:stop, variable][data.mask[:stop, variable]].astype(np.float64)
            if not len(observed):
                raise ValueError("Training-only metric scale unavailable")
            epsilons[source].append(float(max(1e-12, 1e-6*max(np.std(observed),
                np.percentile(observed, 75)-np.percentile(observed, 25)))))
            for length, horizon in TEMPLATES:
                origins = data.candidates(variable, length, horizon, "validation")
                if not len(origins):
                    raise ValueError(f"Missing required source validation variable/template: {source}/{variable}")
                validation.append(asdict(Query(source, variable, int(rng.choice(origins)), length, horizon)))
    atomic_json(output/"train_queries.json", train)
    atomic_json(output/"validation_queries.json", validation)
    atomic_json(output/"metric_epsilons.json", epsilons)
    return {"train_per_source": per_source, "train_total": sum(map(len, train.values())),
        "validation_total": len(validation), "seed": seed,
        "hashes": {name: sha(output/name) for name in
            ("train_queries.json", "validation_queries.json", "metric_epsilons.json")}}


def selection_score(query_mae, validation, epsilons):
    grouped = {}
    for error, query in zip(query_mae, validation):
        grouped.setdefault((query["source"], query["variable"]), []).append(float(error))
    if len(query_mae) != len(validation) or any(len(v) != 6 for v in grouped.values()):
        raise ValueError("Only complete six-template validation may select checkpoints")
    values = {source: [] for source in epsilons}
    return {"variable_mae": {f"{s}:{v}": float(np.mean(e, dtype=np.float64)) for (s, v), e in grouped.items()},
            "source_names": list(values)}


def relative_score(current, baseline, epsilons):
    ratios, source_scores, near_zero = {}, {}, {}
    for source, thresholds in epsilons.items():
        logs = []
        near_zero[source] = 0
        for variable, epsilon in enumerate(thresholds):
            key = f"{source}:{variable}"
            a, b = current["variable_mae"][key], baseline["variable_mae"][key]
            if not np.isfinite([a, b]).all() or min(a, b) < 0:
                raise ValueError("Invalid validation MAE")
            ratio = max(a, epsilon)/max(b, epsilon)
            ratios[key] = float(ratio)
            logs.append(np.log(ratio))
            near_zero[source] += int(b <= epsilon)
        source_scores[source] = float(np.exp(np.mean(logs, dtype=np.float64)))
    return {"score": float(np.exp(np.mean(np.log(list(source_scores.values())), dtype=np.float64))),
            "source_scores": source_scores, "variable_ratios": ratios, "near_zero_baselines": near_zero}
