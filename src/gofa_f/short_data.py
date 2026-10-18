"""Read-only raw sources, balanced short queries, and paired source-only selection."""
from dataclasses import asdict
from pathlib import Path

import numpy as np

from gofa_f.generative_data import Query, SOURCES, Series
from gofa_f.prefix_cache import atomic_json


TEMPLATES = ((10, 1), (12, 12), (14, 7), (48, 24))
TARGETS = {
    "etth1": (48, 24, 7), "etth2": (48, 24, 7), "electric": (48, 24, 321),
    "exchange": (14, 7, 8), "nasdaq100": (10, 1, 105), "traffic": (48, 24, 862),
    "pems08": (12, 12, 170), "weather": (48, 24, 21), "noaa2025": (48, 24, 16),
}


def origins(data, variable, length, horizon, split, require_history=True, calendar_breaks=True):
    n = len(data.values)
    boundaries = (0, int(.6*n), int(.8*n), n)
    a, b = {"train": boundaries[:2], "validation": boundaries[1:3], "test": boundaries[2:]}[split]
    points = np.arange(max(a, length), b-horizon+1)
    count = data.prefix_counts[:, variable]
    good = count[points+horizon]-count[points] > 0
    if require_history:
        good &= count[points]-count[points-length] > 0
    if calendar_breaks:
        good &= data.segments[points-length] == data.segments[points+horizon-1]
    return points[good]


def stratified(points, count, horizon, rng, disjoint=False):
    if len(points) < count:
        raise ValueError("Insufficient distinct eligible forecast origins")
    groups = np.array_split(points, count)
    for _ in range(1000):
        result = np.array([rng.choice(g) for g in groups], np.int64)
        if not disjoint or np.all(np.diff(result) >= horizon):
            return result.tolist()
    raise ValueError("Cannot meet fixed temporal strata and disjoint label windows")


def range_record(data, queries):
    """Only source labels may be audited before the final checkpoint is locked."""
    groups = {}
    for q in queries:
        key = f"{q['length']}->{q['horizon']}"
        v = groups.setdefault(key, {"queries": 0, "observed": 0, "missing": 0,
            "outside": 0, "range_error_lower_bound_sum": 0., "future_missing_prefixes": 0})
        o, variable, length, horizon = (q[k] for k in ("origin", "variable", "length", "horizon"))
        history = data.values[o-length:o, variable][data.mask[o-length:o, variable]].astype(np.float64)
        if not len(history):
            raise ValueError("Empty source history")
        extent = float(history.max()-history.min())
        extent = 1. if extent < 10*np.finfo(np.float64).eps else extent
        low, high = history.min()-.4999*extent, history.min()+1.4999*extent
        mask = data.mask[o:o+horizon, variable]
        target = data.values[o:o+horizon, variable][mask].astype(np.float64)
        v["queries"] += 1
        v["observed"] += len(target)
        v["missing"] += int((~mask).sum())
        v["outside"] += int(((target < low) | (target > high)).sum())
        v["range_error_lower_bound_sum"] += float(np.abs(target-np.clip(target, low, high)).sum())
        v["future_missing_prefixes"] += int(np.any(mask & (np.cumsum(~mask) > 0)))
    for v in groups.values():
        v["outside_fraction"] = v["outside"]/v["observed"]
        v["range_only_mae_lower_bound"] = v["range_error_lower_bound_sum"]/v["observed"]
    return groups


def build_queries(cfg, output, series):
    import json
    root, output = Path(cfg["data"]["directory"]), Path(output)
    pools = output/"pools"
    pools.mkdir(parents=True, exist_ok=True)
    old = pools/"queries.json"
    if old.exists():
        record = json.loads(old.read_text())
        if record["policy"] != cfg["sampling"]:
            raise ValueError("Immutable query policy changed")
        return record
    train, validation, audit = {}, [], {}
    for source_index, (source, count) in enumerate(SOURCES.items()):
        data = series[source]
        rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], source_index, 19]))
        rows, seen = [], set()
        for i in range(8192):
            template = i % 4
            variable = (i//4+template) % count
            length, horizon = TEMPLATES[template]
            eligible = origins(data, variable, length, horizon, "train")
            if not len(eligible):
                raise ValueError("Missing training variable/template")
            for _ in range(1000):
                origin = int(rng.choice(eligible))
                key = (variable, length, horizon, origin)
                if key not in seen:
                    break
            else:
                raise ValueError("Insufficient unique training queries")
            seen.add(key)
            rows.append(asdict(Query(source, variable, origin, length, horizon)))
        train[source] = rows
        own_validation = []
        for variable in range(count):
            for length, horizon in TEMPLATES:
                points = origins(data, variable, length, horizon, "validation")
                for origin in stratified(points, cfg["sampling"]["source_origins_per_variable_template"],
                                         horizon, rng, disjoint=True):
                    own_validation.append(asdict(Query(source, variable, origin, length, horizon)))
        validation.extend(own_validation)
        audit[source] = {"train": range_record(data, rows), "validation": range_record(data, own_validation)}
    targets = {}
    for index, (name, (length, horizon, variables)) in enumerate(TARGETS.items()):
        data = Series(root/"targets"/(name+".npz"))
        if data.values.shape[1] != variables:
            raise ValueError("Target variable count changed: "+name)
        selected = [104] if name == "nasdaq100" else list(range(variables))
        budget = max(cfg["sampling"]["target_queries_minimum"],
                     cfg["sampling"]["target_origins_per_variable_minimum"]*len(selected))
        rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], index, 190]))
        order = rng.permutation(selected).tolist()
        rows = []
        for rank, variable in enumerate(order):
            quota = budget//len(order)+int(rank < budget % len(order))
            points = origins(data, variable, length, horizon, "test", require_history=False,
                             calendar_breaks=name != "exchange")
            for origin in stratified(points, quota, horizon, rng):
                rows.append(asdict(Query(name, variable, origin, length, horizon)))
        if len(rows) != budget or set(q["variable"] for q in rows) != set(selected):
            raise ValueError("Target query coverage incomplete")
        targets[name] = rows
    record = {"policy": cfg["sampling"], "train": train, "validation": validation, "targets": targets,
              "target_predictions": 0, "target_label_statistics_audited": False}
    atomic_json(old, record)
    atomic_json(pools/"source_range_audit.json", audit)
    atomic_json(pools/"metric_epsilons.json", json.loads((root/"metric_epsilons.json").read_text()))
    return record


def relative_score(current, baseline, queries, epsilons, repeats, bootstrap_replicates=1000):
    expected = sum(map(len, epsilons.values()))*len(TEMPLATES)*repeats
    if len(queries) != expected or len(current) != len(queries) or len(baseline) != len(queries) or \
            any(x is None for x in current+baseline):
        raise ValueError("Only complete paired source validation may select checkpoints")
    groups = {}
    for a, b, q in zip(current, baseline, queries):
        key = q["source"], q["variable"], q["length"], q["horizon"]
        groups.setdefault(key, []).append((q["origin"], float(a), float(b)))
    scores, ratios, arrays, near = {}, {}, {}, {}
    for source, eps in epsilons.items():
        a, b = np.empty((len(eps), 4, repeats)), np.empty((len(eps), 4, repeats))
        for v in range(len(eps)):
            for k, (length, horizon) in enumerate(TEMPLATES):
                rows = sorted(groups.get((source, v, length, horizon), []))
                if len(rows) != repeats or len({r[0] for r in rows}) != repeats:
                    raise ValueError("Missing variable/template/temporal stratum")
                a[v, k], b[v, k] = np.array(rows)[:, 1:].T
        if not np.isfinite([a, b]).all() or min(a.min(), b.min()) < 0:
            raise ValueError("Invalid source error")
        av, bv = a.mean((1, 2)), b.mean((1, 2))
        ratio = np.maximum(av, eps)/np.maximum(bv, eps)
        scores[source] = float(np.exp(np.log(ratio).mean()))
        near[source] = int((bv <= eps).sum())
        ratios.update({f"{source}:{v}": float(x) for v, x in enumerate(ratio)})
        arrays[source] = a, b, np.asarray(eps)
    rng = np.random.default_rng(19042)
    boot = []
    for _ in range(bootstrap_replicates):
        logs = []
        for a, b, eps in arrays.values():
            variables = rng.integers(len(eps), size=len(eps))
            strata = rng.integers(repeats, size=repeats)
            av = a[variables][:, :, strata].mean((1, 2))
            bv = b[variables][:, :, strata].mean((1, 2))
            logs.append(np.log(np.maximum(av, eps[variables])/np.maximum(bv, eps[variables])).mean())
        boot.append(np.exp(np.mean(logs)))
    return {"score": float(np.exp(np.log(list(scores.values())).mean())), "source_scores": scores,
        "variable_ratios": ratios, "near_zero_baselines": near,
        "paired_bootstrap_95pct": np.quantile(boot, [.025, .975]).tolist() if boot else None,
        "bootstrap_method": "paired variable resampling and common temporal-stratum blocks within each source",
        "bootstrap_replicates": bootstrap_replicates, "bootstrap_seed": 19042,
        "origins_per_variable_template": repeats,
        "uncertainty_note": "One fixed origin per variable/template cannot estimate temporal variability; the interval mainly reflects variable resampling" if repeats == 1 else
            "Fixed temporal strata do not eliminate time dependence or adaptive-validation bias"}
