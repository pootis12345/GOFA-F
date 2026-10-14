"""Audited external spatial graphs, with train-only correlation as an explicit alternative."""
import json
from pathlib import Path

import numpy as np

from gofa_f.data import correlation_graph, file_sha256


def gaussian_graph(distances, threshold=.1, cutoff=None):
    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("Distances must be a square matrix")
    if np.isnan(distances).any() or (distances < 0).any():
        raise ValueError("Distances must be nonnegative; unreachable pairs may be +inf")
    if not 0 <= threshold <= 1:
        raise ValueError("Gaussian threshold must be in [0,1]")
    # Match DCRNN's bandwidth rule, including recorded zero distances.
    finite = distances[np.isfinite(distances)]
    sigma = finite.std() if finite.size else 0
    if not sigma > 0:
        raise ValueError("Distance bandwidth is zero; check spatial data")
    adjacency = np.exp(-np.square(distances / sigma))
    adjacency[adjacency < threshold] = 0
    if cutoff is not None:
        adjacency[distances > cutoff] = 0
    np.fill_diagonal(adjacency, 1)
    return adjacency.astype(np.float32), float(sigma)


def road_distances(frame, node_ids):
    """Road CSV is [source,target,cost]; GraphExchange uses adjacency[target,source]."""
    ids = list(map(str, node_ids))
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sensor IDs")
    available = set(frame["from"].map(str)) | set(frame["to"].map(str))
    if set(ids) - available:
        raise ValueError("Road file is missing data sensor IDs")
    lookup = {node: i for i, node in enumerate(ids)}
    distances = np.full((len(ids), len(ids)), np.inf)
    for source, target, distance in frame[["from", "to", "cost"]].itertuples(index=False, name=None):
        if str(source) in lookup and str(target) in lookup:
            i, j = lookup[str(target)], lookup[str(source)]
            distances[i, j] = min(distances[i, j], float(distance))
    if not np.isfinite(distances).any():
        raise ValueError("No road distances match data sensor IDs")
    return distances


def geographic_distances(stations, node_ids):
    """Haversine distances in km; proximity is not a measured pollutant transport network."""
    stations = stations.copy()
    stations.index = stations.index.map(str)
    ids = list(map(str, node_ids))
    if stations.index.has_duplicates or len(ids) != len(set(ids)):
        raise ValueError("Duplicate station IDs")
    coordinates = stations.loc[ids, ["latitude", "longitude"]].to_numpy(dtype=np.float64)
    if not np.isfinite(coordinates).all() or (np.abs(coordinates[:, 0]) > 90).any():
        raise ValueError("Invalid station coordinates")
    if (np.abs(coordinates[:, 1]) > 180).any():
        raise ValueError("Invalid station coordinates")
    lat, lon = np.deg2rad(coordinates).T
    delta_lat = lat[:, None] - lat[None, :]
    delta_lon = lon[:, None] - lon[None, :]
    a = np.sin(delta_lat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(delta_lon / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def build_graph(series, config):
    settings = config.get("graph", {"kind": "correlation"})
    kind = settings["kind"]
    if kind == "correlation":
        adjacency = correlation_graph(series.values[:series.train_end], series.mask[:series.train_end],
                                      k=config.get("graph_k", 8))
        return adjacency, {"kind": kind, "training_prefix_only": True, "k": config.get("graph_k", 8)}
    if kind not in ("road_distance", "geographic_distance"):
        raise ValueError(f"Unsupported graph kind: {kind}")
    path = Path(settings["path"])
    metadata = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    if metadata["kind"] != kind or metadata["prepared_sha256"] != series.fingerprint:
        raise ValueError("Spatial graph does not match the prepared data version or requested kind")
    if file_sha256(path) != metadata["graph_sha256"]:
        raise ValueError("Spatial graph hash mismatch")
    with np.load(path, allow_pickle=False) as data:
        adjacency = data["adjacency"].astype(np.float32)
        ids = data["node_ids"].astype(str)
    n = series.values.shape[1]
    if adjacency.shape != (n, n) or ids.shape != (n,) or len(set(ids)) != n:
        raise ValueError("Spatial graph shape or node IDs do not match data")
    if not np.isfinite(adjacency).all() or (adjacency < 0).any() or (adjacency.diagonal() <= 0).any():
        raise ValueError("Spatial graph must be finite, nonnegative and contain self loops")
    if ids.tolist() != metadata["node_ids"]:
        raise ValueError("Spatial graph node order differs from its audited mapping")
    return adjacency, {"kind": kind, "path": str(path), "graph_sha256": metadata["graph_sha256"],
                       "metadata_sha256": file_sha256(path.with_suffix(".metadata.json")),
                       "orientation": "target,source", "source_files": metadata["source_files"],
                       "construction": metadata["construction"]}
