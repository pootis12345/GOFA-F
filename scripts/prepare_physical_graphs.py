"""Prepare new, ID-aligned spatial graph artifacts on C without rewriting any existing series."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from gofa_f.data import file_sha256
from gofa_f.graphs import gaussian_graph, geographic_distances, road_distances


SOURCES = {
    "metr_la": ("traffic/metr_la.zip", "metr_la.h5", "data", "distances_la.csv"),
    "pems_bay": ("traffic/pems_bay.zip", "pems_bay.h5", "data", "distances_bay.csv"),
    "aqi": ("air/air_quality.zip", "full437.h5", "pm25", None),
    "aqi_36": ("air/air_quality.zip", "small36.h5", "pm25", None),
}


def prepare(name, output):
    archive, member, key, distance_csv = SOURCES[name]
    raw = Path("data/raw/tsl")
    manifest = json.loads((raw / "download_manifest.json").read_text())
    archive_path = raw / archive
    if file_sha256(archive_path) != manifest["files"][archive]["sha256"]:
        raise ValueError("Raw archive hash differs from download manifest")
    extracted = raw / "extracted" / member
    frame = pd.read_hdf(extracted, key=key)
    prepared = Path("data/processed") / f"{name}.npz"
    previous = json.loads(prepared.with_suffix(".metadata.json").read_text())
    fingerprint = file_sha256(prepared)
    if previous["hdf_sha256"] != file_sha256(extracted) or previous["prepared_sha256"] != fingerprint:
        raise ValueError("Prepared data and original HDF provenance differ")
    # Verify the exact column/value mapping, rather than assuming equal node counts imply alignment.
    values = frame.to_numpy(dtype=np.float32)
    mask = np.isfinite(values)
    if previous["zero_is_missing"]:
        mask &= values != 0
    with np.load(prepared, allow_pickle=False) as data:
        if not np.array_equal(data["mask"], mask) or not np.array_equal(data["values"], np.where(mask, values, 0)):
            raise ValueError("Prepared column/value order differs from original HDF")
    ids = list(map(str, frame.columns))
    source_files = {str(archive_path): file_sha256(archive_path), str(extracted): file_sha256(extracted)}
    if distance_csv:
        with zipfile.ZipFile(archive_path) as z:
            csv_bytes = z.read(distance_csv)
            distances = road_distances(pd.read_csv(z.open(distance_csv), dtype={"from": str, "to": str}), ids)
        source_files[f"{archive_path}!{distance_csv}"] = hashlib.sha256(csv_bytes).hexdigest()
        adjacency, sigma = gaussian_graph(distances)
        kind = "road_distance"
        construction = {"kernel": "exp(-(distance/sigma)^2)", "sigma": sigma,
                        "distance_unit": "source CSV unit", "threshold": .1,
                        "directed": True, "orientation": "target,source", "self_loops": True}
    else:
        stations = pd.read_hdf(extracted, key="stations")
        distances = geographic_distances(stations, ids)
        adjacency, sigma = gaussian_graph(distances, cutoff=40.)
        kind = "geographic_distance"
        construction = {"kernel": "exp(-(distance_km/sigma_km)^2)", "sigma_km": sigma,
                        "cutoff_km": 40., "threshold": .1, "directed": False,
                        "orientation": "target,source", "self_loops": True,
                        "note": "Static geographic proximity; not observed pollutant transport edges. "
                                "Bandwidth computed from this dataset's static distances."}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"{name}.npz"
    if target.exists() or target.with_suffix(".metadata.json").exists():
        raise FileExistsError(f"Preserve existing graph artifacts; choose a new directory: {target}")
    np.savez_compressed(target, adjacency=adjacency, node_ids=np.asarray(ids))
    metadata = {"dataset": name, "kind": kind, "prepared_sha256": fingerprint,
                "node_ids": ids, "source_files": source_files, "construction": construction,
                "graph_sha256": file_sha256(target), "nodes": len(ids),
                "nonself_directed_entries": int(np.count_nonzero(adjacency) - len(ids))}
    target.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {k: metadata[k] for k in ("dataset", "kind", "nodes", "nonself_directed_entries", "graph_sha256")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/graphs/protocol-v2")
    args = parser.parse_args()
    for name in SOURCES:
        print(json.dumps(prepare(name, args.output)), flush=True)


if __name__ == "__main__":
    main()
