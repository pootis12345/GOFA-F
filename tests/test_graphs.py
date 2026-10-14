import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from gofa_f.benchmark import expand
from gofa_f.data import Series, file_sha256
from gofa_f.graphs import build_graph, gaussian_graph, geographic_distances, road_distances


def test_road_graph_matches_sensor_order_and_message_direction():
    frame = pd.DataFrame({"from": ["a", "b", "a", "b"], "to": ["a", "b", "b", "c"],
                          "cost": [0., 0., 1., 2.]})
    distances = road_distances(frame, ["c", "a", "b"])
    assert distances[2, 1] == 1  # a -> b, row target b / column source a
    assert distances[0, 2] == 2
    assert np.isinf(distances[1, 2])  # No reverse b -> a edge is invented.
    adjacency, _ = gaussian_graph(distances, threshold=0)
    assert adjacency[2, 1] > 0 and adjacency[1, 2] == 0
    assert np.all(adjacency.diagonal() == 1)
    with pytest.raises(ValueError, match="Duplicate"):
        road_distances(frame, ["a", "a"])


def test_geographic_graph_aligns_ids_and_applies_distance_cutoff():
    stations = pd.DataFrame({"latitude": [0., 0., 0.], "longitude": [0., .1, 1.]}, index=["a", "b", "c"])
    distances = geographic_distances(stations, ["c", "b", "a"])
    assert distances[1, 2] == pytest.approx(11.1195, rel=1e-4)
    adjacency, _ = gaussian_graph(distances, cutoff=40.)
    assert adjacency[1, 2] > 0 and adjacency[0, 2] == 0
    assert np.allclose(adjacency, adjacency.T)
    with pytest.raises(KeyError):
        geographic_distances(stations, ["missing"])


def external_fixture(tmp_path):
    data = tmp_path / "series.npz"
    values = np.arange(60, dtype=np.float32).reshape(30, 2)
    np.savez(data, values=values, mask=np.ones_like(values, bool))
    series = Series.load(data)
    path = tmp_path / "graph.npz"
    np.savez(path, adjacency=np.array([[1, .5], [0, 1]], dtype=np.float32), node_ids=np.array(["a", "b"]))
    metadata = {"kind": "road_distance", "prepared_sha256": series.fingerprint,
                "graph_sha256": file_sha256(path), "node_ids": ["a", "b"],
                "source_files": {"source": "hash"}, "construction": {"orientation": "target,source"}}
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata))
    return series, path, metadata


def test_external_graph_is_bound_to_data_version_content_and_node_order(tmp_path):
    series, path, metadata = external_fixture(tmp_path)
    config = {"graph": {"kind": "road_distance", "path": str(path)}}
    adjacency, provenance = build_graph(series, config)
    assert adjacency[0, 1] == .5 and adjacency[1, 0] == 0
    assert provenance["graph_sha256"] == metadata["graph_sha256"]
    for field, value, match in (("prepared_sha256", "other", "data version"),
                                ("graph_sha256", "other", "hash"),
                                ("node_ids", ["b", "a"], "node order")):
        modified = dict(metadata)
        modified[field] = value
        path.with_suffix(".metadata.json").write_text(json.dumps(modified))
        with pytest.raises(ValueError, match=match):
            build_graph(series, config)


def test_missing_external_graph_never_silently_falls_back(tmp_path):
    series, _, _ = external_fixture(tmp_path)
    with pytest.raises(FileNotFoundError):
        build_graph(series, {"graph": {"kind": "road_distance", "path": str(tmp_path / "absent.npz")}})


def test_registered_contexts_and_graph_kinds_match_audited_sources():
    spec = yaml.safe_load(Path("configs/benchmark.yaml").read_text(encoding="utf-8"))
    audit = yaml.safe_load(Path("configs/protocol_sources.yaml").read_text(encoding="utf-8"))
    for _, config in expand(spec):
        source = audit["datasets"][config["dataset"]]
        assert config["context"] == source["context"]
        assert config["horizon"] == source["horizon"]
        assert config["graph"]["kind"] == source["graph_kind"]
        assert source["source"].startswith("https://")
        assert source["horizon_source"] == source["source"]
    old = yaml.safe_load(Path("configs/benchmark_stage1.yaml").read_text(encoding="utf-8"))
    assert all(info["context"] == 96 for info in old["datasets"].values())
    assert all(info["horizons"] == [12, 24, 48] for info in old["datasets"].values())
