import copy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from gofa_f.generative_data import Query, Series, save_series
from gofa_f.graph_benchmark_data import TARGETS, aggregate_meteonet, geo_graph, road_graph, save
from gofa_f.graph_benchmark_state import query_pool, validate_config


def test_six_scope_and_checkpoint_are_fixed():
    cfg = yaml.safe_load((Path(__file__).parents[1]/'configs/stage21_graph_benchmark.yaml').read_text())
    validate_config(cfg)
    assert len(TARGETS) == 6 and 'aqi36' not in TARGETS
    for key, value in [('adapter_sha256', 'wrong'), ('calibration_enabled', True),
                       ('training_enabled', True), ('queries_per_dataset', 129)]:
        altered = copy.deepcopy(cfg)
        altered[key] = value
        with pytest.raises(ValueError):
            validate_config(altered)
    altered = copy.deepcopy(cfg)
    altered['resources']['admission_policy'] = 'memory_cap_only'
    with pytest.raises(ValueError):
        validate_config(altered)


def test_published_road_direction_and_identity():
    roads = pd.DataFrame([[101, 205, 4], [205, 205, 0], [999, 101, 1]], columns=['from', 'to', 'cost'])
    graph, skipped = road_graph(roads, [205, 101])
    np.testing.assert_array_equal(graph, [[0, 1], [0, 0]])
    assert skipped == 1
    roads.loc[0, 'cost'] = -1
    with pytest.raises(ValueError):
        road_graph(roads, [205, 101])


def test_geographic_four_neighbours_and_permutation():
    coords = np.array([[50, i*.1] for i in range(7)])
    ids = list('abcdefg')
    graph = geo_graph(coords, ids)
    np.testing.assert_array_equal(graph, graph.T)
    assert not np.diag(graph).any() and (graph.sum(1) >= 4).all()
    order = [6, 3, 0, 5, 2, 1, 4]
    np.testing.assert_array_equal(geo_graph(coords[order], [ids[i] for i in order]), graph[order][:, order])
    with pytest.raises(ValueError):
        geo_graph([[500, 0], [500, 1]], ['a', 'b'])


def meteo_rows():
    times = pd.date_range('2018-01-01', periods=10, freq='6min')
    return pd.concat([pd.DataFrame(dict(number_sta=str(i), lat=49+i*.1, lon=0., date=times,
        t=np.r_[np.full(8, 273.15+i), [np.nan, np.nan]])) for i in range(4)], ignore_index=True)


def test_meteonet_kelvin_hour_ending_missing_and_chunk_equivalence():
    rows = meteo_rows()
    a, coords = aggregate_meteonet([rows])
    b, _ = aggregate_meteonet([rows.iloc[:13], rows.iloc[13:]])
    pd.testing.assert_frame_equal(a, b)
    np.testing.assert_allclose(a.iloc[0], [0, 1, 2, 3], atol=1e-10)
    assert a.index[0] == pd.Timestamp('2018-01-01 01:00') and len(a) == 8760
    assert a.iloc[1:].isna().all().all() and coords.shape == (4, 2)
    less = rows.copy()
    less.loc[0, 't'] = np.nan
    extra = rows.copy()
    extra['date'] += pd.Timedelta(hours=1)
    result, _ = aggregate_meteonet([less, extra])
    assert pd.isna(result.iloc[0, 0])


def test_meteonet_collapses_identical_duplicates_and_freezes_first_location():
    rows = meteo_rows()
    duplicate = pd.concat([rows.iloc[:2], rows.iloc[1:2], rows.iloc[2:]], ignore_index=True)
    result, _ = aggregate_meteonet([duplicate.iloc[:2], duplicate.iloc[2:]])
    assert result.attrs['duplicate_rows_collapsed'] == 1
    assert result.attrs['conflicting_duplicate_times_masked'] == 0
    rows.loc[1, 'lat'] += .01
    frame, positions = aggregate_meteonet([rows])
    assert positions[0, 0] == 49. and '0' in frame.attrs['later_coordinate_variants']


def test_fixed_sample_budget_mask_and_label_blindness(tmp_path):
    series = {}
    for name, (_, nodes, _, _) in TARGETS.items():
        n = nodes or 140
        path = tmp_path/name
        values = np.ones((10000, n), np.float32)
        values[-1, 0] = np.nan
        save_series(path, name, values, pd.date_range('2018-01-01', periods=10000, freq='h'),
            [str(i) for i in range(n)], {'test_fixture': True})
        series[name] = Series(path/(name+'.npz'))
    a = query_pool(series)
    assert sum(map(len, a['targets'].values())) == 768
    assert all(len(rows) == 128 for rows in a['targets'].values())
    for name, rows in a['targets'].items():
        data = series[name]
        selected = {q['variable'] for q in rows}
        assert len(selected) == min(128, data.values.shape[1])
        for item in rows:
            q = Query(**item)
            assert q.origin >= 8000 and q.origin+q.horizon <= 10000
            assert asdict(q) == item
        data.values *= -22 # Values/errors must never influence fixed identities.
    assert query_pool(series) == a
    assert a['coverage']['kelmarsh2019']['variables'] == 6


def test_save_preserves_observed_zero_and_missing_mask(tmp_path):
    frame = pd.DataFrame([[0., np.nan], [1., 2.]], index=pd.date_range('2018-01-01', periods=2, freq='h'),
        columns=['a', 'b'])
    meta = save(tmp_path, 'meteonet_nw2018', frame, np.array([[0, 1], [1, 0]]), {'unit': 'Celsius'})
    data = Series(tmp_path/'targets/meteonet_nw2018.npz')
    assert data.values[0, 0] == 0 and data.mask[0, 0] and not data.mask[0, 1]
    assert meta['variables'] == ['a', 'b']
    frame.index = pd.DatetimeIndex([frame.index[0]]*2)
    with pytest.raises(ValueError):
        save(tmp_path, 'meteonet_nw2018', frame, np.eye(2), {})


def test_stage21_progress_resume_counts_twelve_phases(tmp_path):
    from gofa_f.graph_benchmark_progress import Progress, render
    progress = Progress(tmp_path, 100, clock=lambda: 10.)
    progress.completed['finished'] = dict(numeric_slots_in_eight_paths=20, active_seconds=10.)
    p = progress.evaluation(1, 2, 10, 20, 5., evaluation_kind='current')
    assert p['overall_work_done'] == 30 and p['overall_fraction'] == .3
    assert p['overall_eta_seconds'] == 35. and p['evaluation_eta_seconds'] == 5.
    assert p['total_model_datasets'] == 12 and '/12' in render(p)
    assert Progress(tmp_path, 100).state['overall_work_done'] == 30


def test_prediction_entry_has_no_training_and_keeps_source_flags():
    import ast
    source = (Path(__file__).parents[1]/'src/gofa_f/graph_benchmark_run.py').read_text()
    tree = ast.parse(source)
    assert not any(isinstance(n, ast.Attribute) and n.attr in ('backward', 'step') for n in ast.walk(tree))
    assert 'model.adapter.requires_grad' not in source
    assert 'model.native.training' in source and '@torch.no_grad()' in source
