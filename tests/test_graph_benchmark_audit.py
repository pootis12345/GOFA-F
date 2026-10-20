import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from gofa_f.graph_benchmark import audit, compare_observations, sha, stage20_stopped, validate_config


def frame(values):
    index = pd.date_range('2014-05-01', periods=len(values), freq='h')
    return pd.DataFrame(np.asarray(values)[:, None].repeat(36, 1), index=index,
        columns=list(map(str, range(36))))


def test_alias_with_clock_label_shift_and_missing():
    data = frame(np.arange(200, dtype=float))
    source = pd.Series(np.arange(200, dtype=float), index=data.index-pd.Timedelta(hours=1))
    data.iloc[5:9, 0] = np.nan
    found = compare_observations(data, source, shifts=[-1, 0, 1])
    row = next(v for v in found if v['station'] == '0' and v['diagnostic_source_offset_hours'] == -1)
    assert row['possible_same_observations'] and row['exact_matches'] == 196
    assert row['planned_test_exact_matches'] == 40
    assert len(row['witnesses']) == 5 and pd.isna(data.iloc[5, 0])


def test_constant_coincidences_are_not_identity_evidence():
    data = frame(np.zeros(200))
    rows = compare_observations(data, data.iloc[:, 0], shifts=[0])
    assert not any(r['possible_same_observations'] for r in rows)
    assert all(r['correlation'] is None for r in rows)


def test_high_correlation_without_same_values_is_not_overlap_proof():
    data = frame(np.arange(200, dtype=float))
    source = data.iloc[:, 0] + .01
    assert not any(r['possible_same_observations'] for r in compare_observations(data, source, [0]))


def test_duplicate_timestamps_not_silently_fixed():
    data = frame(np.arange(200, dtype=float))
    data.index = pd.DatetimeIndex([data.index[0]]*200)
    with pytest.raises(ValueError, match='Duplicate'):
        compare_observations(data, data.iloc[:, 0])


def test_fixed_scope_and_no_fallback_replacement():
    cfg = yaml.safe_load((Path(__file__).parents[1]/'configs/stage21_graph_benchmark.yaml').read_text())
    validate_config(cfg)
    changed = copy.deepcopy(cfg)
    changed['datasets'].pop('aqi36')
    with pytest.raises(ValueError, match='scope'):
        validate_config(changed)
    changed = copy.deepcopy(cfg)
    changed['resources']['admission_policy'] = 'memory_cap_only'
    with pytest.raises(ValueError, match='resource'):
        validate_config(changed)


def test_no_preparation_while_stage20_active(tmp_path):
    (tmp_path/'activity.json').write_text(json.dumps({'worker': {'pid': 1}}))
    (tmp_path/'launcher.json').write_text(json.dumps({'state': 'running'}))
    with pytest.raises(ValueError, match='explicitly stopped'):
        stage20_stopped(tmp_path)


@pytest.mark.parametrize('identical', [True, False])
def test_audit_never_passes_without_exclusion_proof(tmp_path, monkeypatch, identical):
    pytest.importorskip('tables')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    cfg = yaml.safe_load((Path(__file__).parents[1]/'configs/stage21_graph_benchmark.yaml').read_text())
    previous, parent, sources, raw, output = [tmp_path/p for p in ('previous', 'parent', 'data/sources', 'new/raw', 'out')]
    for p in (previous, parent, sources, raw):
        p.mkdir(parents=True)
    (previous/'activity.json').write_text(json.dumps({'worker': None}))
    (previous/'launcher.json').write_text(json.dumps({'state': 'stopped_by_user'}))
    Path(str(previous)+'.launch.json').write_text(json.dumps({'pid': 999999999, 'start_ticks': '1'}))
    times = pd.date_range('2014-05-01', periods=500, freq='h')
    source = sources/'aq_shunyi.npz'
    np.savez(source, values=np.arange(500, dtype=float)[:, None], mask=np.ones((500, 1), bool),
        times=times.asi8)
    source.with_suffix('.json').write_text(json.dumps({'variables': ['PM2.5'],
        'provenance': {'station': 'Shunyi'}}))
    target = frame(np.arange(200, dtype=float)+(.01 if not identical else 0))
    target.to_hdf(raw/'small36.h5', key='pm25')
    pd.DataFrame({'latitude': [40.]*36, 'longitude': [116.]*36}, index=target.columns).to_hdf(
        raw/'small36.h5', key='stations', mode='a')
    (raw/'full437.h5').write_bytes(b'only hashed; never used as a second target')
    (raw/'air_quality.zip').write_bytes(b'archive fixture')
    (parent/'manifest.json').write_text(json.dumps({'source_configuration': {
        'data': {'directory': str(sources.parent)}}, 'source_raw_hashes': {'aq_shunyi': sha(source)}}))
    # Checkpoint binding remains strict; mocking the hash here avoids a multi-MB fixture.
    from gofa_f import graph_benchmark
    real_sha = graph_benchmark.sha
    monkeypatch.setattr(graph_benchmark, 'sha', lambda p: cfg['adapter_sha256']
        if Path(p).name == 'adapter.pt' else real_sha(p))
    cfg.update(stage20_output=str(previous), stage19_output=str(parent),
        stage19_manifest_sha256=sha(parent/'manifest.json'), data_directory=str(raw.parent))
    result = audit(cfg, output)
    assert result['state'] == 'blocked_data_audit'
    assert result['reason'] == ('overlap_evidence_detected' if identical else 'source_target_overlap_unresolved')
    assert not result['gpu_started'] and result['target_predictions'] == 0
    assert not (output/'manifest.json').exists() and not (output/'adapter.pt').exists()
    assert not result['stations_removed'] and not result['dataset_replaced']
