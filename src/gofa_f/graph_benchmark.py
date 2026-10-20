"""Stage21 preparation gate. Overlap blocks the entire six-dataset experiment.

This entry deliberately has no GPU/optimizer imports. With the approved AQI-36
selection rejected, prediction implementation/deployment must not proceed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import yaml


DATASETS = {
    'pems03': ('traffic', 358, 12, 12),
    'pems_bay': ('traffic', 325, 12, 12),
    'aqi36': ('environment', 36, 48, 24),
    'engrad': ('environment', 487, 48, 24),
    'sdwpf': ('wind', 134, 48, 24),
    'kelmarsh2019': ('wind', 6, 48, 24),
}
ADAPTER_SHA = '816d38b577a6664f30dd6e6b6d50986a77b098ce9c4c0b0ffb63d70078b26f15'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(2**20):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def validate_config(cfg):
    expected = {name: dict(domain=domain, nodes=nodes, history=length, horizon=horizon)
                for name, (domain, nodes, length, horizon) in DATASETS.items()}
    if cfg['protocol'] != 'stage21-graph-benchmark-v1' or cfg['seed'] != 42 \
            or cfg['datasets'] != expected or cfg['queries_per_dataset'] != 128 \
            or cfg['adapter_sha256'] != ADAPTER_SHA \
            or cfg['conditions'] != ['native', 'gofa'] \
            or cfg['training_enabled'] or cfg['calibration_enabled']:
        raise ValueError('Approved six-dataset scope/checkpoint must not change')
    expected_resources = dict(admission_policy='idle', admission_samples=3,
        resource_check_seconds=10, memory_cap_gib=40, torch_cap_gib=38,
        activity_limit_seconds=None, wall_deadline=None, yield_on_foreign=False,
        leave_free_gib=0)
    if cfg['resources'] != expected_resources:
        raise ValueError('Approved idle admission/resource policy changed')


def compare_observations(target, source, shifts=range(-24, 25)):
    """Data identity check, NOT a forecasting score or an adopted time correction.

    Only the source training partition is supplied by the caller. Correlation
    and exact matches locate possible aliases/clock-label differences. A
    negative numeric match is never proof of absent exposure.
    """
    if target.index.has_duplicates or source.index.has_duplicates \
            or not target.index.is_monotonic_increasing or not source.index.is_monotonic_increasing:
        raise ValueError('Duplicate/unordered original timestamps cannot be silently repaired')
    if target.shape[1] != 36:
        raise ValueError('Expected the original full 36-node AQI selection')
    values = target.to_numpy(dtype=np.float64)
    rows = []
    for shift in shifts:
        aligned = source.reindex(target.index+pd.Timedelta(hours=shift)).to_numpy(dtype=np.float64)
        for i, station in enumerate(target.columns):
            observed = np.isfinite(values[:, i]) & np.isfinite(aligned)
            a, b = values[observed, i], aligned[observed]
            exact = a == b
            correlation = (float(np.corrcoef(a, b)[0, 1])
                if len(a) > 1 and a.std() > 0 and b.std() > 0 else None)
            distinct = int(len(np.unique(a)))
            suspicious = (len(a) >= 128 and distinct >= 20 and exact.sum() >= 128
                and exact.mean() >= .5 and correlation is not None and correlation >= .99)
            tail = observed.copy()
            tail[:int(.8*len(target))] = False
            tail_exact = (values[tail, i] == aligned[tail])
            witness = np.flatnonzero(observed & (values[:, i] == aligned))[:5]
            rows.append(dict(station=str(station), diagnostic_source_offset_hours=int(shift),
                compared=int(len(a)), exact_matches=int(exact.sum()),
                exact_fraction=float(exact.mean()) if len(a) else None,
                distinct_target_values=distinct, correlation=correlation,
                possible_same_observations=bool(suspicious),
                planned_test_compared=int(tail.sum()), planned_test_exact_matches=int(tail_exact.sum()),
                witnesses=[dict(target_timestamp=str(target.index[j]),
                    source_timestamp=str(target.index[j]+pd.Timedelta(hours=shift)),
                    pm25=float(values[j, i])) for j in witness]))
    return sorted(rows, key=lambda v: -(v['exact_fraction'] or 0))


def stage20_stopped(root):
    root = Path(root)
    ledger, launcher = read(root/'activity.json'), read(root/'launcher.json')
    if ledger.get('worker') is not None or launcher['state'] != 'stopped_by_user':
        raise ValueError('Stage20 must be explicitly stopped before Stage21 preparation')
    for record in (read(Path(str(root)+'.launch.json')),):
        stat = Path('/proc')/str(record['pid'])/'stat'
        if stat.exists():
            fields = stat.read_text().rsplit(')', 1)[1].split()
            if fields[19] == str(record['start_ticks']) and fields[0] != 'Z':
                raise ValueError('Stage20 supervisor still alive')
    if (root/'coefficient_lock.json').exists() or (root/'test_replay.json').exists():
        raise ValueError('Unexpected Stage20 coefficient/test completion; audit before proceeding')
    return dict(state=launcher['state'], worker=None, activity=ledger,
        coefficients_sealed=False, test_replayed=False)


def audit(cfg, output):
    validate_config(cfg)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Preparation must explicitly disable CUDA')
    output = Path(output)
    stop = stage20_stopped(cfg['stage20_output'])
    parent = Path(cfg['stage19_output'])
    if sha(parent/'manifest.json') != cfg['stage19_manifest_sha256'] \
            or sha(parent/'adapter.pt') != cfg['adapter_sha256']:
        raise ValueError('Source-selected checkpoint/provenance binding changed')
    prior = read(parent/'manifest.json')
    source = Path(prior['source_configuration']['data']['directory'])/'sources/aq_shunyi.npz'
    if sha(source) != prior['source_raw_hashes']['aq_shunyi']:
        raise ValueError('Previously accepted Shunyi source observations changed')
    metadata = read(source.with_suffix('.json'))
    if metadata['variables'][0] != 'PM2.5' or metadata['provenance']['station'] != 'Shunyi':
        raise ValueError('Source PM2.5/station identity differs')
    raw = Path(cfg['data_directory'])/'raw'
    raw_paths = [raw/'air_quality.zip', raw/'small36.h5', raw/'full437.h5']
    missing = [p.name for p in raw_paths if not p.exists()]
    if missing:
        value = dict(state='blocked_data_audit', reason='original_aqi_files_missing', missing=missing,
            gpu_started=False, target_predictions=0, automatic_resume_allowed=False)
        atomic(output/'audit.json', value)
        atomic(output/'status.json', value)
        return value
    target = pd.read_hdf(raw/'small36.h5', 'pm25')
    stations = pd.read_hdf(raw/'small36.h5', 'stations')
    if list(map(str, target.columns)) != list(map(str, stations.index)) \
            or list(stations.columns) != ['latitude', 'longitude'] \
            or not np.isfinite(stations.to_numpy()).all():
        raise ValueError('AQI station order or original geographic metadata differs')
    with np.load(source, allow_pickle=False) as data:
        cut = int(.6*len(data['values']))
        times = pd.to_datetime(data['times'][:cut])
        source_series = pd.Series(np.where(data['mask'][:cut, 0], data['values'][:cut, 0], np.nan),
            index=times)
    comparisons = compare_observations(target, source_series)
    matches = [v for v in comparisons if v['possible_same_observations']]
    # Even without a match the missing original station-ID exclusion proof is
    # unresolved, never "passed" solely because numeric arrays differ.
    verdict = 'overlap_evidence_detected' if matches else 'source_target_overlap_unresolved'
    value = dict(protocol=cfg['protocol'], state='blocked_data_audit', reason=verdict,
        at=time.time(), configuration_sha256=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        source_selected_update=3500, adapter_sha256=cfg['adapter_sha256'],
        raw_sha256={p.name: sha(p) for p in raw_paths}, source_raw_sha256=sha(source),
        original_aqi=dict(rows=len(target), nodes=target.shape[1],
            start=str(target.index[0]), end=str(target.index[-1]),
            missing_values=int(target.isna().to_numpy().sum()), imputation_applied=False,
            original_timestamps_modified=False),
        source_training_partition=dict(rows=cut, start=str(times[0]), end=str(times[-1])),
        overlap_evidence=matches, strongest_comparisons=comparisons[:10],
        matched_station_coordinates={v['station']: {
            k: float(stations.loc[next(s for s in stations.index if str(s) == v['station']), k])
            for k in stations.columns} for v in matches},
        interpretation='Clock offsets are audit hypotheses only; no shift is adopted for forecasting. '
            'These are raw-data identity diagnostics, not target predictions or exact adapter exposure counts.',
        stage20_stop_verified=stop, target_predictions=0, gpu_started=False, optimizer_updates=0,
        dataset_replaced=False, stations_removed=False, automatic_resume_allowed=False,
        remaining_work='Six data converters, query sealing and GPU deployment are suspended at the required overlap gate.')
    atomic(output/'audit.json', value)
    atomic(output/'status.json', dict(state=value['state'], reason=verdict, updated_at=value['at'],
        gpu_started=False, target_predictions=0, automatic_resume_allowed=False))
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--audit', action='store_true')
    modes.add_argument('--prepare', action='store_true')
    modes.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    value = read(Path(args.output)/'audit.json') if args.report_only else audit(cfg, args.output)
    print(json.dumps({k: value[k] for k in ('state', 'reason', 'gpu_started', 'target_predictions')}, indent=2))
    # A blocked preparation must never look like a successful launch preflight.
    return 2 if value['state'] == 'blocked_data_audit' else 0


if __name__ == '__main__':
    raise SystemExit(main())
