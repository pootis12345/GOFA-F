"""CPU-only original-data conversion for the six fixed Stage21 v2 targets."""
import io
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from gofa_f.generative_data import save_series, sha


TARGETS = {
    'pems03': ('traffic', 358, 12, 12),
    'pems_bay': ('traffic', 325, 12, 12),
    'meteonet_nw2018': ('environment', None, 48, 24),
    'engrad': ('environment', 487, 48, 24),
    'sdwpf': ('wind', 134, 48, 24),
    'kelmarsh2019': ('wind', 6, 48, 24),
}


def geo_graph(coordinates, ids, geographic=True):
    coords = np.asarray(coordinates, np.float64)
    if coords.shape != (len(ids), 2) or not np.isfinite(coords).all() or len(ids) < 2:
        raise ValueError('Finite ordered coordinates for every node required')
    if geographic:
        if (np.abs(coords[:, 0]) > 90).any() or (np.abs(coords[:, 1]) > 180).any():
            raise ValueError('Coordinates must be verified decimal degrees')
        lat, lon = np.radians(coords).T
        distance = np.sin((lat[:, None]-lat[None])/2)**2 + np.cos(lat[:, None])*np.cos(lat[None]) \
            * np.sin((lon[:, None]-lon[None])/2)**2
    else:
        distance = ((coords[:, None]-coords[None])**2).sum(-1)
    graph = np.zeros((len(ids), len(ids)), np.float32)
    for i in range(len(ids)):
        order = sorted((j for j in range(len(ids)) if j != i), key=lambda j: (distance[i, j], str(ids[j])))
        graph[i, order[:4]] = 1
    return np.maximum(graph, graph.T)


def road_graph(frame, ids, gaussian=False):
    if list(frame.columns[:2]) != ['from', 'to']:
        raise ValueError('Unverified original road schema')
    mapping = {int(v): i for i, v in enumerate(ids)}
    distance = np.full((len(ids), len(ids)), np.inf)
    skipped = 0
    for sender, receiver, cost in frame.iloc[:, :3].itertuples(index=False, name=None):
        if int(sender) not in mapping or int(receiver) not in mapping:
            skipped += 1
            continue
        if not np.isfinite(cost) or cost < 0:
            raise ValueError('Invalid published road cost')
        distance[mapping[int(sender)], mapping[int(receiver)]] = float(cost)
    if gaussian:
        sigma = distance[np.isfinite(distance)].std()
        if sigma <= 0:
            raise ValueError('Invalid official distance scale')
        adjacency = np.exp(-(distance/sigma)**2) >= .1
    else:
        adjacency = np.isfinite(distance)
    np.fill_diagonal(adjacency, False)
    # Files use sender->receiver. OriginGraph expects receiver rows.
    return adjacency.T.astype(np.float32), skipped


def extract_one(archive, basename, directory):
    with zipfile.ZipFile(archive) as z:
        matches = [n for n in z.namelist() if Path(n).name == basename]
        if len(matches) != 1:
            raise ValueError('Expected one original archive member: '+basename)
        target = Path(directory)/basename
        if not target.exists():
            temporary = target.with_suffix(target.suffix+'.part')
            with z.open(matches[0]) as source, temporary.open('wb') as output:
                import shutil
                shutil.copyfileobj(source, output)
            temporary.replace(target)
        return target


def save(root, name, frame, graph, provenance):
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing \
            or frame.columns.has_duplicates or not np.isfinite(graph).all() \
            or graph.shape != (len(frame.columns), len(frame.columns)):
        raise ValueError('Synchronized identities/timestamps/topology are invalid')
    if frame.index.tz is not None:
        frame = frame.copy()
        frame.index = frame.index.tz_convert('UTC').tz_localize(None)
    expected = TARGETS[name][1]
    if expected is not None and frame.shape[1] != expected:
        raise ValueError('Published node count differs: '+name)
    provenance.update(converter_sha256=sha(Path(__file__)), external_graph_binarized=True,
        topology='receiver_rows_sender_columns; no target-value fitting',
        numerical_imputation='none', zeros_preserved=True)
    result = save_series(Path(root)/'targets', name, frame.to_numpy(), frame.index,
        list(map(str, frame.columns)), provenance, graph)
    return result


def convert_roads(root, name):
    raw = Path(root)/'raw'
    archive = raw/(name+'.zip')
    if name == 'pems03':
        with zipfile.ZipFile(archive) as z:
            ids = [v.strip() for v in z.read('index.txt').decode().splitlines() if v.strip()]
            with np.load(io.BytesIO(z.read('pems03.npz')), allow_pickle=False) as data:
                values = data['data'][:, :, 0]
            roads = pd.read_csv(io.BytesIO(z.read('distances.csv')))
        frame = pd.DataFrame(values, index=pd.date_range('2018-09-01', periods=len(values), freq='5min'), columns=ids)
        graph, skipped = road_graph(roads, ids)
        note = 'Original release already imputed; raw imputation mask unavailable; zero flow is observed'
        unit, channel = 'vehicles per 5-minute bin', 'flow'
    else:
        frame = pd.read_hdf(extract_one(archive, 'pems_bay.h5', raw))
        if frame.index.has_duplicates:
            raise ValueError('Duplicate original Bay timestamps')
        frame = frame.reindex(pd.date_range(frame.index[0], frame.index[-1], freq='5min'))
        # DCRNN/TSL zero-speed sentinel is missing; do not forward/back fill it.
        frame = frame.mask(frame.eq(0))
        with zipfile.ZipFile(archive) as z:
            roads = pd.read_csv(io.BytesIO(z.read('distances_bay.csv')))
        graph, skipped = road_graph(roads, frame.columns, gaussian=True)
        note = 'Published DCRNN exp(-(distance/std)^2)>=0.1 support, binarized; zero sentinel masked'
        unit, channel = 'mph', 'speed'
    return save(root, name, frame, graph, dict(raw_sha256={archive.name: sha(archive)},
        source='TSL ProcessedDatasets v1.0.0 author distribution', channel=channel, unit=unit,
        graph_note=note, unmapped_distance_rows=skipped, same_entity_other_times_possible=True))


def convert_engrad(root):
    path = Path(root)/'raw/engrad.h5'
    if hashlib.md5(path.read_bytes()).hexdigest() != 'e311f7587447154859ce361eea51409a':
        raise ValueError('EngRAD Zenodo official checksum differs')
    frame, metadata = pd.read_hdf(path, 'data'), pd.read_hdf(path, 'metadata')
    frame = frame.xs('temperature_2m', axis=1, level=1).loc[:, metadata.index]
    coordinates = metadata[['lat', 'lon']].to_numpy()
    graph = geo_graph(coordinates, frame.columns)
    return save(root, 'engrad', frame, graph, dict(raw_sha256={path.name: sha(path)},
        source='Zenodo 12760772 version1.0.0, 2018-2020', unit='Celsius', channel='temperature_2m',
        data_kind='ECMWF IFS model-derived historical field, not independent station measurements',
        coordinates=coordinates.tolist(), clock='published synchronous index; no unverified offset'))


def aggregate_meteonet(chunks):
    """Hourly ending labels; [h-1h,h), >=8 of ten six-minute samples, no fill."""
    summaries, coordinates, last, changes = [], {}, {}, {}
    off_grid, duplicate_rows, conflicting = 0, 0, 0
    pending = None

    def consume(raw, final=False):
        nonlocal pending, off_grid, duplicate_rows, conflicting
        chunk = raw.copy()
        if not final:
            chunk['number_sta'] = chunk['number_sta'].astype(str)
            chunk['date'] = pd.to_datetime(chunk['date'], errors='raise')
            if not chunk['date'].dt.year.eq(2018).all():
                raise ValueError('Expected original2018 observations')
            off_grid += int(((chunk['date'].astype('int64') % (6*60*10**9)) != 0).sum())
            if pending is not None:
                chunk = pd.concat([pending, chunk], ignore_index=True)
        # Duplicate original timestamps occur in the release. Collapse equal
        # finite values; conflicting finite readings become missing, never an
        # averaged concentration/temperature. Delayed last rows also catch
        # duplicates crossing chunk boundaries.
        keys = ['number_sta', 'date']
        duplicated = chunk.duplicated(keys, keep=False)
        if duplicated.any():
            group = chunk.loc[duplicated].groupby(keys, sort=False)['t'].agg(['first', 'nunique'])
            unique = chunk.loc[duplicated].drop_duplicates(keys).set_index(keys)
            unique['t'] = group['first'].where(group['nunique'] <= 1)
            duplicate_rows += int(duplicated.sum()-len(unique))
            conflicting += int((group['nunique'] > 1).sum())
            chunk = pd.concat([chunk.loc[~duplicated], unique.reset_index()], ignore_index=True)
        chunk = chunk.sort_values(['date', 'number_sta'], kind='stable')
        if not final:
            hold = chunk.groupby('number_sta', sort=False).tail(1).index
            pending = chunk.loc[hold].copy()
            chunk = chunk.drop(index=hold)
        for station, rows in chunk.groupby('number_sta', sort=False):
            if station in last and rows['date'].iloc[0] <= last[station]:
                raise ValueError('Nonlocal duplicate timestamp needs audit; no silent overwrite')
            last[station] = rows['date'].iloc[-1]
            pairs = rows[['lat', 'lon']].drop_duplicates().to_numpy(dtype=np.float64)
            coordinates.setdefault(station, pairs[0])
            for pair in pairs:
                if not np.allclose(coordinates[station], pair, rtol=0, atol=1e-5):
                    changes.setdefault(station, set()).add(tuple(map(float, pair)))
        chunk['hour'] = chunk['date'].dt.floor('h')+pd.Timedelta(hours=1)
        summaries.append(chunk.groupby(['hour', 'number_sta'])['t'].agg(['sum', 'count']))

    for chunk in chunks:
        consume(chunk)
    if pending is not None:
        consume(pending, final=True)
    totals = pd.concat(summaries).groupby(level=[0, 1]).sum()
    values = (totals['sum']/totals['count']).where(totals['count'] >= 8)-273.15
    frame = values.unstack('number_sta').reindex(pd.date_range('2018-01-01 01:00', '2019-01-01', freq='h'))
    # Sensors which never measure this channel are not temperature nodes.
    selected = [c for c in sorted(frame.columns, key=str) if frame[c].iloc[:int(.6*len(frame))].notna().any()]
    frame = frame.loc[:, selected]
    frame.attrs['later_coordinate_variants'] = {s: sorted(map(list, p)) for s, p in changes.items() if s in selected}
    frame.attrs['original_off_six_minute_rows'] = off_grid
    frame.attrs['duplicate_rows_collapsed'] = duplicate_rows
    frame.attrs['conflicting_duplicate_times_masked'] = conflicting
    positions = np.asarray([coordinates[c] for c in selected])
    if len(selected) < 4 or not ((positions[:, 0] >= 43) & (positions[:, 0] <= 52)
            & (positions[:, 1] >= -6) & (positions[:, 1] <= 4)).all():
        raise ValueError('Original NW station coordinates are not confirmed decimal degrees')
    return frame, positions


def convert_meteonet(root):
    path = Path(root)/'raw/meteonet_NW2018.csv.zip'
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if Path(n).name == 'NW_Ground_Stations_2018.csv']
        if len(names) != 1:
            raise ValueError('Expected original complete NW 2018 observations, not a sample')
        with z.open(names[0]) as stream:
            chunks = pd.read_csv(stream, usecols=['number_sta', 'lat', 'lon', 'date', 't'],
                dtype={'number_sta': str}, chunksize=200000)
            frame, positions = aggregate_meteonet(chunks)
    graph = geo_graph(positions, frame.columns)
    return save(root, 'meteonet_nw2018', frame, graph, dict(raw_sha256={path.name: sha(path)},
        source='Meteo-France official documentation-linked katerpillar/meteonet NW_Ground_Stations_2018.csv',
        channel='t', raw_unit='Kelvin', unit='Celsius', coordinates=positions.tolist(),
        coordinate_policy='static first published location before test; later location changes do not update graph',
        later_coordinate_variants=frame.attrs['later_coordinate_variants'],
        original_off_six_minute_rows=frame.attrs['original_off_six_minute_rows'],
        duplicate_rows_collapsed=frame.attrs['duplicate_rows_collapsed'],
        conflicting_duplicate_times_masked=frame.attrs['conflicting_duplicate_times_masked'],
        duplicate_policy='equal finite readings collapsed; conflicting readings masked; cross-chunk last rows delayed',
        aggregation='mean of original 6-minute values in [h-1h,h), hour-ending label; >=8 finite samples',
        node_eligibility='at least one valid temperature in first60%; no error-based selection',
        clock='publisher timestamps retained; no timezone offset inferred from forecasts'))


def convert_sdwpf(root):
    import pyarrow.parquet as pq
    raw = Path(root)/'raw'
    path = raw/'sdwpf_2001_2112_full.parquet'
    schema = pq.ParquetFile(path).schema_arrow.names
    # Verify the original schema rather than guessing an anonymous day origin.
    temporal = 'Tmstamp' if 'Tmstamp' in schema else None
    if temporal is None or 'TurbID' not in schema or 'Patv' not in schema:
        raise ValueError('Unverified dated SDWPF full schema: '+str(schema))
    data = pd.read_parquet(path, columns=['TurbID', temporal, 'Patv'])
    data[temporal] = pd.to_datetime(data[temporal], errors='raise')
    if data.duplicated(['TurbID', temporal]).any():
        raise ValueError('Duplicate SDWPF turbine times cannot be silently averaged')
    locations = pd.read_csv(raw/'sdwpf_turb_location_elevation.csv').set_index('TurbID')
    ids = sorted(locations.index)
    frame = data.pivot(index=temporal, columns='TurbID', values='Patv').loc[:, ids]
    if frame.index[0] != pd.Timestamp('2020-01-01 00:10') or frame.index[-1] != pd.Timestamp('2022-01-01 00:10'):
        raise ValueError('The approved SDWPF full 2020-2021 release is required')
    # The full release uses 15-minute observations in2020, then10-minute
    # observations in2021. Keep every original timestamp. save_series marks
    # non10-minute intervals as breaks; the frozen test tail is ten-minute.
    intervals = pd.Series(frame.index).diff().value_counts()
    test = frame.index[int(.8*len(frame))-48:]
    if not (np.diff(test.asi8) == 10*60*10**9).all():
        raise ValueError('Approved 48->24 ten-minute test windows are not regular')
    positions = locations.loc[ids, ['x', 'y']].to_numpy()
    graph = geo_graph(positions, ids, geographic=False)
    return save(root, 'sdwpf', frame, graph, dict(raw_sha256={p.name: sha(p) for p in
        [path, raw/'sdwpf_turb_location_elevation.csv']}, source='Figshare24798654 sdwpf_full',
        unit='kW', channel='Patv', coordinates=positions.tolist(),
        weather_or_future_status_inputs=False, negative_finite_power_preserved=True,
        timestamp_policy='all original timestamps preserved; 2020 15min, 2021 10min; no resampling',
        interval_counts={str(k): int(v) for k, v in intervals.items()},
        original_boundary_labels='release includes2022-01-01 00:00 and00:10 interval-end labels'))


def convert_kelmarsh(root):
    raw = Path(root)/'raw'
    path = raw/'Kelmarsh_SCADA_2019_3085.zip'
    if hashlib.md5(path.read_bytes()).hexdigest() != '14348a1572b722e1be10e53cd145c39a':
        raise ValueError('Kelmarsh2019 Zenodo official checksum differs')
    metadata = pd.read_csv(raw/'Kelmarsh_WT_static.csv')
    columns, ids = [], []
    with zipfile.ZipFile(path) as z:
        names = sorted(n for n in z.namelist() if 'Turbine_Data' in Path(n).name and n.endswith('.csv'))
        if len(names) != 6:
            raise ValueError('Exactly six original Kelmarsh2019 turbine data files required')
        for name in names:
            with z.open(name) as stream:
                prefix = [stream.readline().decode('utf-8-sig') for _ in range(15)]
            header = next((i for i, line in enumerate(prefix) if 'Date and time' in line and 'Power (kW)' in line), None)
            if header is None:
                raise ValueError('Unverified original Kelmarsh time/power schema')
            with z.open(name) as stream:
                table = pd.read_csv(stream, skiprows=header, usecols=lambda c: c.strip('# ') in ['Date and time', 'Power (kW)'])
            table.columns = [c.strip('# ') for c in table.columns]
            index = pd.to_datetime(table['Date and time'], errors='raise')
            if index.duplicated().any() or not index.is_monotonic_increasing or not index.dt.year.eq(2019).all():
                raise ValueError('Unverified/duplicate Kelmarsh2019 timestamps')
            turbine = next((str(i) for i in range(1, 7) if f'Kelmarsh_{i}_2019' in name), None)
            if turbine is None:
                raise ValueError('Turbine filename does not bind to the static identity')
            ids.append(turbine)
            columns.append(pd.Series(table['Power (kW)'].to_numpy(), index=pd.DatetimeIndex(index), name=turbine))
    frame = pd.concat(columns, axis=1).reindex(pd.date_range('2019-01-01', '2019-12-31 23:50', freq='10min'))
    ordered = metadata.set_index('Title').loc[['Kelmarsh '+i for i in ids]]
    positions = ordered[['Latitude', 'Longitude']].to_numpy()
    graph = geo_graph(positions, ids)
    return save(root, 'kelmarsh2019', frame, graph, dict(raw_sha256={p.name: sha(p) for p in
        [path, raw/'Kelmarsh_WT_static.csv']}, source='Zenodo16807551 v4, fixed2019 archive',
        unit='kW', channel='Power (kW)', coordinates=positions.tolist(),
        negative_finite_power_preserved=True, future_events_or_status_inputs=False))


def convert_all(root):
    root = Path(root)
    records = {}
    functions = dict(pems03=lambda: convert_roads(root, 'pems03'), pems_bay=lambda: convert_roads(root, 'pems_bay'),
        meteonet_nw2018=lambda: convert_meteonet(root), engrad=lambda: convert_engrad(root),
        sdwpf=lambda: convert_sdwpf(root), kelmarsh2019=lambda: convert_kelmarsh(root))
    for name, function in functions.items():
        print('converting '+name, flush=True)
        records[name] = function()
        print(json.dumps(dict(dataset=name, shape=records[name]['shape'], complete=True)), flush=True)
    return records
