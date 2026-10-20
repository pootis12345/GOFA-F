"""Stage23 original-data audit; source eligibility never reads prediction errors."""
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from gofa_f.generative_data import save_series, sha, directed_roads
from gofa_f.graph_benchmark_data import geo_graph
from gofa_f.prefix_cache import atomic_json

VIEWS = {'pems04_flow': (307, 12, 12), 'pems04_speed': (307, 12, 12),
         'boston2017': (16, 48, 24), 'penmanshiel2017': (14, 48, 24)}
NCEI = 'https://www.ncei.noaa.gov/pub/data/noaa/'
PENMANSHIEL = {
    'Penmanshiel_SCADA_2017_WT01-10_3114.zip': 'c8279e34a5e33d688e2219aa9abab74e',
    'Penmanshiel_SCADA_2017_WT11-15_3115.zip': 'd41874e3d195e88506949492b5758cf6',
    'Penmanshiel_WT_static.csv': 'c4cd4191234c1a67a391fe5d2978256b',
    'Penmanshiel_WT_dataSignalMapping.xlsx': '2eae1b84a014619268a6e4f53e6b1c91'}


def download(url, path, md5=None):
    """Atomic download, fail closed on HTTP/checksum errors; never replace originals."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_name(path.name+'.part')
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        req = urllib.request.Request(url, headers={'User-Agent': 'GOFA-F-Stage23-research'})
        with opener.open(req, timeout=45) as source, temporary.open('wb') as out:
            while chunk := source.read(2**20):
                out.write(chunk)
        if md5 and file_md5(temporary) != md5:
            raise ValueError('Publisher MD5 mismatch: '+path.name)
        temporary.replace(path)
    if md5 and file_md5(path) != md5:
        raise ValueError('Publisher MD5 mismatch: '+path.name)
    return path


def file_md5(path):
    value = hashlib.md5()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            value.update(chunk)
    return value.hexdigest()


def distance_km(a, b):
    a, b = np.radians(a), np.radians(b)
    x = np.sin((a[0]-b[0])/2)**2+np.cos(a[0])*np.cos(b[0])*np.sin((a[1]-b[1])/2)**2
    return float(6371.0088*2*np.arcsin(np.sqrt(np.clip(x, 0, 1))))


def station_candidates(metadata, index):
    available = set(re.findall(r'(\d{6}-\d{5})-2017\.gz', index))
    rows = []
    for row in csv.DictReader(io.StringIO(metadata)):
        key = row['USAF']+'-'+row['WBAN']
        if key not in available or row['BEGIN'] > '20171231' or row['END'] < '20170101':
            continue
        try:
            coordinates = [float(row['LAT']), float(row['LON'])]
        except ValueError:
            continue
        distance = distance_km([42.3601, -71.0589], coordinates)
        if distance <= 150:
            rows.append(dict(id=key, name=row['STATION NAME'], coordinates=coordinates,
                             distance_km=distance, begin=row['BEGIN'], end=row['END']))
    return sorted(rows, key=lambda r: (r['distance_km'], r['id']))


def isd_temperature(content):
    rows = []
    for line in content.splitlines():
        fields = line.split()
        if len(fields) != 12:
            raise ValueError('ISD-Lite must have twelve documented fields')
        year, month, day, hour, temperature = map(int, fields[:5])
        if year != 2017 or not 0 <= hour <= 23:
            raise ValueError('Unverified ISD year/hour')
        rows.append((pd.Timestamp(year=year, month=month, day=day, hour=hour),
                     np.nan if temperature == -9999 else temperature/10))
    frame = pd.DataFrame(rows, columns=['time', 'temperature'])
    if frame.time.duplicated().any():
        raise ValueError('Duplicate publisher ISD hours require explicit audit')
    if not frame.time.is_monotonic_increasing:
        raise ValueError('Nonmonotonic publisher ISD hours')
    return frame.set_index('time').temperature.reindex(pd.date_range('2017-01-01', '2017-12-31 23:00', freq='h'))


def prepare_boston(root):
    root, raw = Path(root), Path(root)/'raw/noaa'
    metadata = download(NCEI+'isd-history.csv', raw/'isd-history.csv')
    index = download(NCEI+'isd-lite/2017/', raw/'index2017.html')
    for name in ('isd-lite-format.txt', 'isd-lite-technical-document.txt'):
        download(NCEI+'isd-lite/'+name, raw/name)
    candidates = station_candidates(metadata.read_text(), index.read_text())
    accepted, columns, audit = [], [], []
    for row in candidates:
        # Metadata duplicate filtering does not inspect held-out observations.
        if any(distance_km(row['coordinates'], v['coordinates']) < 1 for v in accepted):
            audit.append(dict(**row, selected=False, reason='within_1km_of_selected_site'))
            continue
        path = download(NCEI+'isd-lite/2017/'+row['id']+'-2017.gz', raw/(row['id']+'-2017.gz'))
        series = isd_temperature(gzip.decompress(path.read_bytes()).decode('ascii'))
        coverage = float(series.iloc[:int(.6*len(series))].notna().mean())
        record = dict(**row, training_coverage=coverage, sha256=sha(path), selected=coverage >= .9)
        audit.append(record)
        print(json.dumps(dict(stage='boston_station_audit', **record)), flush=True)
        if coverage >= .9:
            accepted.append(record)
            columns.append(series.rename(row['id']))
        if len(accepted) == 16:
            break
    atomic_json(root/'boston_station_audit.json', dict(candidates=len(candidates), examined=audit,
        selected=accepted, passed=len(accepted) == 16, selection_uses_first60_percent_only=True))
    if len(accepted) != 16:
        raise ValueError('Preparation blocked: fewer than16 eligible Boston stations; no substitution')
    frame = pd.concat(columns, axis=1)
    positions = [r['coordinates'] for r in accepted]
    return save_series(root/'sources', 'boston2017', frame.to_numpy(), frame.index, list(frame.columns),
        dict(source=NCEI+'isd-lite/2017/', field=5, raw_scale=10, unit='Celsius', missing_sentinel=-9999,
             zeros_preserved=True, imputation='none', coordinates=positions,
             clock='UTC publisher nominal whole-hour labels; near-hour observations, not hourly means',
             publisher_preprocessing='ISD-Lite selected rounded-hour observations; QC flags not retained',
             selection=accepted, station_audit_sha256=sha(root/'boston_station_audit.json'),
             document_hashes={p.name: sha(p) for p in (metadata, index, raw/'isd-lite-format.txt',
                                                       raw/'isd-lite-technical-document.txt')},
             converter_sha256=sha(Path(__file__))), geo_graph(positions, frame.columns))


def prepare_penmanshiel(root):
    root, raw = Path(root), Path(root)/'raw/penmanshiel'
    for name, digest in PENMANSHIEL.items():
        print(json.dumps(dict(stage='wind_download', file=name)), flush=True)
        download('https://zenodo.org/records/8253010/files/'+name+'?download=1', raw/name, digest)
    metadata = pd.read_csv(raw/'Penmanshiel_WT_static.csv')
    mapping = pd.read_excel(raw/'Penmanshiel_WT_dataSignalMapping.xlsx', sheet_name=None)
    evidence = {k: v.fillna('').astype(str).values.tolist() for k, v in mapping.items()}
    power_mapping = [r for rows in evidence.values() for r in rows if len(r) >= 4
                     and r[1] == 'Power' and r[2] == 'Actual value of active power' and r[3] == 'kW']
    if len(power_mapping) != 1 or float(power_mapping[0][0]) != 5:
        raise ValueError('Publisher mapping must uniquely bind signal5 to active Power in kW')
    atomic_json(root/'wind_signal_mapping.json', evidence)
    columns = {}
    schemas = []
    for name in PENMANSHIEL:
        if not name.endswith('.zip'):
            continue
        with zipfile.ZipFile(raw/name) as archive:
            names = sorted(n for n in archive.namelist() if 'Turbine_Data' in Path(n).name and n.endswith('.csv'))
            for member in names:
                with archive.open(member) as stream:
                    prefix = [stream.readline().decode('utf-8-sig') for _ in range(15)]
                header = next((i for i, line in enumerate(prefix) if 'Date and time' in line and 'Power (kW)' in line), None)
                if header is None:
                    raise ValueError('Unverified Penmanshiel mean active power/time schema')
                if not any('Time zone: UTC' in line for line in prefix) or not all(
                        f'Power, {statistic} (kW)' in prefix[header]
                        for statistic in ('Standard deviation', 'Minimum', 'Maximum')):
                    raise ValueError('Publisher UTC and mean/SD/min/max power-family schema required')
                with archive.open(member) as stream:
                    frame = pd.read_csv(stream, skiprows=header,
                        usecols=lambda c: c.strip('# ') in ['Date and time', 'Power (kW)'])
                frame.columns = [c.strip('# ') for c in frame.columns]
                index = pd.to_datetime(frame['Date and time'], errors='raise')
                turbine = re.search(r'Penmanshiel_(\d+)_2017', member)
                if not turbine or index.duplicated().any() or not index.is_monotonic_increasing \
                        or not index.dt.year.eq(2017).all() or len(frame.columns) != 2:
                    raise ValueError('Unverified Penmanshiel turbine/time mapping')
                key = str(int(turbine[1]))
                if key in columns:
                    raise ValueError('Duplicate turbine across archives')
                columns[key] = pd.Series(frame['Power (kW)'].to_numpy(), index=pd.DatetimeIndex(index), name=key)
                schemas.append(dict(member=member, header=prefix[header].strip(), turbine=key,
                                    rows=len(frame), negative_count=int((frame['Power (kW)'] < 0).sum())))
    ids = [str(i) for i in range(1, 16) if i != 3]
    if set(columns) != set(ids):
        raise ValueError('Exactly14 published turbines required, WT03 absent')
    ordered = metadata.set_index('Title').loc[[f'Penmanshiel {int(i):02d}' for i in ids]]
    positions = ordered[['Latitude', 'Longitude']].to_numpy()
    frame = pd.concat([columns[i] for i in ids], axis=1)
    grid = pd.date_range('2017-01-01', '2017-12-31 23:50', freq='10min')
    if not frame.index.isin(grid).all():
        raise ValueError('Off-grid Penmanshiel timestamps require audit')
    frame = frame.reindex(grid)
    atomic_json(root/'wind_field_audit.json', dict(passed=True, schemas=schemas,
        signal_mapping_sha256=sha(root/'wind_signal_mapping.json'), turbines=ids,
        mean_power_column='Power (kW)', primary_scada_mapping=power_mapping[0],
        statistic_evidence='Unqualified10-minute Power column with separate SD/min/max; active-power signal5',
        independent_primary_research_mean_description=
            'https://www.research.unipd.it/retrieve/23e39655-5bbb-4051-934e-6ae1c9eb2275/1-s2.0-S0306261925007834-main.pdf',
        negative_finite_power_preserved=True))
    return save_series(root/'sources', 'penmanshiel2017', frame.to_numpy(), grid, ids,
        dict(source='Zenodo8253010 version0.1.0', unit='kW', channel='Power (kW)',
             coordinates=positions.tolist(), raw_md5=PENMANSHIEL,
             raw_sha256={n: sha(raw/n) for n in PENMANSHIEL},
             imputation='none; absent timestamps masked on original10-minute grid',
             clock='publisher Date and time UTC; no inferred offset',
             future_status_or_events=False, negative_finite_power_preserved=True,
             field_audit_sha256=sha(root/'wind_field_audit.json'), converter_sha256=sha(Path(__file__))),
        geo_graph(positions, ids))


def prepare_traffic(root, old_raw):
    raw = Path(old_raw)
    path, roads = raw/'PEMS04.npz', raw/'PEMS04.csv'
    if sha(path) != '95a3c9b720fffdb85f0330d09bfab41b0b3cad0ca86c0d7d5f3accacb4ac999a' \
            or sha(roads) != '68ffaa80447eb34428a9999dc28363c61406ec0fd714686d0582d2386ac6f5f7':
        raise ValueError('Pinned original PEMS04 inputs changed')
    with np.load(path, allow_pickle=False) as source:
        values = source['data']
    if values.shape != (16992, 307, 3):
        raise ValueError('Original three-channel PEMS04 required')
    graph = directed_roads(pd.read_csv(roads), 307)
    records = {}
    for name, channel in [('pems04_flow', 0), ('pems04_speed', 2)]:
        records[name] = save_series(Path(root)/'sources', name, values[:, :, channel],
            pd.date_range('2018-01-01', periods=len(values), freq='5min'), list(map(str, range(307))),
            dict(source='ASTGNN pinned original PEMS04 release', channel_index=channel,
                 channel='flow' if channel == 0 else 'speed',
                 unit='publisher native speed unit; no unverified conversion' if channel else 'vehicles per5-minute bin',
                 channel_order=['flow', 'occupancy', 'speed'], original_zero_values_preserved=True,
                 preprocessing='Publisher preprocessing/imputation retained; original missing mask unavailable',
                 raw_sha256={path.name: sha(path), roads.name: sha(roads)},
                 topology='original direction; receiver rows / sender columns', converter_sha256=sha(Path(__file__))), graph)
    return records


def prepare_sources(root, old_raw):
    Path(root).mkdir(parents=True, exist_ok=True)
    records = prepare_traffic(root, old_raw)
    records['boston2017'] = prepare_boston(root)
    records['penmanshiel2017'] = prepare_penmanshiel(root)
    return records
