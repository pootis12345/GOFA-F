"""Independently derived past-SCADA cache; never change Stage21 targets/masks."""
from pathlib import Path
import sys

import numpy as np

from gofa_f.allnode_stage import read
from gofa_f.generative_data import Series, sha
from gofa_f.prefix_cache import atomic_json
from gofa_f.scada_wind_models import CHANNELS

RAW_CHANNELS = ('Wspd', 'Etmp', 'Itmp', 'Pab1', 'Prtv', 'Wdir', 'Ndir')


def convert(data_directory, output, dependency_directory):
    output = Path(output)
    path = output/'scada.npz'
    raw = Path(data_directory)/'raw/sdwpf_2001_2112_full.parquet'
    target = Series(Path(data_directory)/'targets/sdwpf.npz')
    expected = target.meta['provenance']['raw_sha256'][raw.name]
    if sha(raw) != expected:
        raise ValueError('Audited original parquet changed')
    if path.exists():
        proof = read(output/'scada.json')
        if not proof or proof['raw_sha256'] != expected or proof['sha256'] != sha(path):
            raise ValueError('Derived SCADA cache binding changed')
        return proof
    # New isolated dependency directory; no package installation into old environments.
    sys.path.insert(0, str(Path(dependency_directory)))
    import pandas as pd
    import pyarrow
    import pyarrow.parquet as pq
    if pyarrow.__version__ != '17.0.0':
        raise ValueError('Pinned isolated PyArrow17 required')
    schema = pq.ParquetFile(raw).schema_arrow.names
    if not set(RAW_CHANNELS+('TurbID', 'Tmstamp', 'Patv')).issubset(schema):
        raise ValueError('Missing preregistered measured SCADA field')
    table = pd.read_parquet(raw, columns=['TurbID', 'Tmstamp', 'Patv', *RAW_CHANNELS])
    table['Tmstamp'] = pd.to_datetime(table['Tmstamp'], errors='raise')
    if table.duplicated(['TurbID', 'Tmstamp']).any():
        raise ValueError('Ambiguous SCADA turbine/timestamp duplicate')
    ids = [int(v) for v in target.ids]
    clock = pd.to_datetime(target.times)
    def aligned(channel):
        pivot = table.pivot(index='Tmstamp', columns='TurbID', values=channel)
        if list(pivot.index.asi8) != list(target.times) or list(pivot.columns) != ids:
            raise ValueError('SCADA timestamp/node mapping mismatch')
        return pivot.loc[clock, ids].to_numpy(dtype=np.float32)
    power = aligned('Patv')
    if not np.array_equal(np.isfinite(power), target.mask) \
            or not np.array_equal(np.where(np.isfinite(power), power, 0), target.values):
        raise ValueError('Raw power and frozen targets differ')
    channels = [aligned(c) for c in RAW_CHANNELS[:5]]
    for c in RAW_CHANNELS[5:]:
        degrees = aligned(c).astype(np.float64)
        channels.extend([np.sin(np.deg2rad(degrees)), np.cos(np.deg2rad(degrees))])
    values = np.stack(channels, axis=-1).astype(np.float32)
    values[~np.isfinite(values)] = np.nan
    output.mkdir(parents=True, exist_ok=True)
    temporary = output/'scada.part'
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, values=values)
    temporary.replace(path)
    proof = dict(raw_sha256=expected, sha256=sha(path), channels=list(CHANNELS),
        shape=list(values.shape), raw_schema=schema, timestamp_alignment_exact=True,
        frozen_power_exact=True, future_covariates=False, era5_used=False,
        no_target_mask_changes=True, no_outlier_or_negative_power_removal=True,
        source='https://pmc.ncbi.nlm.nih.gov/articles/PMC11187227/',
        field_units=dict(Wspd='m/s', Etmp='C', Itmp='C', Pab1='degrees',
                         Prtv='kW', Wdir='degrees relative', Ndir='degrees true north'),
        finite_fraction={n: float(np.isfinite(values[:, :, i]).mean()) for i, n in enumerate(CHANNELS)})
    atomic_json(output/'scada.json', proof)
    return proof
