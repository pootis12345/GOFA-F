"""Bounded public downloads on B; C independently parses/audits these raw bytes.

This helper needs only the standard library. No model, CUDA or training imports.
"""
import argparse
import csv
from datetime import datetime
import gzip
import hashlib
import io
import math
from pathlib import Path
import re
import subprocess
import urllib.request


NCEI = 'https://www.ncei.noaa.gov/pub/data/noaa/'
FILES = {
    'Penmanshiel_SCADA_2017_WT01-10_3114.zip': 'c8279e34a5e33d688e2219aa9abab74e',
    'Penmanshiel_SCADA_2017_WT11-15_3115.zip': 'd41874e3d195e88506949492b5758cf6',
    'Penmanshiel_WT_static.csv': 'c4cd4191234c1a67a391fe5d2978256b',
    'Penmanshiel_WT_dataSignalMapping.xlsx': '2eae1b84a014619268a6e4f53e6b1c91'}


def md5(path):
    h = hashlib.md5()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, path, checksum=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_name(path.name+'.part')
        result = subprocess.run(['curl.exe', '--fail', '--location', '--connect-timeout', '20', '--max-time', '300',
                                 '--retry', '1', '--silent', '--show-error', '--output', str(temporary), url])
        if result.returncode == 35:
            # Windows revocation servers can be offline. Python still verifies
            # the HTTPS certificate chain and hostname; no TLS checks disabled.
            with urllib.request.urlopen(url, timeout=60) as source, temporary.open('wb') as out:
                while chunk := source.read(2**20):
                    out.write(chunk)
        elif result.returncode:
            raise subprocess.CalledProcessError(result.returncode, result.args)
        if checksum and md5(temporary) != checksum:
            raise ValueError('Publisher checksum mismatch: '+path.name)
        temporary.replace(path)
    if checksum and md5(path) != checksum:
        raise ValueError('Publisher checksum mismatch: '+path.name)
    return path


def distance(a, b):
    a, b = list(map(math.radians, a)), list(map(math.radians, b))
    x = math.sin((a[0]-b[0])/2)**2+math.cos(a[0])*math.cos(b[0])*math.sin((a[1]-b[1])/2)**2
    return 6371.0088*2*math.asin(math.sqrt(min(1, max(0, x))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    raw = root/'noaa'
    metadata = fetch(NCEI+'isd-history.csv', raw/'isd-history.csv')
    index = fetch(NCEI+'isd-lite/2017/', raw/'index2017.html')
    for name in ('isd-lite-format.txt', 'isd-lite-technical-document.txt'):
        fetch(NCEI+'isd-lite/'+name, raw/name)
    available = set(re.findall(r'(\d{6}-\d{5})-2017\.gz', index.read_text()))
    rows = []
    for row in csv.DictReader(io.StringIO(metadata.read_text())):
        key = row['USAF']+'-'+row['WBAN']
        if key not in available or row['BEGIN'] > '20171231' or row['END'] < '20170101':
            continue
        try:
            coordinates = [float(row['LAT']), float(row['LON'])]
        except ValueError:
            continue
        d = distance([42.3601, -71.0589], coordinates)
        if d <= 150:
            rows.append((d, key, coordinates))
    accepted = []
    for d, key, coordinates in sorted(rows):
        if any(distance(coordinates, v) < 1 for v in accepted):
            continue
        p = fetch(NCEI+'isd-lite/2017/'+key+'-2017.gz', raw/(key+'-2017.gz'))
        observed = set()
        for line in gzip.decompress(p.read_bytes()).decode('ascii').splitlines():
            fields = list(map(int, line.split()))
            if len(fields) != 12 or fields[0] != 2017:
                raise ValueError('Unverified original ISD format')
            date = datetime(*fields[:4])
            hour = int((date-datetime(2017, 1, 1)).total_seconds()/3600)
            if hour < 5256 and fields[4] != -9999:
                observed.add(hour)
        coverage = len(observed)/5256
        print(f'NOAA {key} distance={d:.3f}km train_coverage={coverage:.6f}', flush=True)
        if coverage >= .9:
            accepted.append(coordinates)
        if len(accepted) == 16:
            break
    if len(accepted) != 16:
        raise ValueError('Fewer than16 eligible Boston stations; stop, no replacement')
    for name, checksum in FILES.items():
        print('Downloading '+name, flush=True)
        fetch('https://zenodo.org/records/8253010/files/'+name+'?download=1', root/'penmanshiel'/name, checksum)
    print('Raw download complete; CPU scientific/field/isolation acceptance still required', flush=True)


if __name__ == '__main__':
    main()
