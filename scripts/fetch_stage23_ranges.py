"""Resume the two pinned Zenodo archives with checked, bounded HTTP byte ranges.

Only transport changes; final publisher MD5 remains mandatory before commit.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import time
import urllib.request

from fetch_stage23_sources import FILES, md5


def fetch(name, digest, root):
    path = root/name
    if path.exists():
        if md5(path) != digest:
            raise ValueError('Pinned archive changed: '+name)
        return
    url = 'https://zenodo.org/api/records/8253010/files/'+name+'/content'
    request = urllib.request.Request(url, headers={'Range': 'bytes=0-0'})
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 206 or not response.headers['Content-Range'].startswith('bytes 0-0/'):
            raise ValueError('Origin does not honor byte ranges')
        size = int(response.headers['Content-Range'].split('/')[-1])
        if len(response.read()) != 1:
            raise ValueError('Invalid ranged response size')
    parts = root/(name+'.ranges')
    parts.mkdir(parents=True, exist_ok=True)
    block = 4*2**20
    old = root/(name+'.part')
    if old.exists():
        with old.open('rb') as stream:
            for i in range(old.stat().st_size//block):
                own = parts/f'{i:05d}'
                chunk = stream.read(block)
                if not own.exists() and len(chunk) == block:
                    own.write_bytes(chunk)
    def one(i):
        start, end = i*block, min((i+1)*block, size)-1
        own = parts/f'{i:05d}'
        if own.exists() and own.stat().st_size == end-start+1:
            return
        error = None
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
                with urllib.request.urlopen(req, timeout=45) as r:
                    expected = f'bytes {start}-{end}/{size}'
                    if r.status != 206 or r.headers.get('Content-Range') != expected:
                        raise ValueError('Byte-range identity changed')
                    chunk = r.read(end-start+2)
                if len(chunk) != end-start+1:
                    raise ValueError('Truncated archive range')
                temporary = own.with_suffix('.tmp')
                temporary.write_bytes(chunk)
                temporary.replace(own)
                print(name, f'part{i+1}/{(size+block-1)//block}', flush=True)
                return
            except Exception as exc:
                error = exc
                time.sleep(min(attempt+1, 3))
        raise RuntimeError('Bounded range download failed: '+str(error))
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(one, range((size+block-1)//block)))
    temporary = root/(name+'.assembled')
    h = hashlib.md5()
    with temporary.open('wb') as output:
        for i in range((size+block-1)//block):
            chunk = (parts/f'{i:05d}').read_bytes()
            h.update(chunk)
            output.write(chunk)
    if temporary.stat().st_size != size or h.hexdigest() != digest:
        raise ValueError('Publisher MD5 mismatch; no archive committed')
    temporary.replace(path)
    print(name, 'MD5 PASSED', size, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    for name, digest in FILES.items():
        if name.endswith('.zip'):
            fetch(name, digest, root)


if __name__ == '__main__':
    main()
