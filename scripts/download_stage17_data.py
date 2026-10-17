"""Public-file transport helper for B; does not run models or training.

Large GitHub objects use bounded ranges and are verified against the pinned Git
blob identity, avoiding stale proxy responses to whole-object requests.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import urllib.request
import time


def ranged(url, output, size, git_blob=None):
    output = Path(output)
    parts = output.parent/(output.name+".ranges")
    parts.mkdir(parents=True, exist_ok=True)
    chunk_size = 2**20
    def part(index):
        start, end = index*chunk_size, min(size, (index+1)*chunk_size)-1
        path = parts/str(index)
        if path.exists() and path.stat().st_size == end-start+1:
            return path
        address = url+("&" if "?" in url else "?")+f"stage17_part={index}&transport={time.time_ns()}"
        request = urllib.request.Request(address, headers={"User-Agent": "GOFA-F-Stage17",
                                                          "Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request, timeout=40) as response:
            if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                raise ValueError("Range response identity differs")
            content = response.read(end-start+2)
        if len(content) != end-start+1:
            raise ValueError("Truncated public file range")
        path.write_bytes(content)
        print(f"{output.name} {end+1}/{size}", flush=True)
        return path
    with ThreadPoolExecutor(max_workers=6) as pool:
        files = list(pool.map(part, range((size+chunk_size-1)//chunk_size)))
    temporary = output.with_suffix(".joining")
    with temporary.open("wb") as stream:
        for path in files:
            stream.write(path.read_bytes())
    if git_blob:
        digest = hashlib.sha1(f"blob {size}\0".encode())
        with temporary.open("rb") as stream:
            while block := stream.read(2**20):
                digest.update(block)
        if digest.hexdigest() != git_blob:
            raise ValueError("Downloaded object differs from pinned official Git blob")
    temporary.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    ranged("https://raw.githubusercontent.com/guoshnBJTU/ASTGNN/"
           "9c2e19b98c4cedf1f35214d8789685b6381b3aad/data/PEMS04/PEMS04.npz",
           root/"PEMS04.npz", 32956284, "a1a39eac4b86963b1ac22874fd1cc693eb642cd2")


if __name__ == "__main__":
    main()
