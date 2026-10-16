"""Resume public, officially hashed ChatTime weights over bounded parallel ranges.

Transport mirror is never trusted for provenance: the assembled SHA256 must equal
the official snapshot obtained on B. No CUDA, proxy configuration, or old files.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import threading
import time

import requests

REPO = "ChengsenWang/ChatTime-1-7B-Chat"
REVISION = "20baa88d6ab838b34abbd1d2b1f4cea57c8ff870"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for value in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(value)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="data/weights/chattime-7b-20baa88")
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("At most eight download connections")
    manifest = json.loads(Path(args.manifest).read_text())
    if manifest["repo"] != REPO or manifest["revision"] != REVISION:
        raise ValueError("Wrong official manifest")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    part_root = output/".parts"
    part_root.mkdir(exist_ok=True)
    block_size = 32*1024*1024
    pending, files = [], []
    transferred = [0]
    lock = threading.Lock()
    status_path = output/"download_progress.json"
    for name, record in manifest["files"].items():
        if "/" in name or "\\" in name or name.startswith("."):
            raise ValueError("Unsafe official filename")
        path = output/name
        if path.exists() and path.stat().st_size == record["size"] and sha(path) == record["sha256"]:
            transferred[0] += record["size"]
            continue
        directory = part_root/name
        directory.mkdir(exist_ok=True)
        old = path.with_name(name+".partial")
        if not old.exists() and path.exists() and path.stat().st_size < record["size"]:
            old = path
        prefix_size = old.stat().st_size if old.exists() else 0
        for index, start in enumerate(range(0, record["size"], block_size)):
            end = min(record["size"], start+block_size)
            part = directory/f"{index:05d}.part"
            if not part.exists() and prefix_size >= end:
                with old.open("rb") as source:
                    source.seek(start)
                    part.write_bytes(source.read(end-start))
            if part.exists() and part.stat().st_size == end-start:
                transferred[0] += end-start
            else:
                pending.append((name, start, end, part))
        files.append((name, record, directory))
    total = sum(v["size"] for v in manifest["files"].values())

    def save_progress(state):
        temp = status_path.with_suffix(".tmp")
        temp.write_text(json.dumps({"state": state, "completed_range_bytes": transferred[0],
            "total_bytes": total, "workers": args.workers, "updated_at": time.time()})+"\n")
        temp.replace(status_path)

    def one(item):
        name, start, end, part = item
        session = requests.Session()
        session.trust_env = False
        for attempt in range(12):
            temporary = part.with_suffix(".tmp")
            try:
                url = f"{args.endpoint}/{REPO}/resolve/{REVISION}/{name}?download=true"
                with session.get(url, headers={"Range": f"bytes={start}-{end-1}"},
                                 stream=True, timeout=(25, 90)) as response:
                    ranged = response.status_code == 206 and response.headers.get("Content-Range", "").startswith(
                        f"bytes {start}-{end-1}/")
                    whole_small_file = response.status_code == 200 and start == 0 and end == manifest["files"][name]["size"]
                    if not (ranged or whole_small_file):
                        raise ValueError("Server did not honor the registered byte range")
                    with temporary.open("wb") as stream:
                        for chunk in response.iter_content(1024*1024):
                            stream.write(chunk)
                if temporary.stat().st_size != end-start:
                    raise ValueError("Incomplete range")
                temporary.replace(part)
                with lock:
                    transferred[0] += end-start
                    save_progress("downloading")
                    if (transferred[0]//block_size) % 8 == 0:
                        print(json.dumps({"completed_bytes": transferred[0], "total_bytes": total}), flush=True)
                return
            except (requests.RequestException, OSError, ValueError) as exc:
                if attempt == 11:
                    raise RuntimeError(f"Download failed: {name} range {start}, {type(exc).__name__}") from None
                time.sleep(min(20, 2*(attempt+1)))
    save_progress("downloading")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, item) for item in pending]
        for future in as_completed(futures):
            future.result()
    for name, record, directory in files:
        target = output/name
        temporary = output/(name+".assembling")
        with temporary.open("wb") as stream:
            for part in sorted(directory.glob("*.part")):
                with part.open("rb") as source:
                    for block in iter(lambda: source.read(8*1024*1024), b""):
                        stream.write(block)
        if temporary.stat().st_size != record["size"] or sha(temporary) != record["sha256"]:
            raise ValueError("Assembled file differs from official SHA256: "+name)
        temporary.replace(target)
        print(json.dumps({"file": name, "official_sha256_verified": True}), flush=True)
    manifest["transport_endpoint"] = args.endpoint
    manifest["transport"] = "bounded eight-connection ranges; original official checksums"
    (output/"snapshot.json").write_text(json.dumps(manifest, indent=2)+"\n")
    # Reclaim only derived files in this new snapshot, after every official hash
    # has passed. Never recursively remove a directory or touch older weights.
    for name in manifest["files"]:
        directory = part_root/name
        if directory.exists():
            if not directory.resolve().is_relative_to(output):
                raise ValueError("Download temporary directory escapes this snapshot")
            for item in list(directory.glob("*.part"))+list(directory.glob("*.tmp")):
                item.unlink()
            if not any(directory.iterdir()):
                directory.rmdir()
        partial = output/(name+".partial")
        if partial.exists():
            partial.unlink()
    if part_root.exists() and not any(part_root.iterdir()):
        part_root.rmdir()
    save_progress("complete")
    print(json.dumps({"complete": True, "bytes": total}), flush=True)


if __name__ == "__main__":
    main()
