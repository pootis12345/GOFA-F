"""Download the registered public ChatTime snapshot; no torch or CUDA imports.

Run on B when C cannot reach Hugging Face, then transfer the verified directory.
Existing complete files are checked; interrupted shard downloads resume by Range.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

REPO = "ChengsenWang/ChatTime-1-7B-Chat"
REVISION = "20baa88d6ab838b34abbd1d2b1f4cea57c8ff870"
SOURCE = "8c2d4c209302d2b2bd6cc3c154586842341e3247"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def download(url, destination, expected=None, size=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (size is None or destination.stat().st_size == size):
        checksum = sha256(destination)
        if expected is None or checksum == expected:
            return checksum
        raise ValueError("Existing file checksum mismatch: "+str(destination))
    partial = destination.with_name(destination.name+".partial")
    if destination.exists() and size is not None and destination.stat().st_size < size and not partial.exists():
        # A stopped SCP transfer is a valid resumable prefix; final SHA still checks all bytes.
        destination.replace(partial)
    for attempt in range(12):
        offset = partial.stat().st_size if partial.exists() else 0
        try:
            request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
            with urllib.request.urlopen(request, timeout=120) as response:
                resumed = response.status == 206 and offset > 0
                if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Server returned an inconsistent Range")
                with partial.open("ab" if resumed else "wb") as stream:
                    last = time.monotonic()
                    while True:
                        chunk = response.read(8*1024*1024)
                        if not chunk:
                            break
                        stream.write(chunk)
                        if time.monotonic()-last >= 30:
                            print(json.dumps({"file": destination.name, "downloaded_bytes": stream.tell()}), flush=True)
                            last = time.monotonic()
            if size is not None and partial.stat().st_size != size:
                raise ValueError("Incomplete download: "+destination.name)
            checksum = sha256(partial)
            if expected is not None and checksum != expected:
                raise ValueError("Downloaded checksum mismatch: "+destination.name)
            partial.replace(destination)
            print(json.dumps({"file": destination.name, "complete": True, "sha256": checksum}), flush=True)
            return checksum
        except (OSError, ValueError, urllib.error.URLError) as exc:
            print(json.dumps({"file": destination.name, "attempt": attempt+1, "error": str(exc)}), flush=True)
            if attempt == 11:
                raise
            time.sleep(min(30, 2*(attempt+1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/weights/chattime-7b-20baa88")
    parser.add_argument("--source-output", default="external/ChatTime")
    parser.add_argument("--manifest", help="Previously verified official snapshot.json, for offline metadata")
    parser.add_argument("--endpoint", default="https://huggingface.co", help="Transport only; official hashes stay mandatory")
    parser.add_argument("--direct", action="store_true", help="Ignore proxy settings only inside this process")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.direct:
        urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
    original = json.loads(Path(args.manifest).read_text()) if args.manifest else None
    if original:
        if original["repo"] != REPO or original["revision"] != REVISION or original["source_revision"] != SOURCE:
            raise ValueError("Wrong official metadata snapshot")
        selected = [{"rfilename": name, "size": v["size"], "lfs": {"sha256": v["sha256"]}}
                    for name, v in original["files"].items()]
    else:
        if args.endpoint != "https://huggingface.co":
            raise ValueError("Mirror transport requires previously verified official file hashes")
        info = fetch_json(f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true")
        if info["sha"] != REVISION:
            raise ValueError("Wrong Hugging Face snapshot")
        selected = [v for v in info["siblings"] if v["rfilename"].endswith((".safetensors", ".json", ".model"))
                    and "/" not in v["rfilename"]]
    if not any(v["rfilename"].endswith(".safetensors") for v in selected):
        raise ValueError("No safetensors weights in registered snapshot")

    def one(record):
        name = record["rfilename"]
        lfs = record.get("lfs", {})
        expected = lfs.get("sha256")
        size = record.get("size", lfs.get("size"))
        if "/" in name or "\\" in name or name.startswith("."):
            raise ValueError("Unsafe snapshot filename")
        checksum = download(f"{args.endpoint}/{REPO}/resolve/{REVISION}/{name}",
                            output/name, expected, size)
        return name, {"sha256": checksum, "size": (output/name).stat().st_size,
                      "upstream_lfs_sha256": original["files"][name]["upstream_lfs_sha256"] if original else expected}

    with ThreadPoolExecutor(max_workers=2) as pool:
        records = dict(pool.map(one, selected))
    source = Path(args.source_output)
    source_records = {}
    for name in ["utils/tools.py", "utils/prompt.py", "model/model.py", "README.md", "requirements.txt"]:
        checksum = download(f"https://raw.githubusercontent.com/ForestsKing/ChatTime/{SOURCE}/{name}", source/name,
                            original["source_files"][name] if original else None)
        source_records[name] = checksum
    manifest = {"repo": REPO, "revision": REVISION, "files": records,
                "source_revision": SOURCE, "source_files": source_records, "transport_endpoint": args.endpoint}
    (output/"snapshot.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    (source/"snapshot.json").write_text(json.dumps({"revision": SOURCE, "files": source_records}, indent=2)+"\n",
                                       encoding="utf-8")
    print(json.dumps({"complete": True, "files": len(records), "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
