"""Extract exact pinned UFGTime source without vendoring it into the project."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

REVISION = "61f55135bc2bd6e9b9e5d592009d358015eea681"
ARCHIVE_SHA256 = "61197707951d436ad6039f16717f1f7c6aa92cbdb8786063a1068eb8d643cabb"
FILES = {
    "src/model.py": "c7229e78ebb136570f783a62d4fd68760d8c36a5b93394a25429b26136de82b1",
    "src/utils.py": "3839d247121677327e415695c867c0ee18415e00b93efb121292de15a4950701",
    "main.py": "cb69e0a577aef24cb48e81abc2f1c0dffbd510a05e5094e6425713f17cc44a68",
    "argument.py": "ddf3eafd6d5bd2f11d6ece951d2369f7ac20441f074c0495c6174491df78605c",
    "config/ufgtime_config.yaml": "9ad855e09491b3603962444347b75285acdb04dc9cdd8fc8bce9b9fe84e11fbc",
    "README.md": "faad6d68ec9da31eb5a550c6d4c03bde09db9db6ac6a557996d818484d1a2619",
}


def prepare(archive, destination):
    archive, destination = Path(archive), Path(destination)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Pinned upstream archive hash mismatch")
    with zipfile.ZipFile(archive) as source:
        for name, digest in FILES.items():
            raw = source.read(f"UFGTIME-{REVISION}/{name}")
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError("Pinned upstream file hash mismatch: " + name)
            target = destination / name
            if target.exists() and target.read_bytes() != raw:
                raise ValueError("Refusing to replace modified upstream source: " + name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
    manifest = {"repository": "https://github.com/WonderHeiYi/UFGTIME", "revision": REVISION,
                "source_sha256": FILES, "source_changes": []}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("--destination", default="external/ufgtime")
    args = parser.parse_args()
    print(json.dumps(prepare(args.archive, args.destination), indent=2))
