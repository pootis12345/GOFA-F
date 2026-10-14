"""Prepare pinned upstream artifacts on C; leaves the old training environment intact.

Create a separate Python 3.11 venv and install .[native,dev] before using this
script on a new machine. B-to-C offline transfers are also supported via --assets.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile


def download(url, target):
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, help="Directory holding hsdgnn.zip and chronos-bolt-mini/")
    args = parser.parse_args()
    assets = args.assets or Path(".remote/native-assets")
    revision = "3b83c6a8abb2229bc176c97ac71f7676bbf584d1"
    archive = assets / "hsdgnn.zip"
    if not args.assets:
        download(f"https://codeload.github.com/zhourongleiden/HSDGNN/zip/{revision}", archive)
    root = Path("external/hsdgnn")
    with zipfile.ZipFile(archive) as z:
        prefix = f"HSDGNN-{revision}/"
        for name in z.namelist():
            if name.startswith(prefix) and not name.endswith("/"):
                target = root / name[len(prefix):]
                if not target.resolve().is_relative_to(root.resolve()):
                    raise ValueError("Unsafe upstream archive path")
                content = z.read(name)
                if target.exists() and target.read_bytes() != content:
                    raise ValueError(f"Existing upstream source differs: {target}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
    bolt_revision = "251268337516a88e253628c43e1d26ec577b376b"
    weights = Path("data/weights/chronos-bolt-mini-2512683")
    hashes = {"config.json": "4299e93f30bb8d99d8f7dbaa39a1dfc6bf827ae67d4ef4c0b6ba4ca3df514980",
              "model.safetensors": "1a1a4297f132b808c5c7da24e3cce549d519c01ff5c2661fcad404baba018f24"}
    for name, expected in hashes.items():
        source = assets / "chronos-bolt-mini" / name
        if not args.assets:
            download(f"https://huggingface.co/amazon/chronos-bolt-mini/resolve/{bolt_revision}/{name}", source)
        content = source.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Weight hash mismatch: {name}")
        target = weights / name
        if target.exists() and target.read_bytes() != content:
            raise ValueError(f"Existing weight differs: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    (weights / "provenance.json").write_text(json.dumps({"repository": "amazon/chronos-bolt-mini",
                                                       "revision": bolt_revision}))
    print("Pinned upstream sources and weights prepared; old .venv untouched.")


if __name__ == "__main__":
    main()
