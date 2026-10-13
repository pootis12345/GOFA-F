"""Download public TSL-maintained raw archives on the remote server (no gated data)."""
import json
import gzip
from pathlib import Path

from huggingface_hub import hf_hub_download
import numpy as np
import requests

from gofa_f.data import file_sha256

REPOSITORY = "TorchSpatiotemporal/ProcessedDatasets"
REVISION = "e7b92503354c28146e16bcb89bf75b7435b93808"
FILES = ("traffic/metr_la.zip", "traffic/pems_bay.zip", "air/air_quality.zip")
ENERGY_REPOSITORY = "laiguokun/multivariate-time-series-data"
ENERGY_REVISION = "7f402f185cc2435b5e66aed13a3b560ed142e023"
ENERGY_FILES = {"electricity": ("electricity/electricity.txt.gz", "1h"),
                "solar": ("solar-energy/solar_AL.txt.gz", "10min")}


def main():
    revision = REVISION
    manifest = {"repository": REPOSITORY, "revision": revision, "files": {}}
    for name in FILES:
        filename = hf_hub_download(REPOSITORY, name, repo_type="dataset", revision=revision,
                                   local_dir="data/raw/tsl")
        manifest["files"][name] = {"sha256": file_sha256(filename), "bytes": Path(filename).stat().st_size}
    output = Path("data/raw/tsl/download_manifest.json")
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    energy_revision = ENERGY_REVISION
    for name, (filename, frequency) in ENERGY_FILES.items():
        raw_path = Path("data/raw/energy") / Path(filename).name
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        source_url = f"https://raw.githubusercontent.com/{ENERGY_REPOSITORY}/{energy_revision}/{filename}"
        if not raw_path.exists():
            response = requests.get(source_url, timeout=120)
            response.raise_for_status()
            raw_path.write_bytes(response.content)
        with gzip.open(raw_path, "rt") as stream:
            values = np.loadtxt(stream, delimiter=",").astype(np.float32)
        target = Path("data/processed") / f"{name}.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            metadata = json.loads(target.with_suffix(".metadata.json").read_text())
            if (metadata["revision"] != energy_revision or metadata["source_sha256"] != file_sha256(raw_path)
                    or metadata["prepared_sha256"] != file_sha256(target)):
                raise ValueError(f"Existing data differs from recorded source: {target}")
            print(f"Verified existing: {target}", flush=True)
            continue
        mask = np.isfinite(values)
        np.savez_compressed(target, values=np.where(mask, values, 0), mask=mask)
        metadata = {"repository": ENERGY_REPOSITORY, "revision": energy_revision, "url": source_url,
                    "source_sha256": file_sha256(raw_path), "prepared_sha256": file_sha256(target),
                    "shape": list(values.shape), "frequency": frequency,
                    "note": "True zeros are retained; original matrix has no timestamps."}
        target.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
