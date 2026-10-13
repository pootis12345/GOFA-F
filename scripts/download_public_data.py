"""Download public TSL-maintained raw archives on the remote server (no gated data)."""
import json
import gzip
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
import numpy as np
import requests

from gofa_f.data import file_sha256

REPOSITORY = "TorchSpatiotemporal/ProcessedDatasets"
FILES = ("traffic/metr_la.zip", "traffic/pems_bay.zip", "air/air_quality.zip")
ENERGY_REPOSITORY = "laiguokun/multivariate-time-series-data"
ENERGY_FILES = {"electricity": ("electricity/electricity.txt.gz", "1h"),
                "solar": ("solar-energy/solar_AL.txt.gz", "10min")}


def main():
    revision = HfApi().dataset_info(REPOSITORY).sha
    manifest = {"repository": REPOSITORY, "revision": revision, "files": {}}
    for name in FILES:
        filename = hf_hub_download(REPOSITORY, name, repo_type="dataset", revision=revision,
                                   local_dir="data/raw/tsl")
        manifest["files"][name] = {"sha256": file_sha256(filename), "bytes": Path(filename).stat().st_size}
    output = Path("data/raw/tsl/download_manifest.json")
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    response = requests.get(f"https://api.github.com/repos/{ENERGY_REPOSITORY}/commits/master", timeout=60)
    response.raise_for_status()
    energy_revision = response.json()["sha"]
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
            raise FileExistsError(target)
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
