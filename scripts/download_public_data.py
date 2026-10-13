"""Download public TSL-maintained raw archives on the remote server (no gated data)."""
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from gofa_f.data import file_sha256

REPOSITORY = "TorchSpatiotemporal/ProcessedDatasets"
FILES = ("traffic/metr_la.zip", "traffic/pems_bay.zip", "air/air_quality.zip")


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


if __name__ == "__main__":
    main()
