"""C-only Stage17 CPU regression and real-weight source mechanisms."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU preflight must explicitly disable CUDA")
    import torch
    import yaml
    from gofa_f.generative_checks import mechanisms
    from gofa_f.generative_data import SOURCES, Series, sha
    from gofa_f.generative_stage import load_model, prepare, read
    from gofa_f.prefix_cache import atomic_json
    torch.set_num_threads(2)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config).read_text())
    prepare(cfg, output)
    xml = output/"cpu_tests.xml"
    result = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "--disable-warnings",
                             "--junitxml="+str(xml)], check=False)
    if result.returncode:
        atomic_json(output/"cpu_preflight.json", {"passed": False, "phase": "regression"})
        raise SystemExit(result.returncode)
    tests = ET.parse(xml).getroot()
    cases = tests.findall(".//testcase")
    record = {"passed": False, "cases": len(cases),
              "skipped": sum(case.find("skipped") is not None for case in cases),
              "manifest_sha256": sha(output/"manifest.json"), "cuda_disabled": True}
    atomic_json(output/"cpu_preflight.json", record)
    root = Path(cfg["data"]["directory"])
    series = {name: Series(root/"sources"/(name+".npz")) for name in SOURCES}
    model, codec = load_model(cfg, "cpu", dtype=torch.float32)
    record["real_weight_dtype"] = "float32"
    try:
        record["real_weights"] = mechanisms(model, codec, series, read(root/"validation_queries.json"), 1e-6,
            diagnostic=lambda result: atomic_json(output/"cpu_mechanisms.json", result))
        record["parameters"] = model.parameter_counts()
        record["passed"] = True
    except Exception as exc:
        record.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        atomic_json(output/"cpu_preflight.json", record)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
