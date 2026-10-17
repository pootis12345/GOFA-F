"""Stage18 complete CPU regression plus fresh FP32 real-weight all-node acceptance."""
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
    parser.add_argument("--reuse-real-cpu-from")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU preflight must explicitly disable CUDA")
    import torch
    import yaml
    from gofa_f.allnode_checks import mechanisms
    from gofa_f.allnode_stage import load_model, prepare, read
    from gofa_f.generative_data import SOURCES, Series, sha
    from gofa_f.prefix_cache import atomic_json
    torch.set_num_threads(2)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config).read_text())
    manifest = prepare(cfg, output)
    xml = output/"cpu_tests.xml"
    completed = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "--disable-warnings",
                               "--junitxml="+str(xml)], check=False)
    if completed.returncode:
        atomic_json(output/"cpu_preflight.json", {"passed": False, "phase": "regression"})
        raise SystemExit(completed.returncode)
    cases = ET.parse(xml).getroot().findall(".//testcase")
    record = {"passed": False, "cases": len(cases),
        "skipped": sum(case.find("skipped") is not None for case in cases), "cuda_disabled": True,
        "manifest_sha256": sha(output/"manifest.json"), "real_weight_dtype": "float32",
        "stage17_proof_reused": False, "stage17_baseline_imported": False}
    atomic_json(output/"cpu_preflight.json", record)
    if args.reuse_real_cpu_from:
        from gofa_f.allnode_recovery import reuse_real_cpu
        reused = reuse_real_cpu(args.reuse_real_cpu_from, manifest)
        record.update(real_weights=reused["real_weights"], parameters=reused["parameters"],
                      real_cpu_proof_reused=reused, passed=True)
        atomic_json(output/"cpu_mechanisms.json", reused["real_weights"])
        atomic_json(output/"cpu_preflight.json", record)
        print(json.dumps(record, indent=2), flush=True)
        return
    root = Path(cfg["data"]["directory"])
    series = {name: Series(root/"sources"/(name+".npz")) for name in SOURCES}
    model, codec = load_model(cfg, "cpu", dtype=torch.float32)
    try:
        record["real_weights"] = mechanisms(model, codec, series, read(root/"validation_queries.json"),
            1e-6, diagnostic=lambda result: atomic_json(output/"cpu_mechanisms.json", result))
        record["parameters"] = model.parameter_counts()
        if record["parameters"]["trainable"] != 1060998:
            raise AssertionError("Unexpected all-node trainable parameter count")
        record["passed"] = True
    except Exception as exc:
        record.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        atomic_json(output/"cpu_preflight.json", record)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
