"""C-only Stage17 CPU regression and real-weight source mechanisms."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


def reuse_real_check(previous, snapshot, manifest):
    previous, snapshot = Path(previous), Path(snapshot)
    raw = (previous/"manifest.json").read_bytes()
    old = json.loads(raw)
    result = json.loads((previous/"cpu_preflight.json").read_text())
    if not result.get("passed") or result.get("real_weight_dtype") != "float32" \
            or not result.get("real_weights", {}).get("passed") \
            or result["real_weights"]["tolerance"] != 1e-6 \
            or result["manifest_sha256"] != hashlib.sha256(raw).hexdigest():
        raise ValueError("Only an intact complete registered FP32 real-weight check can be reused")
    if (snapshot/"CODE_REVISION").read_text().strip() != old["code_revision"]:
        raise ValueError("Wrong source CPU-check snapshot")
    for key in ("configuration", "weights_sha256", "source_data", "pools", "dependencies"):
        if old[key] != manifest[key]:
            raise ValueError("CPU proof data/runtime changed: "+key)
    if set(old["code"]) != set(manifest["code"]):
        raise ValueError("CPU proof module set changed")
    for name, digest in old["code"].items():
        path = snapshot/"src/gofa_f"/name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Source CPU snapshot was modified")
        if name not in ("generative_stage.py", "generative_launcher.py") and manifest["code"][name] != digest:
            raise ValueError("A checked model/data/cache module changed: "+name)
    def functions(path):
        return {node.name: ast.dump(node, include_attributes=False) for node in
            ast.parse(path.read_text()).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            and node.name not in ("train", "main", "supervisor_alive")}
    if functions(snapshot/"src/gofa_f/generative_stage.py") != \
            functions(Path(__file__).resolve().parents[1]/"src/gofa_f/generative_stage.py"):
        raise ValueError("Model factory, optimizer, recovery or preparation functions changed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reuse-real-from")
    parser.add_argument("--reuse-real-snapshot")
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
    manifest = prepare(cfg, output)
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
    if args.reuse_real_from or args.reuse_real_snapshot:
        if not (args.reuse_real_from and args.reuse_real_snapshot):
            raise ValueError("Both bound proof and its immutable source snapshot are required")
        old = reuse_real_check(args.reuse_real_from, args.reuse_real_snapshot, manifest)
        record.update(real_weights=old["real_weights"], parameters=old["parameters"],
            real_weight_dtype="float32", passed=True, real_weight_check_reused_from={
                "output": str(Path(args.reuse_real_from).resolve()),
                "manifest_sha256": old["manifest_sha256"], "reason":
                "model/data/cache bytes and factory/optimizer/recovery AST unchanged; full current regression rerun"})
        atomic_json(output/"cpu_preflight.json", record)
        print(json.dumps(record, indent=2), flush=True)
        return
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
