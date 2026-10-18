"""Stage19 full regression, exhaustive alphabet, and real FP32 acceptance with CUDA disabled."""
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
        raise RuntimeError("CPU proof must explicitly disable CUDA")
    import torch
    import yaml
    from gofa_f.short_generative_stage import prepare, load_model, read
    from gofa_f.short_checks import mechanisms
    from gofa_f.generative_data import SOURCES, Series, sha
    from gofa_f.prefix_cache import atomic_json
    torch.set_num_threads(2)
    cfg, output = yaml.safe_load(Path(args.config).read_text()), Path(args.output)
    prepare(cfg, output)
    xml = output/"cpu_tests.xml"
    result = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "--disable-warnings",
        "-p", "no:cacheprovider", "--junitxml="+str(xml)], check=False)
    if result.returncode:
        atomic_json(output/"cpu_preflight.json", {"passed": False, "phase": "regression"})
        return result.returncode
    cases = ET.parse(xml).getroot().findall(".//testcase")
    proof = {"passed": False, "cases": len(cases), "skipped": sum(x.find("skipped") is not None for x in cases),
        "cuda_disabled": True, "real_weight_dtype": "float32", "manifest_sha256": sha(output/"manifest.json")}
    if cfg.get("continuation"):
        from gofa_f.short_migration import migrate
        proof["migration"] = migrate(cfg, output)
    atomic_json(output/"cpu_preflight.json", proof)
    model, codec = load_model(cfg, "cpu", torch.float32)
    try:
        proof["alphabet"] = codec.audit_alphabet()
        root = Path(cfg["data"]["directory"])
        series = {s: Series(root/"sources"/(s+".npz")) for s in SOURCES}
        proof["real_weights"] = mechanisms(model, codec, series,
            read(output/"pools/queries.json")["validation"], 1e-6,
            diagnostic=lambda v: atomic_json(output/"cpu_mechanisms.json", v))
        if (output/"last.pt").exists():
            from gofa_f.short_training_checks import optimization_checks
            proof["continued_training_optimizations"] = optimization_checks(model, codec, series,
                read(output/"pools/queries.json")["validation"], 1e-6, checkpoint=output/"last.pt",
                diagnostic=lambda v: atomic_json(output/"cpu_optimization_continued.json", v))
        proof["passed"] = True
    except Exception as exc:
        proof.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        atomic_json(output/"cpu_preflight.json", proof)
    print(json.dumps(proof, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
