"""Isolate frozen-head BF16 layout differences; never updates adapter or training state."""
import argparse
import json
import os
from pathlib import Path
import signal
import time

import torch
from torch.nn import functional as F
import yaml

from gofa_f.idle_launcher import atomic_json


def project(hidden, weight, alphabet, extra_rows=0):
    finite = weight.index_select(0, alphabet)
    if extra_rows:
        # Zero work rows retain odd output width; they never enter the loss.
        finite = F.pad(finite, (0, 0, 0, extra_rows))
    return F.linear(hidden, finite)[..., :len(alphabet)]


def differences(actual, expected, tolerance):
    delta = (actual.float()-expected.float()).abs()
    return {"max_abs": float(delta.max()), "violations": int((
        delta > tolerance+tolerance*expected.float().abs()).sum()), "elements": delta.numel()}


def probe(weight, alphabet, tolerance, check=lambda: None):
    """Same input/weights, native 42001 vs finite 10000 vs tail-one 10001."""
    report = {"tolerance": tolerance, "cases": {}, "finite_tokens": len(alphabet),
        "native_vocabulary": len(weight), "candidate": "finite_plus_one_zero_work_row",
        "no_model_or_training_state_modified": True}
    for count, scale in ((1, .1), (25, 1.), (48, 1.)):
        check()
        generator = torch.Generator(device="cpu").manual_seed(42+count)
        raw = torch.randn(128, weight.shape[1], generator=generator).to(weight.device, weight.dtype)*scale
        labels = torch.arange(count, device=weight.device) % len(alphabet)
        def execute(extra):
            hidden = raw.detach().clone().requires_grad_(True)
            logits = F.linear(hidden, weight).index_select(-1, alphabet) if extra is None else \
                project(hidden, weight, alphabet, extra)
            loss = F.cross_entropy(logits[:count].float(), labels)
            gradient, = torch.autograd.grad(loss, hidden)
            return logits[:count].detach(), loss.detach(), gradient.detach()
        reference = execute(None)
        case = {}
        for label, extra in (("compact_10000", 0), ("tail_one_10001", 1)):
            check()
            candidate = execute(extra)
            case[label] = {name: differences(a, b, tolerance) for name, a, b in
                zip(("logits", "loss", "hidden_gradient"), candidate, reference)}
        report["cases"][str(count)] = case
    report["candidate_passed"] = all(item["violations"] == 0 for case in report["cases"].values()
        for item in case["tail_one_10001"].values())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.environ.get("GOFA_STAGE19_ADMITTED") != "1":
        raise RuntimeError("Projection diagnostics require the Stage19 fully-idle supervisor")
    from safetensors import safe_open
    from transformers import LlamaTokenizer
    from gofa_f.allnode_stage import supervisor_alive
    from gofa_f.generative_data import sha
    from gofa_f.short_codec import ShortCodec
    stopped = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    def check():
        if stopped[0] or not supervisor_alive(os.environ["GOFA_STAGE19_SUPERVISOR_PID"],
                                              os.environ["GOFA_STAGE19_SUPERVISOR_TICKS"]):
            raise RuntimeError("Projection diagnostic stopped cooperatively")
    cfg, output = yaml.safe_load(Path(args.config).read_text()), Path(args.output)
    root = Path(cfg["weights"]["directory"])
    index = json.loads((root/"model.safetensors.index.json").read_text())
    shard = root/index["weight_map"]["lm_head.weight"]
    with safe_open(shard, framework="pt", device="cpu") as saved:
        cpu_weight = saved.get_tensor("lm_head.weight")
    codec = ShortCodec(LlamaTokenizer.from_pretrained(root, local_files_only=True))
    check()
    torch.cuda.set_per_process_memory_fraction(38*2**30/torch.cuda.get_device_properties(0).total_memory)
    weight = cpu_weight.to(device="cuda:0", dtype=torch.bfloat16)
    report = probe(weight, codec.alphabet(weight.device), .002, check)
    report.update(weight_shard_sha256=sha(shard), alphabet_sha256=codec.alphabet_sha,
        dtype=str(weight.dtype), allocated_bytes=torch.cuda.max_memory_allocated(),
        reserved_bytes=torch.cuda.max_memory_reserved(),
        allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
    atomic_json(output/"projection_probe.json", report)
    # Completion is diagnostic completion, not acceptance of a training implementation.
    atomic_json(output/"status.json", {"state": "complete", "phase": "projection_probe",
        "candidate_passed": report["candidate_passed"], "updated_at": time.time()})
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
