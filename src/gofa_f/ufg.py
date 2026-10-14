"""Pinned official UFGTime effective forward, with audited complex parameter counts."""
import importlib.util
import json
import math
from pathlib import Path
import sys
import types

import numpy as np
import torch
from torch import nn

from gofa_f.data import file_sha256

REVISION = "61f55135bc2bd6e9b9e5d592009d358015eea681"
MODEL_SHA256 = "c7229e78ebb136570f783a62d4fd68760d8c36a5b93394a25429b26136de82b1"
UTILS_SHA256 = "3839d247121677327e415695c867c0ee18415e00b93efb121292de15a4950701"
TARGET_ACTIVE_DOF = 108930


def portable_knn_graph(x, k, algorithm="bruteforce-sharemem", **kwargs):
    from dgl import knn_graph
    # The upstream CUDA kernel is not available on CPU. Only the CPU reference dispatch changes.
    if x.device.type == "cpu" and algorithm == "bruteforce-sharemem":
        algorithm = "bruteforce"
    return knn_graph(x, k, algorithm=algorithm, **kwargs)


def official_modules(root):
    root = Path(root).resolve()
    for name, digest in (("model.py", MODEL_SHA256), ("utils.py", UTILS_SHA256)):
        if file_sha256(root / "src" / name) != digest:
            raise ValueError("UFGTime source differs from the pinned official commit")
    package_name = "_gofa_ufg_upstream"
    if package_name + ".model" in sys.modules:
        loaded = sys.modules[package_name + ".model"]
        if Path(loaded.__file__).resolve() != root / "src/model.py":
            raise ValueError("Another UFGTime source directory is already loaded")
        return loaded, sys.modules[package_name + ".utils"]
    package = types.ModuleType(package_name)
    package.__path__ = [str(root / "src")]
    sys.modules[package_name] = package
    for name in ("utils", "model"):
        spec = importlib.util.spec_from_file_location(package_name + "." + name, root / "src" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        # The original source imports src.utils. Restore any pre-existing aliases afterwards.
        old = {key: sys.modules.get(key) for key in ("src", "src.utils")}
        sys.modules["src"] = package
        if name == "model":
            sys.modules["src.utils"] = sys.modules[package_name + ".utils"]
        try:
            spec.loader.exec_module(module)
            if name == "model":
                module.knn_graph = portable_knn_graph
        finally:
            for key, value in old.items():
                if value is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = value
    return sys.modules[package_name + ".model"], sys.modules[package_name + ".utils"]


def real_dof(parameter):
    return parameter.numel() * (2 if parameter.is_complex() else 1)


def active_name(name):
    # Explicitly audited against the pinned forward, not inferred from zero initial gradients.
    if name == "embeddings" or name.startswith("conv2_list."):
        return False
    if not name.startswith(("filters.", "conv1_list.", "clin.", "lin2.", "lin3.", "lin4.", "Linear_Trend.")):
        raise ValueError("Unclassified upstream parameter: " + name)
    return True


def audit_parameters(model):
    base = model.official if isinstance(model, UFGTime) else model
    names = dict(base.named_parameters())
    active = {name: parameter for name, parameter in names.items() if active_name(name)}
    return {
        "registered_numel": sum(p.numel() for p in names.values()),
        "registered_real_dof": sum(real_dof(p) for p in names.values()),
        "active_real_dof": sum(real_dof(p) for p in active.values()),
        "trainable_real_dof": sum(real_dof(p) for p in names.values() if p.requires_grad),
        "inactive_names": [name for name in names if name not in active],
        "complex_counting": "two real scalar degrees of freedom per complex scalar",
    }


def build_official(root, length, horizon, nodes, width, device="cpu"):
    model_module, utilities = official_modules(root)
    approximation = np.array([utilities.cheb_approx(f, 2) for f in utilities.get_frame("Haar")])
    return model_module.FreqTimeUFGV2(
        seq_length=length, signal_length=max(16, length), pred_length=horizon,
        hidden_size=width, embed_size=128, num_ts=nodes, device=str(device),
        approx=approximation, s=2.0, lev=1, num_topk=2,
    )


def matching_width(root, length, horizon, nodes, target=TARGET_ACTIVE_DOF):
    # The effective forward is linear in width; the unused quadratic branch never enters matching.
    with torch.device("meta"):
        one = audit_parameters(build_official(root, length, horizon, nodes, 1))["active_real_dof"]
        two = audit_parameters(build_official(root, length, horizon, nodes, 2))["active_real_dof"]
    slope, intercept = two - one, 2 * one - two
    center = (target - intercept) / slope
    choices = {max(1, min(4096, math.floor(center))), max(1, min(4096, math.ceil(center)))}
    width = min(choices, key=lambda h: (abs(slope * h + intercept - target), h))
    count = slope * width + intercept
    if abs(count / target - 1) > .10:
        raise ValueError("No actual-forward parameter match within allowed width/tolerance")
    return width, count


class UFGTime(nn.Module):
    def __init__(self, root, length, horizon, nodes, scale="S", device="cpu", width=None):
        super().__init__()
        if scale not in ("S", "A"):
            raise ValueError("Unknown pure-graph scale")
        self.width = width or (16 if scale == "S" else matching_width(root, length, horizon, nodes)[0])
        self.official = build_official(root, length, horizon, nodes, self.width, device)
        for name, parameter in self.official.named_parameters():
            parameter.requires_grad_(active_name(name))
        self.provenance = {
            "repository": "https://github.com/WonderHeiYi/UFGTIME", "revision": REVISION,
            "model_sha256": MODEL_SHA256, "utils_sha256": UTILS_SHA256,
            "effective_forward": "unmodified FreqTimeUFGV2", "width": self.width, "scale": scale,
            "cpu_portability": "DGL bruteforce kNN; CUDA retains upstream bruteforce-sharemem",
            "signal_length": max(16, length), "haar": True, "level": 1, "k": 2, "s": 2,
            "cheb_order": 2, "unused_parameters_frozen": True,
            "total_parameter_matching": False, "parameter_audit": audit_parameters(self),
        }

    def forward(self, x, observed):
        # Native graph construction sees only the observed historical window.
        return self.official(x.float().masked_fill(~observed, 0)).transpose(1, 2).contiguous()


def runtime_versions():
    import dgl
    import torch_geometric
    return {"torch": torch.__version__, "dgl": dgl.__version__, "pyg": torch_geometric.__version__,
            "numpy": np.__version__}


def source_manifest(root):
    manifest = json.loads((Path(root) / "manifest.json").read_text())
    if manifest["revision"] != REVISION:
        raise ValueError("Unexpected UFGTime manifest")
    for name, digest in manifest["source_sha256"].items():
        if file_sha256(Path(root) / name) != digest:
            raise ValueError("UFGTime source hash mismatch: " + name)
    return manifest
