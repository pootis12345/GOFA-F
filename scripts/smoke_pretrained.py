"""On-server integration test: REAL pretrained backbones, graph gradients and official GNN."""
import json

import numpy as np
import torch

from gofa_f.baseline import DSTSGNNAdapter
from gofa_f.models import ForecastEncoder, parameter_counts


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU integration smoke requires CUDA")
    torch.manual_seed(42)
    x = torch.randn(2, 96, 3, device="cuda")
    mask = torch.ones_like(x, dtype=torch.bool)
    for backbone in ("google-t5/t5-small", "amazon/chronos-t5-small"):
        for layers in ([], [1, 3]):
            model = ForecastEncoder(96, 12, np.ones((3, 3), np.float32), backbone=backbone,
                                    graph_layers=layers, freeze_backbone=True).cuda()
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
            loss = model(x, mask).square().mean()
            loss.backward()
            assert model.adapter.weight.grad.abs().sum() > 0
            if layers:
                assert model.graph["1"].gate.grad.abs().sum() > 0
            optimizer.step()
            print(json.dumps({"backbone": backbone, "graph_layers": layers,
                              "loss": loss.item(), "parameters": parameter_counts(model),
                              "revision": model.resolved_revision}), flush=True)
            del model, optimizer
    model = DSTSGNNAdapter(96, 12, 3, width=16, hidden_size=64).cuda()
    prediction = model(x, mask)
    assert prediction.shape == (2, 12, 3)
    loss = prediction.square().mean()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    print(json.dumps({"baseline": "DST-SGNN", "loss": loss.item(), "parameters": parameter_counts(model)}))


if __name__ == "__main__":
    main()
