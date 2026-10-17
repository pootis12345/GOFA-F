"""Cap-supervised, source-only layer tracing. Never trains or evaluates targets."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import signal

import numpy as np
import torch
import yaml

from gofa_f.allnode_graph import build_graph
from gofa_f.allnode_stage import load_model, read, supervisor_alive
from gofa_f.generative_data import Query, Series
from gofa_f.generative_model import GenerativePrefixLRU, completion_loss
from gofa_f.prefix_cache import atomic_json


def difference(actual, expected, tolerance=.002):
    actual, expected = actual.float(), expected.float()
    error = (actual-expected).abs()
    allowed = tolerance+tolerance*expected.abs()
    index = int(error.flatten().argmax())
    position = np.unravel_index(index, error.shape)
    return {"max_absolute": float(error.max()), "violations": int((error > allowed).sum()),
            "max_position": [int(i) for i in position],
            "actual_at_max": float(actual[position]), "reference_at_max": float(expected[position]),
            "token_maxima": error.amax((0, 2)).tolist()}


@contextmanager
def trace_layers(model):
    values, handles = {}, []
    def capture(name):
        def hook(module, args, output):
            value = output[0] if isinstance(output, tuple) else output
            values.setdefault(name, []).append(value.detach().cpu().clone())
        return hook
    layers = [("embedding", model.native.model.embed_tokens),
              *[(f"layer_{i:02d}", layer) for i, layer in enumerate(model.native.model.layers)],
              ("final_norm", model.native.model.norm), ("logits", model.native.lm_head)]
    for name, layer in layers:
        handles.append(layer.register_forward_hook(capture(name)))
    try:
        yield values
    finally:
        for handle in handles:
            handle.remove()


def diagnose(model, codec, source, queries, check=lambda: None):
    q = Query(**next(q for q in queries if q["source"] == "pems04" and q["length"] == 10))
    device = next(model.native.parameters()).device
    history, response = source.training_record(q)
    graphs, cache = [build_graph(source, q)], GenerativePrefixLRU("diagnostic", 512*2**20)
    batch = codec.batch([codec.record(history, response[:1])]).to(device)
    optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=2e-4, weight_decay=1e-4)
    for _ in range(3):
        check()
        optimizer.zero_grad(set_to_none=True)
        contexts = model.encode_origins(graphs, codec, cache, check)
        completion_loss(model(batch, contexts=contexts,
            prefix=model.cached_prefix(batch, cache, "teacher_forced")).logits, batch.labels)[0].backward()
        optimizer.step()
        model.mark_updated()
    result = {"diagnostic_only": True, "formal_updates": 0, "target_predictions": 0,
              "dtype": str(model.native.dtype), "tolerance": .002, "views": {}}
    for offset in (0, 16, 32):
        check()
        rolled = np.r_[history, np.full(offset, 1e4 if offset == 16 else -1e4)]
        current = codec.batch([codec.record(rolled, response[:1])]).to(device)
        with torch.no_grad():
            context = model.encode_origins(graphs, codec, cache, check)
            with trace_layers(model) as full:
                model(current, contexts=context)
            cut = int(current.prompt_ends[0])+1
            part = current.first(cut)
            with trace_layers(model) as incremental:
                output = model(part, contexts=context, use_cache=True)
                kv = output.past_key_values
                for index in range(cut, current.input_ids.shape[1]):
                    check()
                    part = part.append(current.input_ids[:, index:index+1])
                    output = model(part, contexts=context, cache=kv, use_cache=True)
                    kv = output.past_key_values
            result["views"][str(offset)] = {"prompt_tokens": cut,
                "total_tokens": current.input_ids.shape[1], "layers": {
                    name: difference(torch.cat(incremental[name], 1), torch.cat(value, 1))
                    for name, value in full.items()}}
    return result


def main():
    import os
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.environ.get("GOFA_STAGE18_ADMITTED") != "1":
        raise RuntimeError("Diagnostics require the own-memory-cap supervisor")
    stopped = [False]
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stopped.__setitem__(0, True))
    def check():
        if stopped[0] or not supervisor_alive(os.environ["GOFA_STAGE18_SUPERVISOR_PID"],
                                            os.environ["GOFA_STAGE18_SUPERVISOR_TICKS"]):
            raise RuntimeError("Diagnostic cooperatively stopped")
    output, cfg = Path(args.output), yaml.safe_load(Path(args.config).read_text())
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(38*2**30/torch.cuda.get_device_properties(0).total_memory)
    try:
        root = Path(cfg["data"]["directory"])
        source = Series(root/"sources/pems04.npz")
        model, codec = load_model(cfg, "cuda:0")
        result = diagnose(model, codec, source, read(root/"validation_queries.json"), check)
        atomic_json(output/"layer_diagnostics.json", result)
        atomic_json(output/"status.json", {"state": "complete", "diagnostic_only": True,
                                          "formal_updates": 0, "target_predictions": 0})
    except Exception as exc:
        atomic_json(output/"status.json", {"state": "failed", "error": str(exc), "diagnostic_only": True})
        raise


if __name__ == "__main__":
    main()
