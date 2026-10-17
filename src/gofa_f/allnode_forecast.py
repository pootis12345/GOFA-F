"""Matched BF16 greedy systems, immutable origin messages and per-step fallback accounting."""
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from gofa_f.allnode_graph import build_graph
from gofa_f.generative_data import Query, selection_score
from gofa_f.prefix_cache import atomic_json


@torch.no_grad()
def forecast_batch(model, codec, histories, horizon, contexts=None, check=lambda: None,
                   on_step=lambda *_: None, cached=True, trace=None):
    if horizon <= 0:
        raise ValueError("Positive forecast horizon required")
    present = [np.isfinite(row).any() for row in histories]
    if not all(present):
        predictions = np.zeros((len(histories), horizon), np.float32)
        coverage = np.zeros_like(predictions, dtype=bool)
        reasons = [["empty_history"]*horizon for _ in histories]
        selected = [i for i, valid in enumerate(present) if valid]
        if selected:
            p, v, r = forecast_batch(model, codec, [histories[i] for i in selected], horizon,
                [contexts[i] for i in selected] if contexts is not None else None, check, on_step, cached, trace)
            for i, a, b, c in zip(selected, p, v, r):
                predictions[i], coverage[i], reasons[i] = a, b, c
        return predictions, coverage, reasons
    device = next(model.native.parameters()).device
    series = [np.asarray(row, np.float64).copy() for row in histories]
    fallback = [float(row[np.flatnonzero(np.isfinite(row))[-1]]) for row in series]
    predictions, coverages = [], []
    reasons = [[] for _ in series]
    for offset in range(0, horizon, 16):
        count = min(16, horizon-offset)
        records = [codec.record(row) for row in series]
        tokens = codec.batch(records).to(device)
        generated, kv = [], None  # all 32 layers invalidated on every rolling re-encoding
        for step in range(2*count+8):
            check()
            output = model(tokens, contexts=contexts, cache=kv, use_cache=cached)
            if not cached or kv is None:
                # Full recompute includes generated suffix after the original prompt.
                ends = (torch.full_like(tokens.prompt_ends, tokens.full_ids.shape[1]-1)
                        if generated else tokens.prompt_ends)
                logits = output.logits[torch.arange(len(series), device=device), ends].clone()
            else:
                logits = output.logits[:, -1].clone()
            logits[:, codec.tokenizer.eos_token_id] = -torch.inf
            next_ids = logits.argmax(-1)
            generated.append(next_ids)
            kv = output.past_key_values if cached else None
            tokens = tokens.append(next_ids[:, None])
            if not cached:
                tokens.input_ids = tokens.full_ids
                tokens.labels = torch.full_like(tokens.full_ids, -100)
            on_step(offset+min(count, (step+1)//2), horizon)
        texts = codec.tokenizer.batch_decode(torch.stack(generated, 1), skip_special_tokens=False)
        if trace is not None:
            trace({"offset": offset, "ids": torch.stack(generated, 1).cpu().tolist(),
                "prompt_hashes": [r["prompt_hash"] for r in records],
                "value_positions": [r["value_positions"] for r in records]})
        p_rows, valid_rows = [], []
        for i, (text, record) in enumerate(zip(texts, records)):
            p, valid, why = codec.parse_with_reasons(text, *record["scaling"], fallback[i], count)
            p_rows.append(p)
            valid_rows.append(valid)
            reasons[i].extend(why)
            series[i] = np.concatenate((series[i], p))
        predictions.append(np.stack(p_rows))
        coverages.append(np.stack(valid_rows))
    return np.concatenate(predictions, 1), np.concatenate(coverages, 1), reasons


def evaluate_sources(model, codec, series, queries, output, label, binding, progress, check):
    path = Path(output)/"evaluations"/(label+".json")
    path.parent.mkdir(exist_ok=True)
    record = json.loads(path.read_text()) if path.exists() else {"binding": binding,
        "errors": [None]*len(queries), "coverage": [None]*len(queries),
        "fallback_reasons": [None]*len(queries), "active_seconds": 0., "complete": False,
        "protocol": "official_prompt_greedy_bf16_origin_last_fallback_v1", "stage17_imported": False}
    if record["binding"] != binding:
        raise ValueError("Evaluation fragments belong to a different locked protocol")
    if record["complete"]:
        return selection_score(record["errors"], queries, series["epsilons"])
    native = label == "native"
    data, groups = series["data"], {}
    for i, query in enumerate(queries):
        if record["errors"][i] is None:
            groups.setdefault((query["length"], query["horizon"]), []).append(i)
    work_total = sum(query["horizon"] for query in queries)
    microbatch = 16 if native else 1
    for group in groups.values():
        position = 0
        while position < len(group):
            indices = group[position:position+microbatch]
            selected = [Query(**queries[i]) for i in indices]
            started, notified = time.monotonic(), [0.]
            def notify(step, horizon):
                if time.monotonic()-notified[0] < 5:
                    return
                notified[0] = time.monotonic()
                done = sum(value is not None for value in record["errors"])
                work = sum(query["horizon"] for query, value in zip(queries, record["errors"])
                           if value is not None)
                progress.evaluation(done, len(queries), work, work_total,
                    record["active_seconds"]+time.monotonic()-started, evaluation_kind=label,
                    evaluation_current_step=step, evaluation_current_horizon=horizon)
            try:
                with torch.no_grad():
                    contexts = None if native else model.encode_origins(
                        [build_graph(data[q.source], q) for q in selected], codec, check=check)
                    histories = [np.where(data[q.source].mask[q.origin-q.length:q.origin, q.variable],
                        data[q.source].values[q.origin-q.length:q.origin, q.variable], np.nan) for q in selected]
                    predictions, coverage, why = forecast_batch(model, codec, histories,
                        selected[0].horizon, contexts, check, notify)
            except Exception as exc:
                record["active_seconds"] += time.monotonic()-started
                atomic_json(path, record)
                if isinstance(exc, torch.cuda.OutOfMemoryError) and microbatch > 1:
                    microbatch = max(1, microbatch//2)
                    contexts = None
                    torch.cuda.empty_cache()
                    continue
                raise
            for index, q, p, valid, reasons in zip(indices, selected, predictions, coverage, why):
                values = data[q.source]
                target = values.values[q.origin:q.origin+q.horizon, q.variable]
                mask = values.mask[q.origin:q.origin+q.horizon, q.variable]
                if not mask.any():
                    raise ValueError("Validation query has no original observed labels")
                record["errors"][index] = float(np.abs(p.astype(np.float64)-target)[mask].mean())
                record["coverage"][index] = float(valid.mean())
                record["fallback_reasons"][index] = dict(Counter(reasons))
            record["active_seconds"] += time.monotonic()-started
            atomic_json(path, record)
            position += len(indices)
    record["complete"] = True
    atomic_json(path, record)
    progress.evaluation(len(queries), len(queries), work_total, work_total,
        record["active_seconds"], evaluation_kind=label, evaluation_current_step=0, evaluation_current_horizon=0)
    return selection_score(record["errors"], queries, series["epsilons"])
