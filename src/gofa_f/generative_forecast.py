"""Source-only greedy rolling generation; target evaluation is not dispatched here."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from gofa_f.generative_data import Query, selection_score
from gofa_f.generative_model import QueryTokens
from gofa_f.prefix_cache import atomic_json


@torch.no_grad()
def forecast_batch(model, codec, histories, horizon, graphs, check=lambda: None, on_step=lambda *_: None):
    device = next(model.native.parameters()).device
    series = [np.asarray(x, np.float64).copy() for x in histories]
    original_last = [float(x[np.flatnonzero(np.isfinite(x))[-1]]) if np.isfinite(x).any() else 0. for x in series]
    predictions, coverages = [], []
    memories = model.adapter.memories(graphs) if graphs is not None else None
    for offset in range(0, horizon, 16):
        count = min(16, horizon-offset)
        records = [codec.record(x) for x in series]
        tokens = codec.batch(records).to(device)
        generated, cache = [], None
        for step in range(2*count+8):
            check()
            output = model(tokens, graphs, cache=cache, use_cache=True, memories=memories)
            if cache is None:
                logits = output.logits[torch.arange(len(series), device=device), tokens.prompt_ends].clone()
            else:
                logits = output.logits[:, -1].clone()
            logits[:, codec.tokenizer.eos_token_id] = -torch.inf  # official min_new_tokens == max_new_tokens
            next_ids = logits.argmax(-1)
            generated.append(next_ids)
            cache = output.past_key_values
            tokens = QueryTokens(next_ids[:, None], torch.cat((tokens.attention_mask,
                torch.ones((len(series), 1), device=device, dtype=torch.bool)), 1),
                torch.full((len(series), 1), -100, device=device), tokens.prompt_ends)
            on_step(offset+min(count, (step+1)//2), horizon)
        texts = codec.tokenizer.batch_decode(torch.stack(generated, 1), skip_special_tokens=False)
        batch, valid = [], []
        for i, (text, record) in enumerate(zip(texts, records)):
            p, coverage = codec.parse(text, record["inverse_scale"], record["inverse_offset"],
                                      original_last[i], count)
            batch.append(p)
            valid.append(coverage)
            series[i] = np.concatenate((series[i], p))
        predictions.append(np.stack(batch))
        coverages.append(np.stack(valid))
    return np.concatenate(predictions, 1), np.concatenate(coverages, 1)


def evaluate_sources(model, codec, series, queries, output, label, binding, progress, check):
    path = Path(output)/"evaluations"/(label+".json")
    path.parent.mkdir(exist_ok=True)
    record = json.loads(path.read_text()) if path.exists() else {"binding": binding,
        "errors": [None]*len(queries), "coverage": [None]*len(queries), "active_seconds": 0., "complete": False}
    if record["binding"] != binding:
        raise ValueError("Evaluation fragments belong to a different locked model/data/decode")
    if record["complete"]:
        return selection_score(record["errors"], queries, series["epsilons"])
    native = label == "native"
    data = series["data"]
    groups = {}
    for index, q in enumerate(queries):
        if record["errors"][index] is None:
            groups.setdefault((q["length"], q["horizon"]), []).append(index)
    total_work = sum(q["horizon"] for q in queries)
    microbatch = 16
    for group in groups.values():
        position = 0
        while position < len(group):
            indices = group[position:position+microbatch]
            batch_queries = [Query(**queries[i]) for i in indices]
            graphs = None if native else [data[q.source].graph(q).to(next(model.parameters()).device)
                                         for q in batch_queries]
            histories = [np.where(data[q.source].mask[q.origin-q.length:q.origin, q.variable],
                data[q.source].values[q.origin-q.length:q.origin, q.variable], np.nan) for q in batch_queries]
            started = time.monotonic()
            notified = [0.]
            def update(step, horizon):
                if time.monotonic()-notified[0] < 5:
                    return
                notified[0] = time.monotonic()
                done = sum(v is not None for v in record["errors"])
                work = sum(q["horizon"] for q, v in zip(queries, record["errors"]) if v is not None)
                progress.evaluation(done, len(queries), work, total_work,
                    record["active_seconds"]+time.monotonic()-started, evaluation_kind=label,
                    evaluation_current_step=step, evaluation_current_horizon=horizon)
            try:
                predictions, coverage = forecast_batch(model, codec, histories,
                    batch_queries[0].horizon, graphs, check, update)
            except torch.cuda.OutOfMemoryError:
                if microbatch == 1:
                    raise
                microbatch = max(1, microbatch//2)
                torch.cuda.empty_cache()
                continue
            for i, q, p, valid in zip(indices, batch_queries, predictions, coverage):
                values = data[q.source]
                target = values.values[q.origin:q.origin+q.horizon, q.variable]
                mask = values.mask[q.origin:q.origin+q.horizon, q.variable]
                if not mask.any():
                    raise ValueError("Validation query has no original labels")
                record["errors"][i] = float(np.abs(p.astype(np.float64)-target)[mask].mean())
                record["coverage"][i] = float(valid.mean())
            record["active_seconds"] += time.monotonic()-started
            atomic_json(path, record)
            position += len(indices)
    record["complete"] = True
    atomic_json(path, record)
    return selection_score(record["errors"], queries, series["epsilons"])
