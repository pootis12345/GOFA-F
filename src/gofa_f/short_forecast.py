"""Single-origin, single-block constrained generation and resumable query shards."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from gofa_f.allnode_graph import build_graph
from gofa_f.generative_data import Query
from gofa_f.prefix_cache import atomic_json


@torch.no_grad()
def forecast_batch(model, codec, histories, horizon, contexts=None, check=lambda: None,
                   on_step=lambda *_: None, cached=True):
    if not 1 <= horizon <= 24:
        raise ValueError("Stage19 supports one block of 1 to 24 numeric slots")
    present = [np.isfinite(h).any() for h in histories]
    n = len(histories)
    result = {"predictions": np.zeros((n, horizon), np.float32),
              "inverse_valid": np.zeros((n, horizon), bool),
              "boundary": np.zeros((n, horizon), bool), "numeric_ids": [[] for _ in histories],
              "generated_ids": [[] for _ in histories], "format_complete": [False]*n,
              "forward_calls": 0, "min_numeric_margin": None}
    if not all(present):
        selected = [i for i, valid in enumerate(present) if valid]
        if selected:
            sub = forecast_batch(model, codec, [histories[i] for i in selected], horizon,
                [contexts[i] for i in selected] if contexts is not None else None, check, on_step, cached)
            for k in ("predictions", "inverse_valid", "boundary", "numeric_ids", "generated_ids", "format_complete"):
                for j, i in enumerate(selected):
                    result[k][i] = sub[k][j]
            result["forward_calls"], result["min_numeric_margin"] = sub["forward_calls"], sub["min_numeric_margin"]
        return result
    device = next(model.native.parameters()).device
    records = [codec.record(h) for h in histories]
    tokens = codec.batch(records).to(device)
    if codec.first_separator:
        part = torch.tensor(codec.first_separator, device=device).expand(n, -1)
        tokens = tokens.append(part)
        tokens.input_ids = tokens.full_ids
    alphabet = codec.alphabet(device)
    kv, numbers, generated = None, [], [list(codec.first_separator) for _ in histories]
    margin = float("inf")
    for slot in range(horizon):
        check()
        output = model(tokens, contexts=contexts, cache=kv, use_cache=cached)
        if kv is None:
            ends = tokens.prompt_ends+len(codec.first_separator) if cached else \
                torch.full_like(tokens.prompt_ends, tokens.full_ids.shape[1]-1)
            logits = output.logits[torch.arange(n, device=device), ends]
        else:
            logits = output.logits[:, -1]
        scores = logits.float().index_select(-1, alphabet)
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Nonfinite numeric logits are a mechanism failure, not a fallback")
        top = scores.topk(2, dim=-1).values
        margin = min(margin, float((top[:, 0]-top[:, 1]).min()))
        ids = alphabet[scores.argmax(-1)]
        numbers.append(ids.cpu().numpy())
        for row, value in enumerate(ids.tolist()):
            generated[row].append(value)
            if slot+1 < horizon:
                generated[row].extend(codec.separator)
        result["forward_calls"] += 1
        on_step(slot+1, horizon)
        if slot+1 < horizon:
            delimiter = torch.tensor(codec.separator, device=device).expand(n, -1)
            tokens = tokens.append(torch.cat((ids[:, None], delimiter), 1))
            kv = output.past_key_values if cached else None
            if not cached:
                tokens.input_ids = tokens.full_ids
                tokens.labels = torch.full_like(tokens.full_ids, -100)
    numeric = np.stack(numbers, 1)
    for row, (ids, record) in enumerate(zip(numeric, records)):
        p, valid, boundary = codec.decode_ids(ids, *record["scaling"], record["last"])
        result["predictions"][row], result["inverse_valid"][row], result["boundary"][row] = p, valid, boundary
        result["numeric_ids"][row], result["generated_ids"][row] = ids.tolist(), generated[row]
        result["format_complete"][row] = len(ids) == horizon
    result["min_numeric_margin"] = margin
    return result


def evaluate(model, codec, series, queries, output, label, binding, progress, check, native=False):
    root = Path(output)/"evaluations"/label
    root.mkdir(parents=True, exist_ok=True)
    complete = root/"complete.json"
    if complete.exists():
        record = json.loads(complete.read_text())
        if record["binding"] != binding:
            raise ValueError("Evaluation binding changed")
        return record
    entries = [None]*len(queries)
    for index, query in enumerate(queries):
        path = root/f"{index:06d}.json"
        if path.exists():
            record = json.loads(path.read_text())
            if record["binding"] != binding or record["query"] != query:
                raise ValueError("Query shard belongs to a different locked evaluation")
            entries[index] = record
    seconds = sum(v["active_seconds"] for v in entries if v is not None)
    total_slots = sum(q["horizon"] for q in queries)
    for index, query in enumerate(queries):
        if entries[index] is not None:
            continue
        check()
        q = Query(**query)
        data = series[q.source]
        history = np.where(data.mask[q.origin-q.length:q.origin, q.variable],
            data.values[q.origin-q.length:q.origin, q.variable], np.nan)
        started, notified = time.monotonic(), [0.]
        done = sum(v is not None for v in entries)
        slots = sum(v["query"]["horizon"] for v in entries if v is not None)
        def notify(step, horizon):
            if time.monotonic()-notified[0] >= 5:
                progress.evaluation(done, len(queries), slots+step, total_slots,
                    seconds+time.monotonic()-started, evaluation_kind=label,
                    evaluation_current_step=step, evaluation_current_horizon=horizon)
                notified[0] = time.monotonic()
        contexts = None
        if not native and np.isfinite(history).any():
            contexts = model.encode_origins([build_graph(data, q)], codec, check=check)
        prediction = forecast_batch(model, codec, [history], q.horizon, contexts, check, notify)
        y = data.values[q.origin:q.origin+q.horizon, q.variable].astype(np.float64)
        observed = data.mask[q.origin:q.origin+q.horizon, q.variable]
        if not observed.any():
            raise ValueError("Evaluation query has no observed target")
        base = codec.record(history)
        scale, offset = base["scaling"]
        low, high = (-.9999+.5-offset)/scale, (.9999+.5-offset)/scale
        outside = ((y < low) | (y > high)) & observed
        p = prediction["predictions"][0]
        record = {"binding": binding, "query": query, "prediction": p.tolist(),
            "target": y.tolist(), "observed": observed.tolist(), "scaling": list(base["scaling"]),
            "representable_range": [low, high], "true_outside_range": outside.tolist(),
            "range_only_error_lower_bound": np.where(observed, np.abs(y-np.clip(y, low, high)), 0).tolist(),
            "inverse_valid": prediction["inverse_valid"][0].tolist(),
            "boundary": prediction["boundary"][0].tolist(), "format_complete": prediction["format_complete"][0],
            "empty_history": not bool(base["present"]), "numeric_ids": prediction["numeric_ids"][0],
            "generated_ids": prediction["generated_ids"][0],
            "numeric_slots": q.horizon, "generated_token_count": len(prediction["generated_ids"][0]),
            "forward_calls": prediction["forward_calls"], "min_numeric_margin": prediction["min_numeric_margin"],
            "mae": float(np.abs(p.astype(np.float64)-y)[observed].mean()),
            "active_seconds": time.monotonic()-started}
        atomic_json(root/f"{index:06d}.json", record)
        entries[index] = record
        seconds += record["active_seconds"]
    result = {"binding": binding, "complete": True, "queries": len(queries),
              "active_seconds": seconds, "errors": [v["mae"] for v in entries],
              "metrics": summarize(entries), "protocol": "single_block_24_finite_grammar_v1"}
    atomic_json(complete, result)
    progress.evaluation(len(queries), len(queries), total_slots, total_slots, seconds,
                        evaluation_kind=label, evaluation_current_step=0, evaluation_current_horizon=0)
    return result


def summarize(entries, split_templates=True):
    sources = {e["query"]["source"] for e in entries}
    if len(sources) > 1:
        return {"by_source": {s: summarize([e for e in entries if e["query"]["source"] == s])
                              for s in sorted(sources)},
                "note": "Raw-unit errors from different datasets are never pooled"}
    by_variable = {}
    formatting = {"format_completed_queries": 0, "inverse_valid_slots": 0, "numeric_slots": 0,
        "boundary_slots": 0, "true_outside_range": 0, "observed_targets": 0, "fallback_slots": 0,
        "empty_history_queries": 0, "generated_tokens": 0, "forward_calls": 0}
    for entry in entries:
        q = entry["query"]
        key = f"{q['source']}:{q['variable']}"
        slot = by_variable.setdefault(key, [])
        p, y, m = (np.asarray(entry[k]) for k in ("prediction", "target", "observed"))
        if not np.isfinite(p).all():
            raise FloatingPointError("Nonfinite reported system prediction")
        for step, (a, b, seen) in enumerate(zip(p, y, m)):
            slot.append((float(a), float(b), bool(seen), step))
        formatting["format_completed_queries"] += int(entry["format_complete"])
        formatting["inverse_valid_slots"] += sum(entry["inverse_valid"])
        formatting["numeric_slots"] += q["horizon"]
        formatting["boundary_slots"] += sum(entry["boundary"])
        formatting["true_outside_range"] += sum(entry["true_outside_range"])
        formatting["observed_targets"] += sum(entry["observed"])
        formatting["fallback_slots"] += q["horizon"]-sum(entry["inverse_valid"])
        formatting["empty_history_queries"] += int(entry["empty_history"])
        formatting["generated_tokens"] += entry["generated_token_count"]
        formatting["forward_calls"] += entry["forward_calls"]
    def metrics(rows):
        if not rows:
            return {"observations": 0, "mae": None, "rmse": None, "mape_pct": None,
                    "mape_valid_fraction": None, "wape_pct": None}
        a = np.asarray(rows, np.float64)
        p, y, mask = a[:, 0], a[:, 1], a[:, 2].astype(bool)
        p, y = p[mask], y[mask]
        if not len(y):
            return metrics([])
        error = np.abs(p-y)
        valid = np.abs(y) > 1e-5
        denominator = np.abs(y).sum()
        return {"observations": len(y), "mae": float(error.mean()), "rmse": float(np.sqrt(np.square(error).mean())),
            "mape_pct": float(100*np.mean(error[valid]/np.abs(y[valid]))) if valid.any() else None,
            "mape_valid_fraction": float(valid.mean()), "wape_pct": float(100*error.sum()/denominator) if denominator else None}
    variables = {key: dict(metrics(rows), queries=sum(1 for e in entries
        if f"{e['query']['source']}:{e['query']['variable']}" == key)) for key, rows in by_variable.items()}
    micro = metrics([row for rows in by_variable.values() for row in rows])
    macro = {}
    for key in ("mae", "rmse", "mape_pct", "wape_pct", "mape_valid_fraction"):
        values = [v[key] for v in variables.values() if v[key] is not None]
        macro[key] = float(np.mean(values)) if values else None
        macro[key+"_eligible_variables"] = len(values)
    horizon = max(e["query"]["horizon"] for e in entries)
    by_step = [dict(step=i+1, **metrics([r for rows in by_variable.values() for r in rows if r[3] == i]))
               for i in range(horizon)]
    formatting.update(format_completion_fraction=formatting["format_completed_queries"]/len(entries),
        inverse_valid_fraction=formatting["inverse_valid_slots"]/formatting["numeric_slots"],
        fallback_fraction=formatting["fallback_slots"]/formatting["numeric_slots"],
        true_outside_range_fraction=formatting["true_outside_range"]/formatting["observed_targets"],
        boundary_fraction_of_requested_slots=formatting["boundary_slots"]/formatting["numeric_slots"])
    result = {"micro": micro, "macro": macro, "variables": variables, "by_step": by_step,
        "formatting": formatting, "mape_absolute_threshold": 1e-5,
        "macro_rmse_definition": "arithmetic mean of per-variable RMSE; not square root of macro MSE",
        "uncertainty_note": "Per-variable estimates describe the fixed sampled origins; they are not independent full-population estimates"}
    templates = {(e["query"].get("length"), e["query"]["horizon"]) for e in entries}
    if split_templates and len(templates) > 1:
        result["by_template"] = {f"{length}->{horizon}": summarize([e for e in entries
            if (e["query"].get("length"), e["query"]["horizon"]) == (length, horizon)], False)
            for length, horizon in sorted(templates)}
    if horizon == 24 and len(templates) == 1:
        result["steps_1_to_16"] = metrics([r for rows in by_variable.values() for r in rows if r[3] < 16])
        result["steps_17_to_24"] = metrics([r for rows in by_variable.values() for r in rows if r[3] >= 16])
    return result
