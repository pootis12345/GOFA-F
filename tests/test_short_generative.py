from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from transformers import LlamaTokenizer

from gofa_f.generative_data import Query
from gofa_f.allnode_graph import build_graph
from gofa_f.short_codec import ShortCodec, numeric_loss
from gofa_f.short_data import TEMPLATES, origins, relative_score, stratified
from gofa_f.short_forecast import forecast_batch, summarize
from gofa_f.short_generative_stage import config_check, record_validation
from test_allnode import tiny, source


@pytest.fixture(scope="module")
def codec():
    root = Path("/home/dcg/dcg_data/GOFA-F/data/weights/chattime-7b-20baa88")
    if not root.exists():
        pytest.skip("Real official tokenizer not installed on this CPU host")
    return ShortCodec(LlamaTokenizer.from_pretrained(root, local_files_only=True))


@pytest.mark.parametrize("length,horizon", TEMPLATES)
def test_canonical_full_answer_and_finite_alphabet(codec, length, horizon):
    h = np.linspace(0, 10, length)
    row = codec.record(h, np.linspace(-1, 11, horizon))
    suffix = row["ids"][row["prompt_end"]+1:]
    expected = []
    for i, position in enumerate(row["answer_positions"]):
        expected.extend(codec.first_separator if i == 0 else codec.separator)
        expected.append(row["ids"][position])
    assert suffix == expected
    assert sum(x != -100 for x in row["labels"]) == horizon
    assert codec.tokenizer.eos_token_id not in suffix
    assert codec.nan_id not in codec.finite_ids
    assert len(set(codec.finite_ids)) == 10000


def test_missing_targets_only_causal_finite_prefix(codec):
    a = codec.record(np.arange(10.), [np.nan, 99., np.nan, -5.])
    b = codec.record(np.arange(10.), [np.nan, 999., np.nan, -50.])
    assert a["filled_response"] == [9., 99., 99., -5.]
    assert b["filled_response"][0] == 9.
    assert codec.nan_id not in a["ids"][a["prompt_end"]+1:]
    assert sum(x != -100 for x in a["labels"]) == 2
    assert a["scaling"] == b["scaling"]


def test_empty_training_cases_explicitly_rejected(codec):
    with pytest.raises(ValueError, match="observed value"):
        codec.record(np.arange(10.), [np.nan]*24)
    with pytest.raises(ValueError, match="history"):
        codec.record([np.nan]*10, [1.])
    result = forecast_batch(None, codec, [np.full(10, np.nan)], 24)
    assert not result["inverse_valid"].any()
    assert not result["predictions"].any()
    assert result["forward_calls"] == 0


def test_numeric_loss_shift_and_normalization(codec):
    labels = torch.tensor([[-100, codec.finite_ids[1], -100]])
    logits = torch.zeros(1, 3, 42001, requires_grad=True)
    loss, count = numeric_loss(logits, labels, codec)
    assert count == 1
    assert float(loss) == pytest.approx(np.log(10000), rel=1e-6)
    loss.backward()
    assert logits.grad[0, 0, codec.finite_ids[1]] < 0
    assert logits.grad[0, 1:].abs().sum() == 0
    assert logits.grad[0, 0, codec.nan_id] == 0
    with pytest.raises(ValueError, match="Nonnumeric|nonnumeric|Nonfinite"):
        numeric_loss(logits.detach(), torch.tensor([[-100, codec.nan_id, -100]]), codec)
    with pytest.raises(ValueError, match="No observed"):
        numeric_loss(logits.detach(), torch.full((1, 3), -100), codec)


@pytest.mark.parametrize("horizon", [1, 7, 12, 24])
def test_numeric_slots_not_token_budget_and_actual_generation_equivalence(codec, horizon):
    model, data = tiny(), source(9)
    with torch.no_grad():
        for exchange in model.adapter.exchanges:
            exchange.up.weight.normal_(std=.001)
    q = Query("pems04", 0, 500, 48, horizon)
    history = data.values[452:500, 0]
    context = model.encode_origins([build_graph(data, q)], codec)
    a = forecast_batch(model, codec, [history], horizon, context, cached=True)
    b = forecast_batch(model, codec, [history], horizon, context, cached=False)
    assert a["numeric_ids"] == b["numeric_ids"]
    assert a["generated_ids"] == b["generated_ids"]
    assert len(a["numeric_ids"][0]) == horizon
    assert len(a["generated_ids"][0]) == len(codec.first_separator)+horizon+(horizon-1)*len(codec.separator)
    assert a["forward_calls"] == horizon
    assert a["format_complete"] == [True]


def test_constant_range_buffer_and_boundaries(codec):
    row = codec.record(np.full(10, 3.))
    p, valid, boundary = codec.decode_ids([codec.finite_ids[0], codec.finite_ids[-1]], *row["scaling"], 3.)
    np.testing.assert_allclose(p, [2.5001, 4.4999], atol=1e-6)
    assert valid.all() and boundary.all()


def test_temporal_strata_disjoint_and_insufficient():
    points = np.arange(100, 300)
    chosen = stratified(points, 3, 24, np.random.default_rng(42), disjoint=True)
    assert len(set(chosen)) == 3 and min(np.diff(chosen)) >= 24
    with pytest.raises(ValueError, match="Insufficient"):
        stratified(np.array([1]), 3, 1, np.random.default_rng(42))
    data = source(3)
    data.segments = np.zeros(1000, int)
    data.prefix_counts = np.r_[np.zeros((1, 3), int), data.mask.cumsum(0)]
    data.mask[600:624, 0] = False
    data.prefix_counts = np.r_[np.zeros((1, 3), int), data.mask.cumsum(0)]
    assert 600 not in origins(data, 0, 48, 24, "validation")


def test_complete_four_template_three_origin_selection_and_bootstrap():
    queries = []
    for s in ("a", "b", "c", "d"):
        for length, horizon in TEMPLATES:
            for origin in (100, 200, 300):
                queries.append(dict(source=s, variable=0, length=length, horizon=horizon, origin=origin))
    n = len(queries)
    result = relative_score([2.]*n, [1.]*n, queries, {s: [1e-6] for s in "abcd"}, 3, 30)
    assert result["score"] == pytest.approx(2.)
    assert result["paired_bootstrap_95pct"] == pytest.approx([2., 2.])
    with pytest.raises(ValueError, match="complete"):
        relative_score([2.]*(n-1), [1.]*(n-1), queries[:-1], {s: [1e-6] for s in "abcd"}, 3, 0)


def metric_entry(variable, target, prediction):
    n = len(target)
    return dict(query=dict(source="x", variable=variable, horizon=n), prediction=prediction,
        target=target, observed=[True]*n, format_complete=True, inverse_valid=[True]*n,
        boundary=[False]*n, true_outside_range=[False]*n, empty_history=False,
        generated_token_count=2*n-1, forward_calls=n)


def test_macro_micro_mape_zero_and_wape_denominator():
    entries = [metric_entry(0, [0.], [1.]), metric_entry(1, [1.]*3, [4.]*3)]
    result = summarize(entries)
    assert result["macro"]["mae"] == 2.
    assert result["micro"]["mae"] == 2.5
    assert result["variables"]["x:0"]["wape_pct"] is None
    assert result["variables"]["x:0"]["mape_pct"] is None
    assert result["macro"]["mape_pct_eligible_variables"] == 1
    assert result["formatting"]["numeric_slots"] == 4


def test_best_and_early_reference_independent_and_no_duplicate_commit():
    state = dict(best_score=None, early_reference=None, bad_validations=0, last_validated_update=0)
    assert record_validation(state, 1., 500)
    assert record_validation(state, .9995, 1000)
    assert state["best_score"] == .9995 and state["early_reference"] == 1.
    assert state["bad_validations"] == 1
    before = deepcopy(state)
    assert not record_validation(state, .9995, 1000)
    assert state == before


def test_stage19_config_keeps_old_stage_unchanged():
    cfg = yaml.safe_load(Path("configs/stage19_short_generative_chattime.yaml").read_text())
    config_check(cfg)
    cfg["resources"]["admission_policy"] = "memory_cap_only"
    with pytest.raises(ValueError):
        config_check(cfg)


def test_batch_isolation_and_no_target_mask_input(codec):
    model = tiny()
    h = np.linspace(0., 3., 12)
    a = forecast_batch(model, codec, [h, h+100.], 7)
    b = forecast_batch(model, codec, [h], 7)
    assert a["numeric_ids"][0] == b["numeric_ids"][0]
    np.testing.assert_allclose(a["predictions"][0], b["predictions"][0], atol=0, rtol=0)


def test_state_checkpoint_keeps_partial_gradients_rng_and_cursor(codec, tmp_path):
    from gofa_f.allnode_stage import restore_checkpoint, save_checkpoint
    from gofa_f.short_progress import Progress
    model = tiny()
    optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=2e-4)
    tokens = codec.batch([codec.record(np.arange(10.), [8., 9.])])
    data = source(9)
    q = Query("pems04", 0, 500, 10, 2)
    context = model.encode_origins([build_graph(data, q)], codec)
    numeric_loss(model(tokens, contexts=context).logits, tokens.labels, codec)[0].backward()
    optimizer.step()
    context = model.encode_origins([build_graph(data, q)], codec)
    numeric_loss(model(tokens, contexts=context).logits, tokens.labels, codec)[0].backward()
    expected = {k: p.grad.clone() for k, p in model.adapter.named_parameters()}
    state = dict(updates=1, pending=23, pending_tokens=46, pending_loss=12., last_validated_update=0)
    progress = Progress(tmp_path)
    save_checkpoint(tmp_path/"last.pt", model, optimizer, state, progress, "binding")
    saved = torch.load(tmp_path/"last.pt", weights_only=False)
    model.zero_grad(set_to_none=True)
    restored, _ = restore_checkpoint(tmp_path/"last.pt", model, optimizer, "binding")
    assert restored == state
    assert torch.equal(torch.get_rng_state(), saved["rng"]["torch"])
    for name, p in model.adapter.named_parameters():
        torch.testing.assert_close(p.grad, expected[name], rtol=0, atol=0)
    assert len(optimizer.state) == 16


def test_no_future_graph_change_even_when_answer_extremes_change(codec):
    data = source(9)
    q = Query("pems04", 0, 500, 48, 24)
    graph = build_graph(data, q)
    data.values[500:] = 1e8
    data.mask[500:] = False
    assert build_graph(data, q).identity == graph.identity


def test_early_stopping_cannot_trigger_before_complete_sixth_validation():
    state = dict(best_score=None, early_reference=None, bad_validations=0, last_validated_update=0)
    for update in range(500, 3001, 500):
        record_validation(state, 1., update)
        assert (state["bad_validations"] >= 5) == (update == 3000)


def test_near_zero_rule_penalizes_graph_error_without_dropping_variables():
    queries = [dict(source="a", variable=0, length=length, horizon=horizon, origin=o)
               for length, horizon in TEMPLATES for o in (100, 200, 300)]
    a = relative_score([0.]*12, [0.]*12, queries, {"a": [1e-6]}, 3, 0)
    b = relative_score([2e-6]*12, [0.]*12, queries, {"a": [1e-6]}, 3, 0)
    assert a["score"] == 1. and b["score"] == pytest.approx(2.)
    assert b["near_zero_baselines"]["a"] == 1


def test_evaluation_shards_resume_without_repeating_complete_queries(codec, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from gofa_f.short_forecast import evaluate
    import gofa_f.short_forecast as forecasting
    data = source(3)
    qs = [dict(source="pems04", variable=0, origin=o, length=10, horizon=1, chunk=0) for o in (500, 600)]
    progress = SimpleNamespace(evaluation=lambda *a, **k: None)
    model = tiny()
    first = evaluate(model, codec, {"pems04": data}, qs, tmp_path, "native", "bound", progress, lambda: None, True)
    assert first["complete"]
    folder = tmp_path/"evaluations/native"
    (folder/"complete.json").unlink()
    (folder/"000001.json").unlink()
    original, calls = forecasting.forecast_batch, []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(forecasting, "forecast_batch", counted)
    second = evaluate(model, codec, {"pems04": data}, qs, tmp_path, "native", "bound", progress, lambda: None, True)
    assert len(calls) == 1 and second["errors"] == first["errors"]
    with pytest.raises(ValueError, match="binding"):
        evaluate(model, codec, {"pems04": data}, qs, tmp_path, "native", "wrong", progress, lambda: None, True)


def test_targets_cannot_run_before_source_checkpoint_lock(tmp_path, monkeypatch):
    import sys
    from gofa_f.short_generative_stage import main
    monkeypatch.setattr(sys, "argv", ["stage19", "--config", "configs/stage19_short_generative_chattime.yaml",
                                    "--output", str(tmp_path), "--evaluate-targets"])
    with pytest.raises(ValueError, match="sealed"):
        main()


def test_single_block_24_reports_late_steps_separately():
    e = metric_entry(0, [1.]*24, [1.]*16+[3.]*8)
    r = summarize([e])
    assert r["steps_1_to_16"]["mae"] == 0.
    assert r["steps_17_to_24"]["mae"] == 2.
