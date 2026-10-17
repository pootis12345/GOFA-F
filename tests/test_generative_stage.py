from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from gofa_f.generative_audit import align_nasdaq, geographic_stations, noaa_hourly
from gofa_f.generative_launcher import own_usage
from gofa_f.generative_progress import Progress
from gofa_f.generative_stage import (config_check, restore_checkpoint, save_checkpoint, schedule)
from test_generative import graph, tiny, tokens
from gofa_f.generative_model import completion_loss


def test_pinned_config_and_no_time_or_target_shortcuts():
    cfg = yaml.safe_load(open("configs/stage17_generative_chattime.yaml"))
    config_check(cfg)
    for section, key, value in [("data", "target_evaluation_enabled", True),
        ("resources", "wall_deadline", "2099-01-01"), ("resources", "activity_limit_seconds", 28800),
        ("architecture", "graph_layers", [29, 30])]:
        changed = deepcopy(cfg)
        changed[section][key] = value
        with pytest.raises(ValueError):
            config_check(changed)


def test_real_checkpoint_pending_gradient_rng_and_next_update(tmp_path):
    model, batch = tiny(), tokens()
    optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=2e-4)
    graphs = [graph(), graph(5, target=2)]
    completion_loss(model(batch, graphs).logits, batch.labels)[0].backward()
    state = {"updates": 0, "pending": 2, "pending_tokens": 6, "bad_validations": 3}
    progress = Progress(tmp_path)
    save_checkpoint(tmp_path/"last.pt", model, optimizer, state, progress, "bound")
    completion_loss(model(batch, graphs).logits, batch.labels)[0].backward()
    optimizer.step()
    expected = model.trainable_state()
    restored, p = restore_checkpoint(tmp_path/"last.pt", model, optimizer, "bound")
    assert restored == state and p["updates"] == 0
    completion_loss(model(batch, graphs).logits, batch.labels)[0].backward()
    optimizer.step()
    for n, value in model.trainable_state().items():
        torch.testing.assert_close(value, expected[n], rtol=0, atol=0)
    with pytest.raises(ValueError):
        restore_checkpoint(tmp_path/"last.pt", model, optimizer, "different")


def test_schedule_balanced_and_resume_cursor():
    names = ["solar", "nn5", "pems04", "aq_shunyi"]
    pools = {n: [dict(source=n, variable=0, origin=i, length=10, horizon=1, chunk=0)
                 for i in range(8192)] for n in names}
    selected = [schedule(pools, i, 42) for i in range(64)]
    assert all(sum(q.source == name for q in selected) == 16 for name in names)
    assert schedule(pools, 49, 42) == selected[49]


def test_nasdaq_timestamp_alignment_lltc_pseudozero_and_sessions():
    times = pd.date_range("2016-07-26", periods=160, freq="B")+pd.Timedelta(hours=10)
    index = pd.Series(np.arange(160.)+5000, index=times)
    stock_times = times.insert(0, times[0]-pd.Timedelta(minutes=1))
    stocks = {"AAPL": pd.Series(np.arange(161.), index=stock_times),
              "LLTC": pd.Series(np.r_[1., np.ones(157), np.zeros(3)], index=stock_times)}
    aligned, sessions = align_nasdaq(stocks, index)
    assert aligned["AAPL"].iloc[0] == 1
    assert aligned["LLTC"].iloc[156] == 1
    assert aligned["LLTC"].iloc[157:].isna().all()
    assert len(set(sessions)) == 160


def test_noaa_correct_temperature_utc_and_missing():
    frame = pd.DataFrame([[12345, 20250101, 0, 20241231, 1800, 3, -100., 35., 99., 12.5],
                          [12345, 20250101, 100, 20241231, 1900, 3, -100., 35., 99., -9999.]])
    values, metadata = noaa_hourly(frame)
    assert values.iloc[0] == 12.5 and np.isnan(values.iloc[1])
    assert str(values.index[0]) == "2025-01-01 00:00:00+00:00"
    assert metadata["field"] == "T_HR_AVG"


def test_noaa_selection_uses_only_metadata():
    stations = [dict(wban=str(10000+i), latitude=25+i, longitude=-120+i) for i in range(20)]
    chosen = geographic_stations(stations)
    assert len(chosen) == 16 and chosen[0]["wban"] == "10000"
    assert geographic_stations(list(reversed(stations))) == chosen


def test_own_usage_ignores_foreign_allocations(monkeypatch):
    monkeypatch.setattr("gofa_f.generative_launcher.own_gpu_pids", lambda _: {10})
    gpu = {"pids": [10, 20], "process_used_mib": {10: 26000, 20: 50000}}
    assert own_usage(gpu, 10) == 26000
    gpu["process_used_mib"][10] = None
    assert own_usage(gpu, 10) is None


def test_final_partial_epoch_and_early_stop_eta(tmp_path):
    p = Progress(tmp_path)
    p.tick("complete", force=True, updates=4000)
    saved = json.loads((tmp_path/"progress.json").read_text())
    assert saved["epoch"] == 8 and saved["epoch_queries_total"] == 26624
    assert saved["epoch_queries_done"] == 26624 and saved["overall_eta_seconds"] == 0
