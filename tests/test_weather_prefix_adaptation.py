from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from gofa_f.generative_data import Query
from gofa_f.weather_checks import fixtures
from gofa_f.weather_loss import batch, generated_record, loss, prefix_key, records
from gofa_f.weather_policy import COUNTS, confirmation_pass, decision, macro, scheduled, subset
from gofa_f.weather_progress import Progress
from gofa_f.weather_state import validate
from test_allnode import source, tiny
from test_short_training import codec  # noqa: F401
from test_source_retraining import data
from gofa_f.source_retraining_policy import pools


def test_pool_coverage_schedule_and_weather_style_flip():
    original = pools({n: data(c) for n, c in [('pems04_flow', 307), ('pems04_speed', 307),
                                            ('boston2017', 16), ('penmanshiel2017', 14)]})
    result = subset(original['train'])
    assert {n: len(v) for n, v in result.items()} == COUNTS
    assert {q['variable'] for q in result['pems04_flow']} == {q['variable'] for q in result['pems04_speed']}
    for v in range(16):
        points = sorted(q['origin'] for q in result['boston2017'] if q['variable'] == v)
        assert len(points) == 32 and min(np.diff(points)) >= 24
    epochs = []
    for epoch in range(2):
        rows = []
        for update in range(16):
            events = [scheduled(result, ((epoch*16+update)*64)+i) for i in range(64)]
            assert sum(x['kind'] == 'weather_true' for x in events) == 16
            assert sum(x['kind'] == 'weather_generated' for x in events) == 16
            assert sum(x['kind'] == 'keep' for x in events) == 32
            rows.extend(events[:32])
        epochs.append(rows)
    assert len({prefix_key(x['query']) for x in epochs[0]}) == 512
    assert all(a['query'] == b['query'] and a['kind'] != b['kind'] for a, b in zip(*epochs))


def test_point_within_variable_macro_and_fixed_gate_boundaries():
    names = ['pems04_flow', 'pems04_speed', 'boston2017', 'penmanshiel2017']
    qs = [dict(source=n, variable=0) for n in names]+[dict(source='boston2017', variable=0)]
    rows = [dict(query=q, target=[0., 0.], prediction=[1., 3.], observed=[True, True]) for q in qs]
    rows[-1].update(prediction=[8., 999.], observed=[True, False])
    value = macro(rows, qs)
    assert value['views']['boston2017'] == 4.  # point average (1+3+8)/3, not mean of query MAEs5
    assert value['views']['pems04_flow'] == 2.
    native = dict(views={n: 10. for n in names})
    parent = dict(views=dict(native['views'], boston2017=11.))
    assert decision(50, dict(views=dict(native['views'], boston2017=9.9)), native, parent) == 'lock'
    assert decision(50, dict(views=dict(native['views'], boston2017=10.8)), native, parent) == 'continue'
    assert decision(100, dict(views=dict(native['views'], boston2017=10.8)), native, parent) == 'stop'
    bad = dict(views=dict(native['views'], boston2017=9., penmanshiel2017=10.0500001))
    assert decision(50, bad, native, parent) == 'stop'
    assert not confirmation_pass(bad, native, parent)


def test_mixed_prefix_uses_true_labels_and_canonical_shift(codec):  # noqa: F811
    h, y = np.arange(48, dtype=float), np.array([12., np.nan, 17.])
    other = codec.record(h, [42., 43., 44.])
    ids = other['ids'][other['prompt_end']+1:]
    truth = codec.record(h, y)
    row = generated_record(codec, h, 3, ids, y)
    assert row['labels'] == truth['labels'] and row['ids'] != truth['ids']
    assert row['answer_positions'] == truth['answer_positions']
    assert [row['labels'][p] for p in row['answer_positions']][1] == -100
    assert all(row['ids'][p] in codec.finite_set for p in row['answer_positions'])
    with pytest.raises(ValueError):
        generated_record(codec, h, 3, ids+[codec.nan_id], y)
    with pytest.raises(ValueError):
        generated_record(codec, h, 4, ids)


def test_domain_loss_matches_full64_formula_and_no_anchor_ce():
    torch.manual_seed(42)
    events = []
    for name, count in [('boston2017', 32), ('pems04_flow', 8), ('pems04_speed', 8), ('penmanshiel2017', 16)]:
        events.extend(dict(query=dict(source=name), kind='weather_true' if name == 'boston2017' else 'keep',
                           weight=2/3 if name == 'boston2017' else 4/3) for _ in range(count))
    scores, teacher = torch.randn(128, 7, requires_grad=True), torch.randn(128, 7)
    targets, rows, counts = torch.arange(128) % 7, torch.arange(64).repeat_interleave(2), torch.full((64,), 2)
    total, _ = loss((scores, targets, rows, counts), teacher, events)
    per_query = []
    for i, event in enumerate(events):
        own = slice(i*2, i*2+2)
        per_query.append(loss((scores[own], targets[own], torch.zeros(2, dtype=torch.long), torch.tensor([2])),
                              teacher[own], [dict(event, weight=1.)])[0])
    expected = (torch.stack(per_query[:32]).mean()+torch.stack(per_query[32:48]).mean()
                +torch.stack(per_query[48:]).mean())/3
    torch.testing.assert_close(total/64, expected)
    changed = targets.clone()
    changed[64:] = (changed[64:]+3) % 7
    again, _ = loss((scores, changed, rows, counts), teacher, events)
    torch.testing.assert_close(total, again, rtol=0, atol=0)


def test_keep_future_values_do_not_change_inputs_or_kl(codec):  # noqa: F811
    model = tiny()
    teacher = deepcopy(model.adapter).requires_grad_(False)
    series = dict(pems04_flow=source(9))
    query = asdict(Query('pems04_flow', 0, 500, 12, 12))
    h = series['pems04_flow'].values[488:500, 0]
    row = codec.record(h, np.full(12, h[-1]))
    prefixes = {prefix_key(query): dict(query=query, generated_ids=row['ids'][row['prompt_end']+1:])}
    events = [dict(query=query, kind='keep', weight=4/3)]
    a, ga = records(codec, series, events, prefixes)
    value, _, _ = batch(model, teacher, codec, series, events, prefixes)
    series['pems04_flow'].values[500:] = np.nan
    series['pems04_flow'].mask[500:] = False
    b, gb = records(codec, series, events, prefixes)
    after, _, _ = batch(model, teacher, codec, series, events, prefixes)
    assert a[0]['ids'] == b[0]['ids'] and a[0]['labels'] == b[0]['labels']
    assert torch.equal(ga[0].graph.history, gb[0].graph.history)
    torch.testing.assert_close(value, after, rtol=0, atol=0)


def test_mixed_prefix_microbatch_gradients_and_frozen_teacher(codec):  # noqa: F811
    model = tiny()
    with torch.no_grad():
        for layer in model.adapter.exchanges:
            layer.up.weight.normal_(std=.001)
    teacher = deepcopy(model.adapter).requires_grad_(False)
    with torch.no_grad():
        model.adapter.exchanges[0].up.weight.add_(.0001)
    series = {n: source(9) for n in ['boston2017', 'pems04_flow', 'penmanshiel2017']}
    qs = [asdict(Query(n, 0, 500, 48 if n == 'boston2017' else 12, 24 if n == 'boston2017' else 12)) for n in series]
    events, prefixes = fixtures(codec, series, qs)
    value, _, _ = batch(model, teacher, codec, series, events[2:], prefixes)
    value.backward()
    expected = {n: p.grad.clone() for n, p in model.adapter.named_parameters()}
    model.zero_grad(set_to_none=True)
    for event in events[2:]:
        value, _, _ = batch(model, teacher, codec, series, [event], prefixes, shared=False)
        value.backward()
    for n, p in model.adapter.named_parameters():
        torch.testing.assert_close(p.grad, expected[n], rtol=1e-6, atol=1e-6)
    assert all(p.grad is None for p in teacher.parameters())
    assert all(p.grad is None for p in model.native.parameters())


def test_policy_progress_resource_admission_and_target_guard(tmp_path):
    import gofa_f.short_launcher as launcher
    from gofa_f.robust_targets import run
    cfg = yaml.safe_load(Path('configs/stage25_weather_prefix_adaptation.yaml').read_text())
    validate(cfg)
    gate = launcher.admission_gate(cfg['resources'], 'weather_prefix_adaptation')
    assert gate.observe(dict(uuid='GPU-test', compute_mode='Default', pids=[123], used_mib=80000, utilization=100))
    with pytest.raises(ValueError):
        validate(dict(cfg, training=dict(cfg['training'], max_updates=200)))
    t = [0.]
    p = Progress(tmp_path, clock=lambda: t[0])
    p.tick('train', force=True)
    t[0] = 100.
    value = p.tick(updates=16, measured_updates=16, prefix_done=1024, prefix_seconds=500,
                   evaluation_seconds_per_slot=[.03], force=True)
    assert value['views']['boston2017']['epoch'] == 2
    assert value['views']['boston2017']['epoch_queries_done'] == 0
    assert value['overall_eta_seconds'] > value['training_eta_seconds']
    with pytest.raises(ValueError, match='confirmation'):
        run(None, None, {}, tmp_path, None, None, gate_name='source_gate.json')
