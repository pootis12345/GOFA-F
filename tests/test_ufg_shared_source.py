from copy import deepcopy
from pathlib import Path
import pytest
import yaml

from gofa_f.ufg_shared_source import configuration
from gofa_f.dual_memory_policy import SourceSchedule
from gofa_f.dual_memory_data import GROUPS


def test_config_requires_one_shared_model_and_no_target_training(tmp_path,monkeypatch):
    monkeypatch.setenv('GOFA_WORKSPACE','/workspace')
    config=Path(__file__).parents[1]/'configs/stage65_shared_ufg_source.yaml'
    cfg=configuration(config)
    for mutate in [lambda c:c['model'].update(shape_specific_weights=True),
        lambda c:c['model'].update(single_parameter_set=False),
        lambda c:c.update(target_training_enabled=True),
        lambda c:c['training'].update(max_updates=1000)]:
        own=deepcopy(cfg);mutate(own);path=tmp_path/'changed.yaml';path.write_text(yaml.safe_dump(own))
        with pytest.raises(ValueError):configuration(path)


def test_source_schedule_mixes_both_shapes_with_three_domain_balance():
    queries={n:{s:dict(train=[dict(source=n,length=int(s.split('-')[0]),horizon=int(s.split('-')[1]),
        variable=v,origin=100+v) for v in range(4)]) for s in ('12-12','48-24')} for n in GROUPS}
    schedule=SourceSchedule(queries,shape=None,initial=8192,expand=True);seen=[]
    for _ in range(24):
        q,state=schedule.peek();schedule.commit(state);seen.append(q)
    assert {tuple((q['length'],q['horizon'])) for q in seen}=={(12,12),(48,24)}
    assert {d:sum(GROUPS[q['source']]==d for q in seen) for d in ('traffic','weather','wind')}==dict(traffic=8,weather=8,wind=8)
    for n in GROUPS:assert sum(q['source']==n and q['length']==12 for q in seen)==sum(q['source']==n and q['length']==48 for q in seen)
