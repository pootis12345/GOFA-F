from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gofa_f.dual_memory_data import memory_input
from gofa_f.dual_memory_eval_data import read,seal,InputStore
from gofa_f.dual_memory_eval_records import PredictionStore
from gofa_f.dual_memory_eval_prepare import require_confirmation
from gofa_f.dual_memory_evaluation import configuration,guard,summarize
from gofa_f.generative_data import Query,sha
from gofa_f.prefix_cache import atomic_json
from gofa_f.short_sampling_diagnostic import SETTINGS


def example(tmp_path):
    q=asdict(Query('pems04_flow',0,70,12,12));data=SimpleNamespace(values=np.arange(300,dtype=np.float32).reshape(100,3),
        mask=np.ones((100,3),bool),external=np.ones((3,3))-np.eye(3),ids=['a','b','c'],segments=np.zeros(100,int))
    item=memory_input(data,Query(**q))
    manifest=dict(queries=[q],configuration=dict(sampling=SETTINGS),prediction_execution='qa',deployment=dict(weights_sha256='native',
        expert=dict(checkpoint_sha256='expert'),gate=dict(sha256='gate'),ufg={'12-12':dict(checkpoint_sha256='ufg')}))
    atomic_json(tmp_path/'manifest.json',manifest)
    return q,item,PredictionStore(tmp_path)


def test_atomic_prediction_refuses_binding_change_and_nonfinite(tmp_path):
    q,item,store=example(tmp_path)
    p=np.arange(12)
    store.save(q,item,'enhanced',p,.1)
    assert store.get(q,item,'enhanced')['prediction']==p.tolist()
    with pytest.raises(ValueError,match='replaced'):store.save(q,item,'enhanced',p,.1)
    with pytest.raises(ValueError,match='Nonfinite|nonfinite'):store.save(q,item,'fusion',np.full(12,np.nan),.1)
    item.identity='changed'
    with pytest.raises(ValueError,match='binding changed'):store.get(q,item,'enhanced')


def test_native_trajectory_resume_only_fills_missing_paths_and_checks_median(tmp_path):
    q,item,store=example(tmp_path);calls=[]
    def generate(seed):
        calls.append(seed)
        return dict(prediction=[float(len(calls))]*12,numeric_ids=list(range(12)),generated_ids=list(range(12)))
    first,_=store.trajectory(q,item,0,generate);again,_=store.trajectory(q,item,0,generate)
    assert first==again and len(calls)==1
    values=[store.trajectory(q,item,i,generate)[0]['result']['prediction'] for i in range(8)]
    store.save(q,item,'native',np.median(values,axis=0),1.)
    (tmp_path/'inputs').mkdir();atomic_json(tmp_path/'inputs/index.json',dict(qa=True))
    inputs=SimpleNamespace(get=lambda row:(item,'metadata'))
    proof=store.complete('native',inputs)
    assert read(proof)['queries']==1 and len(read(proof)['shards'])==9 and len(calls)==8
    path=store.path(q,'native');changed=read(path);changed['prediction'][0]+=1;atomic_json(path,changed)
    with pytest.raises(ValueError,match='median'):store.complete('native',inputs)


def test_native_trajectory_missing_record_cannot_be_certified(tmp_path):
    q,item,store=example(tmp_path);store.save(q,item,'native',np.zeros(12),1.)
    (tmp_path/'inputs').mkdir();atomic_json(tmp_path/'inputs/index.json',{})
    with pytest.raises(ValueError,match='Missing saved Native'):store.complete('native',SimpleNamespace(get=lambda row:(item,'')))


def test_target_preparation_requires_unchanged_completed_source_confirmation(tmp_path):
    root=tmp_path/'confirmation';root.mkdir();atomic_json(root/'report.json',dict(score=1.));atomic_json(root/'manifest.json',dict(inputs=True))
    locked=dict(expert='locked')
    cert=dict(complete=True,deployment=locked,all_model_groups_complete=True,parameters_modified=False,
        output=str(root),report_sha256=sha(root/'report.json'),manifest_sha256=sha(root/'manifest.json'))
    path=tmp_path/'certificate.json';atomic_json(path,cert)
    assert require_confirmation(path,locked)['complete']
    with pytest.raises(ValueError,match='unchanged-model'):require_confirmation(path,dict(expert='other'))
    atomic_json(root/'report.json',dict(score=0.))
    with pytest.raises(ValueError,match='report changed'):require_confirmation(path,locked)


def test_configuration_forbids_target_training_sampling_changes_and_caps(tmp_path,monkeypatch):
    import yaml
    monkeypatch.setenv('GOFA_WORKSPACE','/qa')
    original=__import__('pathlib').Path(__file__).parents[1]/'configs/stage59_dual_memory_evaluation.yaml'
    cfg=yaml.safe_load(original.read_text());path=tmp_path/'cfg.yaml';path.write_text(yaml.safe_dump(cfg))
    assert configuration(path)['training']['max_updates']==0
    for field,value in [('data',dict(target_training_enabled=True,target_evaluation_enabled=True)),
            ('sampling',dict(SETTINGS,samples=1)),('resources',dict(cfg['resources'],memory_cap_gib=41)),('training',dict(max_updates=1))]:
        changed=dict(cfg);changed[field]=value;path.write_text(yaml.safe_dump(changed))
        with pytest.raises(ValueError):configuration(path)


def test_evaluation_has_no_optimizer_or_backprop_entrypoint():
    import ast,inspect
    import gofa_f.dual_memory_evaluation as module
    tree=ast.parse(inspect.getsource(module))
    assert not [n for n in ast.walk(tree) if isinstance(n,ast.Attribute) and n.attr in ('optim','backward','step')]


def test_readonly_ufg_worker_saves_two_shapes_and_never_reads_labels(tmp_path,monkeypatch):
    import sys,types
    from gofa_f.dual_memory_engines import model_digest
    import gofa_f.dual_memory_evaluation as evaluation
    class TinyUFG(torch.nn.Module):
        def __init__(self,root,length,horizon,device='cpu'):
            super().__init__();self.bias=torch.nn.Parameter(torch.tensor(2.));self.horizon=horizon
        def forward(self,packed):
            return self.bias.double().expand(1,self.horizon)
    monkeypatch.setitem(sys.modules,'gofa_f.ufg_memory_roster',types.SimpleNamespace(MemoryRosterUFG=TinyUFG,pack_roster=lambda item:None))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    q,item,_=example(tmp_path);other=dict(q,length=48,horizon=24)
    base=read(tmp_path/'manifest.json');base['scope']='source_confirmation';base['queries'].append(other)
    for shape in ('12-12','48-24'):
        model=TinyUFG(None,*map(int,shape.split('-')));path=tmp_path/(shape+'.pt');torch.save(dict(model=model.state_dict()),path)
        base['deployment']['ufg'][shape]=dict(path=str(path),checkpoint_sha256=sha(path),model_sha256=model_digest(model))
    atomic_json(tmp_path/'manifest.json',base);atomic_json(tmp_path/'ufg_activity.json',dict(spent_seconds=7.,worker=None))
    (tmp_path/'inputs').mkdir();atomic_json(tmp_path/'inputs/index.json',dict(qa=True))
    outputs=PredictionStore(tmp_path)
    inputs=SimpleNamespace(get=lambda row:(item,'metadata'))
    monkeypatch.setattr(evaluation,'guard',lambda *_:(base,inputs,outputs));monkeypatch.setattr(evaluation,'verify_deployment',lambda *_:None)
    evaluation.run_ufg(dict(ufg_root='not-a-real-model'),tmp_path)
    assert read(tmp_path/'ufg_complete.json')['queries']==2
    assert read(tmp_path/'ufg_status.json')['source_weights_unchanged']
    assert outputs.get(other,item,'ufg')['prediction']==[2.]*24
    assert read(tmp_path/'ufg_activity.json')['worker'] is None and read(tmp_path/'ufg_activity.json')['spent_seconds']>=7.


def test_cuda_oom_stops_for_strategy_audit_without_an_unmodified_restart(tmp_path,monkeypatch):
    import sys
    import gofa_f.dual_memory_evaluation as module
    monkeypatch.setattr(sys,'argv',['evaluation','--config','dummy','--output',str(tmp_path)])
    monkeypatch.setattr(module,'configuration',lambda *_:{})
    monkeypatch.setattr(module,'run_gpu',lambda *_:(_ for _ in ()).throw(torch.OutOfMemoryError('fixed roster needs a memory-strategy repair')))
    with pytest.raises(torch.OutOfMemoryError):module.main()
    assert read(tmp_path/'status.json')['state']=='failed'
    assert len(list(tmp_path.glob('failure-*.json')))==1
