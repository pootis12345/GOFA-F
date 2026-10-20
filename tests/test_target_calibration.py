import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from gofa_f.allnode_stage import read
from gofa_f.generative_data import Query, sha
from gofa_f.prefix_cache import atomic_json
from gofa_f.short_data import TARGETS
from gofa_f.short_forecast import summarize
from gofa_f.short_sampling_diagnostic import SETTINGS, sample_seed
import gofa_f.target_calibration as stage
from gofa_f.target_calibration_math import blend, calibrate_dataset, fit_coefficient, fused_metrics, macro_mae
from gofa_f.target_calibration_state import result_binding, validation_queries


def entry(name='etth1',variable=0,origin=500,native=0.,truth=2.,horizon=1,mask=None):
    q = vars(Query(name,variable,origin,10,horizon))
    p = [native]*horizon
    return dict(query=q,prediction=p,target=[truth]*horizon,observed=mask or [True]*horizon,
        scaling=[1.,0.],representable_range=[-100.,100.],inverse_valid=[True]*horizon,
        boundary=[False]*horizon,true_outside_range=[False]*horizon,
        format_complete=True,empty_history=False,generated_token_count=horizon,
        forward_calls=horizon,mae=abs(native-truth),sampling=SETTINGS,
        seeds=[sample_seed(q,s) for s in range(8)])


@pytest.mark.parametrize('truth,coefficient',[(0.,0.),(2.,.2),(10.,1.),(-2.,0.),(12.,1.)])
def test_breakpoint_solver_and_exact_endpoints(truth,coefficient):
    n, g = [entry(truth=truth)], [entry(native=10.,truth=truth)]
    before = copy.deepcopy((n,g))
    fitted = fit_coefficient(n,g)
    assert fitted['candidate'] == coefficient
    assert fitted['candidate_mae'] <= min(fitted['native_mae'],fitted['graph_mae'])
    assert (n,g) == before
    assert np.array_equal(blend([1e20,-1e20],[.1,.2],0),[1e20,-1e20])
    assert np.array_equal(blend([1e20,-1e20],[.1,.2],1),[.1,.2])


def test_macro_weighting_ties_missing_and_nonfinite_rejection():
    # Two variables, not three points: the entire interval is optimal, choose 0.
    n = [entry(variable=0,truth=10.,horizon=2),entry(variable=1,truth=0.)]
    g = copy.deepcopy(n)
    for row in g:
        row['prediction'] = [10.]*row['query']['horizon']
    assert fit_coefficient(n,g)['candidate'] == 0.
    assert macro_mae(n,g,.3) == pytest.approx(5.)
    a, b = entry(horizon=2,mask=[True,False]), entry(native=10.,horizon=2,mask=[True,False])
    a['target'][1] = b['target'][1] = -1e20
    assert fit_coefficient([a],[b])['candidate'] == .2
    b['scaling'] = [2.,0.]
    with pytest.raises(ValueError,match='scaling'):
        fit_coefficient([a],[b])
    with pytest.raises(ValueError,match='finite'):
        blend([np.nan],[1.],.5)


def test_identical_predictions_zero_error_and_confirmation_reject_without_refit():
    n = [entry(origin=500+i) for i in range(128)]
    g = [entry(origin=500+i,native=10.) for i in range(128)]
    for rows in (n,g):
        for row in rows[96:]:
            row['target'] = [-1.]
    fitted = calibrate_dataset(n,g)
    assert fitted['candidate'] == .2 and fitted['coefficient'] == 0.
    assert not fitted['confirmation_supported'] and not fitted['refit_on_all_validation']
    perfect = [entry(native=2.,origin=500+i) for i in range(128)]
    zero = calibrate_dataset(perfect,copy.deepcopy(perfect))
    assert zero['candidate'] == zero['coefficient'] == 0.
    assert not zero['confirmation_supported']


def test_fusion_is_derived_original_unit_prediction_not_generated_coverage():
    n, g = [entry(horizon=2)], [entry(native=10.,horizon=2)]
    n[0]['inverse_valid'][1] = False
    before = copy.deepcopy((n,g))
    result = fused_metrics(n,g,.2)
    assert result['macro']['mae'] == 0 and 'formatting' not in result
    assert result['derived_prediction']['generation_coverage'] is None
    assert result['derived_prediction']['effective_base_fallback_slots'] == 1
    assert not result['derived_prediction']['generated_text'] and (n,g) == before


def fake_series():
    result = {}
    for name,(_,_,nodes) in TARGETS.items():
        result[name] = SimpleNamespace(values=np.broadcast_to(np.zeros((8000,1)),(8000,nodes)),
            prefix_counts=np.broadcast_to(np.arange(8001)[:,None],(8001,nodes)),segments=np.zeros(8000))
    return result


def test_1152_queries_are_deterministic_balanced_past_only_and_quarter_disjoint():
    series = fake_series()
    a, coverage = validation_queries(series)
    b, _ = validation_queries(series)
    assert a == b and sum(map(len,a.values())) == 1152
    assert sum(q['horizon'] for rows in a.values() for q in rows)*16 == 335872
    for name,rows in a.items():
        cuts = coverage[name]['quarter_boundaries']
        expected = min(128,1 if name == 'nasdaq100' else TARGETS[name][2])
        assert len(set(q['variable'] for q in rows)) == expected
        assert len(set(Query(**q).key for q in rows)) == 128
        for fold in range(4):
            for q in rows[fold*32:(fold+1)*32]:
                assert cuts[fold] <= q['origin'] and q['origin']+q['horizon'] <= cuts[fold+1]
                assert q['origin'] < 6400
        for v in set(q['variable'] for q in rows):
            times = sorted(q['origin'] for q in rows if q['variable'] == v)
            assert np.all(np.diff(times) >= TARGETS[name][1])


def build_archive(tmp_path):
    output, parent = tmp_path/'new', tmp_path/'old-test'
    output.mkdir()
    parent.mkdir()
    cfg = dict(stage19_output=str(parent),stage19_manifest_sha256='old-manifest',adapter_sha256='adapter')
    manifest = dict(configuration=cfg,weights_sha256='backbone',alphabet={'alphabet_sha256':'alphabet'},
        queries={},test_queries={},test_query_hashes={},test_result_hashes={},coverage={})
    atomic_json(output/'manifest.json',{'fixture':True})
    binding = sha(output/'manifest.json')
    complete = {}
    for name in TARGETS:
        queries = [entry(name=name,origin=500+i)['query'] for i in range(128)]
        manifest['queries'][name] = queries
        manifest['test_queries'][name] = [entry(name=name,origin=900+i)['query'] for i in range(2)]
        manifest['test_query_hashes'][name] = {}
        manifest['test_result_hashes'][name] = {}
        for mode,prediction in (('native',0.),('gofa',10.)):
            label = f'validation-{name}-{mode}'
            root = output/'evaluations'/label
            (root/'trajectories').mkdir(parents=True)
            expected = result_binding(manifest,binding,name,mode)
            for i,q in enumerate(queries):
                row = entry(name=name,origin=q['origin'],native=prediction)
                row['binding'] = expected
                atomic_json(root/f'{i:06d}.json',row)
                for s in range(8):
                    atomic_json(root/'trajectories'/f'{i:06d}-{s:02d}.json',
                        dict(query=q,binding=expected,prediction=[prediction],seed=sample_seed(q,s),sample=s))
            result_path = output/(label+'-result.json')
            atomic_json(result_path,dict(binding=expected,complete=True,queries=128,sampling=SETTINGS,
                errors=[abs(prediction-2.)]*128,numeric_slots_in_eight_paths=1024))
            complete[label] = sha(result_path)
            label = f'target-{name}-{mode}'
            root = parent/'evaluations'/label
            root.mkdir(parents=True)
            expected = f"old-manifest:sample8:{label}:"+('backbone' if mode=='native' else 'adapter')+':alphabet'
            records = []
            for i,q in enumerate(manifest['test_queries'][name]):
                row = entry(name=name,origin=q['origin'],native=prediction)
                row['binding'] = expected
                atomic_json(root/f'{i:06d}.json',row)
                records.append(row)
            manifest['test_query_hashes'][name][mode] = [sha(root/f'{i:06d}.json') for i in range(2)]
            file = parent/(label+'-result.json')
            atomic_json(file,dict(complete=True,binding=expected,sampling=SETTINGS,queries=2,metrics=summarize(records)))
            manifest['test_result_hashes'][name][mode] = sha(file)
    atomic_json(output/'validation_complete.json',dict(binding=binding,complete=True,results=complete))
    return cfg,output,parent,manifest


def test_calibration_test_isolation_seal_replay_and_changed_input_rejection(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    cfg, output, parent, manifest = build_archive(tmp_path)
    monkeypatch.setattr(stage,'own_manifest',lambda *_:manifest)
    monkeypatch.setattr(stage,'verify_inference',lambda *_:manifest)
    with pytest.raises(ValueError,match='sealed'):
        stage.replay_test(cfg,output)
    # The complete inherited test directory is unavailable during fitting.
    parent.rename(tmp_path/'inaccessible-test')
    coefficients = stage.calibrate(cfg,output)
    assert all(v['coefficient'] == .2 for v in coefficients['coefficients'].values())
    assert not coefficients['test_records_used']
    (tmp_path/'inaccessible-test').rename(parent)
    value = stage.replay_test(cfg,output)
    assert value['complete'] and value['results']['all_nine_strictly_improved']
    assert not value['strict_zero_shot'] and value['results']['test_gpu_predictions_generated'] == 0
    assert stage.calibrate(cfg,output) == coefficients
    before = (output/'coefficients.json').read_bytes()
    bad = copy.deepcopy(coefficients)
    bad['coefficients']['etth1']['coefficient'] = .3
    atomic_json(output/'coefficients.json',bad)
    with pytest.raises(ValueError,match='sealed'):
        stage.replay_test(cfg,output)
    (output/'coefficients.json').write_bytes(before)
    path = output/'evaluations/validation-etth1-native/trajectories/000000-00.json'
    row = read(path)
    row['prediction'] = [1.] # Median unchanged, but input hash must still reject it.
    atomic_json(path,row)
    with pytest.raises(ValueError,match='Validation changed'):
        stage.replay_test(cfg,output)


def test_no_gpu_without_admission_and_cpu_modes_forbid_context(tmp_path,monkeypatch):
    monkeypatch.delenv('GOFA_STAGE19_ADMITTED',raising=False)
    with pytest.raises(RuntimeError,match='admission'):
        stage.predict_validation({},tmp_path)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-forbidden')
    with pytest.raises(RuntimeError,match='CUDA disabled'):
        stage.cpu_only()


def test_test_results_cannot_reopen_validation_rejected_coefficient(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    cfg = {'adapter_sha256':'locked'}
    manifest = {'configuration':cfg,'coverage':{}}
    atomic_json(tmp_path/'manifest.json',{'fixture':True})
    datasets = {}
    for name in TARGETS:
        n = [entry(name=name,origin=500+i,truth=2. if i<96 else -1.) for i in range(128)]
        g = [entry(name=name,origin=500+i,native=10.,truth=2. if i<96 else -1.) for i in range(128)]
        datasets[name] = dict(native=n,gofa=g)
    monkeypatch.setattr(stage,'own_manifest',lambda *_:manifest)
    monkeypatch.setattr(stage,'verify_inference',lambda *_:manifest)
    monkeypatch.setattr(stage,'validation_records',lambda *_:(datasets,'fixed-validation-only'))
    seal = stage.calibrate(cfg,tmp_path)
    original = (tmp_path/'coefficients.json').read_bytes()
    def records(cfg,manifest,name,mode):
        # A different coefficient would be perfect on test; it must not be chosen.
        rows = [entry(name=name,native=0. if mode=='native' else 10.,truth=2.)]
        return rows,summarize(rows)
    monkeypatch.setattr(stage,'test_records',records)
    result = stage.replay_test(cfg,tmp_path)
    assert not result['results']['all_nine_strictly_improved']
    assert result['results']['improved_datasets'] == 0
    assert all(v['candidate']==.2 and v['coefficient']==0. for v in seal['coefficients'].values())
    assert (tmp_path/'coefficients.json').read_bytes() == original


def test_validation_worker_is_no_grad_preserves_flags_and_stops_before_fitting(tmp_path,monkeypatch):
    from test_allnode import tiny
    model = tiny()
    torch.save(model.trainable_state(),tmp_path/'adapter.pt')
    cfg = {'adapter_sha256':sha(tmp_path/'adapter.pt')}
    manifest = dict(configuration=cfg,source_configuration={'data':{'directory':str(tmp_path)}},
        weights_sha256='weights',alphabet={'alphabet_sha256':'alphabet'},total_numeric_work=18432,
        queries={name:[vars(Query(name,0,500+i,10,1)) for i in range(128)] for name in TARGETS})
    atomic_json(tmp_path/'manifest.json',manifest)
    atomic_json(tmp_path/'cpu_preflight.json',dict(passed=True,manifest_sha256=sha(tmp_path/'manifest.json')))
    for name in ('GOFA_STAGE19_ADMITTED','GOFA_STAGE19_SUPERVISOR_PID','GOFA_STAGE19_SUPERVISOR_TICKS'):
        monkeypatch.setenv(name,'1')
    monkeypatch.setattr(stage.signal,'signal',lambda *_:None)
    monkeypatch.setattr(stage,'verify_inference',lambda *_:manifest)
    monkeypatch.setattr(stage,'load_model',lambda *_:(model,SimpleNamespace(alphabet_sha='alphabet')))
    monkeypatch.setattr(stage,'Series',lambda *_:None)
    monkeypatch.setattr(stage,'supervisor_alive',lambda *_:True)
    monkeypatch.setattr(torch.cuda,'set_per_process_memory_fraction',lambda *_:None)
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda *_:SimpleNamespace(total_memory=80*2**30))
    for name in ('max_memory_allocated','max_memory_reserved','reset_peak_memory_stats'):
        monkeypatch.setattr(torch.cuda,name,lambda:0)
    events = []
    def acceptance(*_,**kwargs):
        events.append('acceptance')
        return dict(passed=True,gradients_created=False)
    monkeypatch.setattr(stage,'mechanisms',acceptance)
    monkeypatch.setattr(stage,'calibrate',lambda *_:pytest.fail('GPU worker fitted coefficients'))
    def evaluate(*args):
        assert events[0] == 'acceptance' and not torch.is_grad_enabled()
        assert model.training and model.adapter.training and not model.native.training
        assert all(p.requires_grad for p in model.adapter.parameters())
        events.append(args[5])
        return dict(complete=True,queries=128,numeric_slots_in_eight_paths=1024,
            active_seconds=1.,binding=args[6],sampling=SETTINGS)
    monkeypatch.setattr(stage,'evaluate',evaluate)
    stage.predict_validation(cfg,tmp_path)
    assert len(events) == 19 and read(tmp_path/'status.json')['state'] == 'validation_complete'
    assert not (tmp_path/'coefficients.json').exists() and not (tmp_path/'test_replay.json').exists()
    assert not any(p.grad is not None for p in model.parameters())
    stage.predict_validation(cfg,tmp_path)
    assert events.count('acceptance') == 1 # Bound proof survives a resource restart.


def test_launcher_cpu_finalization_does_not_query_or_start_gpu(tmp_path,monkeypatch):
    import gofa_f.short_launcher as launcher
    cfg = yaml.safe_load(Path('configs/stage20_target_calibration.yaml').read_text())
    config = tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump(cfg))
    atomic_json(tmp_path/'status.json',dict(state='validation_complete'))
    events = []
    monkeypatch.setattr(launcher.signal,'signal',lambda *_:None)
    monkeypatch.setattr(launcher,'query_gpu',lambda:pytest.fail('CPU finalization queried GPU admission'))
    def run(command,**kwargs):
        assert command[-1] == '--finish-cpu' and kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
        assert read(tmp_path/'activity.json')['worker'] is None
        events.append(command)
        atomic_json(tmp_path/'status.json',dict(state='complete',goal_achieved=False))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(launcher.subprocess,'run',run)
    launcher.supervise(tmp_path,config,Path('.'),'target_calibration')
    assert len(events) == 1 and read(tmp_path/'launcher.json')['state'] == 'complete'
    assert not read(tmp_path/'launcher.json')['goal_achieved']
