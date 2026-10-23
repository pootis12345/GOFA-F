"""Conditional, matched source-group transfer comparison: TSGFM versus UFG.

The run-bound CPU coordinator first waits for verified Stage61/62 handles.
No target evaluation, neural Memory integration or new tuning is performed.
"""
import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

from gofa_f.dual_memory_data import GROUPS,SHAPES
from gofa_f.dual_memory_engines import UFGEngine
from gofa_f.dual_memory_fitting import fit,evaluate,read
from gofa_f.dual_memory_policy import DOMAINS,digest,summarize
from gofa_f.dual_memory_store import SourceStore
from gofa_f.dual_memory_warmup import cpu_only
from gofa_f.generative_data import sha,Query
from gofa_f.graph_memory_data import flatten_queries,fresh_queries
from gofa_f.idle_launcher import process_identity
from gofa_f.prefix_cache import atomic_json
from gofa_f.tsgfm_expert import TSGFMEngine
from gofa_f.tsgfm_source import screen,RECIPE

PROTOCOL='stage63-matched-source-crossfit-v1'
FILES=('tsgfm_source_crossfit.py','tsgfm_expert.py','tsgfm_source.py',
    'dual_memory_fitting.py','dual_memory_policy.py','dual_memory_store.py',
    'dual_memory_engines.py','dual_memory_data.py','dual_memory_warmup.py',
    'graph_memory_data.py','generative_data.py','ufg_memory_roster.py','ufg_subgraph.py','ufg.py','prefix_cache.py')


def configuration(path):
    cfg=yaml.safe_load(Path(path).read_text().replace('${GOFA_WORKSPACE}',os.environ['GOFA_WORKSPACE']))
    if cfg.get('protocol')!=PROTOCOL or cfg.get('held_groups')!=list(DOMAINS):
        raise ValueError('Fixed whole-source group comparison required')
    if cfg.get('target_training_enabled') is not False or cfg.get('target_evaluation_enabled') is not False:
        raise ValueError('This stage has no target access')
    if cfg.get('ufg_maximum_updates')!=20000 or cfg.get('resources')!=dict(cuda_enabled=False,cpu_threads=2,deadline=None):
        raise ValueError('Fixed sufficiently budgeted CPU comparison required')
    return cfg


def runtime():return {n:sha(Path(__file__).parent/n) for n in FILES}


def live(handle):
    if not handle or process_identity(handle['pid'])!=handle['start_ticks']:return False
    stat=Path('/proc')/str(handle['pid'])/'stat'
    return stat.read_text().split(') ',1)[1].split()[0]!='Z'


def rows_for(manifest,held,shape=None,split='development'):
    if held not in DOMAINS or shape not in (None,'12-12','48-24') or split not in ('train','development'):
        raise ValueError('Unknown complete held source group or task shape')
    return [q for name in GROUPS if GROUPS[name]==held for s,values in manifest['queries'][name].items()
        if shape is None or s==shape for q in values[split]]


def prepare(cfg,output):
    cpu_only();ts=Path(cfg['expert_output']);ufg=Path(cfg['ufg_adequacy_output']);base=Path(cfg['source_cache_output'])
    for root,key in ((ts,'expert_manifest_sha256'),(ufg,'ufg_adequacy_manifest_sha256'),(base,'source_cache_manifest_sha256')):
        if sha(root/'manifest.json')!=cfg[key]:raise ValueError('Preregistered source lineage changed')
    original=read(ts/'manifest.json');other=read(ufg/'manifest.json');common=read(base/'manifest.json')
    for key in ('source_hashes','metadata_hashes','sampler','partitions','training_unit_normalizers'):
        if original[key]!=other[key] or original[key]!=common[key]:raise ValueError('Source input/scale mismatch')
    for name in GROUPS:
        for shape in ('12-12','48-24'):
            for split in ('train','development'):
                if original['queries'][name][shape][split]!=other['queries'][name][shape][split] or \
                    original['queries'][name][shape][split]!=common['queries'][name][shape][split]:
                    raise ValueError('Both models must use identical source query eligibility')
    if original['configuration']['training']!=RECIPE:raise ValueError('Standalone expert recipe changed')
    m=deepcopy(original);m['configuration']['ufg']=deepcopy(other['configuration']['ufg'])
    m['matched_configuration']=cfg;m['matched_runtime_code']=runtime()
    m['source_cache_index_sha256']=sha(base/'source_inputs/index.json')
    # Reserve an unopened source confirmation BEFORE crossfit/Memory selection.
    exclusions=list(flatten_queries(common['queries']))
    warm=Path(cfg['prior_source_reservation'])
    if sha(warm/'manifest.json')!=cfg['prior_source_reservation_manifest_sha256']:
        raise ValueError('Other predeclared source windows changed')
    exclusions.extend(flatten_queries(read(warm/'manifest.json')['queries']))
    previous=common['configuration'];prior=Path(previous['source_output'])/'manifest.json'
    if sha(prior)!=previous['source_manifest_sha256']:raise ValueError('Audited older source lineage changed')
    for split in ('validation','confirmation','diagnostic'):exclusions.extend(flatten_queries(read(prior)['pools'][split]))
    reserved=Path(previous['reservation_output'])/'manifest.json'
    if sha(reserved)!=common['prior_reservations_sha256']:raise ValueError('Older unopened reservation changed')
    for q in read(reserved)['queries'].values():exclusions.extend(q['development']);exclusions.extend(q['reserved'])
    store=SourceStore(m,base/'source_inputs')
    for name in GROUPS:
        data=store.source(name)
        for length,horizon in SHAPES:
            rows=fresh_queries(data,name,length,horizon,'test',32,
                m['partitions'][name]['confirmation'],exclusions,63)
            m['queries'][name][f'{length}-{horizon}']['confirmation']=rows;exclusions.extend(rows)
    m['source_confirmation_labels_opened']=False;m['target_labels_opened']=False
    output.mkdir(parents=True,exist_ok=True)
    if (output/'manifest.json').exists() and read(output/'manifest.json')!=m:raise ValueError('Matched source manifest is immutable')
    atomic_json(output/'manifest.json',m)
    atomic_json(output/'status.json',dict(state='prepared',phase='conditional_source_comparison',
        goal_achieved=False,target_predictions=0,training_started=False))
    return m


def require_proof(output,cfg):
    m=read(output/'manifest.json');binding=sha(output/'manifest.json');proof=read(output/'cpu_preflight.json',{})
    if m['matched_configuration']!=cfg or m['matched_runtime_code']!=runtime():raise ValueError('Matched runtime changed')
    if not proof.get('passed') or proof.get('manifest_sha256')!=binding or proof.get('runtime_code')!=runtime():
        raise ValueError('Related CPU proof must bind both source workers')
    if sha(Path(cfg['source_cache_output'])/'source_inputs/index.json')!=m['source_cache_index_sha256']:
        raise ValueError('Shared source input cache changed')
    return m,binding


def worker(cfg,output,kind,held,shape):
    cpu_only();m,binding=require_proof(output,cfg)
    if kind not in ('tsgfm','ufg') or held not in DOMAINS or \
        (kind=='tsgfm' and shape is not None) or (kind=='ufg' and shape not in ('12-12','48-24')):
        raise ValueError('Only fresh matched held-source fits are permitted')
    path=output/'fits'/f'{kind}-{held}-{shape or "shared"}';path.mkdir(parents=True,exist_ok=True)
    lock=(path/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=read(path/'status.json',{})
    if previous.get('state')=='complete':return
    if previous.get('state')=='failed':raise ValueError('A failed worker requires audit')
    stopped=[False]
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    identity=dict(pid=os.getpid(),start_ticks=process_identity(os.getpid()))
    start=time.monotonic();elapsed=read(path/'activity.json',{}).get('spent_cpu_residency_seconds',0.);visible={};engine=None
    def check():
        if stopped[0]:raise InterruptedError('Matched CPU worker paused cooperatively')
    def progress(**fields):
        if fields.get('phase')!=visible.get('phase'):visible.clear()
        visible.update(fields)
        atomic_json(path/'status.json',dict(state='running',model=kind,held_group=held,task=shape or 'shared',
            target_predictions=0,goal_achieved=False,**visible))
        atomic_json(path/'activity.json',dict(worker=identity,spent_cpu_residency_seconds=elapsed+time.monotonic()-start,
            new_gpu_seconds=0,cuda_initialized=False,threads=2))
    try:
        progress(phase='fresh_held_source_initialization')
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        store=SourceStore(m,Path(cfg['source_cache_output'])/'source_inputs',held_group=held)
        engine=TSGFMEngine(m,store) if kind=='tsgfm' else UFGEngine(cfg['ufg_root'],*map(int,shape.split('-')),m,store,microbatch=64)
        transaction=digest([binding,kind,held,shape]);fit_id='hold-'+held
        best=fit(engine,m,cfg,path,fit_id,None,transaction,check,progress,ufg=kind=='ufg')
        store.permit_out_of_fit(path/'selection.json')
        rows=rows_for(m,held,shape)
        records=evaluate(engine,rows,'development',path/'out_of_fit',transaction,best['model_sha256'],check,progress,role='out_of_fit')
        atomic_json(path/'result.json',dict(binding=binding,model=kind,held_group=held,shape=shape,
            records=records,selection_sha256=sha(path/'selection.json'),selection=read(path/'selection.json'),
            source_confirmation_labels_used=False,target_labels_used=False,
            initialization='fresh_seed42_or_exact_same_fit_resume_not_all_source_weights'))
        atomic_json(path/'status.json',dict(state='complete',phase='held_source_predictions_sealed',model=kind,
            held_group=held,task=shape or 'shared',queries=len(records),goal_achieved=False,target_predictions=0))
    except InterruptedError:
        atomic_json(path/'status.json',dict(state='paused',goal_achieved=False));raise SystemExit(75)
    except BaseException as exc:
        atomic_json(path/f'failure-{time.time_ns()}.json',dict(error=repr(exc)))
        atomic_json(path/'status.json',dict(state='failed',error=repr(exc),goal_achieved=False));raise
    finally:
        if engine is not None:engine.close()
        atomic_json(path/'activity.json',dict(worker=None,spent_cpu_residency_seconds=elapsed+time.monotonic()-start,
            new_gpu_seconds=0,cuda_initialized=False,threads=2,
            peak_rss_kib=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss))


def parent_complete(root,expected_manifest,expected_module):
    root=Path(root)
    if sha(root/'manifest.json')!=expected_manifest:raise ValueError('Parent manifest changed')
    state=read(root/'status.json');activity=read(root/'activity.json',{})
    handle=read(Path(str(root)+'.launch.json'))
    if handle is None:raise ValueError('Parent process launch identity is missing')
    if state['state']=='failed':raise ValueError('Parent implementation failure requires audit')
    if state['state']=='complete':
        return not live(handle) and activity.get('worker') is None
    if not live(handle):raise ValueError('Unfinished parent has no verified live handle')
    command=(Path('/proc')/str(handle['pid'])/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
    if expected_module not in command or str(root) not in command:raise ValueError('Parent process ownership changed')
    return False


def recheck_full_source(cfg,output,m,binding,check,progress):
    """Compare the frozen expert to the adequately stopped all-source UFG."""
    expert=Path(cfg['expert_output']);adequacy=Path(cfg['ufg_adequacy_output'])
    lock=read(adequacy/'ufg_source_lock.json')
    if lock['binding']!=cfg['ufg_adequacy_manifest_sha256'] or not lock['both_shapes_stopped_by_source_criterion'] or lock['target_labels_used']:
        return dict(passed=False,reason='official_UFG_adequacy_not_met')
    store=SourceStore(m,Path(cfg['source_cache_output'])/'source_inputs')
    selected=read(expert/'all/selection.json');proof=[]
    def records(folder,selection,shape=None,fallback=None):
        if sha(folder/'best.pt')!=selection['checkpoint_sha256'] or selection['target_labels_used']:
            raise ValueError('Frozen all-source checkpoint changed')
        directory=folder/'evaluations'/f"update-{selection['best_update']:06d}"
        if not directory.exists():
            if fallback is None or read(fallback/'selection.json')['model_sha256']!=selection['model_sha256']:
                raise ValueError('Missing source predictions for the exact selected UFG tensor')
            directory=fallback/'evaluations'/f"update-{selection['best_update']:06d}"
        expected={Query(**q).key:q for held in DOMAINS for q in rows_for(m,held,shape)}
        found={}
        for path in sorted(directory.glob('*.json')):
            check();r=read(path);key=Query(**r['query']).key
            if key not in expected or key in found or r['identity']['checkpoint_sha']!=selection['model_sha256']:
                raise ValueError('All-source baseline prediction scope/checkpoint changed')
            item,y,mask=store.get(expected[key],'development')
            if r['input_sha256']!=item.identity or not np.array_equal(r['target'],y[0].numpy()) or not np.array_equal(r['observed'],mask[0].numpy()):
                raise ValueError('All-source comparator input/labels changed')
            found[key]=r;proof.append(dict(path=str(path),sha256=sha(path)))
        if set(found)!=set(expected):raise ValueError('Incomplete all-source comparator')
        return list(found.values())
    ts=records(expert/'all',selected);ufg=[]
    for shape,entry in lock['models'].items():
        folder=Path(entry['directory']);actual=read(folder/'selection.json')
        if actual!=entry['selection'] or actual['stop_reason']!='source_early_stop':raise ValueError('Adequacy source selection changed')
        ufg.extend(records(folder,actual,shape,Path(cfg['source_cache_output'])/('ufg-'+shape)))
    metrics={name:summarize(rows,m['training_unit_normalizers']) for name,rows in (('tsgfm',ts),('ufg',ufg))}
    gate=screen(metrics['tsgfm'],metrics['ufg'])
    gate.update(scope='included_source_recheck_after_official_UFG_adequacy',fixed_six_target_superiority_proven=False)
    result=dict(passed=gate['passed'],reason='rechecked_source_gate',gate=gate,metrics=metrics,
        source_prediction_provenance=proof,adequacy_lock_sha256=sha(adequacy/'ufg_source_lock.json'),
        expert_selection_sha256=sha(expert/'all/selection.json'))
    previous=read(output/'baseline_recheck.json')
    if previous is not None and previous!=result:raise ValueError('Frozen source recheck changed')
    atomic_json(output/'baseline_recheck.json',result)
    return result


def paired_metrics(ts_records,ufg_records,manifest,store=None):
    expected={Query(**q).key:q for held in DOMAINS for q in rows_for(manifest,held)}
    models={}
    for name,records in (('tsgfm',ts_records),('ufg',ufg_records)):
        mapped={Query(**r['query']).key:r for r in records}
        if len(mapped)!=len(records) or set(mapped)!=set(expected):raise ValueError('Incomplete/duplicate held-source comparison')
        models[name]=mapped
    for key in expected:
        a,b=models['tsgfm'][key],models['ufg'][key]
        if a['query']!=b['query'] or a['input_sha256']!=b['input_sha256'] or \
            a['target']!=b['target'] or a['observed']!=b['observed']:
            raise ValueError('Matched models differ in actual history roster, truth or valid slots')
        if store is not None:
            item,y,mask=store.get(expected[key],'development')
            if item.identity!=a['input_sha256'] or not np.array_equal(a['target'],y[0].numpy()) or not np.array_equal(a['observed'],mask[0].numpy()):
                raise ValueError('Matched records disagree with the audited source cache')
    metrics={k:summarize(list(v.values()),manifest['training_unit_normalizers']) for k,v in models.items()}
    gate=screen(metrics['tsgfm'],metrics['ufg'])
    gate.update(scope='matched_whole_source_group_out_of_fit',fixed_six_target_superiority_proven=False,
        held_group_labels_used_for_fitting_or_checkpoint_selection=False)
    return dict(metrics=metrics,gate=gate,queries=len(expected),same_real_history_rosters_and_valid_slots=True)


def controller(cfg,output,snapshot,config):
    cpu_only();lock=(output/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    m,binding=require_proof(output,cfg);stopped=[False];children=[]
    old_activity=read(output/'activity.json',{});waited=old_activity.get('coordinator_wait_seconds',0.)
    prior=old_activity.get('coordinator_residency_seconds',0.);started=time.monotonic()
    identity=dict(pid=os.getpid(),start_ticks=process_identity(os.getpid()))
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    def check():
        if stopped[0]:raise InterruptedError('Source comparison coordinator paused')
    def activity(active):
        atomic_json(output/'activity.json',dict(worker=identity if active else None,
            coordinator_wait_seconds=waited,coordinator_residency_seconds=prior+time.monotonic()-started,
            model_cpu_residency_seconds={p.parent.name:read(p)['spent_cpu_residency_seconds']
                for p in (output/'fits').glob('*/activity.json')},
            coordinator_overlaps_child_residency_do_not_sum=True,new_gpu_seconds=0,
            source_parent_costs_not_reset_or_added_twice=True,cuda_initialized=False,threads=2))
    def progress(phase,**fields):
        atomic_json(output/'status.json',dict(state='running',phase=phase,
            goal_achieved=False,target_predictions=0,**fields));activity(True)
    def finish(phase,**fields):
        atomic_json(output/'report.json',dict(binding=binding,phase=phase,goal_achieved=False,
            target_predictions=0,source_confirmation_opened=False,controller_wait_seconds=waited,
            parent_manifest_sha256={k:cfg[k] for k in ('expert_manifest_sha256','ufg_adequacy_manifest_sha256','source_cache_manifest_sha256')},
            new_model_cpu_residency_seconds={p.parent.name:read(p)['spent_cpu_residency_seconds']
                for p in (output/'fits').glob('*/activity.json')},**fields))
        atomic_json(output/'status.json',dict(state='complete',phase=phase,goal_achieved=False,target_predictions=0))
    try:
        expert=Path(cfg['expert_output']);adequacy=Path(cfg['ufg_adequacy_output'])
        while not parent_complete(expert,cfg['expert_manifest_sha256'],'gofa_f.tsgfm_source'):
            check();progress('waiting_for_verified_independent_expert',training_started=False)
            start=time.monotonic();time.sleep(10);waited+=time.monotonic()-start
        initial=read(expert/'report.json')
        if not initial['screen']['passed']:
            finish('not_started_standalone_source_screen_failed',source_screen=initial['screen'],
                expert_report_sha256=sha(expert/'report.json'),new_training_updates=0);return
        while not parent_complete(adequacy,cfg['ufg_adequacy_manifest_sha256'],'gofa_f.ufg_source_adequacy'):
            check();progress('waiting_for_verified_official_UFG_adequacy',training_started=False)
            start=time.monotonic();time.sleep(10);waited+=time.monotonic()-start
        prerequisite=recheck_full_source(cfg,output,m,binding,check,progress)
        if not prerequisite['passed']:finish('not_started_source_recheck_or_adequacy_failed',prerequisite=prerequisite,new_training_updates=0);return
        # Crossfits use fresh initialization for BOTH models. The frozen all-
        # source checkpoints are NOT imported into held-source workers.
        results={'tsgfm':[],'ufg':[]};sufficient=True
        for held in DOMAINS:
            for kind,shape in (('tsgfm',None),('ufg','12-12'),('ufg','48-24')):
                check();name=f'{kind}-{held}-{shape or "shared"}';folder=output/'fits'/name
                result=folder/'result.json';handle_path=Path(str(folder)+'.launch.json')
                if not result.exists():
                    handle=read(handle_path) if handle_path.exists() else None
                    if not live(handle):
                        state=read(folder/'status.json',{})
                        if state.get('state') in ('failed','running'):raise ValueError('Audit stopped matched worker before restart')
                        python=cfg['native_python'] if kind=='tsgfm' else cfg['ufg_python']
                        command=[python,'-u','-m','gofa_f.tsgfm_source_crossfit','--config',str(config),
                            '--output',str(output),'--worker',kind,'--held-group',held]
                        if shape:command+=['--shape',shape]
                        folder.parent.mkdir(parents=True,exist_ok=True)
                        environment=dict(os.environ,PYTHONPATH=str(snapshot/'src'),CUDA_VISIBLE_DEVICES='',
                            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',DGLBACKEND='pytorch')
                        environment.pop('GOFA_AUTOMATION_DEADLINE_UNIX',None)
                        with Path(str(folder)+'.log').open('ab') as log:
                            child=subprocess.Popen(command,cwd=snapshot,env=environment,stdin=subprocess.DEVNULL,
                                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        handle=dict(pid=child.pid,start_ticks=process_identity(child.pid),command=command)
                        atomic_json(handle_path,handle)
                    children=[handle]
                    while live(handle):
                        check();progress('matched_source_fitting',current_fit=name,
                            worker_status=read(folder/'status.json',{}));time.sleep(10)
                    children=[]
                value=read(result)
                if value is None or read(folder/'activity.json')['worker'] is not None:
                    raise ValueError('Matched worker incomplete after process exit')
                if value['binding']!=binding or value['held_group']!=held or value['model']!=kind:
                    raise ValueError('Matched worker result binding changed')
                if value['selection']!=read(folder/'selection.json') or value['selection_sha256']!=sha(folder/'selection.json') or \
                    value['selection']['checkpoint_sha256']!=sha(folder/'best.pt'):
                    raise ValueError('Matched worker source selection changed')
                results[kind].extend(value['records'])
                if kind=='ufg' and value['selection']['stop_reason']!='source_early_stop':sufficient=False
        comparison=paired_metrics(results['tsgfm'],results['ufg'],m,
            SourceStore(m,Path(cfg['source_cache_output'])/'source_inputs'))
        finish('matched_source_gate_passed' if sufficient and comparison['gate']['passed'] else 'matched_source_gate_failed',
            comparison=comparison,all_UFG_crossfits_met_source_early_stop=sufficient,
            memory_integration_not_started=True)
    except InterruptedError:
        atomic_json(output/'status.json',dict(state='paused',goal_achieved=False));raise SystemExit(75)
    except BaseException as exc:
        atomic_json(output/f'failure-{time.time_ns()}.json',dict(error=repr(exc)))
        atomic_json(output/'status.json',dict(state='failed',error=repr(exc),goal_achieved=False));raise
    finally:
        for handle in children:
            if live(handle):os.kill(handle['pid'],signal.SIGTERM)
        until=time.monotonic()+60
        while any(live(h) for h in children) and time.monotonic()<until:time.sleep(1)
        activity(False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--snapshot')
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker',choices=('tsgfm','ufg'))
    parser.add_argument('--held-group',choices=DOMAINS);parser.add_argument('--shape',choices=('12-12','48-24'));args=parser.parse_args()
    cfg=configuration(args.config);output=Path(args.output)
    if args.prepare:prepare(cfg,output)
    elif args.worker:worker(cfg,output,args.worker,args.held_group,args.shape)
    else:controller(cfg,output,Path(args.snapshot),Path(args.config))


if __name__=='__main__':main()
