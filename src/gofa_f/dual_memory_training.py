"""Formal source cross-fitting/depth selection and independent official-UFG fits.

No target-data path is accepted. Confirmation/target evaluation require separate
sealed deployment execution and cannot be confused with completion of training.
"""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
import yaml

from gofa_f.dual_memory_data import GROUPS, SHAPES
from gofa_f.dual_memory_policy import DOMAINS,digest,summarize
from gofa_f.dual_memory_store import SourceStore,HistoryCache,atomic_torch
from gofa_f.dual_memory_fitting import fit,evaluate,read
from gofa_f.dual_memory_engines import MemoryEngine,UFGEngine
from gofa_f.generative_data import Query,sha
from gofa_f.prefix_cache import atomic_json

FILES=('dual_memory_policy.py','dual_memory_store.py','dual_memory_engines.py',
       'dual_memory_fitting.py','dual_memory_fusion.py','dual_memory_training.py')


def configuration(path):
    cfg=yaml.safe_load(Path(path).read_text().replace('${GOFA_WORKSPACE}',os.environ['GOFA_WORKSPACE']))
    if cfg['protocol']!='stage59-dual-memory-source-v1' or cfg['training']['max_updates']!=2000 or any(cfg['data'].values()):
        raise ValueError('Fixed source-only formal-training protocol required')
    if cfg['resources']!=dict(admission_policy='memory_cap_only',admission_samples=1,resource_check_seconds=10,
        memory_cap_gib=40,torch_cap_gib=38,activity_limit_seconds=None,wall_deadline=None,yield_on_foreign=False,leave_free_gib=0):
        raise ValueError('Source resource policy changed')
    return cfg


def prepare(cfg,output):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():raise ValueError('CPU-only preparation required')
    parent=Path(cfg['parent_output'])
    if sha(parent/'manifest.json')!=cfg['parent_manifest_sha256'] or sha(parent/'mechanisms.json')!=cfg['acceptance_sha256']:
        raise ValueError('Accepted source preparation or real7Bproof changed')
    old=read(parent/'manifest.json');proof=read(parent/'mechanisms.json')
    if not proof['passed'] or set(proof['depths'])!={'3','6'} or read(parent/'activity.json')['worker'] is not None:
        raise ValueError('Completed dual-depth proof and exited previous worker required')
    for name,value in old['code'].items():
        if sha(Path(__file__).parent/name)!=value:raise ValueError('Accepted tensor/input core changed: '+name)
    m=dict(old,formal_configuration=cfg,formal_runtime_code={n:sha(Path(__file__).parent/n) for n in FILES},
        accepted_mechanisms_sha256=cfg['acceptance_sha256'],parent_manifest_sha256=cfg['parent_manifest_sha256'],
        inherited_gpu_seconds=read(parent/'activity.json')['spent_seconds'],formal_target_labels_used=False)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'manifest.json').exists() and read(output/'manifest.json')!=m:raise ValueError('Formal source binding changed')
    atomic_json(output/'manifest.json',m)
    store=SourceStore(m,output/'source_inputs')
    def progress(**fields):atomic_json(output/'data_preparation.json',dict(state='preparing',**fields))
    store.build(progress=progress)
    atomic_json(output/'data_preparation.json',dict(state='complete',records=len(store.index),
        cache_index_sha256=sha(output/'source_inputs/index.json'),confirmation_labels_opened=False,target_labels_used=False))
    atomic_json(output/'status.json',dict(state='prepared',phase='formal_source_training',goal_achieved=False,target_predictions=0))


def development_rows(m,names=None,shape=None):
    names=list(GROUPS) if names is None else names
    return [q for name in names for length,horizon in SHAPES for q in m['queries'][name][f'{length}-{horizon}']['development']
            if shape is None or shape==f'{length}-{horizon}']


def native_predictions(engine,m,rows,output,binding,check,progress):
    from gofa_f.short_sampling_diagnostic import generate,sample_seed,SETTINGS
    root=output/'native/development';root.mkdir(parents=True,exist_ok=True);records=[]
    for index,row in enumerate(rows):
        key=Query(**row).key;name=digest(key);path=root/(name+'.json');value=read(path)
        identity=dict(binding=binding,query=row,weights=m['weights_sha256'],sampling=SETTINGS)
        if value is None:
            item,target,mask=engine.store.get(row,'development')
            history=np.where(item.observed[0].numpy(),item.history[0].numpy(),np.nan)
            trajectories=[];start=time.monotonic()
            folder=root/'trajectories';folder.mkdir(exist_ok=True)
            for sample in range(8):
                check();seed=sample_seed(row,sample);own=folder/f'{name}-{sample}.json';saved=read(own)
                if saved is None:
                    result=generate(engine.model.native_view(),engine.codec,history,row['horizon'],seed=seed,check=check)
                    saved=dict(identity=identity,seed=seed,result=result);atomic_json(own,saved)
                if saved['identity']!=identity or saved['seed']!=seed:raise ValueError('Native trajectory binding changed')
                trajectories.append(saved['result']['prediction'])
                progress(phase='native_source_sampling',completed_queries=index,total_queries=len(rows),sample=sample+1)
            prediction=np.median(np.asarray(trajectories,np.float64),axis=0)
            value=dict(identity=identity,query=row,query_key=key,prediction=prediction.tolist(),
                target=target[0].numpy().tolist(),observed=mask[0].numpy().tolist(),
                input_sha256=item.identity,generation_seconds=time.monotonic()-start)
            atomic_json(path,value)
        elif value['identity']!=identity:raise ValueError('Native source median binding changed')
        records.append(value)
    return records


def ratios(records,native,normalizers):
    actual=summarize(records,normalizers);reference=summarize(native,normalizers)
    view={v:actual['views'][v]['macro_mae']/max(reference['views'][v]['macro_mae'],1e-12,normalizers[v]*1e-9) for v in actual['views']}
    domains={g:float(np.exp(np.mean([np.log(view[v]) for v in view if GROUPS[v]==g]))) for g in DOMAINS}
    return dict(views=view,domains=domains,worst=max(domains.values()),geometric=float(np.exp(np.mean(np.log(list(domains.values()))))))


def verify_runtime(m,cfg):
    if cfg!=m['formal_configuration'] or m['formal_runtime_code']!={n:sha(Path(__file__).parent/n) for n in FILES}:
        raise ValueError('Formal source runtime/config changed')
    for n,value in m['code'].items():
        if sha(Path(__file__).parent/n)!=value:raise ValueError('Frozen accepted core changed')


def run_native(cfg,output):
    from transformers import LlamaForCausalLM,LlamaTokenizer
    from gofa_f.short_codec import ShortCodec
    from gofa_f.allnode_stage import supervisor_alive,Paused
    from gofa_f.dual_memory_fusion import fit_gate
    if os.environ.get('GOFA_STAGE19_ADMITTED')!='1':raise ValueError('Cap-supervised source worker required')
    m=read(output/'manifest.json');verify_runtime(m,cfg);binding=sha(output/'manifest.json')
    if not read(output/'formal_cpu_preflight.json',{}).get('passed'):raise ValueError('Related training CPU proof required')
    stopped=[False]
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    def check():
        if stopped[0] or not supervisor_alive(os.environ['GOFA_STAGE19_SUPERVISOR_PID'],os.environ['GOFA_STAGE19_SUPERVISOR_TICKS']):raise Paused('cooperative_resource_pause')
    visible={}
    def progress(**fields):
        if fields.get('phase')!=visible.get('phase'):visible.clear()
        visible.update(fields)
        atomic_json(output/'status.json',dict(state='running',goal_achieved=False,target_predictions=0,**visible))
    engine=None
    try:
        torch.cuda.set_per_process_memory_fraction(38*2**30/torch.cuda.get_device_properties(0).total_memory)
        root=Path(m['source_configuration']['weights']['directory'])
        if sha(root/'snapshot.json')!=m['weights_sha256']:raise ValueError('Frozen ChatTime snapshot changed')
        progress(phase='loading_frozen_chattime')
        native=LlamaForCausalLM.from_pretrained(root,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda:0')
        versions={n:p._version for n,p in native.named_parameters()}
        tokenizer=LlamaTokenizer.from_pretrained(root,local_files_only=True);codec=ShortCodec(tokenizer)
        cache=HistoryCache(output/'history_cache',int(cfg['prefix_cache']['ram_gib']*2**30),int(cfg['prefix_cache']['disk_gib']*2**30))
        store=SourceStore(m,output/'source_inputs')
        engine=MemoryEngine(native,tokenizer,codec,m,store,cache,6)
        probes=[q for name in GROUPS for l,h in SHAPES for q in m['queries'][name][f'{l}-{h}']['development'][:1]]
        native_predictions(engine,m,probes,output,binding,check,progress)
        engine.close();engine=None
        crossfit={}
        for depth in (6,3):
            combined=[]
            for held in DOMAINS:
                check();torch.manual_seed(42);np.random.seed(42);random.seed(42)
                fit_id='hold-'+held;folder=output/f'depth{depth}'/fit_id
                own_store=SourceStore(m,output/'source_inputs',held_group=held)
                engine=MemoryEngine(native,tokenizer,codec,m,own_store,cache,depth)
                best=fit(engine,m,cfg,folder,fit_id,depth,digest([binding,depth,fit_id]),check,progress)
                own_store.permit_out_of_fit(folder/'selection.json')
                names=[v for v in GROUPS if GROUPS[v]==held]
                records=evaluate(engine,development_rows(m,names),'development',folder/'out_of_fit',binding,
                    best['model_sha256'],check,progress,role='out_of_fit')
                for r in records:r['query_key']=Query(**r['query']).key
                combined.extend(records);engine.close();engine=None
            crossfit[depth]=combined
            atomic_json(output/f'crossfit-depth{depth}.json',dict(binding=binding,records=combined,
                traffic_channels_held_out_together=True,target_labels_used=False))
        store=SourceStore(m,output/'source_inputs');engine=MemoryEngine(native,tokenizer,codec,m,store,cache,6)
        baseline=native_predictions(engine,m,development_rows(m),output,binding,check,progress)
        engine.close();engine=None
        scores={depth:ratios(rows,baseline,m['training_unit_normalizers']) for depth,rows in crossfit.items()}
        chosen=min((6,3),key=lambda d:(scores[d]['worst'],scores[d]['geometric'],d))
        selection=dict(binding=binding,depth=chosen,source_out_of_fit_scores=scores,selection_uses_target_labels=False)
        existing=read(output/'depth_selection.json')
        if existing is not None and existing!=selection:raise ValueError('Global source depth seal changed')
        atomic_json(output/'depth_selection.json',selection)
        torch.manual_seed(42);np.random.seed(42);random.seed(42)
        engine=MemoryEngine(native,tokenizer,codec,m,store,cache,chosen)
        best=fit(engine,m,cfg,output/'final-source-model','all',chosen,digest([binding,chosen,'all']),check,progress)
        gate=fit_gate(crossfit[chosen],baseline,store,cfg['fusion'])
        gate.update(binding=binding,depth=chosen,source_expert_sha256=best['model_sha256'])
        existing=read(output/'shared_gate.json')
        if existing is not None and existing!=gate:raise ValueError('Sealed shared gate changed')
        atomic_json(output/'shared_gate.json',gate)
        lock=dict(binding=binding,depth=chosen,expert_checkpoint_sha256=sha(output/'final-source-model/best.pt'),
            expert_model_sha256=best['model_sha256'],shared_gate_sha256=sha(output/'shared_gate.json'),
            source_confirmation_labels_used=False,target_labels_used=False)
        atomic_json(output/'source_model_lock.json',lock)
        if versions!={n:p._version for n,p in native.named_parameters()} or any(p.grad is not None for p in native.parameters()):raise ValueError('Frozen backbone changed')
        atomic_json(output/'status.json',dict(state='complete',phase='source_model_and_shared_fusion_locked_confirmation_pending',
            goal_achieved=False,target_predictions=0,source_confirmation_opened=False,selection=selection,
            cache_hits=cache.hits,cache_misses=cache.misses))
    except Paused:
        atomic_json(output/'status.json',dict(state='resource_paused',phase='formal_source_training',goal_achieved=False));raise SystemExit(75)
    except BaseException as exc:
        atomic_json(output/f'failure-{time.time_ns()}.json',dict(error=repr(exc),binding=binding))
        atomic_json(output/'status.json',dict(state='failed',phase='formal_source_training',error=repr(exc),goal_achieved=False));raise
    finally:
        if engine is not None:engine.close()


def run_ufg(cfg,output):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():raise ValueError('UFG source runner is CPU-only')
    lock=(output/'ufg.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    m=read(output/'manifest.json');verify_runtime(m,cfg);binding=sha(output/'manifest.json');stopped=[False]
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    started=time.monotonic();old=read(output/'ufg_activity.json',{});prior=old.get('spent_seconds',0.)
    from gofa_f.idle_launcher import process_identity
    def progress(**fields):
        atomic_json(output/'ufg_status.json',dict(state='running',goal_achieved=False,target_predictions=0,**fields))
        atomic_json(output/'ufg_activity.json',dict(spent_seconds=prior+time.monotonic()-started,
            worker=dict(pid=os.getpid(),start_ticks=process_identity(os.getpid())),cuda_initialized=False,threads=2))
    def check():
        if stopped[0]:raise InterruptedError('UFG cooperative CPU pause')
    try:
        for length,horizon in SHAPES:
            random.seed(42);np.random.seed(42);torch.manual_seed(42)
            store=SourceStore(m,output/'source_inputs');engine=UFGEngine(cfg['ufg_root'],length,horizon,m,store)
            fit(engine,m,cfg,output/f'ufg-{length}-{horizon}','all',None,
                digest([binding,'ufg',length,horizon]),check,progress,ufg=True)
            engine.close()
        atomic_json(output/'ufg_status.json',dict(state='complete',phase='two_shape_source_bests_locked',target_predictions=0))
    except InterruptedError:
        atomic_json(output/'ufg_status.json',dict(state='paused',phase='source_training'));raise SystemExit(75)
    except BaseException as exc:
        atomic_json(output/'ufg_status.json',dict(state='failed',error=repr(exc)));raise
    finally:
        atomic_json(output/'ufg_activity.json',dict(spent_seconds=prior+time.monotonic()-started,worker=None,
            cuda_initialized=False,threads=2,peak_rss_kib=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--ufg',action='store_true');args=parser.parse_args()
    cfg=configuration(args.config);output=Path(args.output)
    if args.prepare:prepare(cfg,output)
    elif args.ufg:run_ufg(cfg,output)
    else:run_native(cfg,output)


if __name__=='__main__':main()
