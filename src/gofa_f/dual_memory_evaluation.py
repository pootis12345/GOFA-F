"""Read-only source confirmation and frozen six-target evaluation; no optimizer."""
import argparse
import gc
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
import yaml

from gofa_f.dual_memory_eval_data import (MODELS,TARGETS,InputStore,read,seal,truth_records,verify_deployment)
from gofa_f.dual_memory_eval_prepare import verify_runtime
from gofa_f.dual_memory_eval_records import PredictionStore
from gofa_f.generative_data import sha
from gofa_f.prefix_cache import atomic_json


def configuration(path):
    cfg=yaml.safe_load(Path(path).read_text().replace('${GOFA_WORKSPACE}',os.environ['GOFA_WORKSPACE']))
    if cfg['protocol']!='stage59-dual-memory-evaluation-v1' or cfg['training']!={'max_updates':0} or \
            cfg['data']!={'target_training_enabled':False,'target_evaluation_enabled':True}:
        raise ValueError('Only frozen evaluation without neural/target training is permitted')
    if cfg['resources']!=dict(admission_policy='memory_cap_only',admission_samples=1,resource_check_seconds=10,
            memory_cap_gib=40,torch_cap_gib=38,activity_limit_seconds=None,wall_deadline=None,
            yield_on_foreign=False,leave_free_gib=0):raise ValueError('Evaluation resource contract changed')
    if cfg['sampling']!=dict(samples=8,top_k=100,temperature=1.,top_p=1.,seed=42,
            aggregate='pointwise_median',execution='serial_single_query_independent_kv'):
        raise ValueError('Native sampling protocol changed')
    return cfg


def guard(root,cfg):
    m=read(root/'manifest.json');verify_runtime(m)
    if cfg!=m['configuration']:raise ValueError('Evaluation configuration changed')
    if m['scope']=='targets':
        if set(m['datasets'])!=set(TARGETS) or len(m['queries'])!=768:raise ValueError('Complete fixed six-target budget required')
        for name in TARGETS:
            rows=[q for q in m['queries'] if q['source']==name]
            shape=(12,12) if name in ('pems03','pems_bay') else (48,24)
            if len(rows)!=128 or any((q['length'],q['horizon'])!=shape for q in rows):raise ValueError('A target shape/query budget changed')
    elif m['scope']!='source_confirmation' or len(m['queries'])!=256:raise ValueError('Sealed source confirmation budget required')
    proof=read(root/'cpu_preflight.json')
    if not proof.get('passed') or proof['binding']!=sha(root/'manifest.json'):
        raise ValueError('Related bound CPU evaluation proof required')
    store=InputStore(root)
    if sha(root/'inputs/index.json')!=read(root/'preparation.json')['input_index_sha256']:
        raise ValueError('History input index changed')
    return m,store,PredictionStore(root)


def signal_checker(gpu=False):
    stopped=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    def check():
        if stopped[0]:raise InterruptedError('cooperative_evaluation_pause')
        if gpu:
            from gofa_f.allnode_stage import supervisor_alive
            if not supervisor_alive(os.environ['GOFA_STAGE19_SUPERVISOR_PID'],os.environ['GOFA_STAGE19_SUPERVISOR_TICKS']):
                raise InterruptedError('evaluation_supervisor_exited')
    return check


def progress(root,model,index,total,records,**fields):
    seconds=sum(v.get('seconds',0.) for v in records)
    scope=read(root/'manifest.json')['scope'];average=seconds/len(records) if records else None
    atomic_json(root/('ufg_status.json' if model=='ufg' else 'status.json'),dict(state='running',phase='frozen_predictions',
        scope=scope,model=model,completed=index,total=total,
        mean_query_seconds=average,eta_current_model_seconds=None if average is None else (total-index)*average,
        conditional_remaining_pipeline_query_seconds=None if average is None else (total-index+(768 if scope=='source_confirmation' else 0))*average,
        eta_excludes_unknown_resource_pauses=True,goal_achieved=False,**fields))


def source_replays(model,codec,parent,check):
    """Reuse immutable historical references; new origin optimization must replay."""
    from gofa_f.generative_data import Query
    from gofa_f.graph_memory_stage import load_sources
    from gofa_f.short_sampling_diagnostic import generate,sample_seed
    from gofa_f.dual_memory_data import GROUPS
    old=Path(os.environ['GOFA_WORKSPACE'])/'outputs/stage57-chattime-graph-memory-source-pilot-v1'
    manifest=read(old/'manifest.json');sources=load_sources(parent)
    flattened=[q for name in GROUPS for q in manifest['queries'][name]['development']]
    result=[]
    for i in (0,16,32,48):
        check();row=flattened[i];q=Query(**row);data=sources[q.source]
        history=np.where(data.mask[q.origin-q.length:q.origin,q.variable],data.values[q.origin-q.length:q.origin,q.variable],np.nan)
        reference_path=old/'evaluations/native/trajectories'/f'{i:04d}-0.json';reference=read(reference_path)
        actual=generate(model.native_view(),codec,history,q.horizon,seed=sample_seed(row,0),check=check)
        if actual['generated_ids']!=reference['generated_ids']:raise ValueError('Existing source sampled-token replay failed')
        result.append(dict(query=row,reference_sha256=sha(reference_path),sampled_tokens_equal=True))
    return result


@torch.no_grad()
def execution_acceptance(root,model,tokenizer,codec,manifest,inputs,check):
    """Only the new sharing path needs real-weight proof; frozen core is reused."""
    from gofa_f.dual_memory_origin import NativeOrigin,SharedPrefixBuilder,select_rows
    from gofa_f.dual_memory_run import make_tokens,metadata_tokens
    from gofa_f.short_sampling_diagnostic import generate,sample_seed
    path=root/'mechanisms.json';binding=sha(root/'manifest.json')
    if path.exists():
        proof=read(path)
        if proof['binding']!=binding or not proof['passed']:raise ValueError('Wrong evaluation mechanism proof')
        return
    parent=read(Path(manifest['deployment']['source_directory'])/'manifest.json')
    replays=source_replays(model,codec,parent,check);cases=[]
    unique={}
    for row in manifest['queries']:unique.setdefault((row['length'],row['horizon']),row)
    for shape,row in unique.items():
        check();item,text=inputs.get(row);item=item.to('cuda:0');tokens=make_tokens(codec,item,'cuda:0')
        meta=model.metadata_vector(metadata_tokens(tokenizer,text,'cuda:0'))
        origin=NativeOrigin(model,select_rows(tokens,[0],crop=True),codec.first_separator)
        independent=model.frozen_history(tokens);shared=SharedPrefixBuilder(model,origin).frozen_history(tokens)
        torch.testing.assert_close(shared.hidden[tokens.attention_mask],independent.hidden[tokens.attention_mask],rtol=.002,atol=.002)
        full,_=model.enhanced(tokens,item,meta,prefix=independent,optimized=False)
        actual,_=model.enhanced(tokens,item,meta,prefix=shared,optimized=True)
        torch.testing.assert_close(actual,full,rtol=.002,atol=.002)
        history=np.where(item.observed[0].cpu(),item.history[0].cpu(),np.nan)
        maximum=0.
        rng=torch.get_rng_state().clone();crng=torch.cuda.get_rng_state_all()
        keys=[v.clone() for v in origin.kv.key_cache]
        for sample in range(8):
            check();seed=sample_seed(row,sample);a=[];b=[];c=[]
            reference=generate(model.native_view(),codec,history,row['horizon'],seed=seed,check=check,trace=a)
            actual_path=generate(origin,codec,history,row['horizon'],seed=seed,check=check,trace=b)
            # First trajectory additionally checks independent complete-prefix
            # inference against incremental decoding; neither uses graph KV.
            if sample==0:
                whole=generate(model.native_view(),codec,history,row['horizon'],seed=seed,cached=False,check=check,trace=c)
                torch.testing.assert_close(torch.cat(a),torch.cat(c),rtol=.002,atol=.002)
                if whole['generated_ids']!=reference['generated_ids']:raise ValueError('Full/incremental Native sampled tokens differ')
            torch.testing.assert_close(torch.cat(a),torch.cat(b),rtol=.002,atol=.002)
            maximum=max(maximum,float((torch.cat(a)-torch.cat(b)).abs().max()))
            if actual_path['generated_ids']!=reference['generated_ids'] or actual_path['prediction']!=reference['prediction']:
                raise ValueError('Shared-origin sampling differs from independent Native')
        if not torch.equal(rng,torch.get_rng_state()) or any(not torch.equal(a,b) for a,b in zip(crng,torch.cuda.get_rng_state_all())):
            raise ValueError('Evaluation sampling modified global RNG')
        if any(not torch.equal(a,b) for a,b in zip(keys,origin.kv.key_cache)):raise ValueError('A trajectory modified shared prompt KV')
        cases.append(dict(shape=list(shape),query=row,real_nodes=len(item.node_ids),input_sha256=item.identity,
            enhanced_max_absolute_difference=float((actual-full).abs().max()),sampled_logit_max_difference=maximum,
            independent_paths=8,sampled_tokens_equal=True,full_incremental_seed0_equal=True))
    seal(path,dict(binding=binding,passed=True,cases=cases,source_replays=replays,cpu_tolerance=1e-6,
        cuda_tolerance=.002,sampling_rng_isolated=True,training_updates=0))


@torch.no_grad()
def run_gpu(cfg,root):
    from transformers import LlamaForCausalLM,LlamaTokenizer
    from gofa_f.dual_memory_model import DualMemoryGOFA
    from gofa_f.dual_memory_origin import NativeOrigin,SharedPrefixBuilder,select_rows
    from gofa_f.dual_memory_run import make_tokens,metadata_tokens
    from gofa_f.dual_memory_store import HistoryCache
    from gofa_f.dual_memory_engines import model_digest
    from gofa_f.dual_memory_fusion import blend
    from gofa_f.short_codec import ShortCodec
    from gofa_f.short_sampling_diagnostic import generate
    if os.environ.get('GOFA_STAGE19_ADMITTED')!='1':raise ValueError('Resource supervisor admission required')
    m,inputs,outputs=guard(root,cfg);check=signal_checker(True);check()
    torch.cuda.set_per_process_memory_fraction(38*2**30/torch.cuda.get_device_properties(0).total_memory)
    source=Path(m['deployment']['source_directory']);parent=read(source/'manifest.json')
    weights=Path(parent['source_configuration']['weights']['directory'])
    if sha(weights/'snapshot.json')!=m['deployment']['weights_sha256']:raise ValueError('Frozen ChatTime weights changed')
    native=LlamaForCausalLM.from_pretrained(weights,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda:0')
    tokenizer=LlamaTokenizer.from_pretrained(weights,local_files_only=True);codec=ShortCodec(tokenizer)
    model=DualMemoryGOFA(native,depth=m['deployment']['depth'],weight_binding=m['deployment']['weights_sha256']).to('cuda:0').eval()
    checkpoint=torch.load(m['deployment']['expert']['path'],map_location='cpu',weights_only=False)
    model.adapter.load_state_dict(checkpoint['model'])
    for p in model.adapter.parameters():p.requires_grad_(False)
    versions={n:p._version for n,p in native.named_parameters()};expert_hash=model_digest(model.adapter)
    if expert_hash!=m['deployment']['expert']['model_sha256']:raise ValueError('Loaded expert differs from sealed source model')
    execution_acceptance(root,model,tokenizer,codec,m,inputs,check)
    gate=read(m['deployment']['gate']['path']);cache=HistoryCache(root/'history_cache',2*2**30,8*2**30);metadata={};records=[]
    for index,row in enumerate(m['queries']):
        check();cpu_item,text=inputs.get(row)
        n=outputs.get(row,cpu_item,'native');g=outputs.get(row,cpu_item,'enhanced');f=outputs.get(row,cpu_item,'fusion')
        if n is None or g is None:
            started=time.monotonic();item=cpu_item.to('cuda:0');tokens=make_tokens(codec,item,'cuda:0')
            origin=NativeOrigin(model,select_rows(tokens,[0],crop=True),codec.first_separator)
            if n is None:
                history=np.where(cpu_item.observed[0],cpu_item.history[0],np.nan);trajectories=[];hashes=[]
                native_record=codec.record(history)
                bounds=codec.decode_ids([codec.finite_ids[0],codec.finite_ids[-1]],*native_record['scaling'],native_record['last'])[0]
                for sample in range(8):
                    check();value,path=outputs.trajectory(row,cpu_item,sample,
                        lambda seed:generate(origin,codec,history,row['horizon'],seed=seed,check=check))
                    trajectories.append(value['result']);hashes.append(sha(path))
                    progress(root,'native',index,len(m['queries']),records,sample=sample+1)
                n=outputs.save(row,cpu_item,'native',np.median(np.asarray([t['prediction'] for t in trajectories],np.float64),axis=0),
                    time.monotonic()-started,trajectory_sha256=hashes,format_complete=True,inverse_transform_complete=True,
                    boundary_fraction=float(np.mean([t['boundary'] for t in trajectories])),fallback_count=0,
                    representable_raw_range=np.asarray(bounds).tolist())
            if g is None:
                started=time.monotonic()
                if text not in metadata:metadata[text]=model.metadata_vector(metadata_tokens(tokenizer,text,'cuda:0')).detach()
                prefix=cache.get(SharedPrefixBuilder(model,origin),tokens)
                prediction,_=model.enhanced(tokens,item,metadata[text],prefix=prefix,optimized=True)
                g=outputs.save(row,cpu_item,'enhanced',prediction[0].cpu().numpy(),time.monotonic()-started,
                    derived_continuous_forecast=True,language_generation_coverage=None,fallback_count=0)
            del tokens,item,origin
        if f is None:
            started=time.monotonic();prediction,alpha=blend(gate,cpu_item,n['prediction'],g['prediction'])
            f=outputs.save(row,cpu_item,'fusion',prediction,time.monotonic()-started,enhanced_weight=alpha.tolist(),
                native_record_sha256=sha(outputs.path(row,'native')),enhanced_record_sha256=sha(outputs.path(row,'enhanced')),
                hard_native_fallback=bool(np.all(alpha==0)),derived_continuous_forecast=True,language_generation_coverage=None)
        records.append(dict(seconds=n['seconds']+g['seconds']+f['seconds']))
        progress(root,'fusion',index+1,len(m['queries']),records,cache_hits=cache.hits,cache_misses=cache.misses)
    for name in ('native','enhanced','fusion'):outputs.complete(name,inputs)
    if expert_hash!=model_digest(model.adapter) or versions!={n:p._version for n,p in native.named_parameters()} or \
            any(p.grad is not None for p in model.parameters()):raise ValueError('Frozen evaluation modified weights or accumulated gradients')
    verify_deployment(m['deployment'])
    seal(root/'frozen_gpu_weights.json',dict(binding=outputs.binding,native_unchanged=True,expert_unchanged=True,
        gradients_disabled=True,expert_model_sha256=expert_hash,native_parameters=sum(p.numel() for p in native.parameters()),
        added_expert_parameters=sum(p.numel() for p in model.adapter.parameters()),
        peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),peak_torch_reserved_bytes=torch.cuda.max_memory_reserved()))
    atomic_json(root/'status.json',dict(state='complete',phase='gpu_predictions_complete_cpu_scoring_pending',
        queries=len(records),goal_achieved=False,target_results_not_yet_scored=True))


@torch.no_grad()
def run_ufg(cfg,root):
    from gofa_f.dual_memory_engines import model_digest
    from gofa_f.ufg_memory_roster import MemoryRosterUFG,pack_roster
    from gofa_f.idle_launcher import process_identity
    if torch.cuda.is_initialized() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('UFG evaluation is CPU-only')
    m,inputs,outputs=guard(root,cfg);check=signal_checker();models={};hashes={};records=[];started=time.monotonic()
    prior=read(root/'ufg_activity.json')['spent_seconds'] if (root/'ufg_activity.json').exists() else 0.
    try:
        for index,row in enumerate(m['queries']):
            check();item,_=inputs.get(row);value=outputs.get(row,item,'ufg');shape=f"{row['length']}-{row['horizon']}"
            if shape not in models:
                entry=m['deployment']['ufg'][shape]
                model=MemoryRosterUFG(cfg['ufg_root'],row['length'],row['horizon'],'cpu').eval()
                model.load_state_dict(torch.load(entry['path'],map_location='cpu')['model'])
                for p in model.parameters():p.requires_grad_(False)
                hashes[shape]=model_digest(model)
                if hashes[shape]!=entry['model_sha256']:raise ValueError('Loaded UFG checkpoint differs from source best')
                models[shape]=model
            if value is None:
                started_query=time.monotonic();prediction=models[shape](pack_roster(item))[0].numpy()
                value=outputs.save(row,item,'ufg',prediction,time.monotonic()-started_query,
                    trailing_padded_roles=31-len(item.node_ids),derived_continuous_forecast=True,
                    language_generation_coverage=None,fallback_count=0)
            records.append(value);progress(root,'ufg',index+1,len(m['queries']),records)
            atomic_json(root/'ufg_activity.json',dict(spent_seconds=prior+time.monotonic()-started,
                worker=dict(pid=os.getpid(),start_ticks=process_identity(os.getpid())),cuda_initialized=False,threads=2))
        outputs.complete('ufg',inputs)
        if any(model_digest(v)!=hashes[k] or any(p.grad is not None for p in v.parameters()) for k,v in models.items()):
            raise ValueError('UFG evaluation changed frozen source parameters')
        verify_deployment(m['deployment'])
        atomic_json(root/'ufg_status.json',dict(state='complete',queries=len(records),source_weights_unchanged=True))
        seal(root/'ufg_frozen_weights.json',dict(binding=outputs.binding,shape_specific=True,
            checkpoints={shape:dict(model_sha256=hashes[shape],parameters=sum(p.numel() for p in model.parameters())) for shape,model in models.items()}))
    finally:
        atomic_json(root/'ufg_activity.json',dict(spent_seconds=prior+time.monotonic()-started,worker=None,
            cuda_initialized=False,threads=2,peak_rss_kib=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss))


def summarize(cfg,root):
    from gofa_f.dual_memory_eval_metrics import compare_dataset,spatial_blocks,macro_metrics
    from gofa_f.generative_data import Series
    m,inputs,outputs=guard(root,cfg);path=root/'report.json'
    if path.exists():return read(path)
    frozen=read(root/'frozen_gpu_weights.json')
    if frozen['binding']!=outputs.binding or not frozen['native_unchanged'] or not frozen['expert_unchanged'] or not frozen['gradients_disabled']:
        raise ValueError('Completed frozen-weight execution proof required before scoring')
    if not read(root/'ufg_status.json').get('source_weights_unchanged'):raise ValueError('UFG frozen-weight proof missing')
    if read(root/'activity.json').get('worker') is not None or read(root/'ufg_activity.json').get('worker') is not None:
        raise ValueError('Prediction workers must exit before CPU scoring')
    seals={name:root/f'{name}_complete.json' for name in MODELS}
    truths=truth_records(root,seals);results={};prediction_map={name:[] for name in MODELS}
    for row in m['queries']:
        item,_=inputs.get(row)
        for name in MODELS:prediction_map[name].append(outputs.get(row,item,name)['prediction'])
    names=TARGETS if m['scope']=='targets' else list(m['datasets'])
    for name in names:
        data=Series(m['datasets'][name]['path']);spatial=spatial_blocks(data.external,data.ids)
        shapes=sorted({(r['query']['length'],r['query']['horizon']) for r in truths if r['query']['source']==name})
        for length,horizon in shapes:
            selected=[i for i,r in enumerate(truths) if r['query']['source']==name and (r['query']['length'],r['query']['horizon'])==(length,horizon)]
            records=[truths[i] for i in selected];predictions={n:[p[i] for i in selected] for n,p in prediction_map.items()}
            result=compare_dataset(records,predictions,spatial)
            for model,p in predictions.items():
                y=np.asarray([r['target'] for r in records]);mask=np.asarray([r['observed'] for r in records]);p=np.asarray(p)
                variables=np.asarray([r['query']['variable'] for r in records]);per_step={}
                for h in range(horizon):
                    valid=mask[:,h]
                    if valid.any():per_step[str(h+1)]=macro_metrics(p[valid,h:h+1],y[valid,h:h+1],mask[valid,h:h+1],variables[valid])['macro']
                result['metrics'][model]['per_step']=per_step
            result['coverage']=dict(queries=len(selected),variables=len({r['query']['variable'] for r in records}),
                dataset_variables=len(data.ids),real_node_counts=[len(inputs.get(m['queries'][i])[0].node_ids) for i in selected],
                language_generation_coverage_only_applies_to_native=True,enhanced_and_fusion_are_continuous_predictions=True)
            result['output_audit']={}
            for model in MODELS:
                saved=[outputs.get(m['queries'][i],inputs.get(m['queries'][i])[0],model) for i in selected]
                outside=0;slots=0
                for i,value in zip(selected,saved):
                    item,_=inputs.get(m['queries'][i]);native_record=outputs.get(m['queries'][i],item,'native')
                    low,high=native_record['representable_raw_range'];p=np.asarray(value['prediction'])
                    outside+=int(((p<low)|(p>high)).sum());slots+=len(p)
                result['output_audit'][model]=dict(finite_prediction_fraction=1.,
                    outside_native_representable_range_fraction=outside/slots,
                    language_generation_coverage=1. if model=='native' else None,
                    inverse_transform_complete=True if model=='native' else None,
                    hard_native_fallback_queries=sum(bool(v.get('hard_native_fallback')) for v in saved),
                    fallback_count=sum(v.get('fallback_count',0) for v in saved))
            results[name if m['scope']=='targets' else f'{name}/{length}-{horizon}']=result
    verify_deployment(m['deployment'])
    achieved=m['scope']=='targets' and set(results)==set(TARGETS) and all(v['passes_both'] for v in results.values())
    report=dict(binding=outputs.binding,scope=m['scope'],deployment=m['deployment'],results=results,goal_achieved=achieved,
        all_model_groups_complete=True,parameters_modified=False,target_labels_used_for_fitting=False,
        source_confirmation_is_not_candidate_reselection=True,old_target_development_exposure=True,
        single_training_seed=True,pretraining_exposure_unverifiable=True,
        observed_samples_do_not_establish_general_stability=True,
        ufg_shape_specific_weights_and_padding_disclosed=True,proxy_weather_wind_graphs_disclosed=True,
        no_extra_scada_inputs=True,prediction_completion_sha256={n:sha(p) for n,p in seals.items()},
        gpu_activity=read(root/'activity.json'),cpu_ufg_activity=read(root/'ufg_activity.json'))
    report['frozen_weight_proof']=frozen;report['ufg_weight_proof']=read(root/'ufg_frozen_weights.json')
    seal(path,report)
    if m['scope']=='source_confirmation':
        seal(root/'confirmation_certificate.json',dict(complete=True,output=str(root),deployment=m['deployment'],
            report_sha256=sha(path),manifest_sha256=outputs.binding,all_model_groups_complete=True,
            parameters_modified=False,confirmation_used_to_refit=False))
    atomic_json(root/'final_status.json',dict(state='complete',phase='source_confirmed' if m['scope']=='source_confirmation' else 'frozen_six_target_evaluation',
        goal_achieved=achieved,report_sha256=sha(path)))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--ufg',action='store_true');parser.add_argument('--summarize',action='store_true');args=parser.parse_args()
    cfg=configuration(args.config);root=Path(args.output)
    try:
        if args.summarize:summarize(cfg,root)
        elif args.ufg:run_ufg(cfg,root)
        else:run_gpu(cfg,root)
    except InterruptedError as exc:
        atomic_json(root/('ufg_status.json' if args.ufg else 'status.json'),dict(state='resource_paused',phase='frozen_evaluation',error=str(exc)))
        raise SystemExit(75)
    except BaseException as exc:
        atomic_json(root/f'failure-{time.time_ns()}.json',dict(error=repr(exc),training_updates=0))
        atomic_json(root/('ufg_status.json' if args.ufg else 'status.json'),dict(state='failed',error=repr(exc),goal_achieved=False));raise


if __name__=='__main__':main()
