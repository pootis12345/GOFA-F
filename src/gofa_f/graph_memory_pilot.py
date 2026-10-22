"""One fixed-budget, source-only actual ChatTime graph-memory experiment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
import yaml

from gofa_f.allnode_stage import Paused,supervisor_alive
from gofa_f.generative_data import Query,sha
from gofa_f.generative_model import GenerativePrefixLRU
from gofa_f.graph_memory_data import CONTROLS,memory_graph
from gofa_f.graph_memory_model import GraphMemoryGOFA
from gofa_f.graph_memory_stage import FILES as MEMORY_FILES,load_sources
from gofa_f.prefix_cache import atomic_json
from gofa_f.short_codec import ShortCodec
from gofa_f.short_forecast import summarize
from gofa_f.short_sampling_diagnostic import SETTINGS,generate,sample_seed
from gofa_f.short_training import compact_forward,compact_loss,teacher_prefix
from gofa_f.source_graph_probe import digest,immutable,read,save_torch
from gofa_f.source_retraining_data import VIEWS

TRAINING=dict(max_updates=100,effective_batch=8,microbatch=1,queries_per_view=64,
    learning_rate=.0001,warmup_updates=10,weight_decay=.0001,gradient_clip=1.,validation_at=[100])
POLICY=dict(source_training_enabled=True,target_training_enabled=False,target_evaluation_enabled=False,
    hourly_followup=False,mechanism_checkpoint_reuse=False,selected_parent_adapter_reuse=False)
GATE=dict(traffic_view_improvement=.01,maximum_time_block_degradation=.02,
    maximum_other_view_native_degradation=.02,uncertainty_draws=2000)
CODE=tuple(dict.fromkeys((*MEMORY_FILES,'graph_memory_pilot.py','generative_model.py','chattime.py',
    'allnode_graph.py','allnode_codec.py','graph_probe_data.py','source_sparse_normalized_probe.py')))


def configuration(path):
    cfg=yaml.safe_load(Path(path).read_text().replace('${GOFA_WORKSPACE}',os.environ['GOFA_WORKSPACE']))
    if cfg['protocol']!='stage57-chattime-graph-memory-source-pilot-v1' or cfg['training']!=TRAINING \
            or cfg['seed']!=42 or cfg['conditions']!=list(CONTROLS) or cfg['policy']!=POLICY \
            or cfg['gate']!=GATE or cfg['sampling']!=SETTINGS or cfg['resources']['wall_deadline'] is not None \
            or cfg['resources']['activity_limit_seconds'] is not None or cfg['data']!={'target_training_enabled':False,'target_evaluation_enabled':False}:
        raise ValueError('Fixed source-only graph-memory pilot protocol changed')
    return cfg


def tensor_digest(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        h.update(name.encode());h.update(str((tuple(value.shape),value.dtype)).encode())
        h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def batch_queries(manifest,update):
    """Eight slots: two per view; weights make each of three domains equal."""
    result=[];epoch,offset=divmod(update*2,64)
    for view,name in enumerate(VIEWS):
        rows=manifest['queries'][name]['train']
        order=np.random.default_rng(np.random.SeedSequence([42,57,view,epoch])).permutation(64)
        for i in range(2):
            result.append(dict(query=rows[int(order[offset+i])],weight=1/12 if view<2 else 1/6))
    return result


def prepare(cfg,output):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():raise ValueError('CPU preparation required')
    parent=Path(cfg['mechanism_output']);m=read(parent/'manifest.json');proof=read(parent/'mechanisms.json')
    if sha(parent/'manifest.json')!=cfg['mechanism_manifest_sha256'] or not proof['passed'] \
            or sha(parent/'mechanisms.json')!=cfg['mechanisms_sha256'] or proof['manifest_sha256']!=sha(parent/'manifest.json'):
        raise ValueError('Real ChatTime mechanism evidence changed')
    for name,value in m['code'].items():
        if sha(Path(__file__).parent/name)!=value:raise ValueError('Accepted model/numerical kernel changed: '+name)
    sources=load_sources(m);graphs={}
    for name in VIEWS:
        graphs[name]=[]
        for split,key in [('train','train'),('development','validation')]:
            for q in m['queries'][name][split]:
                query=Query(**q);data=sources[name]
                if query.variable not in m['partitions'][name][key]:raise ValueError('Source node partition changed')
                lo,hi=(0,int(.6*len(data.values))) if split=='train' else (int(.6*len(data.values)),int(.8*len(data.values)))
                if not max(lo,query.length)<=query.origin or query.origin+query.horizon>hi \
                        or data.segments[query.origin-query.length]!=data.segments[query.origin+query.horizon-1]:
                    raise ValueError('Source query crosses time split/segment')
                items=[memory_graph(data,query,m['partitions'][name][key],c) for c in CONTROLS]
                first=items[0]
                for item in items[1:]:
                    if item.variable_ids!=first.variable_ids or any(not torch.equal(getattr(first.graph,k),getattr(item.graph,k)) for k in ('history','observed','edges')):
                        raise ValueError('Controls changed input nodes/history/scales')
                graphs[name].append(dict(split=split,query=q,roster=list(first.variable_ids),
                    controls={c:v.identity for c,v in zip(CONTROLS,items)}))
    result=dict(configuration=cfg,source_preparation=m,queries=m['queries'],partitions=m['partitions'],
        graph_identities=graphs,mechanism_manifest_sha256=sha(parent/'manifest.json'),
        mechanism_proof_sha256=sha(parent/'mechanisms.json'),reserved_source_queries_opened=0,target_queries_opened=0,
        code={name:sha(Path(__file__).parent/name) for name in CODE},
        source_only=True,same_rosters_labels_and_initialization=True,strict_target_zero_shot_not_yet_tested=True)
    output.mkdir(parents=True,exist_ok=True);immutable(output/'manifest.json',result)
    atomic_json(output/'status.json',dict(state='prepared',stage=57,phase='source_pilot',target_predictions=0))
    return result


def history_target(data,query,labels):
    q=Query(**query)
    history=np.where(data.mask[q.origin-q.length:q.origin,q.variable],data.values[q.origin-q.length:q.origin,q.variable],np.nan)
    target=None if not labels else np.where(data.mask[q.origin:q.origin+q.horizon,q.variable],data.values[q.origin:q.origin+q.horizon,q.variable],np.nan)
    return history,target


def saved_rng():
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        numpy=np.random.get_state(),python=random.getstate())


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])
    np.random.set_state(state['numpy']);random.setstate(state['python'])


def checkpoint(model,optimizer,state,path):
    value=dict(state,adapter=model.trainable_state(),optimizer=optimizer.state_dict(),rng=saved_rng(),
        parameter_version=model.parameter_version,cache_generation=model.cache_generation,
        gradients={n:None if p.grad is None else p.grad.detach().cpu().clone() for n,p in model.adapter.named_parameters()})
    save_torch(path,value)


def restore_checkpoint(model,optimizer,path,binding,condition):
    value=torch.load(path,map_location='cpu',weights_only=False)
    if value['binding']!=binding or value['condition']!=condition:raise ValueError('Training recovery binding changed')
    model.load_trainable(value['adapter']);optimizer.load_state_dict(value['optimizer'])
    model.parameter_version=value['parameter_version'];model.cache_generation=value['cache_generation']
    for name,p in model.adapter.named_parameters():p.grad=None if value['gradients'][name] is None else value['gradients'][name].to(p.device)
    restore_rng(value['rng'])
    return {k:v for k,v in value.items() if k not in ('adapter','optimizer','rng','gradients','parameter_version','cache_generation')}


def train_condition(model,codec,sources,m,condition,initial,output,binding,cache,check,progress):
    root=output/'conditions'/condition;root.mkdir(parents=True,exist_ok=True);path=root/'last.pt'
    model.load_trainable(initial['adapter']);restore_rng(initial['rng']);model.train()
    optimizer=torch.optim.AdamW(model.adapter.parameters(),lr=TRAINING['learning_rate'],weight_decay=TRAINING['weight_decay'])
    state=dict(binding=binding,condition=condition,updates=0,pending=0,partial_losses=[],losses=[],training_seconds=0.)
    if path.exists():state=restore_checkpoint(model,optimizer,path,binding,condition)
    else:optimizer.zero_grad(set_to_none=True);checkpoint(model,optimizer,state,path)
    started=time.monotonic();prior_seconds=state['training_seconds']
    def save():
        state['training_seconds']=prior_seconds+time.monotonic()-started;checkpoint(model,optimizer,state,path)
    try:
        while state['updates']<TRAINING['max_updates']:
            for group in optimizer.param_groups:group['lr']=TRAINING['learning_rate']*min(1.,(state['updates']+1)/TRAINING['warmup_updates'])
            records=batch_queries(m,state['updates'])
            for index in range(state['pending'],len(records)):
                check();row=records[index];q=Query(**row['query']);data=sources[q.source]
                history,target=history_target(data,row['query'],True)
                tokens=codec.batch([codec.record(history,target)]).to(next(model.native.parameters()).device)
                prefix=teacher_prefix(model,tokens,cache)
                graph=memory_graph(data,q,m['partitions'][q.source]['train'],condition)
                contexts=model.encode_origins([graph],codec,cache,check)
                loss=compact_loss(compact_forward(model,codec,tokens,contexts,prefix))[0]
                if not loss.isfinite():raise FloatingPointError('Nonfinite source teacher loss')
                (loss*row['weight']).backward()
                state['partial_losses'].append(dict(source=q.source,ce=float(loss.detach()),query=row['query']))
                state['pending']=index+1
                del loss,contexts,prefix,tokens
                progress('source_train',condition=condition,updates=state['updates'],pending=state['pending'])
            check()
            if any(p.grad is not None and not p.grad.isfinite().all() for p in model.adapter.parameters()):raise FloatingPointError('Nonfinite graph-memory gradient')
            if state['updates'] in (0,TRAINING['max_updates']-1):
                immutable(root/f'gradients-{state["updates"]+1:03d}.json',dict(update=state['updates']+1,
                    norms={n:0. if p.grad is None else float(p.grad.norm()) for n,p in model.adapter.named_parameters()}))
            torch.nn.utils.clip_grad_norm_(model.adapter.parameters(),TRAINING['gradient_clip'])
            optimizer.step();model.mark_updated();optimizer.zero_grad(set_to_none=True)
            state['updates']+=1;state['losses'].append(state['partial_losses']);state['partial_losses']=[];state['pending']=0
            save()
        final=dict(binding=binding,condition=condition,updates=state['updates'],checkpoint_sha256=sha(path),
            adapter_sha256=tensor_digest(model.trainable_state()),initial_adapter_sha256=initial['adapter_sha256'],
            query_order_sha256=digest([batch_queries(m,u) for u in range(TRAINING['max_updates'])]),
            training_seconds=state['training_seconds'],selection='fixed final update100; no intermediate accuracy selection')
        immutable(root/'locked.json',final)
        progress('source_train',condition=condition,updates=state['updates'],pending=0)
        return final
    except Paused:
        save();raise


@torch.no_grad()
def evaluate(model,codec,sources,m,label,condition,output,binding,model_sha,cache,check,progress):
    root=output/'evaluations'/label;root.mkdir(parents=True,exist_ok=True);paths=root/'trajectories';paths.mkdir(exist_ok=True)
    queries=[q for name in VIEWS for q in m['queries'][name]['development']]
    identity=dict(binding=binding,model_sha256=model_sha,condition=condition,sampling=SETTINGS)
    complete=read(root/'complete.json')
    if complete:
        if complete['identity']!=identity:raise ValueError('Evaluation completion binding changed')
        progress('source_sample8',condition=label,evaluation_work_done=sum(q['horizon'] for q in queries)*8)
        return complete
    entries=[];model.eval()
    def checked(path,query,**fields):
        value=read(path)
        if value is not None and (value['identity']!=identity or value['query']!=query or any(value[k]!=v for k,v in fields.items())):
            raise ValueError('Sample trajectory or query binding changed')
        return value
    for i,query in enumerate(queries):
        check();entry=checked(root/f'{i:04d}.json',query)
        if entry:entries.append(entry);continue
        q=Query(**query);data=sources[q.source];history,_=history_target(data,query,False);base=codec.record(history)
        samples=[checked(paths/f'{i:04d}-{sample}.json',query,seed=sample_seed(query,sample),sample=sample) for sample in range(8)]
        contexts=None
        if condition is not None and any(v is None for v in samples):
            graph=memory_graph(data,q,m['partitions'][q.source]['validation'],condition)
            contexts=model.encode_origins([graph],codec,cache,check)
        for sample in range(8):
            check()
            if samples[sample] is None:
                started=time.monotonic();seed=sample_seed(query,sample)
                item=generate(model,codec,history,q.horizon,contexts,seed=seed,check=check)
                item.update(identity=identity,query=query,seed=seed,sample=sample,active_seconds=time.monotonic()-started)
                immutable(paths/f'{i:04d}-{sample}.json',item);samples[sample]=item
            work=8*sum(row['horizon'] for row in queries[:i])+(sample+1)*q.horizon
            progress('source_sample8',condition=label,queries_done=i,queries_total=len(queries),sample=sample+1,evaluation_work_done=work)
        # The generation loop has no access to future truth. Labels are read only now.
        prediction=np.median(np.asarray([v['prediction'] for v in samples],np.float64),axis=0)
        target=data.values[q.origin:q.origin+q.horizon,q.variable].astype(np.float64);observed=data.mask[q.origin:q.origin+q.horizon,q.variable]
        if not observed.any() or not np.isfinite(prediction).all():raise ValueError('Invalid sampled source result')
        scale,offset=base['scaling'];low,high=(-.9999+.5-offset)/scale,(.9999+.5-offset)/scale
        centers=np.median([[codec.id_centers[t] for t in v['numeric_ids']] for v in samples],axis=0)
        entry=dict(identity=identity,query=query,prediction=prediction.tolist(),target=target.tolist(),observed=observed.tolist(),
            scaling=list(base['scaling']),representable_range=[low,high],true_outside_range=(((target<low)|(target>high))&observed).tolist(),
            range_only_error_lower_bound=np.where(observed,np.abs(target-np.clip(target,low,high)),0).tolist(),
            boundary=((centers==-.9999)|(centers==.9999)).tolist(),inverse_valid=[True]*q.horizon,
            format_complete=True,empty_history=False,numeric_slots=q.horizon,generated_token_count=sum(len(v['generated_ids']) for v in samples),
            forward_calls=8*q.horizon,mae=float(np.abs(prediction-target)[observed].mean()),
            active_seconds=sum(v['active_seconds'] for v in samples),seeds=[v['seed'] for v in samples])
        immutable(root/f'{i:04d}.json',entry);entries.append(entry)
        del contexts
    result=dict(identity=identity,complete=True,queries=len(entries),metrics=summarize(entries),
        query_shards_sha256={f'{i:04d}':sha(root/f'{i:04d}.json') for i in range(len(entries))})
    immutable(root/'complete.json',result)
    progress('source_sample8',condition=label,queries_done=len(queries),queries_total=len(queries),
        evaluation_work_done=sum(q['horizon'] for q in queries)*8)
    return result


def variable_mae(entries):
    groups={}
    for row in entries:
        key=row['query']['variable'];mask=np.asarray(row['observed'],bool)
        error=np.abs(np.asarray(row['prediction'])-np.asarray(row['target']))[mask]
        groups.setdefault(key,[]).extend(error.tolist())
    return {key:float(np.mean(value)) for key,value in groups.items()}


def gate_report(records):
    table={label:{name:variable_mae([r for r in rows if r['query']['source']==name]) for name in VIEWS} for label,rows in records.items()}
    mae={label:{name:float(np.mean(list(values.values()))) for name,values in views.items()} for label,views in table.items()}
    checks=[];ratios={};intervals={}
    for name in list(VIEWS)[:2]:
        physical=[r for r in records['physical'] if r['query']['source']==name]
        order=np.argsort([r['query']['origin'] for r in physical]);blocks=np.array_split(order,4)
        for label in ('native','self','rewired'):
            baseline=mae[label][name];value=mae['physical'][name]
            ok=value<=.99*baseline and value<baseline-max(1e-12,1e-9*baseline)
            checks.append(dict(source=name,comparison=label,scope='variable_equal_mae',passed=bool(ok)))
            ratios[name,label]=value/max(baseline,1e-12)
            for block,indices in enumerate(blocks):
                a=float(np.mean(list(variable_mae([physical[int(i)] for i in indices]).values())))
                other=[r for r in records[label] if r['query']['source']==name]
                b=float(np.mean(list(variable_mae([other[int(i)] for i in indices]).values())))
                checks.append(dict(source=name,comparison=label,scope='time_block_'+str(block),passed=bool(a<=1.02*b),physical=a,baseline=b))
            keys=sorted(table['physical'][name]);rng=np.random.default_rng(np.random.SeedSequence([42,57,list(VIEWS).index(name),list(records).index(label)]))
            a=np.array([table['physical'][name][k] for k in keys]);b=np.array([table[label][name][k] for k in keys])
            draws=rng.integers(len(keys),size=(GATE['uncertainty_draws'],len(keys)))
            values=a[draws].mean(1)/np.maximum(b[draws].mean(1),1e-12)
            intervals[name+'_'+label]=dict(ratio=ratios[name,label],ratio_95pct=np.quantile(values,[.025,.975]).tolist(),
                variable_clusters=len(keys),small_development_sample=True)
    for name in list(VIEWS)[2:]:
        checks.append(dict(source=name,comparison='native',scope='no_more_than_2pct_degradation',
            passed=bool(mae['physical'][name]<=1.02*mae['native'][name])))
    return dict(passed=all(c['passed'] for c in checks),checks=checks,macro_mae_by_view=mae,
        paired_variable_cluster_intervals=intervals,target_predictions=0,goal_achieved=False,
        next_action='replicate fixed recipe and source holdout before targets' if all(c['passed'] for c in checks)
            else 'seal scientific failure; diagnose actual graph-memory source evidence before any larger training or targets',
        caveat='Single-seed small source pilot; coordinate4NN graphs are proxies; no target or recent-GNN performance claim')


def finish_report(output,binding):
    records={}
    for label in ('native',*CONTROLS):
        root=output/'evaluations'/label;complete=read(root/'complete.json')
        if not complete or complete['identity']['binding']!=binding:raise ValueError('All four complete paired source evaluations required')
        rows=[read(root/f'{i:04d}.json') for i in range(64)]
        for i,row in enumerate(rows):
            if sha(root/f'{i:04d}.json')!=complete['query_shards_sha256'][f'{i:04d}']:raise ValueError('Source query shard changed')
            samples=[read(root/'trajectories'/f'{i:04d}-{sample}.json') for sample in range(8)]
            for sample,value in enumerate(samples):
                if value is None or value['identity']!=row['identity'] or value['query']!=row['query'] \
                        or value['seed']!=sample_seed(row['query'],sample) or value['sample']!=sample \
                        or len(value['numeric_ids'])!=row['query']['horizon']:
                    raise ValueError('Incomplete or wrongly bound raw sampled trajectories')
            point=np.median(np.asarray([v['prediction'] for v in samples],np.float64),0)
            np.testing.assert_allclose(point,row['prediction'],rtol=0,atol=1e-6)
        replay=summarize(rows)
        for name in VIEWS:
            if abs(replay['by_source'][name]['macro']['mae']-complete['metrics']['by_source'][name]['macro']['mae'])>1e-6:
                raise ValueError('Independent raw-unit metric replay failed')
        records[label]=rows
    if any([r['query'] for r in rows]!=[r['query'] for r in records['native']] for rows in records.values()):raise ValueError('Unpaired source query identities')
    report=dict(binding=binding,**gate_report(records),raw_metric_replay_passed=True)
    immutable(output/'report.json',report)
    atomic_json(output/'status.json',dict(state='complete',stage=57,phase='source_pilot_complete',
        source_gate_passed=report['passed'],target_predictions=0,goal_achieved=False))
    return report


def run(cfg,output):
    if os.environ.get('GOFA_STAGE19_ADMITTED')!='1':raise ValueError('Memory-cap supervisor admission required')
    m=read(output/'manifest.json');binding=sha(output/'manifest.json');proof=read(output/'cpu_preflight.json')
    if m['configuration']!=cfg or m['code']!={name:sha(Path(__file__).parent/name) for name in CODE} \
            or not proof['passed'] or proof['manifest_sha256']!=binding:raise ValueError('Bound source code/CPU acceptance required')
    if read(output/'status.json').get('state')=='complete':raise ValueError('Sealed pilot must not rerun')
    stop=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    def check():
        if stop[0] or not supervisor_alive(os.environ['GOFA_STAGE19_SUPERVISOR_PID'],os.environ['GOFA_STAGE19_SUPERVISOR_TICKS']):raise Paused('cooperative_resource_pause')
    last=[0.];first=time.monotonic();observed={};current={}
    train_units={c:0. for c in CONTROLS};eval_units={c:0 for c in ('native',*CONTROLS)}
    rate=dict(source_train=[0.,0.],source_sample8=[0.,0.]);anchor=[None,None,first,0.]
    evaluation_total=4*8*sum(q['horizon'] for name in VIEWS for q in m['queries'][name]['development'])
    def progress(phase,**fields):
        now=time.monotonic();condition=fields.get('condition')
        units=fields.get('updates',0)+fields.get('pending',0)/8 if phase=='source_train' else fields.get('evaluation_work_done',0)
        if phase in rate and anchor[:2]==[phase,condition] and units>=anchor[3]:
            rate[phase][0]+=now-anchor[2];rate[phase][1]+=units-anchor[3]
        anchor[:]=[phase,condition,now,units]
        if phase=='source_train' and condition in train_units:train_units[condition]=units
        if phase=='source_sample8' and condition in eval_units:eval_units[condition]=units
        current.update(phase=phase,**fields)
        if now-last[0]<5:return
        last[0]=now
        train_rate=rate['source_train'][0]/rate['source_train'][1] if rate['source_train'][1] else None
        eval_rate=rate['source_sample8'][0]/rate['source_sample8'][1] if rate['source_sample8'][1] else None
        left_train=300-sum(train_units.values());left_eval=evaluation_total-sum(eval_units.values())
        eta=(left_train*train_rate+left_eval*eval_rate if train_rate is not None and eval_rate is not None else None)
        atomic_json(output/'progress.json',dict(current,updated_at=time.time(),worker_seconds=time.monotonic()-first,
            target_predictions=0,maximum_updates_per_condition=100,conditions=CONTROLS,
            completed_updates_by_condition=train_units,sampled_numeric_slots_done=sum(eval_units.values()),sampled_numeric_slots_total=evaluation_total,
            train_seconds_per_update=train_rate,evaluation_seconds_per_numeric_slot=eval_rate,
            eta_seconds=eta,eta_note='Measured training plus remaining sample8 source evaluation; excludes unknown resource waits'))
        atomic_json(output/'status.json',dict(state='running',stage=57,**current))
    try:
        check();torch.cuda.set_per_process_memory_fraction(38*2**30/torch.cuda.get_device_properties(0).total_memory)
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        from transformers import LlamaForCausalLM,LlamaTokenizer
        parent=m['source_preparation'];root=Path(parent['source_configuration']['weights']['directory'])
        if sha(root/'snapshot.json')!=parent['weights_sha256']:raise ValueError('Frozen native weights snapshot changed')
        codec=ShortCodec(LlamaTokenizer.from_pretrained(root,local_files_only=True))
        native=LlamaForCausalLM.from_pretrained(root,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True)
        model=GraphMemoryGOFA(native,codec).to('cuda:0');sources=load_sources(parent)
        versions={name:p._version for name,p in model.native.named_parameters()}
        initial_path=output/'initial.pt'
        if initial_path.exists():
            initial=torch.load(initial_path,map_location='cpu',weights_only=False)
            if initial['binding']!=binding or tensor_digest(initial['adapter'])!=initial['adapter_sha256']:raise ValueError('Fresh initialization binding changed')
        else:
            initial=dict(binding=binding,adapter=model.trainable_state(),rng=saved_rng())
            initial['adapter_sha256']=tensor_digest(initial['adapter']);save_torch(initial_path,initial)
        model.load_trainable(initial['adapter'])
        cache=GenerativePrefixLRU(binding,max_bytes=2*2**30,disk_directory=output/'frozen_prefixes',disk_limit=8*2**30)
        progress('native_source_sample8')
        evaluate(model,codec,sources,m,'native',None,output,binding,parent['weights_sha256'],cache,check,progress)
        for condition in CONTROLS:
            locked=read(output/'conditions'/condition/'locked.json')
            if locked:
                path=output/'conditions'/condition/'last.pt'
                if locked['binding']!=binding or locked['checkpoint_sha256']!=sha(path) or locked['initial_adapter_sha256']!=initial['adapter_sha256']:
                    raise ValueError('Locked control checkpoint changed')
                saved=torch.load(path,map_location='cpu',weights_only=False);model.load_trainable(saved['adapter'])
                if tensor_digest(model.trainable_state())!=locked['adapter_sha256']:raise ValueError('Locked adapter hash changed')
                progress('source_train',condition=condition,updates=100,pending=0)
            else:locked=train_condition(model,codec,sources,m,condition,initial,output,binding,cache,check,progress)
            evaluate(model,codec,sources,m,condition,condition,output,binding,locked['checkpoint_sha256'],cache,check,progress)
        if any(p.requires_grad or p.grad is not None or p._version!=versions[name] for name,p in model.native.named_parameters()):raise ValueError('Frozen backbone changed')
        atomic_json(output/'cache.json',dict(hits=cache.hits,misses=cache.misses,peak_cpu_bytes=cache.peak,disk_bytes=cache.disk_used,
            graph_conditioned_states_cached=False,frozen_first30_layers_only=True))
        finish_report(output,binding)
    except Paused as error:
        atomic_json(output/'status.json',dict(state='resource_paused',stage=57,**current,reason=str(error)));raise SystemExit(75)
    except Exception as error:
        immutable(output/f'failure-{time.time_ns()}.json',dict(current,error=repr(error)))
        atomic_json(output/'status.json',dict(state='failed',stage=57,**current,error=repr(error)));raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--report',action='store_true');args=parser.parse_args()
    cfg=configuration(args.config)
    if args.prepare:prepare(cfg,args.output)
    elif args.report:print(json.dumps(finish_report(args.output,sha(args.output/'manifest.json'))))
    else:run(cfg,args.output)


if __name__=='__main__':main()
