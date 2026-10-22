"""One fixed source experiment: graph-only innovations, aligned Native prefixes."""
from contextlib import contextmanager
import argparse
import gc
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
import yaml

from gofa_f import graph_memory_stage as mechanism
from gofa_f.allnode_stage import Paused,supervisor_alive
from gofa_f.cross_innovation import CrossInnovationGOFA
from gofa_f.generative_data import Query,sha
from gofa_f.generative_model import GenerativePrefixLRU
from gofa_f.graph_memory_data import memory_graph
from gofa_f.graph_memory_diagnostic import scoring_tokens
from gofa_f.graph_memory_pilot import (TRAINING,batch_queries,checkpoint,gate_report,history_target,
    restore_checkpoint,restore_rng,saved_rng,tensor_digest)
from gofa_f.graph_proposal_probe import coupled_draw
from gofa_f.prefix_cache import atomic_json
from gofa_f.short_codec import ShortCodec
from gofa_f.short_forecast import summarize
from gofa_f.short_sampling_diagnostic import SETTINGS,generate,sample_seed
from gofa_f.short_training import compact_forward,compact_loss,teacher_prefix
from gofa_f.source_graph_probe import digest,immutable,read,save_torch

EXTRA=('cross_innovation.py','source_cross_innovation.py','graph_memory_pilot.py',
       'graph_memory_diagnostic.py','graph_proposal_probe.py')
QA_FILES=tuple(dict.fromkeys((*mechanism.FILES,*EXTRA)))


@contextmanager
def acceptance_factory():
    """Reuse the independent real-model proof runner with the new exact operator."""
    previous=mechanism.GraphMemoryGOFA,mechanism.FILES
    mechanism.GraphMemoryGOFA=CrossInnovationGOFA;mechanism.FILES=QA_FILES
    try:yield
    finally:mechanism.GraphMemoryGOFA,mechanism.FILES=previous


def configuration(path):
    cfg=yaml.safe_load(Path(path).read_text().replace('${GOFA_WORKSPACE}',os.environ['GOFA_WORKSPACE']))
    if cfg['protocol']!='stage58-cross-innovation-v1' or cfg['training']!=TRAINING or cfg['conditions']!=['physical','self','rewired'] \
            or cfg['data']!={'target_training_enabled':False,'target_evaluation_enabled':False} \
            or cfg['resources']['wall_deadline'] is not None or cfg['resources']['activity_limit_seconds'] is not None:
        raise ValueError('Single preregistered Stage58source recipe required')
    return cfg


def prepare(cfg,output):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():raise ValueError('CPU preparation required')
    parent=Path(cfg['parent_source_output']);old=read(parent/'manifest.json');proposal=Path(cfg['proposal_output'])
    if sha(parent/'manifest.json')!=cfg['parent_manifest_sha256'] or read(parent/'status.json')['state']!='complete' \
            or read(parent/'activity.json')['worker'] is not None:raise ValueError('Exact terminated source parent required')
    for name,value in old['code'].items():
        if sha(Path(__file__).parent/name)!=value:raise ValueError('Accepted inherited numerical source changed')
    proposals=read(proposal/'manifest.json');report=read(proposal/'report.json')
    if sha(proposal/'manifest.json')!=cfg['proposal_manifest_sha256'] or sha(proposal/'report.json')!=cfg['proposal_report_sha256'] or report['native_off_exact_paths']!=512 \
            or proposals['parent_manifest_sha256']!=cfg['parent_manifest_sha256'] or read(proposal/'activity.json')['worker'] is not None:
        raise ValueError('Exact512-path Native reference proof required')
    for name,value in proposals['code'].items():
        if sha(Path(__file__).parent/name)!=value:raise ValueError('Accepted proposal helper changed')
    expected=[q for split in old['queries'].values() for q in split['development']]
    if proposals['queries']!=expected:raise ValueError('Development query order changed')
    for i,query in enumerate(expected):
        for sample,ref in enumerate(proposals['proposals'][i]):
            path=parent/'evaluations/native/trajectories'/f'{i:04d}-{sample}.json'
            if sha(path)!=ref['sha256'] or ref['seed']!=sample_seed(query,sample):raise ValueError('Frozen Native development reference changed')
    codes={name:sha(Path(__file__).parent/name) for name in QA_FILES}
    m=dict(configuration=cfg,source_preparation=old['source_preparation'],queries=old['queries'],partitions=old['partitions'],
        parent_manifest_sha256=sha(parent/'manifest.json'),proposal_manifest_sha256=sha(proposal/'manifest.json'),
        development_queries=proposals['queries'],development_proposals=proposals['proposals'],inherited_code=old['code'],
        code=codes,source_holdout_opened=0,target_queries_opened=0,old_adapter_loaded=False,qa_weights_are_discarded=True,
        parent_source_gpu_seconds=read(parent/'activity.json')['spent_seconds'],
        parent_proposal_gpu_seconds=read(proposal/'activity.json')['spent_seconds'])
    output.mkdir(parents=True,exist_ok=True);immutable(output/'manifest.json',m)
    qa=output/'qa';qa.mkdir(exist_ok=True);qm=dict(old['source_preparation'],configuration=cfg,code=codes)
    immutable(qa/'manifest.json',qm)
    atomic_json(output/'status.json',dict(state='prepared',stage=58,phase='source_cross_innovation',target_predictions=0))
    return m


def training_references(model,codec,sources,m,output,binding,check,progress):
    model.eval();result={};paths={};done=0;seconds=0.
    for name,split in m['queries'].items():
        root=output/'native_training_references'/name;root.mkdir(parents=True,exist_ok=True)
        for i,query in enumerate(split['train']):
            check();q=Query(**query);path=root/f'{i:03d}.json';seed=sample_seed(query,0);value=read(path)
            if value:
                if value['binding']!=binding or value['query']!=query or value['seed']!=seed:raise ValueError('Training Native reference changed')
            else:
                history,_=history_target(sources[name],query,False);tick=time.monotonic()
                value=generate(model,codec,history,q.horizon,contexts=None,seed=seed,check=check)
                value.update(binding=binding,query=query,active_seconds=time.monotonic()-tick,source_future_labels_used=False)
                immutable(path,value)
            result[q.key]=value;paths[q.key]=sha(path);done+=1;seconds+=value['active_seconds']
            progress('native_references',reference_done=done,reference_total=256,reference_seconds=seconds)
    reference=dict(binding=binding,paths_sha256=paths,queries=256,frozen_native_only=True)
    immutable(output/'native_references.json',reference)
    return result,sha(output/'native_references.json')


def train(model,codec,sources,m,condition,initial,references,reference_hash,output,binding,cache,check,progress):
    directory=output/'conditions'/condition;directory.mkdir(parents=True,exist_ok=True);path=directory/'last.pt'
    model.set_condition(condition);model.load_trainable(initial['adapter']);restore_rng(initial['rng']);model.train()
    optimizer=torch.optim.AdamW(model.adapter.parameters(),lr=TRAINING['learning_rate'],weight_decay=TRAINING['weight_decay'])
    state=dict(binding=binding,condition=condition,updates=0,pending=0,partial_losses=[],losses=[],training_seconds=0.,reference_hash=reference_hash)
    if path.exists():state=restore_checkpoint(model,optimizer,path,binding,condition)
    else:optimizer.zero_grad(set_to_none=True);checkpoint(model,optimizer,state,path)
    if state['reference_hash']!=reference_hash:raise ValueError('Frozen training proposal set changed')
    started=time.monotonic();prior=state['training_seconds']
    def save():
        state['training_seconds']=prior+time.monotonic()-started;checkpoint(model,optimizer,state,path)
    try:
        while state['updates']<100:
            for group in optimizer.param_groups:group['lr']=TRAINING['learning_rate']*min(1.,(state['updates']+1)/10)
            rows=batch_queries(m,state['updates'])
            for i in range(state['pending'],8):
                check();row=rows[i];q=Query(**row['query']);history,target=history_target(sources[q.source],row['query'],True)
                ref=references[q.key];tokens,_=scoring_tokens(codec,history,target,ref['generated_ids']);tokens=tokens.to('cuda:0')
                prefix=teacher_prefix(model,tokens,cache);graph=memory_graph(sources[q.source],q,m['partitions'][q.source]['train'],condition)
                contexts=model.encode_origins([graph],codec,cache,check)
                loss=compact_loss(compact_forward(model,codec,tokens,contexts,prefix))[0]
                if not loss.isfinite():raise FloatingPointError('Nonfinite proposal-conditioned source loss')
                (loss*row['weight']).backward();state['pending']=i+1
                state['partial_losses'].append(dict(source=q.source,ce=float(loss.detach()),query=row['query']))
                del loss,contexts,prefix,tokens
                progress('source_train',condition=condition,updates=state['updates'],pending=state['pending'],training_seconds=prior+time.monotonic()-started)
            check()
            if any(p.grad is not None and not p.grad.isfinite().all() for p in model.adapter.parameters()):raise FloatingPointError('Nonfinite innovation gradient')
            if state['updates'] in (0,99):immutable(directory/f'gradients-{state["updates"]+1:03d}.json',
                dict(update=state['updates']+1,norms={n:0. if p.grad is None else float(p.grad.norm()) for n,p in model.adapter.named_parameters()}))
            torch.nn.utils.clip_grad_norm_(model.adapter.parameters(),1.);optimizer.step();model.mark_updated();optimizer.zero_grad(set_to_none=True)
            state['updates']+=1;state['losses'].append(state['partial_losses']);state['partial_losses']=[];state['pending']=0;save()
        lock=dict(binding=binding,condition=condition,updates=100,checkpoint_sha256=sha(path),adapter_sha256=tensor_digest(model.trainable_state()),
            initial_adapter_sha256=initial['adapter_sha256'],query_order_sha256=digest([batch_queries(m,u) for u in range(100)]),
            reference_hash=reference_hash,training_seconds=state['training_seconds'],operator='rich_temporal' if condition=='self' else 'cross_innovation')
        immutable(directory/'locked.json',lock);progress('source_train',condition=condition,updates=100,pending=0,training_seconds=state['training_seconds'])
        return lock
    except Paused:
        save();raise


@torch.no_grad()
def evaluate(model,codec,sources,m,condition,output,binding,cache,check,progress):
    model.set_condition(condition);model.eval();weight=m['source_preparation']['weights_sha256']
    if condition!='native':
        directory=output/'conditions'/condition;lock=read(directory/'locked.json');weight=sha(directory/'last.pt')
        if weight!=lock['checkpoint_sha256']:raise ValueError('Locked development model changed')
        saved=torch.load(directory/'last.pt',map_location='cpu',weights_only=False);model.load_trainable(saved['adapter']);del saved
    root=output/'evaluations'/condition;paths=root/'trajectories';paths.mkdir(parents=True,exist_ok=True);rows=[]
    identity=dict(binding=binding,condition=condition,weights_sha256=weight,sampling=SETTINGS,policy='one_coupled_native_prefix_step')
    for i,query in enumerate(m['development_queries']):
        check();q=Query(**query);data=sources[q.source];history,_=history_target(data,query,False);contexts=None;samples=[]
        for sample,proposal in enumerate(m['development_proposals'][i]):
            check();path=paths/f'{i:04d}-{sample}.json';value=read(path)
            if value:
                if value['identity']!=identity or value['query']!=query or value['proposal_sha256']!=proposal['sha256'] or value['seed']!=proposal['seed']:
                    raise ValueError('Development replay binding changed')
            else:
                if condition!='native' and contexts is None:
                    graph=memory_graph(data,q,m['partitions'][q.source]['validation'],condition);contexts=model.encode_origins([graph],codec,cache,check)
                value=coupled_draw(model,codec,history,q.horizon,proposal,contexts,cache,check)
                if condition=='native' and any(value[k]!=proposal[k] for k in ('numeric_ids','generated_ids','prediction')):
                    immutable(output/f'native-mismatch-{i:04d}-{sample}.json',dict(expected=proposal,actual=value))
                    raise ValueError('Native-off exact proposal identity failed')
                value.update(identity=identity,query=query,sample=sample,proposal_sha256=proposal['sha256']);immutable(path,value)
            samples.append(value);progress('source_evaluation',condition=condition,evaluation_query=i,evaluation_sample=sample+1)
        p=np.median(np.asarray([v['prediction'] for v in samples],np.float64),0)
        y=data.values[q.origin:q.origin+q.horizon,q.variable].astype(np.float64);mask=data.mask[q.origin:q.origin+q.horizon,q.variable]
        base=codec.record(history);scale,offset=base['scaling'];lo,hi=(-.9999+.5-offset)/scale,(.9999+.5-offset)/scale
        centers=np.median([[codec.id_centers[v] for v in path['numeric_ids']] for path in samples],0)
        row=dict(identity=identity,query=query,prediction=p.tolist(),target=y.tolist(),observed=mask.tolist(),
            true_outside_range=(((y<lo)|(y>hi))&mask).tolist(),boundary=((centers==-.9999)|(centers==.9999)).tolist(),
            inverse_valid=[True]*q.horizon,format_complete=True,empty_history=False,numeric_slots=q.horizon,
            generated_token_count=sum(len(v['generated_ids']) for v in samples),forward_calls=8,mae=float(np.abs(p-y)[mask].mean()))
        immutable(root/f'{i:04d}.json',row);rows.append(row);del contexts
    complete=dict(complete=True,binding=binding,identity=identity,queries=64,metrics=summarize(rows),
        query_shards_sha256={f'{i:04d}':sha(root/f'{i:04d}.json') for i in range(64)})
    immutable(root/'complete.json',complete)
    return rows


def run(cfg,output):
    if os.environ.get('GOFA_STAGE19_ADMITTED')!='1':raise ValueError('Cap-supervised source pipeline required')
    m=read(output/'manifest.json');binding=sha(output/'manifest.json');cpu=read(output/'cpu_preflight.json')
    if m['configuration']!=cfg or not cpu['passed'] or cpu['manifest_sha256']!=binding:raise ValueError('Bound CPU acceptance required')
    for name,value in dict(m['inherited_code'],**m['code']).items():
        if sha(Path(__file__).parent/name)!=value:raise ValueError('Fixed source code changed')
    stop=[False]
    def signals():
        for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    signals()
    def check():
        if stop[0] or not supervisor_alive(os.environ['GOFA_STAGE19_SUPERVISOR_PID'],os.environ['GOFA_STAGE19_SUPERVISOR_TICKS']):raise Paused('cooperative_resource_pause')
    timings=read(output/'phase_times.json',{});phase=[None,time.monotonic()];last=[0.];state={};began=time.monotonic()
    train_units={c:0. for c in cfg['conditions']};eval_units={c:0 for c in ('native',*cfg['conditions'])}
    rates=dict(references=.69,training=1.,evaluation=.04);references_done=[0]
    def change(new):
        now=time.monotonic()
        if phase[0] is not None:timings[phase[0]]=timings.get(phase[0],0.)+now-phase[1]
        phase[:]=[new,now];atomic_json(output/'phase_times.json',timings)
    def progress(kind,**fields):
        now=time.monotonic();condition=fields.get('condition')
        if kind=='native_references':
            references_done[0]=fields['reference_done'];rates['references']=fields['reference_seconds']/max(1,fields['reference_done'])
        if kind=='source_train' and condition in train_units:
            train_units[condition]=fields.get('updates',0)+fields.get('pending',0)/8
            if train_units[condition] and fields.get('training_seconds'):
                rates['training']=fields['training_seconds']/train_units[condition]
        if kind=='source_evaluation' and condition in eval_units:
            eval_units[condition]=fields['evaluation_query']*8+fields['evaluation_sample']
            key='native_identity' if condition=='native' else 'evaluation_'+condition
            rates['evaluation']=(timings.get(key,0.)+now-phase[1])/max(1,eval_units[condition])
        if state.get('phase')!=kind:state.clear()
        state.update(phase=kind,**fields)
        if now-last[0]<5:return
        last[0]=now
        eta=((256-references_done[0])*rates['references']+(300-sum(train_units.values()))*rates['training']+
            (2048-sum(eval_units.values()))*rates['evaluation']) if kind!='cuda_acceptance' else None
        atomic_json(output/'progress.json',dict(state,updated_at=time.time(),worker_seconds=now-began,
            target_queries=0,completed_updates_by_condition=train_units,evaluation_paths_by_condition=eval_units,
            eta_seconds=eta,eta_note='Phase-measured rates; historical estimates until measured; includes remaining references/train/sample8, excludes unknown resource waits'))
        atomic_json(output/'status.json',dict(state='running',stage=58,**state))
    try:
        check();change('cuda_acceptance');progress('cuda_acceptance')
        qa=output/'qa'
        if not (qa/'mechanisms.json').exists():
            atomic_json(qa/'cpu_preflight.json',dict(cpu,manifest_sha256=sha(qa/'manifest.json'),main_manifest_sha256=binding))
            try:
                with acceptance_factory():mechanism.gpu_acceptance(cfg,qa)
            finally:signals()
        if not read(qa/'mechanisms.json')['passed']:raise ValueError('New innovation CUDA mechanism failed')
        gc.collect();torch.cuda.empty_cache();check();change('model_load')
        from transformers import LlamaForCausalLM,LlamaTokenizer
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        root=Path(m['source_preparation']['source_configuration']['weights']['directory'])
        if sha(root/'snapshot.json')!=m['source_preparation']['weights_sha256']:raise ValueError('Frozen Native weights changed')
        torch.cuda.set_per_process_memory_fraction(38*2**30/torch.cuda.get_device_properties(0).total_memory)
        codec=ShortCodec(LlamaTokenizer.from_pretrained(root,local_files_only=True))
        native=LlamaForCausalLM.from_pretrained(root,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True)
        model=CrossInnovationGOFA(native,codec).to('cuda:0');versions={n:p._version for n,p in model.native.named_parameters()}
        sources=mechanism.load_sources(m['source_preparation']);initial_path=output/'initial.pt'
        if initial_path.exists():
            initial=torch.load(initial_path,map_location='cpu',weights_only=False)
            if initial['binding']!=binding or tensor_digest(initial['adapter'])!=initial['adapter_sha256']:raise ValueError('Fresh source initialization changed')
        else:
            initial=dict(binding=binding,adapter=model.trainable_state(),rng=saved_rng());initial['adapter_sha256']=tensor_digest(initial['adapter']);save_torch(initial_path,initial)
        cache=GenerativePrefixLRU(binding,max_bytes=2*2**30,disk_directory=output/'frozen_prefixes',disk_limit=8*2**30)
        change('native_references');refs,ref_hash=training_references(model,codec,sources,m,output,binding,check,progress)
        change('native_identity');records={'native':evaluate(model,codec,sources,m,'native',output,binding,cache,check,progress)}
        for condition in cfg['conditions']:
            change('training_'+condition);progress('source_train',condition=condition,updates=0,pending=0)
            lock=read(output/'conditions'/condition/'locked.json')
            if lock:
                if lock['binding']!=binding or lock['checkpoint_sha256']!=sha(output/'conditions'/condition/'last.pt') or lock['reference_hash']!=ref_hash:
                    raise ValueError('Locked source state changed')
                progress('source_train',condition=condition,updates=100,pending=0,training_seconds=lock['training_seconds'])
            else:train(model,codec,sources,m,condition,initial,refs,ref_hash,output,binding,cache,check,progress)
            change('evaluation_'+condition);records[condition]=evaluate(model,codec,sources,m,condition,output,binding,cache,check,progress)
        if any(p.requires_grad or p.grad is not None or p._version!=versions[n] for n,p in model.native.named_parameters()):raise ValueError('Frozen backbone changed')
        change(None)
        result=dict(binding=binding,**gate_report(records),stage=58,formal_updates_per_condition=100,discarded_qa_updates=3,
            native_off_exact_paths=512,native_training_references=256,reference_hash=ref_hash,phase_times=timings,
            metrics={c:summarize(rows) for c,rows in records.items()},source_holdout_opened=0,target_queries=0,native_frozen=True,
            fresh_initialization_sha256=initial['adapter_sha256'],cache_hits=cache.hits,cache_misses=cache.misses,cache_peak_cpu_bytes=cache.peak,
            temporal_control_is_richer_operator=True,physical_vs_permuted_only_support_differs=True)
        immutable(output/'report.json',result)
        atomic_json(output/'status.json',dict(state='complete',stage=58,phase='source_cross_innovation',source_gate_passed=result['passed'],goal_achieved=False,eta_seconds=None))
    except SystemExit as error:
        change(None)
        if error.code==75:atomic_json(output/'status.json',dict(state='resource_paused',stage=58,phase='cuda_acceptance'))
        else:atomic_json(output/'status.json',dict(state='failed',stage=58,error=repr(error)))
        raise
    except Paused as error:
        change(None);atomic_json(output/'status.json',dict(state='resource_paused',stage=58,**state,reason=str(error)));raise SystemExit(75)
    except Exception as error:
        change(None);immutable(output/f'failure-{time.time_ns()}.json',dict(state,error=repr(error)))
        atomic_json(output/'status.json',dict(state='failed',stage=58,**state,error=repr(error)));raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--prepare',action='store_true');args=parser.parse_args();cfg=configuration(args.config)
    if args.prepare:prepare(cfg,args.output)
    else:run(cfg,args.output)


if __name__=='__main__':main()
