"""Fixed source-only conditional/prefix diagnosis; no optimizer or generation."""
from contextlib import contextmanager
import argparse
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
import yaml

from gofa_f.allnode_stage import Paused,supervisor_alive
from gofa_f.generative_data import Query,sha
from gofa_f.generative_model import GenerativePrefixLRU
from gofa_f.graph_memory import linear,normalize
from gofa_f.graph_memory_data import memory_graph
from gofa_f.graph_memory_model import GraphMemoryGOFA
from gofa_f.graph_memory_pilot import history_target,tensor_digest
from gofa_f.graph_memory_stage import load_sources
from gofa_f.prefix_cache import atomic_json
from gofa_f.short_codec import ShortCodec
from gofa_f.short_sampling_diagnostic import sample_seed
from gofa_f.short_training import teacher_prefix
from gofa_f.source_graph_probe import digest,immutable,read


def config(path):
    cfg=yaml.safe_load(Path(path).read_text().replace('${GOFA_WORKSPACE}',os.environ['GOFA_WORKSPACE']))
    if cfg['protocol']!='stage57-graph-memory-fixed-diagnostic-v1' or cfg['training']!={'max_updates':0} \
            or cfg['queries_per_view']!=4 or cfg['data']!={'target_training_enabled':False,'target_evaluation_enabled':False} \
            or cfg['resources']['wall_deadline'] is not None or cfg['resources']['activity_limit_seconds'] is not None:
        raise ValueError('Fixed no-training source diagnostic required')
    return cfg


def scoring_tokens(codec,history,target,generated=None):
    record=codec.record(history,target);tokens=codec.batch([record])
    if generated is not None:
        numbers=[int(i) for i in generated if int(i) in codec.finite_set]
        expected=[]
        for i,value in enumerate(numbers):expected.extend(codec.first_separator if i==0 else codec.separator);expected.append(value)
        if expected!=list(generated) or len(numbers)!=len(target) \
                or len(generated)!=len(record['ids'])-record['prompt_end']-1:
            raise ValueError('Saved native prefix numeric slots/grammar changed')
        tokens.input_ids[0,record['prompt_end']+1:]=torch.tensor(generated,dtype=torch.long)
        tokens.full_ids=tokens.input_ids.clone()
    return tokens,record


def score_distribution(logits,tokens,codec,record,target):
    positions=torch.nonzero(tokens.labels[0]!=-100).flatten()
    scores=logits[0,positions-1].float().index_select(-1,codec.alphabet(logits.device))
    if not torch.isfinite(scores).all():raise FloatingPointError('Nonfinite conditional numeric scores')
    labels=codec.lookup(logits.shape[-1],logits.device)[tokens.labels[0,positions]]
    if (labels<0).any():raise ValueError('Nonnumeric target label')
    logp=scores.log_softmax(-1);p=logp.exp();cdf=p.cumsum(-1)
    indices=torch.arange(p.shape[-1]-1,device=p.device)
    crps=((cdf[:,:-1]-(indices[None]>=labels[:,None]).float())**2).mean(-1)
    quantile=(cdf<.5).sum(-1).clamp_max(p.shape[-1]-1)
    _,top=scores.topk(100,-1)
    truncated=torch.zeros_like(p).scatter_(-1,top,p.gather(-1,top))
    truncated=truncated/truncated.sum(-1,keepdim=True)
    top_median=(truncated.cumsum(-1)<.5).sum(-1).clamp_max(p.shape[-1]-1)
    centers=torch.tensor([codec.id_centers[i] for i in codec.finite_ids],dtype=torch.float64,device=p.device)
    scale,offset=record['scaling']
    def inverse(index):return (centers[index]+.5-offset)/scale
    observed=np.isfinite(target);truth=np.asarray(target,np.float64)[observed]
    median=inverse(quantile).cpu().numpy();argmax=inverse(scores.argmax(-1)).cpu().numpy();cut=inverse(top_median).cpu().numpy()
    ce=-logp.gather(-1,labels[:,None]).squeeze(-1);entropy=-(p*logp).sum(-1)
    steps=[]
    for i,h in enumerate(np.flatnonzero(observed)):
        steps.append(dict(step=int(h)+1,ce=float(ce[i]),cdf_score=float(crps[i]),
            median_prediction=float(median[i]),argmax_prediction=float(argmax[i]),top100_median_prediction=float(cut[i]),
            target=float(truth[i]),median_error=float(abs(median[i]-truth[i])),argmax_error=float(abs(argmax[i]-truth[i])),
            top100_median_error=float(abs(cut[i]-truth[i])),entropy=float(entropy[i]),
            endpoint_probability=float(p[i,0]+p[i,-1]),true_label_top100=bool((top[i]==labels[i]).any())))
    return dict(observed=observed.tolist(),steps=steps,numeric_vocabulary=10000,
        label_shift='logits[j-1] predicts observed label[j]',conditional_diagnostic_not_free_forecast=True)


def stats(value):
    value=value.float()
    if not value.numel():return None
    return dict(count=value.numel(),mean=float(value.mean()),maximum=float(value.max()),
        quantiles=torch.quantile(value,torch.tensor([.05,.5,.95],device=value.device)).cpu().tolist())


@contextmanager
def observe_reads(model,rows,phase):
    originals=[]
    try:
        for slot,block in enumerate(model.adapter.exchanges):
            had='read' in block.__dict__;old=block.__dict__.get('read');original=block.read
            originals.append((block,had,old))
            def capture(hidden,times,receivers,bank,graph,original=original,block=block,slot=slot):
                delta=original(hidden,times,receivers,bank,graph)
                with torch.no_grad(),torch.autocast(hidden.device.type,enabled=False):
                    ratio=delta.to(hidden.dtype).float().norm(dim=-1)/hidden.float().norm(dim=-1).clamp_min(1e-30)
                    z=linear(block.down,normalize(block.norm,hidden.float()))
                    gate=linear(block.gate,z).sigmoid().squeeze(-1)
                    rows.append(dict(layer=model.graph_layers[slot],phase=phase[0],all_positions=stats(ratio),
                        target_positions=stats(ratio[receivers==0]),auxiliary_positions=stats(ratio[receivers!=0]),gates=stats(gate)))
                return delta
            block.read=capture
        yield
    finally:
        for block,had,old in originals:
            if had:block.read=old
            else:delattr(block,'read')


def work_items(queries):
    result=[]
    for weights in ('native','physical','self','rewired'):
        for prefix in ('teacher','native_sample0'):
            for i,row in enumerate(queries):result.append(dict(index=i,weights=weights,support=None if weights=='native' else weights,prefix=prefix,label=weights))
    for support in ('self','rewired'):
        for i,row in enumerate(queries):result.append(dict(index=i,weights='physical',support=support,prefix='teacher',label='physical_support_'+support))
    return result


def prepare(cfg,output):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():raise ValueError('CPU preparation required')
    root=Path(cfg['source_output']);parent=read(root/'manifest.json')
    if sha(root/'manifest.json')!=cfg['source_manifest_sha256'] or sha(root/'report.json')!=cfg['source_report_sha256'] \
            or read(root/'status.json')['state']!='complete' or read(root/'activity.json')['worker'] is not None:
        raise ValueError('Exact completed failed source pilot required')
    for name,value in parent['code'].items():
        if sha(Path(__file__).parent/name)!=value:raise ValueError('Accepted model/numerical source changed: '+name)
    rows=[];all_queries=[q for name in parent['queries'] for q in parent['queries'][name]['development']]
    for name,split in parent['queries'].items():
        for query in split['development'][:4]:
            index=all_queries.index(query);path=root/'evaluations/native/trajectories'/f'{index:04d}-0.json';item=read(path)
            if item['query']!=query or item['identity']['binding']!=cfg['source_manifest_sha256'] or item['seed']!=sample_seed(query,0):
                raise ValueError('Wrong saved native prefix/query/seed')
            rows.append(dict(query=query,parent_index=index,native_path_sha256=sha(path),native_generated_ids=item['generated_ids']))
    locks={c:read(root/'conditions'/c/'locked.json') for c in ('physical','self','rewired')}
    for condition,value in locks.items():
        if sha(root/'conditions'/condition/'last.pt')!=value['checkpoint_sha256']:raise ValueError('Locked checkpoint changed')
    free={}
    for label in ('native','physical','self','rewired'):
        free[label]=[]
        for row in rows:
            path=root/'evaluations'/label/f'{row["parent_index"]:04d}.json';item=read(path)
            if item['query']!=row['query'] or item['identity']['binding']!=cfg['source_manifest_sha256']:
                raise ValueError('Saved free-forecast binding changed')
            free[label].append(dict(query=item['query'],prediction=item['prediction'],target=item['target'],observed=item['observed'],sha256=sha(path)))
    manifest=dict(configuration=cfg,parent_manifest_sha256=sha(root/'manifest.json'),parent_report_sha256=sha(root/'report.json'),
        queries=rows,source_preparation=parent['source_preparation'],partitions=parent['partitions'],locks=locks,
        inherited_model_code=parent['code'],work=work_items(rows),existing_free_subset=free,
        existing_full_source_macro_mae=read(root/'report.json')['macro_mae_by_view'],
        diagnostic_code_sha256=sha(Path(__file__)),parameter_updates=0,new_free_predictions=0,target_queries=0,reserved_source_queries=0)
    output.mkdir(parents=True,exist_ok=True);immutable(output/'manifest.json',manifest)
    atomic_json(output/'status.json',dict(state='prepared',phase='fixed_source_diagnostic',work_total=160))
    return manifest


def summarize_records(records):
    result={};metrics=('ce','cdf_score','median_error','argmax_error','top100_median_error','entropy','endpoint_probability','true_label_top100')
    for row in records:
        key=row['work']['label']+'/'+row['work']['prefix'];source=row['query']['source']
        result.setdefault(key,{}).setdefault(source,[]).append(row)
    summary={}
    for condition,sources in result.items():
        summary[condition]={}
        for source,rows in sources.items():
            def aggregate(low=1,high=24):
                values={}
                for metric in metrics:
                    means=[np.mean([v[metric] for v in row['scores']['steps'] if low<=v['step']<=high]) for row in rows
                        if any(low<=v['step']<=high for v in row['scores']['steps'])]
                    values[metric]=float(np.mean(means)) if means else None
                return values
            summary[condition][source]=dict(queries=len(rows),all_steps=aggregate(),first_step=aggregate(1,1),
                later_steps=aggregate(2,24),steps1_16=aggregate(1,16),steps17_24=aggregate(17,24))
    return summary


@torch.no_grad()
def run(cfg,output):
    if os.environ.get('GOFA_STAGE19_ADMITTED')!='1':raise ValueError('Cap-supervised admission required')
    m=read(output/'manifest.json');binding=sha(output/'manifest.json');proof=read(output/'cpu_preflight.json')
    if m['configuration']!=cfg or m['diagnostic_code_sha256']!=sha(Path(__file__)) or not proof['passed'] or proof['manifest_sha256']!=binding:
        raise ValueError('CPU proof/config/code binding changed')
    for name,value in m['inherited_model_code'].items():
        if sha(Path(__file__).parent/name)!=value:raise ValueError('Accepted inherited source changed')
    stop=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    def check():
        if stop[0] or not supervisor_alive(os.environ['GOFA_STAGE19_SUPERVISOR_PID'],os.environ['GOFA_STAGE19_SUPERVISOR_TICKS']):raise Paused('cooperative_resource_pause')
    started=time.monotonic();records=[];parent=Path(cfg['source_output']);current=None
    try:
        check();torch.cuda.set_per_process_memory_fraction(38*2**30/torch.cuda.get_device_properties(0).total_memory)
        from transformers import LlamaForCausalLM,LlamaTokenizer
        root=Path(m['source_preparation']['source_configuration']['weights']['directory'])
        if sha(root/'snapshot.json')!=m['source_preparation']['weights_sha256']:raise ValueError('Native snapshot changed')
        codec=ShortCodec(LlamaTokenizer.from_pretrained(root,local_files_only=True))
        native=LlamaForCausalLM.from_pretrained(root,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True)
        model=GraphMemoryGOFA(native,codec).to('cuda:0').eval().requires_grad_(False)
        versions={n:p._version for n,p in model.native.named_parameters()};sources=load_sources(m['source_preparation'])
        cache=GenerativePrefixLRU(binding,max_bytes=2**30)
        directory=output/'records';directory.mkdir(exist_ok=True)
        for i,work in enumerate(m['work']):
            check();path=directory/f'{i:03d}.json';selected=m['queries'][work['index']];query=selected['query']
            saved=read(path)
            if saved:
                if saved['binding']!=binding or saved['work']!=work or saved['query']!=query:raise ValueError('Diagnostic row binding changed')
                records.append(saved);continue
            if current!=work['weights']:
                current=work['weights']
                if current!='native':
                    state_path=parent/'conditions'/current/'last.pt'
                    if sha(state_path)!=m['locks'][current]['checkpoint_sha256']:raise ValueError('Checkpoint changed during diagnostic')
                    state=torch.load(state_path,map_location='cpu',weights_only=False);model.load_trainable(state['adapter'])
                    if tensor_digest(model.trainable_state())!=m['locks'][current]['adapter_sha256']:raise ValueError('Adapter tensor binding changed')
                    del state
            q=Query(**query);history,target=history_target(sources[q.source],query,True)
            generated=None if work['prefix']=='teacher' else selected['native_generated_ids']
            tokens,record=scoring_tokens(codec,history,target,generated);tokens=tokens.to('cuda:0')
            prefix=teacher_prefix(model,tokens,cache);observations=[];phase=['source_history'];contexts=None
            with observe_reads(model,observations,phase):
                if work['support'] is not None:
                    graph=memory_graph(sources[q.source],q,m['partitions'][q.source]['validation'],work['support'])
                    contexts=model.encode_origins([graph],codec,cache,check)
                phase[0]='target_decoder';logits=model(tokens,contexts=contexts,prefix=prefix).logits
            scores=score_distribution(logits,tokens,codec,record,target)
            value=dict(binding=binding,work=work,query=query,scores=scores,memory_observations=observations,
                inputs_sha256=tensor_digest({'ids':tokens.full_ids,'mask':tokens.attention_mask}),native_prefix_sha256=selected['native_path_sha256'])
            immutable(path,value);records.append(value)
            atomic_json(output/'status.json',dict(state='running',phase='fixed_source_diagnostic',done=i+1,total=160,current=work,
                active_seconds=time.monotonic()-started,eta_seconds=(time.monotonic()-started)/(i+1)*(160-i-1)))
            del contexts,logits,prefix,tokens
        if any(p._version!=versions[n] or p.grad is not None or p.requires_grad for n,p in model.native.named_parameters()):raise ValueError('Frozen native parameters changed')
        result=dict(binding=binding,complete=True,parameter_updates=0,new_free_predictions=0,target_queries=0,
            source_gate_remains_failed=True,conditional_metrics_only=True,summary=summarize_records(records),
            native_frozen=True,existing_full_source_macro_mae=m['existing_full_source_macro_mae'],
            existing_free_subset=m['existing_free_subset'],
            record_hashes={p.name:sha(p) for p in directory.glob('*.json')},
            model_code_seconds=time.monotonic()-started,torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            cache_hits=cache.hits,cache_misses=cache.misses)
        immutable(output/'report.json',result)
        atomic_json(output/'status.json',dict(state='complete',phase='fixed_source_diagnostic',done=160,total=160,eta_seconds=None,goal_achieved=False))
    except Paused as error:
        atomic_json(output/'status.json',dict(state='resource_paused',reason=str(error),done=len(records),total=160));raise SystemExit(75)
    except Exception as error:
        immutable(output/f'failure-{time.time_ns()}.json',dict(error=repr(error),done=len(records)))
        atomic_json(output/'status.json',dict(state='failed',error=repr(error)));raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--prepare',action='store_true');args=parser.parse_args();cfg=config(args.config)
    if args.prepare:prepare(cfg,args.output)
    else:run(cfg,args.output)


if __name__=='__main__':main()
