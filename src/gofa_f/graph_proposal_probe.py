"""One source-only frozen-Native-proposal correction; no optimizer or fitting."""
import argparse
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
from gofa_f.graph_memory_data import memory_graph
from gofa_f.graph_memory_diagnostic import scoring_tokens
from gofa_f.graph_memory_model import GraphMemoryGOFA
from gofa_f.graph_memory_pilot import gate_report,history_target,tensor_digest
from gofa_f.graph_memory_stage import load_sources
from gofa_f.prefix_cache import atomic_json
from gofa_f.short_codec import ShortCodec
from gofa_f.short_forecast import summarize
from gofa_f.short_sampling_diagnostic import SETTINGS,choose,sample_seed
from gofa_f.short_training import teacher_prefix
from gofa_f.source_graph_probe import immutable,read


def configuration(path):
    cfg=yaml.safe_load(Path(path).read_text().replace('${GOFA_WORKSPACE}',os.environ['GOFA_WORKSPACE']))
    if cfg['protocol']!='stage57-native-proposal-replay-v1' or cfg['training']!={'max_updates':0} \
            or cfg['data']!={'target_training_enabled':False,'target_evaluation_enabled':False} \
            or cfg['resources']['wall_deadline'] is not None or cfg['resources']['activity_limit_seconds'] is not None:
        raise ValueError('Fixed source-only one-step proposal replay required')
    return cfg


@torch.no_grad()
def coupled_draw(model,codec,history,horizon,proposal,contexts=None,cache=None,check=lambda:None):
    """History/proposal only; corrected draws never become conditioning input."""
    check();history=np.asarray(history,np.float64)
    if not np.isfinite(history).any():raise ValueError('Observed origin history required')
    dummy=np.full(horizon,history[np.flatnonzero(np.isfinite(history))[-1]])
    tokens,record=scoring_tokens(codec,history,dummy,proposal['generated_ids'])
    device=next(model.native.parameters()).device;tokens=tokens.to(device)
    prefix=teacher_prefix(model,tokens,cache)
    output=model(tokens,contexts=contexts,prefix=prefix).logits
    positions=torch.nonzero(tokens.labels[0]!=-100).flatten()
    if len(positions)!=horizon:raise ValueError('Unknown future mask must not choose forecast slots')
    scores=output[0,positions-1].float().index_select(-1,codec.alphabet(device))
    generator=torch.Generator(device=device).manual_seed(proposal['seed'])
    numbers=[];generated=list(codec.first_separator)
    for step in range(horizon):
        check();selected=choose(scores[step:step+1],generator)
        value=int(codec.alphabet(device)[selected].item());numbers.append(value);generated.append(value)
        if step+1<horizon:generated.extend(codec.separator)
    prediction,valid,boundary=codec.decode_ids(numbers,*record['scaling'],record['last'])
    if not valid.all():raise FloatingPointError('Nonfinite corrected prediction')
    return dict(numeric_ids=numbers,generated_ids=generated,prediction=prediction.tolist(),boundary=boundary.tolist(),
        seed=proposal['seed'],no_corrected_output_feedback=True,full_forwards=1)


def prepare(cfg,output):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():raise ValueError('CPU preparation required')
    parent=Path(cfg['source_output']);m=read(parent/'manifest.json')
    if sha(parent/'manifest.json')!=cfg['source_manifest_sha256'] or sha(parent/'report.json')!=cfg['source_report_sha256'] \
            or read(parent/'status.json')['state']!='complete' or read(parent/'activity.json')['worker'] is not None:
        raise ValueError('Exact terminated source pilot required')
    for name,value in m['code'].items():
        if sha(Path(__file__).parent/name)!=value:raise ValueError('Accepted numerical/model file changed')
    queries=[q for name in m['queries'] for q in m['queries'][name]['development']];proposals=[]
    for i,q in enumerate(queries):
        paths=[]
        for sample in range(8):
            path=parent/'evaluations/native/trajectories'/f'{i:04d}-{sample}.json';item=read(path)
            if item['query']!=q or item['sample']!=sample or item['seed']!=sample_seed(q,sample) \
                    or item['identity']['binding']!=cfg['source_manifest_sha256']:raise ValueError('Native proposal binding/seed changed')
            paths.append(dict(sha256=sha(path),generated_ids=item['generated_ids'],numeric_ids=item['numeric_ids'],
                prediction=item['prediction'],seed=item['seed'],sample=sample))
        proposals.append(paths)
    locks={c:read(parent/'conditions'/c/'locked.json') for c in ('physical','self','rewired')}
    for c,value in locks.items():
        if sha(parent/'conditions'/c/'last.pt')!=value['checkpoint_sha256']:raise ValueError('Locked source weights changed')
    result=dict(configuration=cfg,parent_manifest_sha256=sha(parent/'manifest.json'),source_preparation=m['source_preparation'],
        queries=queries,partitions=m['partitions'],proposals=proposals,locks=locks,accepted_model_code=m['code'],
        code={name:sha(Path(__file__).parent/name) for name in ('graph_proposal_probe.py','graph_memory_diagnostic.py')},
        sampling=SETTINGS,parameter_updates=0,new_native_autoregressive_generation=0,original_source_screen_passed=False,
        source_holdout_opened=0,target_queries=0,planned_full_forwards=2048,inherited_source_gpu_seconds=read(parent/'activity.json')['spent_seconds'])
    output.mkdir(parents=True,exist_ok=True);immutable(output/'manifest.json',result)
    atomic_json(output/'status.json',dict(state='prepared',phase='source_proposal_replay',completed_paths=0,total_paths=2048))
    return result


@torch.no_grad()
def run(cfg,output):
    if os.environ.get('GOFA_STAGE19_ADMITTED')!='1':raise ValueError('Cap-only GPU supervisor required')
    m=read(output/'manifest.json');binding=sha(output/'manifest.json');proof=read(output/'cpu_preflight.json')
    if m['configuration']!=cfg or not proof['passed'] or proof['manifest_sha256']!=binding:raise ValueError('Bound CPU acceptance required')
    for name,value in dict(m['accepted_model_code'],**m['code']).items():
        if sha(Path(__file__).parent/name)!=value:raise ValueError('Accepted source changed')
    stop=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    def check():
        if stop[0] or not supervisor_alive(os.environ['GOFA_STAGE19_SUPERVISOR_PID'],os.environ['GOFA_STAGE19_SUPERVISOR_TICKS']):raise Paused('cooperative_resource_pause')
    started=time.monotonic();done=0;parent=Path(cfg['source_output'])
    try:
        check();torch.cuda.set_per_process_memory_fraction(38*2**30/torch.cuda.get_device_properties(0).total_memory)
        from transformers import LlamaForCausalLM,LlamaTokenizer
        source=m['source_preparation'];root=Path(source['source_configuration']['weights']['directory'])
        if sha(root/'snapshot.json')!=source['weights_sha256']:raise ValueError('Native snapshot changed')
        codec=ShortCodec(LlamaTokenizer.from_pretrained(root,local_files_only=True))
        native=LlamaForCausalLM.from_pretrained(root,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True)
        model=GraphMemoryGOFA(native,codec).to('cuda:0').eval().requires_grad_(False)
        versions={n:p._version for n,p in model.native.named_parameters()};sources=load_sources(source)
        cache=GenerativePrefixLRU(binding,max_bytes=2*2**30);records={}
        for condition in ('native','physical','self','rewired'):
            weight_sha=source['weights_sha256']
            if condition!='native':
                path=parent/'conditions'/condition/'last.pt';weight_sha=sha(path)
                if weight_sha!=m['locks'][condition]['checkpoint_sha256']:raise ValueError('Frozen source checkpoint changed')
                state=torch.load(path,map_location='cpu',weights_only=False);model.load_trainable(state['adapter'])
                if tensor_digest(model.trainable_state())!=m['locks'][condition]['adapter_sha256']:raise ValueError('Adapter tensor mismatch')
                del state
            directory=output/'evaluations'/condition;paths=directory/'trajectories';paths.mkdir(parents=True,exist_ok=True)
            identity=dict(binding=binding,condition=condition,weight_sha256=weight_sha,mode='one_coupled_proposal_step',sampling=SETTINGS)
            entries=[]
            for i,query in enumerate(m['queries']):
                check();q=Query(**query);data=sources[q.source];history,_=history_target(data,query,False);contexts=None;samples=[]
                for sample,proposal in enumerate(m['proposals'][i]):
                    check();path=paths/f'{i:04d}-{sample}.json';item=read(path)
                    if item:
                        if item['identity']!=identity or item['query']!=query or item['proposal_sha256']!=proposal['sha256'] \
                                or item['sample']!=sample or item['seed']!=proposal['seed']:raise ValueError('Saved proposal replay binding changed')
                    else:
                        if condition!='native' and contexts is None:
                            graph=memory_graph(data,q,m['partitions'][q.source]['validation'],condition)
                            contexts=model.encode_origins([graph],codec,cache,check)
                        tick=time.monotonic();item=coupled_draw(model,codec,history,q.horizon,proposal,contexts,cache,check)
                        if condition=='native' and (item['numeric_ids']!=proposal['numeric_ids'] or item['generated_ids']!=proposal['generated_ids'] \
                                or item['prediction']!=proposal['prediction']):
                            immutable(output/f'native-identity-failure-{i:04d}-{sample}.json',dict(query=query,sample=sample,expected=proposal,actual=item))
                            raise ValueError('Native graph-off coupled identity failed; do not relax or continue')
                        item.update(identity=identity,query=query,proposal_sha256=proposal['sha256'],sample=sample,active_seconds=time.monotonic()-tick,
                            native_off_identity_passed=condition=='native')
                        immutable(path,item)
                    samples.append(item);done+=1
                    atomic_json(output/'status.json',dict(state='running',phase='source_proposal_replay',condition=condition,query=i,sample=sample,
                        completed_paths=done,total_paths=2048,eta_seconds=(time.monotonic()-started)/done*(2048-done)))
                # Unknown target values/missingness do not enter any forecast input.
                prediction=np.median(np.asarray([v['prediction'] for v in samples],np.float64),0)
                target=data.values[q.origin:q.origin+q.horizon,q.variable].astype(np.float64);mask=data.mask[q.origin:q.origin+q.horizon,q.variable]
                base=codec.record(history);scale,offset=base['scaling'];lo,hi=(-.9999+.5-offset)/scale,(.9999+.5-offset)/scale
                centers=np.median([[codec.id_centers[t] for t in v['numeric_ids']] for v in samples],0)
                row=dict(identity=identity,query=query,prediction=prediction.tolist(),target=target.tolist(),observed=mask.tolist(),
                    true_outside_range=(((target<lo)|(target>hi))&mask).tolist(),range_only_error_lower_bound=np.where(mask,np.abs(target-np.clip(target,lo,hi)),0).tolist(),
                    boundary=((centers==-.9999)|(centers==.9999)).tolist(),inverse_valid=[True]*q.horizon,format_complete=True,empty_history=False,
                    generated_token_count=sum(len(v['generated_ids']) for v in samples),forward_calls=8,numeric_slots=q.horizon,
                    mae=float(np.abs(prediction-target)[mask].mean()),prediction_process='nonrecursive_frozen_proposal_conditioning')
                immutable(directory/f'{i:04d}.json',row);entries.append(row)
                del contexts
            complete=dict(complete=True,identity=identity,queries=64,metrics=summarize(entries),query_shards_sha256={f'{i:04d}':sha(directory/f'{i:04d}.json') for i in range(64)})
            immutable(directory/'complete.json',complete);records[condition]=entries
        if any(p.requires_grad or p.grad is not None or p._version!=versions[n] for n,p in model.native.named_parameters()):raise ValueError('Frozen Native changed')
        report=dict(binding=binding,**gate_report(records),original_source_screen_passed=False,native_off_exact_paths=512,
            parameter_updates=0,new_native_autoregressive_generation=0,completed_full_forwards=2048,source_queries=64,
            metrics={c:summarize(rows) for c,rows in records.items()},model_code_seconds=time.monotonic()-started,
            cache_hits=cache.hits,cache_misses=cache.misses,torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            interpretation='New source inference policy on exposed development windows; not original graph-autoregressive decoding or target evidence')
        immutable(output/'report.json',report)
        atomic_json(output/'status.json',dict(state='complete',phase='source_proposal_replay',completed_paths=2048,total_paths=2048,
            source_replay_gate_passed=report['passed'],original_source_screen_passed=False,goal_achieved=False,eta_seconds=None))
    except Paused as error:
        atomic_json(output/'status.json',dict(state='resource_paused',reason=str(error),completed_paths=done,total_paths=2048));raise SystemExit(75)
    except Exception as error:
        immutable(output/f'failure-{time.time_ns()}.json',dict(error=repr(error),completed_paths=done))
        atomic_json(output/'status.json',dict(state='failed',error=repr(error),completed_paths=done));raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--prepare',action='store_true');args=parser.parse_args();cfg=configuration(args.config)
    if args.prepare:prepare(cfg,args.output)
    else:run(cfg,args.output)


if __name__=='__main__':main()
