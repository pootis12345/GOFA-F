"""Source-only preparation/real-ChatTime mechanism acceptance for graph memories."""
import argparse
from dataclasses import asdict
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
from transformers import LlamaForCausalLM,LlamaTokenizer

from gofa_f.generative_data import Query,Series,sha
from gofa_f.graph_memory_data import memory_graph,flatten_queries,fresh_queries
from gofa_f.graph_memory_model import GraphMemoryGOFA
from gofa_f.short_codec import ShortCodec,numeric_loss
from gofa_f.short_training import teacher_prefix,compact_forward,compact_loss
from gofa_f.source_graph_probe import read,immutable,digest,save_torch
from gofa_f.source_retraining_data import VIEWS
from gofa_f.prefix_cache import atomic_json

FILES=('graph_memory.py','graph_memory_model.py','graph_memory_data.py','graph_memory_stage.py',
       'allnode_model.py','allnode_kernel.py','generative_kernel.py','short_codec.py','short_training.py','short_sampling_diagnostic.py')


@torch.no_grad()
def verify_sampled_paths(model,codec,history,horizon,contexts,seed,check=lambda:None,rtol=.002,atol=.002):
    """Compare actual local-RNG top100 trajectories, not teacher argmax proxies."""
    from gofa_f.short_sampling_diagnostic import generate
    traces=[];paths=[]
    for cached in (False,True):
        trace=[]
        paths.append(generate(model,codec,history,horizon,contexts,seed=seed,cached=cached,check=check,trace=trace))
        traces.append(torch.cat(trace))
    torch.testing.assert_close(traces[0],traces[1],rtol=rtol,atol=atol)
    if paths[0]['generated_ids']!=paths[1]['generated_ids'] or paths[0]['numeric_ids']!=paths[1]['numeric_ids']:
        raise ValueError('Nonzero full/incremental sampled tokens differ')
    return dict(seed=seed,numeric_slots=horizon,generated_ids=paths[0]['generated_ids'],
        numeric_ids=paths[0]['numeric_ids'],sampled_tokens_equal=True,
        numerical_logits_max_absolute_difference=float((traces[0]-traces[1]).abs().max()),
        full_logits_sha256=hashlib.sha256(traces[0].numpy().tobytes()).hexdigest(),
        incremental_logits_sha256=hashlib.sha256(traces[1].numpy().tobytes()).hexdigest())


def configuration(path):
    cfg=yaml.safe_load(Path(path).read_text().replace('${GOFA_WORKSPACE}',os.environ['GOFA_WORKSPACE']))
    if cfg['protocol']!='stage56-chattime-graph-memory-v1' or cfg['target_training_enabled'] or cfg['target_evaluation_enabled'] \
            or cfg['resources']['wall_deadline'] is not None or cfg['resources']['activity_limit_seconds'] is not None:
        raise ValueError('Fixed source-only graph-memory mechanism protocol required')
    return cfg


def prepare(cfg,output):
    if torch.cuda.is_initialized() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU preparation required')
    parent=Path(cfg['source_output']);old=read(parent/'manifest.json')
    if sha(parent/'manifest.json')!=cfg['source_manifest_sha256']:raise ValueError('Audited source lineage changed')
    data_root=Path(old['configuration']['data']['directory'])/'sources'
    raw={};metadata={};queries={};partitions={};exclusions=[]
    for key in ('validation','confirmation','diagnostic'):exclusions.extend(flatten_queries(old['pools'][key]))
    source38=Path(cfg['traffic_partition_output']);partition=read(source38/'manifest.json')['partition']
    history_inventory={'stage23':sha(parent/'manifest.json'),'stage38':sha(source38/'manifest.json')}
    previous=read(source38/'manifest.json')['queries']
    exclusions.extend(flatten_queries(previous['validation']));exclusions.extend(flatten_queries(previous['confirmation']))
    source35=Path(os.environ['GOFA_WORKSPACE'])/'outputs/stage35-source-graph-audit-v1/manifest.json'
    if source35.exists():
        exclusions.extend(flatten_queries(read(source35)));history_inventory['stage35']=sha(source35)
    for name,(nodes,length,horizon) in VIEWS.items():
        data=Series(data_root/(name+'.npz'));raw[name]=sha(data.path);metadata[name]=sha(data.path.with_suffix('.json'))
        if raw[name]!=old['source_hashes'][name] or metadata[name]!=old['metadata_hashes'][name]:raise ValueError('Source observations/metadata changed')
        own=partition if name.startswith('pems04') else dict(train=list(range(nodes)),validation=list(range(nodes)),confirmation=list(range(nodes)))
        partitions[name]=own
        pool=[q for q in old['pools']['train'][name] if q['variable'] in own['train']]
        order=np.random.default_rng(np.random.SeedSequence([42,56,list(VIEWS).index(name)])).permutation(len(pool))
        train=[pool[int(i)] for i in order[:64]]
        if len(train)!=64:raise ValueError('Fixed source support pool insufficient')
        # Do not read source confirmation labels; merely reserve new query identities.
        dev=fresh_queries(data,name,length,horizon,'validation',16,own['validation'],exclusions,56)
        reserve=fresh_queries(data,name,length,horizon,'test',32,own['confirmation'],exclusions+dev,56)
        queries[name]=dict(train=train,development=dev,reserved=reserve)
    m=dict(configuration=cfg,source_parent_manifest_sha256=sha(parent/'manifest.json'),
        source_configuration=old['source_configuration'],weights_sha256=old['weights_sha256'],weight_stats=old['weight_stats'],
        source_directory=str(data_root),source_hashes=raw,metadata_hashes=metadata,partitions=partitions,queries=queries,
        source_exclusion_inventory=history_inventory,source_exclusions_sha256=digest(exclusions),
        fresh_label_windows_from_recorded_source_evaluations=True,entire_source_datasets_previously_exposed=True,
        graph_contract='publisher_bidirectional_traffic_coordinate4NN_other',
        code={name:sha(Path(__file__).parent/name) for name in FILES},
        formal_source_training_updates=0,target_queries_opened=0,old_checkpoints_loaded=False)
    output.mkdir(parents=True,exist_ok=True);immutable(output/'manifest.json',m)
    atomic_json(output/'status.json',dict(state='prepared',phase='source_reservations',stage=56,target_queries=0))
    return m


def load_sources(m):
    result={}
    for name in VIEWS:
        path=Path(m['source_directory'])/(name+'.npz')
        if sha(path)!=m['source_hashes'][name] or sha(path.with_suffix('.json'))!=m['metadata_hashes'][name]:raise ValueError('Source hash changed')
        data=Series(path)
        if name.startswith('pems04'):data.external=np.maximum(data.external,data.external.T)
        result[name]=data
    return result


def gpu_acceptance(cfg,output):
    if os.environ.get('GOFA_STAGE19_ADMITTED')!='1':raise ValueError('Cap-supervised GPU admission required')
    m=read(output/'manifest.json');binding=sha(output/'manifest.json')
    if m['configuration']!=cfg or m['code']!={name:sha(Path(__file__).parent/name) for name in FILES}:raise ValueError('Fixed code/config binding changed')
    proof=read(output/'cpu_preflight.json')
    if not proof or not proof['passed'] or proof['manifest_sha256']!=binding:raise ValueError('Bound CPU acceptance required')
    if (output/'mechanisms.json').exists():raise ValueError('Sealed mechanism result must not rerun')
    stopped=[False];model=optimizer=None;phase='load';started=time.monotonic()
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    def check():
        from gofa_f.allnode_stage import Paused,supervisor_alive
        if stopped[0] or not supervisor_alive(os.environ['GOFA_STAGE19_SUPERVISOR_PID'],os.environ['GOFA_STAGE19_SUPERVISOR_TICKS']):raise Paused('cooperative_resource_pause')
    def status(state,**fields):atomic_json(output/'status.json',dict(state=state,phase=phase,stage=56,**fields))
    try:
        check();torch.cuda.set_per_process_memory_fraction(38*2**30/torch.cuda.get_device_properties(0).total_memory)
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        root=Path(m['source_configuration']['weights']['directory'])
        if sha(root/'snapshot.json')!=m['weights_sha256']:raise ValueError('Frozen ChatTime weights snapshot changed')
        tokenizer=LlamaTokenizer.from_pretrained(root,local_files_only=True);codec=ShortCodec(tokenizer)
        native=LlamaForCausalLM.from_pretrained(root,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True)
        model=GraphMemoryGOFA(native,codec).to('cuda:0').train();sources=load_sources(m)
        versions={n:p._version for n,p in model.native.named_parameters()}
        zero_saved=read(output/'zero_cases.json',dict(binding=binding,results={}))
        if zero_saved['binding']!=binding:raise ValueError('Zero-case binding changed')
        results=zero_saved['results']
        cases=[(name,m['queries'][name]['train'][0]) for name in VIEWS]
        phase='zero_native_and_source_shapes';status('running')
        for name,query in cases:
            if name in results:continue
            check();q=Query(**query);data=sources[name];history=np.where(data.mask[q.origin-q.length:q.origin,q.variable],data.values[q.origin-q.length:q.origin,q.variable],np.nan)
            target=np.where(data.mask[q.origin:q.origin+q.horizon,q.variable],data.values[q.origin:q.origin+q.horizon,q.variable],np.nan)
            tokens=codec.batch([codec.record(history,target)]).to('cuda:0');graph=memory_graph(data,q,m['partitions'][name]['train'])
            with torch.no_grad():
                contexts=model.encode_origins([graph],codec,check=check)
                off=model(tokens).logits;zero=model(tokens,contexts=contexts).logits
                torch.testing.assert_close(zero,off,rtol=.002,atol=.002)
                alphabet=codec.alphabet(zero.device);positions=tokens.labels[:,1:]!=-100
                a=zero[:,:-1].float().index_select(-1,alphabet)[positions];b=off[:,:-1].float().index_select(-1,alphabet)[positions]
                if not torch.equal(a.argmax(-1),b.argmax(-1)):raise ValueError('Zero-init numeric tokens differ from native')
            results[name]=dict(nodes=contexts[0].nodes,length=q.length,horizon=q.horizon,
                native_max_absolute_difference=float((zero-off).abs().max()),zero_numeric_argmax_equal=True)
            atomic_json(output/'zero_cases.json',dict(binding=binding,results=results))
            del zero,off,contexts,a,b,tokens
        # Discarded mechanism optimizer: exactly3nonzero updates, never a trained candidate.
        phase='three_discarded_updates';status('running');optimizer=torch.optim.AdamW(model.adapter.parameters(),lr=1e-4,weight_decay=1e-4)
        name,query=cases[0];q=Query(**query);data=sources[name]
        history=np.where(data.mask[q.origin-q.length:q.origin,q.variable],data.values[q.origin-q.length:q.origin,q.variable],np.nan)
        target=np.where(data.mask[q.origin:q.origin+q.horizon,q.variable],data.values[q.origin:q.origin+q.horizon,q.variable],np.nan)
        tokens=codec.batch([codec.record(history,target)]).to('cuda:0');graph=memory_graph(data,q,m['partitions'][name]['train'])
        prefix=teacher_prefix(model,tokens);losses=[];completed=0
        if (output/'mechanism-state.pt').exists():
            saved=torch.load(output/'mechanism-state.pt',map_location='cpu',weights_only=False)
            if saved['binding']!=binding or not saved['discarded_candidate']:raise ValueError('Mechanism recovery binding changed')
            model.load_trainable(saved['adapter']);optimizer.load_state_dict(saved['optimizer'])
            model.parameter_version=saved['parameter_version'];model.cache_generation=saved['cache_generation']
            torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
            completed=saved['updates'];losses=saved['losses']
        for update in range(completed,3):
            check();optimizer.zero_grad(set_to_none=True);contexts=model.encode_origins([graph],codec,check=check)
            loss=compact_loss(compact_forward(model,codec,tokens,contexts,prefix))[0]
            loss.backward()
            if not loss.isfinite() or any(p.grad is not None and not p.grad.isfinite().all() for p in model.adapter.parameters()):raise ValueError('Nonfinite memory source gradient')
            torch.nn.utils.clip_grad_norm_(model.adapter.parameters(),1.);optimizer.step();model.mark_updated();losses.append(float(loss.detach()))
            save_torch(output/'mechanism-state.pt',dict(binding=binding,adapter=model.trainable_state(),optimizer=optimizer.state_dict(),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),updates=update+1,losses=losses,
                parameter_version=model.parameter_version,cache_generation=model.cache_generation,discarded_candidate=True))
        del prefix,tokens
        contexts=loss=None
        replay=read(output/'nonzero_cases.json',dict(binding=binding,results={}))
        if replay['binding']!=binding:raise ValueError('Nonzero-case binding changed')
        nonzero=replay['results']
        # Cover both supported real-source length/horizon shapes after the same
        # three discarded updates; neither case contributes model selection.
        for name,query in (cases[0],cases[2]):
            if name in nonzero:continue
            phase='nonzero_full_prefix_incremental_'+name;status('running');check();model.zero_grad(set_to_none=True)
            q=Query(**query);data=sources[name]
            history=np.where(data.mask[q.origin-q.length:q.origin,q.variable],data.values[q.origin-q.length:q.origin,q.variable],np.nan)
            target=np.where(data.mask[q.origin:q.origin+q.horizon,q.variable],data.values[q.origin:q.origin+q.horizon,q.variable],np.nan)
            tokens=codec.batch([codec.record(history,target)]).to('cuda:0');graph=memory_graph(data,q,m['partitions'][name]['train'])
            contexts=model.encode_origins([graph],codec,check=check)
            full=model(tokens,contexts=contexts).logits;numeric_loss(full,tokens.labels,codec)[0].backward();full=full.detach()
            gradients={n:p.grad.detach().cpu().clone() if p.grad is not None else None for n,p in model.adapter.named_parameters()}
            model.zero_grad(set_to_none=True);contexts=model.encode_origins([graph],codec,check=check)
            compact=compact_forward(model,codec,tokens,contexts,teacher_prefix(model,tokens));compact_loss(compact)[0].backward()
            for n,p in model.adapter.named_parameters():
                if gradients[n] is None:
                    if p.grad is not None:raise ValueError('Gradient branch mismatch')
                else:torch.testing.assert_close(p.grad.cpu(),gradients[n],rtol=.002,atol=.002)
            del compact,gradients,contexts
            model.zero_grad(set_to_none=True)
            with torch.no_grad():
                contexts=model.encode_origins([graph],codec,check=check)
                cut=int(tokens.prompt_ends[0])+1;part=tokens.first(cut);out=model(part,contexts=contexts,use_cache=True)
                pieces=[out.logits];kv=out.past_key_values
                for index in range(cut,tokens.input_ids.shape[1]):
                    check();part=part.append(tokens.input_ids[:,index:index+1]);out=model(part,contexts=contexts,cache=kv,use_cache=True)
                    pieces.append(out.logits);kv=out.past_key_values
                incremental=torch.cat(pieces,1);torch.testing.assert_close(incremental,full,rtol=.002,atol=.002)
                alphabet=codec.alphabet(full.device)
                if not torch.equal(incremental.index_select(-1,alphabet).argmax(-1),full.index_select(-1,alphabet).argmax(-1)):
                    raise ValueError('Nonzero full/incremental numeric argmax tokens differ')
                from gofa_f.short_sampling_diagnostic import sample_seed
                sampled=verify_sampled_paths(model,codec,history,q.horizon,contexts,sample_seed(query,0),check)
                # After adaptation, explicit graph-off still returns the same native
                # result checked before all optimizer steps.
                off=model(tokens).logits
                native_difference=float((off-full).abs().max())
            nonzero[name]=dict(query=query,nodes=contexts[0].nodes,full_compact_gradients_passed=True,
                full_incremental_max_absolute_difference=float((incremental-full).abs().max()),numeric_argmax_equal=True,
                nonzero_graph_effect_max_absolute_difference=native_difference,sampling=sampled)
            if native_difference==0:raise ValueError('Discarded updates did not exercise the graph-memory path')
            atomic_json(output/'nonzero_cases.json',replay)
            del full,incremental,pieces,kv,out,off,part,contexts,tokens
        if any(p.requires_grad or p.grad is not None or p._version!=versions[n] for n,p in model.native.named_parameters()):raise ValueError('Frozen backbone changed')
        value=dict(passed=True,manifest_sha256=binding,real_chattime=True,source_cases=results,discarded_optimizer_updates=3,
            losses=losses,nonzero_source_cases=nonzero,
            numeric_argmax_equal=True,sampled_tokens_equal=True,full_compact_gradients_passed=True,native_frozen=True,
            trainable_parameters=sum(p.numel() for p in model.adapter.parameters()),target_predictions=0,
            gpu_active_seconds=time.monotonic()-started,torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),next_action='fixed source pilot; do not claim forecast improvement')
        immutable(output/'mechanisms.json',value);status('complete',mechanisms_passed=True,formal_source_updates=0,target_predictions=0)
    except Exception as error:
        from gofa_f.allnode_stage import Paused
        if isinstance(error,Paused):status('resource_paused',reason=str(error));raise SystemExit(75)
        immutable(output/f'failure-{time.time_ns()}.json',dict(phase=phase,error=repr(error),active_seconds=time.monotonic()-started))
        status('failed',error=repr(error));raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--prepare',action='store_true');args=parser.parse_args();cfg=configuration(args.config)
    if args.prepare:prepare(cfg,args.output)
    else:gpu_acceptance(cfg,args.output)


if __name__=='__main__':main()
