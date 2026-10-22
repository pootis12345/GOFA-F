"""Dual-Memory source preparation and one real-weight execution acceptance.

This entry currently exposes preparation/acceptance/report only. Formal source
cross-fitting and target evaluation remain explicitly incomplete, never implied
by a green mechanism report. Existing experiment results are not overwritten.
"""
import argparse
import gc
import hashlib
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
import yaml

from gofa_f.dual_memory_data import GROUPS, SHAPES, memory_input
from gofa_f.dual_memory_model import DualMemoryGOFA
from gofa_f.generative_data import Query, sha
from gofa_f.graph_memory_stage import load_sources
from gofa_f.prefix_cache import atomic_json
from gofa_f.source_graph_probe import read, immutable, save_torch


def configuration(path):
    cfg = yaml.safe_load(Path(path).read_text().replace('${GOFA_WORKSPACE}',os.environ['GOFA_WORKSPACE']))
    if cfg['protocol'] != 'stage59-dual-memory-v1' or cfg['data']['target_training_enabled'] or \
            cfg['data']['target_evaluation_enabled'] or cfg['architecture'] != dict(depths=[6,3],memory_slots=4,graph_width=128,continuous_shared_head=True):
        raise ValueError('Fixed source-only dual-Memory protocol required')
    r = cfg['resources']
    if r != dict(admission_policy='memory_cap_only',admission_samples=1,resource_check_seconds=10,
            memory_cap_gib=40,torch_cap_gib=38,activity_limit_seconds=None,wall_deadline=None,
            yield_on_foreign=False,leave_free_gib=0):
        raise ValueError('Shared resource contract changed')
    return cfg


def make_tokens(codec,item,device):
    histories = np.where(item.observed.cpu().numpy(),item.history.cpu().numpy(),np.nan)
    return codec.batch([codec.record(x) for x in histories]).to(device)


def metadata_tokens(tokenizer,text,device):
    from gofa_f.allnode_codec import AllNodeTokens
    ids = tokenizer(text,return_tensors='pt',add_special_tokens=True)['input_ids'].to(device)
    mask = torch.ones_like(ids,dtype=torch.bool)
    return AllNodeTokens(ids,mask,torch.full_like(ids,-100),torch.tensor([ids.shape[1]-1],device=device),
        (torch.empty(0,dtype=torch.long,device=device),),(hashlib.sha256(text.encode()).hexdigest(),),ids.clone())


def acceptance(cfg,output):
    from transformers import LlamaForCausalLM,LlamaTokenizer
    from gofa_f.allnode_stage import Paused,supervisor_alive
    from gofa_f.dual_memory_data import all_node_labels
    from gofa_f.short_codec import ShortCodec
    from gofa_f.short_sampling_diagnostic import generate,sample_seed
    if os.environ.get('GOFA_STAGE19_ADMITTED') != '1':
        raise ValueError('CPU cap-supervisor admission required')
    m = read(output/'manifest.json'); binding = sha(output/'manifest.json')
    if m['configuration'] != cfg or m['code'] != {f:sha(Path(__file__).parent/f) for f in m['code']}:
        raise ValueError('Prepared code/config changed')
    proof = read(output/'cpu_preflight.json')
    if not proof or not proof['passed'] or proof['manifest_sha256'] != binding:
        raise ValueError('Bound CPU execution proof required')
    if (output/'mechanisms.json').exists():
        raise ValueError('Completed acceptance cannot be rerun')
    stopped = [False]
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    def check():
        if stopped[0] or not supervisor_alive(os.environ['GOFA_STAGE19_SUPERVISOR_PID'],os.environ['GOFA_STAGE19_SUPERVISOR_TICKS']):
            raise Paused('cooperative_resource_pause')
    def status(phase,**fields):
        atomic_json(output/'status.json',dict(state='running',phase=phase,stage=59,target_predictions=0,
            formal_updates=0,goal_achieved=False,**fields))
    started = time.monotonic(); model = None
    try:
        check(); torch.cuda.set_per_process_memory_fraction(38*2**30/torch.cuda.get_device_properties(0).total_memory)
        root = Path(m['source_configuration']['weights']['directory'])
        if sha(root/'snapshot.json') != m['weights_sha256']:
            raise ValueError('Pinned ChatTime snapshot changed')
        torch.manual_seed(42); np.random.seed(42); random.seed(42)
        tokenizer = LlamaTokenizer.from_pretrained(root,local_files_only=True); codec = ShortCodec(tokenizer)
        status('loading_frozen_chattime')
        native = LlamaForCausalLM.from_pretrained(root,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda:0')
        versions = {n:p._version for n,p in native.named_parameters()}
        sources = load_sources(m)
        partial = read(output/'acceptance_partial.json',dict(binding=binding,depths={},native_replays={}))
        if partial['binding'] != binding: raise ValueError('Acceptance recovery binding changed')
        old_root = Path(os.environ['GOFA_WORKSPACE'])/'outputs/stage57-chattime-graph-memory-source-pilot-v1'
        old = read(old_root/'manifest.json')
        flattened = [q for name in GROUPS for q in old['queries'][name]['development']]
        for depth in cfg['architecture']['depths']:
            if str(depth) in partial['depths']: continue
            check(); torch.manual_seed(42)
            model = DualMemoryGOFA(native,depth=depth,weight_binding=m['weights_sha256']).to('cuda:0')
            optimizer = torch.optim.AdamW(model.adapter.parameters(),lr=1e-4)
            # Three discarded interface updates on source labels only.
            q = Query(**m['queries']['pems04_flow']['12-12']['train'][0])
            item = memory_input(sources[q.source],q,m['partitions'][q.source]['train']).to('cuda:0')
            ts = make_tokens(codec,item,'cuda:0'); prefix = model.frozen_history(ts)
            meta = model.metadata_vector(metadata_tokens(tokenizer,m['texts'][q.source]['12-12'],'cuda:0'))
            target,mask = all_node_labels(sources[q.source],q,item); target=target.to('cuda:0');mask=mask.to('cuda:0')&item.present[:,None]
            state_path=output/f'qa-depth{depth}.pt'; completed=0; losses=[]
            if state_path.exists():
                saved=torch.load(state_path,map_location='cpu',weights_only=False)
                if saved['binding']!=binding or not saved['discarded']:raise ValueError('QA state binding changed')
                model.load_trainable(saved['adapter']);optimizer.load_state_dict(saved['optimizer']);completed=saved['updates'];losses=saved['losses']
                torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
            status('discarded_memory_updates',depth=depth,discarded_updates=completed)
            for update in range(completed,3):
                check();optimizer.zero_grad(set_to_none=True)
                out,_=model.enhanced(ts,item,meta,prefix=prefix,optimized=True,checkpoint_tail=True)
                loss=((out-target).abs().masked_fill(~mask,0).sum(1)/mask.sum(1).clamp_min(1))[mask.any(1)].mean()/m['training_unit_normalizers'][q.source]
                loss.backward()
                if not loss.isfinite() or any(p.grad is not None and not p.grad.isfinite().all() for p in model.adapter.parameters()):raise FloatingPointError('Nonfinite QA gradients')
                torch.nn.utils.clip_grad_norm_(model.adapter.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
                save_torch(state_path,dict(binding=binding,discarded=True,updates=update+1,losses=losses,
                    adapter=model.trainable_state(),optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
            # Explicitly nonzero graph read/write QA, discarded after proof.
            with torch.no_grad():
                for e in model.adapter.exchanges:e.gate.fill_(.15);e.ff_gate.fill_(.1)
                model.adapter.head[-1].weight.normal_(std=.002)
            cases=[]
            for name in GROUPS:
                for length,horizon in SHAPES:
                    check();shape=f'{length}-{horizon}';q=Query(**m['queries'][name][shape]['train'][0])
                    qa_maximum = name=='pems04_flow' and length==48
                    if qa_maximum:
                        # Only execution QA:31distinct real source histories,
                        # constructed three-level topology that reaches the cap.
                        # Never enters a source fit, score, pool or selection.
                        from types import SimpleNamespace
                        data=sources[name];a=np.zeros((len(data.ids),len(data.ids)),np.float32)
                        for node in range(1,11):a[0,node]=1;a[node,node+10]=1;a[node+10,node+20]=1
                        synthetic=SimpleNamespace(values=data.values,mask=data.mask,external=a,ids=data.ids,segments=data.segments)
                        qq=Query(name,0,q.origin,length,horizon)
                        item=memory_input(synthetic,qq).to('cuda:0')
                        if len(item.node_ids)!=31:raise AssertionError('Maximum31real-node QA did not execute')
                    else:
                        item=memory_input(sources[name],q,m['partitions'][name]['train']).to('cuda:0')
                    ts=make_tokens(codec,item,'cuda:0');prefix=model.frozen_history(ts)
                    meta=model.metadata_vector(metadata_tokens(tokenizer,m['texts'][name][shape],'cuda:0'))
                    status('nonzero_full_cached_gradients',depth=depth,source=name,shape=shape,nodes=len(item.node_ids))
                    model.zero_grad(set_to_none=True)
                    full,mem=model.enhanced(ts,item,meta,prefix=prefix,optimized=False)
                    (full/item.extent[:,None]).square().mean().backward()
                    gradients={n:None if p.grad is None else p.grad.detach().clone() for n,p in model.adapter.named_parameters()}
                    model.zero_grad(set_to_none=True)
                    cached,cmem=model.enhanced(ts,item,meta,prefix=prefix,optimized=True,checkpoint_tail=True)
                    torch.testing.assert_close(cached,full,rtol=.002,atol=.002)
                    torch.testing.assert_close(cmem,mem,rtol=.002,atol=.002)
                    (cached/item.extent[:,None]).square().mean().backward()
                    difference=0.
                    for n,p in model.adapter.named_parameters():
                        if gradients[n] is None or p.grad is None:raise ValueError('An intended branch did not participate in autograd: '+n)
                        torch.testing.assert_close(p.grad,gradients[n],rtol=.002,atol=.002)
                        difference=max(difference,float((p.grad-gradients[n]).abs().max()))
                    cases.append(dict(source=name,shape=shape,nodes=len(item.node_ids),input_sha256=item.identity,
                        constructed_topology_for_maximum_shape_only=qa_maximum,
                        full_cached_raw_max_difference=float((cached-full).abs().max()),
                        full_cached_memory_max_difference=float((cmem-mem).abs().max()),gradient_max_difference=difference))
                    del full,mem,cached,cmem,gradients,prefix,meta
            # Reuse exact historical Native token references, one path per source.
            if not partial['native_replays']:
                for i in (0,16,32,48):
                    check();row=flattened[i];q=Query(**row);data=sources[q.source]
                    history=np.where(data.mask[q.origin-q.length:q.origin,q.variable],data.values[q.origin-q.length:q.origin,q.variable],np.nan)
                    refpath=old_root/'evaluations/native/trajectories'/f'{i:04d}-0.json';reference=read(refpath)
                    seed=sample_seed(row,0);rng=torch.get_rng_state().clone();crng=torch.cuda.get_rng_state_all()
                    result=generate(model.native_view(),codec,history,q.horizon,seed=seed,check=check)
                    if result['generated_ids']!=reference['generated_ids'] or seed!=reference['seed']:raise ValueError('Protected Native failed exact historical sampled-token replay')
                    if not torch.equal(rng,torch.get_rng_state()) or any(not torch.equal(a,b) for a,b in zip(crng,torch.cuda.get_rng_state_all())):raise ValueError('Native local sample RNG leaked')
                    partial['native_replays'][q.source]=dict(reference_sha256=sha(refpath),query=row,seed=seed,sampled_tokens_equal=True)
                    atomic_json(output/'acceptance_partial.json',partial)
            partial['depths'][str(depth)]=dict(discarded_updates=3,losses=losses,cases=cases,nonzero_graph_gates=.15)
            atomic_json(output/'acceptance_partial.json',partial)
            del optimizer,model;model=None;gc.collect();torch.cuda.empty_cache()
        if versions!={n:p._version for n,p in native.named_parameters()} or any(p.grad is not None for p in native.parameters()):raise ValueError('Frozen Native weights/gradients changed')
        # One connected64query throughput measurement at the primary depth.
        model=DualMemoryGOFA(native,depth=6,weight_binding=m['weights_sha256']).to('cuda:0').eval()
        profile=read(output/'throughput_partial.json',dict(binding=binding,records=[]))
        if profile['binding']!=binding:raise ValueError('Profile binding changed')
        rows=[(name,f'{l}-{h}',r) for name in GROUPS for l,h in SHAPES for r in m['queries'][name][f'{l}-{h}']['train'][:8]]
        for index,(name,shape,row) in enumerate(rows):
            if index<len(profile['records']):continue
            check();q=Query(**row);item=memory_input(sources[name],q,m['partitions'][name]['train']).to('cuda:0')
            ts=make_tokens(codec,item,'cuda:0');meta=model.metadata_vector(metadata_tokens(tokenizer,m['texts'][name][shape],'cuda:0'))
            torch.cuda.synchronize();start=time.monotonic()
            with torch.no_grad():
                prefix=model.frozen_history(ts);out,_=model.enhanced(ts,item,meta,prefix=prefix,optimized=True)
            torch.cuda.synchronize();elapsed=time.monotonic()-start
            if not torch.isfinite(out).all():raise FloatingPointError('Nonfinite direct forecast')
            profile['records'].append(dict(query=row,real_nodes=len(item.node_ids),seconds=elapsed,
                prefix_bytes=prefix.hidden.numel()*prefix.hidden.element_size(),input_sha256=item.identity))
            atomic_json(output/'throughput_partial.json',profile)
            status('throughput64',completed=index+1,total=64)
        result=dict(binding=binding,passed=True,depths=partial['depths'],native_replays=partial['native_replays'],
            cpu_tolerance=1e-6,cuda_tolerance=.002,frozen_native_unchanged=True,
            throughput_queries=64,throughput_seconds=sum(x['seconds'] for x in profile['records']),
            peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(),
            formal_training_updates=0,target_predictions=0,goal_achieved=False,active_worker_seconds=time.monotonic()-started)
        immutable(output/'mechanisms.json',result)
        atomic_json(output/'status.json',dict(state='complete',phase='mechanisms_complete_formal_training_pending',stage=59,
            formal_training_updates=0,target_predictions=0,goal_achieved=False,source_training_pending=True))
    except Paused:
        atomic_json(output/'status.json',dict(state='resource_paused',phase='dual_memory_acceptance',goal_achieved=False))
        raise SystemExit(75)
    except BaseException as exc:
        atomic_json(output/'failure.json',dict(type=type(exc).__name__,message=str(exc),binding=binding,at=time.time()))
        atomic_json(output/'status.json',dict(state='failed',phase='dual_memory_acceptance',goal_achieved=False))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--report-only',action='store_true');args=p.parse_args()
    cfg=configuration(args.config);output=Path(args.output)
    if args.prepare:
        from gofa_f.dual_memory_prepare import prepare
        prepare(cfg,output)
    elif args.report_only:
        print(read(output/'status.json'))
    else:
        acceptance(cfg,output)


if __name__=='__main__':main()
