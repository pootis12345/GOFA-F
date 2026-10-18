"""Rebaseline sample8, exact 1177 continuation, review after sampled update1500.

No target evaluation and no automatic continuation beyond the authorized review.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import time

import torch
import yaml

from gofa_f.allnode_model import frozen_cache_binding
from gofa_f.allnode_stage import (Paused,apply_update,cpu_copy,read,restore_checkpoint,
                                 restore_rng,rng_state,schedule,supervisor_alive)
from gofa_f.generative_data import SOURCES,Series,sha
from gofa_f.generative_model import GenerativePrefixLRU
from gofa_f.prefix_cache import atomic_json
from gofa_f.short_data import relative_score
from gofa_f.short_generative_stage import load_model,record_validation
from gofa_f.short_migration import assert_equal
from gofa_f.short_sampled_forecast import evaluate
from gofa_f.short_sampled_progress import Progress
from gofa_f.short_sampled_state import atomic_torch,prepare,save,temporary_adapter,verify
from gofa_f.short_training import training_batch


def commit_validation(output,ledger,score,update,candidate):
    record=dict(updates=update,checkpoint_sha256=sha(candidate),**score,
                selection_protocol='complete_source_sample8_median')
    old=read(output/f'validation-{update:06d}.json',record)
    if old!=record:
        raise ValueError('Already completed sampled score changed')
    atomic_json(output/f'validation-{update:06d}.json',record)
    if ledger['last_validated_update']!=update:
        best=record_validation(ledger,score['score'],update)
        ledger['validated_updates'].append(update)
        if best:
            ledger['best_update']=update
            shutil.copyfile(candidate,output/'sampled_best.pt')
            atomic_json(output/'sampled_best_validation.json',record)
    return record


def report(output):
    checkpoint=output/'last.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    values={str(u):read(output/f'validation-{u:06d}.json') for u in (1000,1500)}
    complete=saved['state']['updates']==1500 and saved['state']['pending']==0 and \
        saved['sampling_ledger']['last_validated_update']==1500 and values['1500'] is not None
    value=dict(complete=complete,authorized_segment='1177_to_1500_then_review',
        updates=saved['state']['updates'],pending=saved['state']['pending'],
        sampling_ledger=saved['sampling_ledger'],validations=values,
        sampled_best_sha256=sha(output/'sampled_best.pt') if (output/'sampled_best.pt').exists() else None,
        last_sha256=sha(checkpoint),activity=read(output/'activity.json'),
        target_evaluations=0,training_requires_new_authorization_after_review=True,
        no_final_target_checkpoint_sealed=True)
    atomic_json(output/'report.json',value)
    return value


def run(cfg,output,resume=False):
    if os.environ.get('GOFA_STAGE19_ADMITTED')!='1':
        raise RuntimeError('Sampled continuation requires fully-idle CPU supervisor admission')
    manifest=verify(cfg,output)
    binding=sha(output/'manifest.json')
    proof=read(output/'cpu_preflight.json',{})
    if not proof.get('passed') or proof['manifest_sha256']!=binding:
        raise ValueError('Code/config/migration-bound CPU proof required')
    if not resume:
        raise ValueError('The trained continuation always requires explicit --resume')
    stopped=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    def check():
        if stopped[0] or not supervisor_alive(os.environ['GOFA_STAGE19_SUPERVISOR_PID'],
                                              os.environ['GOFA_STAGE19_SUPERVISOR_TICKS']):
            raise Paused('cooperative_resource_or_user_stop')
    saved=torch.load(output/'last.pt',map_location='cpu',weights_only=False)
    state,ledger=saved['state'],saved['sampling_ledger']
    def status(kind,**fields):
        atomic_json(output/'status.json',dict(state=kind,updates=state['updates'],
            pending_queries=state['pending'],updated_at=time.time(),**fields))
    model=optimizer=progress=None
    try:
        status('running',phase='restore_exact_state')
        torch.cuda.set_per_process_memory_fraction(38*2**30/torch.cuda.get_device_properties(0).total_memory)
        base=manifest['source_configuration']
        model,codec=load_model(base,torch.device('cuda:0'))
        if codec.alphabet_sha!=proof['alphabet']['alphabet_sha256']:
            raise ValueError('CPU/GPU finite alphabet differs')
        optimizer=torch.optim.AdamW(model.adapter.parameters(),lr=2e-4,weight_decay=1e-4)
        state,restored_progress=restore_checkpoint(output/'last.pt',model,optimizer,binding)
        for key,actual in [('adapter',model.trainable_state()),('optimizer',cpu_copy(optimizer.state_dict())),
            ('gradients',{n:cpu_copy(p.grad) for n,p in model.adapter.named_parameters()}),('rng',rng_state())]:
            assert_equal(saved[key],actual)
        assert model.parameter_version==saved['parameter_version']
        atomic_json(output/'restore_audit.json',dict(passed=True,updates=state['updates'],pending=state['pending'],
            pending_tokens=state['pending_tokens'],state_exact=True,cuda_bitwise_future_replay_claimed=False))
        del saved
        progress=Progress(output,restored_progress)
        pools=read(output/'pools/queries.json')
        series={s:Series(Path(base['data']['directory'])/'sources'/(s+'.npz')) for s in SOURCES}
        eps=read(output/'pools/metric_epsilons.json')
        cache=GenerativePrefixLRU(frozen_cache_binding(model,manifest['weights_sha256']),max_bytes=32*2**30)
        cache_base=dict(state.get('cache_totals',dict(hits=0,misses=0,peak_bytes=0)))
        def memory(phase):
            item=state['peak_memory'].setdefault(phase,dict(allocated_bytes=0,reserved_bytes=0))
            item['allocated_bytes']=max(item['allocated_bytes'],torch.cuda.max_memory_allocated())
            item['reserved_bytes']=max(item['reserved_bytes'],torch.cuda.max_memory_reserved())
            state['cache_totals']=dict(hits=cache_base['hits']+cache.hits,misses=cache_base['misses']+cache.misses,
                peak_bytes=max(cache_base['peak_bytes'],cache.peak))
            atomic_json(output/'cuda_memory.json',state['peak_memory'])
            atomic_json(output/'cache.json',dict(state['cache_totals'],current_bytes=cache.used,
                prefix_layers=30,ram_limit_gib=32,disk_limit_gib=128,disk_cache_enabled=False,
                graph_kv_cross_update=False))
        def checkpoint():
            save(output/'last.pt',model,optimizer,state,progress,ledger,binding)
        def evaluation(label,native=False,checkpoint_path=None):
            status('running',phase=label)
            progress.tick(label,force=True)
            digest=manifest['weights_sha256'] if native else sha(checkpoint_path)
            result=evaluate(model,codec,series,pools['validation'],output,label,
                binding+':sample8:'+label+':'+digest+':'+codec.alphabet_sha,progress,check,native)
            progress.state.setdefault('sampled_evaluation_durations',{})[label]=result['active_seconds']
            memory(label)
            return result
        # Neither reference evaluation may discard the 1177 partial gradients/RNG.
        with temporary_adapter(model):
            baseline=evaluation('sampled-native',True)
        reference=torch.load(output/'reference-001000.pt',map_location='cpu',weights_only=False)
        with temporary_adapter(model,reference):
            anchor=evaluation('sampled-update-001000',checkpoint_path=output/'reference-001000.pt')
        anchor_score=relative_score(anchor['errors'],baseline['errors'],pools['validation'],eps,1)
        if ledger['last_validated_update']<=1000:
            commit_validation(output,ledger,anchor_score,1000,output/'reference-001000.pt')
        checkpoint()
        last_save=time.monotonic()
        while state['updates']<1500:
            check()
            status('running',phase='train')
            progress.tick('train',updates=state['updates'],pending_queries=state['pending'])
            previous_update=state['updates']
            if state['pending']==64:
                apply_update(model,optimizer,state,progress)
            else:
                size=min(state['microbatch'],64-state['pending'])
                queries=[schedule(pools['train'],state['updates']*64+state['pending']+i,42) for i in range(size)]
                progress.tick(current_microbatch=size,current_history=[q.length for q in queries],
                              current_horizon=[q.horizon for q in queries])
                gradients={n:cpu_copy(p.grad) for n,p in model.adapter.named_parameters()}
                before_rng=rng_state()
                try:
                    loss,count=training_batch(model,codec,series,queries,cache,check)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite unchanged teacher-forcing loss')
                    loss.backward()
                except Exception as exc:
                    for n,p in model.adapter.named_parameters():
                        p.grad=gradients[n].to(p.device) if gradients[n] is not None else None
                    restore_rng(before_rng)
                    if isinstance(exc,torch.cuda.OutOfMemoryError) and size>1:
                        state['microbatch']=1
                        state.setdefault('microbatch_fallbacks',[]).append(dict(updates=state['updates'],
                            pending=state['pending'],reason='cuda_oom_rolled_back',from_size=size,to_size=1))
                        torch.cuda.empty_cache()
                        continue
                    raise
                state['pending']+=size
                state['pending_tokens']+=count
                state['pending_loss']+=float(loss.detach())
                if state['pending']==64:
                    apply_update(model,optimizer,state,progress)
            if state['updates']!=previous_update:
                with (output/'training_history.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(updates=state['updates'],loss=progress.state.get('loss'),
                        learning_rate=progress.state.get('learning_rate'),at=time.time(),
                        parameter_version=model.parameter_version))+'\n')
            if time.monotonic()-last_save>=30 or state['updates']==1500:
                memory('train')
                checkpoint()
                last_save=time.monotonic()
        if state['updates']!=1500 or state['pending']:
            raise ValueError('Authorized review cannot exceed update1500 or retain an extra batch')
        candidate=output/'candidate-001500.pt'
        if not candidate.exists():
            atomic_torch(candidate,model.trainable_state())
        assert_equal(torch.load(candidate,map_location='cpu',weights_only=False),model.trainable_state())
        current=evaluation('sampled-update-001500',checkpoint_path=candidate)
        score=relative_score(current['errors'],baseline['errors'],pools['validation'],eps,1)
        commit_validation(output,ledger,score,1500,candidate)
        state['stop_reason']='review_at_1500'
        progress.tick('review_paused',force=True)
        checkpoint()
        report(output)
        status('complete',phase='review_paused',stop_reason='review_at_1500',target_evaluations=0,
               requires_new_authorization=True)
    except Paused as exc:
        if model is not None and optimizer is not None and progress is not None:
            save(output/'last.pt',model,optimizer,state,progress,ledger,binding)
        status('resource_paused',reason=str(exc))
    except torch.cuda.OutOfMemoryError as exc:
        key=f"{state['updates']}:{state['pending']}:{read(output/'status.json',{}).get('phase')}"
        attempts=state.setdefault('sampled_oom_attempts',{})
        attempts[key]=attempts.get(key,0)+1
        if model is not None and optimizer is not None and progress is not None:
            save(output/'last.pt',model,optimizer,state,progress,ledger,binding)
        if attempts[key]<3:
            status('resource_paused',reason='cuda_oom_rolled_back',attempts=attempts[key])
        else:
            atomic_json(output/'failures.json',dict(type=type(exc).__name__,error=str(exc),
                reason='three_identical_ooms_after_idle_readmission',updates=state['updates']))
            status('failed',error='Repeated identical OOM; state retained, no blind restart')
            raise
    except Exception as exc:
        if model is not None and optimizer is not None and progress is not None:
            save(output/'last.pt',model,optimizer,state,progress,ledger,binding)
        atomic_json(output/'failures.json',dict(type=type(exc).__name__,error=str(exc),updates=state['updates']))
        status('failed',error=str(exc))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare',action='store_true')
    mode.add_argument('--report-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    cfg,output=yaml.safe_load(Path(args.config).read_text()),Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    if args.prepare:
        print(json.dumps(prepare(cfg,output)))
    elif args.report_only:
        print(json.dumps(report(output)))
    else:
        run(cfg,output,args.resume)


if __name__=='__main__':
    main()
