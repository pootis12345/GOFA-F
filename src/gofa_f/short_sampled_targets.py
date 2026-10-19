"""Read-only nine-target evaluation of one source-locked adapter, sample8 median."""
import argparse
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
import yaml

from gofa_f.allnode_graph import build_graph
from gofa_f.allnode_stage import Paused,read,supervisor_alive
from gofa_f.generative_data import Query,Series,sha
from gofa_f.prefix_cache import atomic_json
from gofa_f.short_data import TARGETS
from gofa_f.short_generative_stage import load_model
from gofa_f.short_migration import assert_equal
from gofa_f.short_sampled_forecast import evaluate
from gofa_f.short_sampling_diagnostic import SETTINGS,generate,sample_seed
from gofa_f.short_source_diagnostic import compare_scores
from gofa_f.short_target_progress import Progress
from gofa_f.short_target_state import prepare,verify


@torch.no_grad()
def mechanisms(model,codec,manifest,check,rtol=2e-3,atol=2e-3,audit=lambda _:None):
    root=Path(manifest['source_configuration']['data']['directory'])
    records=[]
    for probe in manifest['probes']:
        q=Query(**probe['query'])
        if sha(root/'sources'/(q.source+'.npz'))!=manifest['source_raw_hashes'][q.source]:
            raise ValueError('Source mechanism observations changed')
        data=Series(root/'sources'/(q.source+'.npz'))
        history=np.where(data.mask[q.origin-q.length:q.origin,q.variable],
            data.values[q.origin-q.length:q.origin,q.variable],np.nan)
        for mode in ('native','gofa'):
            ref=probe['references'][mode]
            if sha(Path(ref['path']))!=ref['sha256']:
                raise ValueError('Locked source replay trajectory changed')
            expected=read(ref['path'])
            seed=sample_seed(probe['query'],0)
            if expected.get('query',probe['query'])!=probe['query'] or expected.get('seed',seed)!=seed:
                raise ValueError('Source replay query/seed identity differs')
            context=model.encode_origins([build_graph(data,q)],codec,check=check) if mode=='gofa' else None
            traces,other=[],[]
            kwargs=dict(contexts=context,seed=seed,check=check)
            untraced=generate(model,codec,history,q.horizon,**kwargs)
            cached=generate(model,codec,history,q.horizon,trace=traces,**kwargs)
            full=generate(model,codec,history,q.horizon,cached=False,trace=other,**kwargs)
            equal=cached['generated_ids']==full['generated_ids']==expected['generated_ids']
            # compare_scores uses the pre-registered BF16 tolerance.
            comparison=compare_scores(traces,other,rtol)
            record=dict(query=probe['query'],mode=mode,seed=seed,comparison=comparison,raw_tokens_equal=equal,
                tracing_unchanged=untraced['generated_ids']==cached['generated_ids'],
                expected=expected,cached=cached,full=full,untraced=untraced)
            records.append(record)
            audit(dict(records=records,passed=equal and record['tracing_unchanged'] and not comparison['violations']))
            if comparison['violations'] or not equal or not record['tracing_unchanged']:
                raise ValueError('3500 source replay/full-incremental sampling failed: '+str(dict(
                    query=probe['query'],mode=mode,comparison=comparison,
                    cached_full_equal=cached['generated_ids']==full['generated_ids'],
                    cached_reference_equal=cached['generated_ids']==expected['generated_ids'],
                    tracing_unchanged=record['tracing_unchanged'])))
    return dict(passed=True,rtol=rtol,atol=atol,records=records,
        target_predictions_before_acceptance=0,gradients_created=any(p.grad is not None for p in model.parameters()))


def report(output):
    manifest=read(output/'manifest.json')
    binding=sha(output/'manifest.json')
    adapter_digest=sha(output/'adapter.pt')
    result={}
    for name in TARGETS:
        pair={mode:read(output/f'target-{name}-{mode}-result.json') for mode in ('native','gofa')}
        for mode,value in pair.items():
            if value is None:
                continue
            digest=manifest['weights_sha256'] if mode=='native' else adapter_digest
            expected=f'{binding}:sample8:target-{name}-{mode}:{digest}:'+manifest['alphabet']['alphabet_sha256']
            if value['binding']!=expected or value['sampling']!=SETTINGS \
                    or len(value['errors'])!=len(manifest['queries']['targets'][name]) \
                    or value['numeric_slots_in_eight_paths']!=8*sum(q['horizon'] for q in manifest['queries']['targets'][name]):
                raise ValueError('Target result belongs to a different bound model/query/sampling protocol')
        complete=all(v is not None and v.get('complete') and v['queries']==len(manifest['queries']['targets'][name]) for v in pair.values())
        result[name]=dict(complete=complete,results=pair if complete else None,
            expected_queries=len(manifest['queries']['targets'][name]))
    value=dict(complete=all(v['complete'] for v in result.values()),targets=result,
        selected_update=3500,adapter_sha256=adapter_digest,
        manifest_sha256=sha(output/'manifest.json'),optimizer_updates=0,
        completed_datasets=sum(v['complete'] for v in result.values()),activity=read(output/'activity.json'),
        memory=read(output/'cuda_memory.json'),coverage=manifest['queries']['coverage'],
        limitations=['Traffic/Electric/PEMS08 only one origin per variable',
            'Weather 48->24 is custom, not original paper 288->144',
            'BF16 constrained single-block sample8 differs from official FP16 rolling protocol',
            'single seed; pretrained exposure cannot be independently verified',
            'different units across datasets and Weather variables; no pooled cross-dataset raw-unit score'])
    atomic_json(output/'report.json',value)
    return value


@torch.no_grad()
def run(cfg,output):
    if os.environ.get('GOFA_STAGE19_ADMITTED')!='1':
        raise RuntimeError('Target evaluation requires fully-idle supervised admission')
    manifest=verify(cfg,output)
    binding=sha(output/'manifest.json')
    proof=read(output/'cpu_preflight.json',{})
    if not proof.get('passed') or proof['manifest_sha256']!=binding:
        raise ValueError('Code/config/subset-bound CPU preflight required')
    stopped=[False]
    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    def check():
        if stopped[0] or not supervisor_alive(os.environ['GOFA_STAGE19_SUPERVISOR_PID'],os.environ['GOFA_STAGE19_SUPERVISOR_TICKS']):
            raise Paused('cooperative_resource_or_user_pause')
    progress=Progress(output,manifest['total_numeric_work'])
    phase='restore_frozen_3500'
    def status(state,**fields):
        atomic_json(output/'status.json',dict(state=state,phase=phase,updated_at=time.time(),
            selected_update=3500,optimizer_updates=0,**fields))
    memory=read(output/'cuda_memory.json',{})
    def save_memory():
        old=memory.setdefault(phase,dict(allocated_bytes=0,reserved_bytes=0))
        old['allocated_bytes']=max(old['allocated_bytes'],torch.cuda.max_memory_allocated())
        old['reserved_bytes']=max(old['reserved_bytes'],torch.cuda.max_memory_reserved())
        atomic_json(output/'cuda_memory.json',memory)
    try:
        status('running')
        torch.cuda.set_per_process_memory_fraction(38*2**30/torch.cuda.get_device_properties(0).total_memory)
        model,codec=load_model(manifest['source_configuration'],torch.device('cuda:0'))
        if codec.alphabet_sha!=manifest['alphabet']['alphabet_sha256']:
            raise ValueError('Evaluation numeric alphabet differs')
        adapter=torch.load(output/'adapter.pt',map_location='cpu',weights_only=False)
        model.load_trainable(adapter)
        model.parameter_version=3500
        model.eval().requires_grad_(False)
        assert_equal(adapter,model.trainable_state())
        atomic_json(output/'restore_audit.json',dict(passed=True,adapter_sha256=sha(output/'adapter.pt'),
            selected_update=3500,all_parameters_frozen=True,optimizer_updates=0))
        phase='source_replay_mechanisms'
        status('running')
        old=read(output/'mechanisms.json')
        if old is None:
            old=mechanisms(model,codec,manifest,check,
                audit=lambda value:atomic_json(output/'mechanism_diagnostics.json',value))
            old['binding']=binding
            atomic_json(output/'mechanisms.json',old)
        if not old['passed'] or old['binding']!=binding or old['gradients_created']:
            raise ValueError('Source replay acceptance missing or changed')
        save_memory()
        root=Path(manifest['source_configuration']['data']['directory'])
        for name in TARGETS:
            data={name:Series(root/'targets'/(name+'.npz'))}
            for native in (True,False):
                mode='native' if native else 'gofa'
                phase=f'target-{name}-{mode}'
                check()
                status('running',dataset=name,model=mode)
                progress.tick(phase,force=True,dataset=name,model=mode)
                weight_hash=manifest['weights_sha256'] if native else cfg['adapter_sha256']
                result=evaluate(model,codec,data,manifest['queries']['targets'][name],output,phase,
                    binding+':sample8:'+phase+':'+weight_hash+':'+codec.alphabet_sha,progress,check,native)
                progress.completed[phase]=result
                progress.evaluation(result['queries'],result['queries'],result['numeric_slots_in_eight_paths'],
                    result['numeric_slots_in_eight_paths'],result['active_seconds'],evaluation_kind=phase,evaluation_sample=8)
                atomic_json(output/(phase+'-result.json'),result)
                save_memory()
                report(output)
            verify(cfg,output)
        assert_equal(adapter,model.trainable_state())
        if any(p.grad is not None for p in model.parameters()):
            raise ValueError('Read-only target evaluation created gradients')
        progress.tick('complete',force=True,completed_model_datasets=18)
        if not report(output)['complete']:
            raise ValueError('Paired target coverage is incomplete')
        status('complete',target_datasets_complete=9)
    except Paused as exc:
        status('resource_paused',reason=str(exc))
    except torch.cuda.OutOfMemoryError as exc:
        attempts=read(output/'oom_attempts.json',{})
        key=phase+':'+str(progress.state.get('evaluation_done'))+':'+str(progress.state.get('evaluation_sample'))
        attempts[key]=attempts.get(key,0)+1
        atomic_json(output/'oom_attempts.json',attempts)
        if attempts[key]<3:
            status('resource_paused',reason='cuda_oom; atomic paths retained',attempts=attempts[key])
        else:
            atomic_json(output/'failures.json',dict(type=type(exc).__name__,phase=phase,error=str(exc)))
            status('failed',error='Repeated identical OOM; no blind restart')
            raise
    except Exception as exc:
        save_memory()
        atomic_json(output/'failures.json',dict(type=type(exc).__name__,phase=phase,error=str(exc)))
        status('failed',error=str(exc))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--prepare',action='store_true')
    modes.add_argument('--report-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    cfg,output=yaml.safe_load(Path(args.config).read_text()),Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    if args.prepare:
        prepare(cfg,output)
    elif args.report_only:
        report(output)
    else:
        run(cfg,output)


if __name__=='__main__':
    main()
