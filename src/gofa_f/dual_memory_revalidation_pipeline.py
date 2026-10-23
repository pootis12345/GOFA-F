"""Run-bound Stage60 confirmation/targets/delivery after verified source exit."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from gofa_f.dual_memory_eval_data import read,seal,source_ready,deployment
from gofa_f.dual_memory_eval_pipeline import alive,source_handles
from gofa_f.dual_memory_revalidation import configuration,prepare_scope,adequate_deployment
from gofa_f.generative_data import sha
from gofa_f.idle_launcher import process_identity
from gofa_f.prefix_cache import atomic_json


def launch(snapshot,config,cfg,output,ufg=False):
    path=Path(str(output)+('.ufg.launch.json' if ufg else '.launch.json'))
    if path.exists() and alive(read(path)):return read(path)
    status=output/('ufg_status.json' if ufg else 'status.json')
    if status.exists() and read(status).get('state')=='failed':
        raise ValueError('A failed evaluation requires audit before any restart')
    complete=('ufg_complete.json','ufg_frozen_weights.json') if ufg else \
        ('native_complete.json','enhanced_complete.json','fusion_complete.json','frozen_gpu_weights.json')
    if status.exists() and read(status).get('state')=='complete' and all((output/n).exists() for n in complete):
        return None
    python=cfg['ufg_python'] if ufg else sys.executable
    command=[python,'-u','-m','gofa_f.dual_memory_revalidation','--config',str(config),'--output',str(output),'--ufg'] if ufg else \
        [python,'-u','-m','gofa_f.short_launcher','--config',str(config),'--output',str(output),
         '--snapshot',str(snapshot),'--worker-mode','dual_memory_revalidation']
    env=dict(os.environ,PYTHONPATH=str(snapshot/'src'),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    with Path(str(output)+('.ufg.log' if ufg else '.log')).open('ab') as log:
        child=subprocess.Popen(command,cwd=snapshot,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    handle=dict(pid=child.pid,start_ticks=process_identity(child.pid),command=command,snapshot=str(snapshot),at=time.time())
    atomic_json(path,handle);return handle


def cost_report(source,cfg,locked):
    source=Path(source);manifest=read(source/'manifest.json');activity=read(source/'activity.json')
    confirm=Path(cfg['source_confirmation_output']);target=Path(cfg['target_output'])
    scopes={name:read(path/'activity.json') for name,path in (('source_confirmation',confirm),('targets',target))}
    warmup=Path(manifest['formal_configuration']['warmup_output'])
    warm=read(warmup/'manifest.json');prior=Path(warm['warmup_configuration']['inherited_costs_file'])
    if sha(prior)!=warm['inherited_gpu_costs_sha256']:raise ValueError('Inherited cost evidence changed')
    parent_costs=read(prior);inherited=activity['inherited_prior_gpu_seconds']
    if inherited!=parent_costs['cumulative_gpu_seconds'] or activity['spent_seconds']<inherited:
        raise ValueError('GPU cost reset or incompatible inheritance')
    return dict(state='complete',goal_achieved=read(target/'report.json')['goal_achieved'],deployment=locked,
        source_confirmation_report_sha256=sha(confirm/'report.json'),target_report_sha256=sha(target/'report.json'),
        inherited_source_gpu_seconds=activity['spent_seconds'],new_evaluation_gpu_seconds=sum(v['spent_seconds'] for v in scopes.values()),
        cumulative_gpu_seconds=activity['spent_seconds']+sum(v['spent_seconds'] for v in scopes.values()),
        inherited_prior_stage_gpu_seconds=inherited,new_joint_source_gpu_seconds=activity['spent_seconds']-inherited,
        peak_group_mib=max(activity.get('peak_group_mib',0),*(v.get('peak_group_mib',0) for v in scopes.values())),
        source_ufg_cpu=read(source/'ufg_activity.json'),
        source_temporal_initialization_cpu=read(warmup/'activity.json'),
        inherited_prior_cpu_costs={k:v for k,v in parent_costs.items() if 'cpu' in k},
        evaluation_ufg_cpu={name:read(path/'ufg_activity.json') for name,path in (('source_confirmation',confirm),('targets',target))},
        baseline_adequacy_cpu=read(Path(cfg['ufg_adequacy_output'])/'activity.json'),
        standalone_expert_source_cpu=read(Path(read(Path(cfg['source_alternative_review_output'])/'manifest.json')['matched_configuration']['expert_output'])/'activity.json'),
        source_alternative_review_activity=read(Path(cfg['source_alternative_review_output'])/'activity.json'),
        ufg_source_training_reused_not_repeated=True,target_results_not_used_to_refit=True)


def route_review(cfg):
    """Source-only preregistered preference; no target model comparison."""
    root=Path(cfg['source_alternative_review_output'])
    if sha(root/'manifest.json')!=cfg['source_alternative_review_manifest_sha256']:
        raise ValueError('Source alternative review manifest changed')
    status=read(root/'status.json')
    if status.get('state')=='failed':raise ValueError('Source alternative implementation failure requires audit')
    handle=read(Path(str(root)+'.launch.json'))
    if status.get('state')!='complete' or alive(handle):
        if not alive(handle):raise ValueError('Unfinished source review without verified live handle')
        return 'waiting'
    if read(root/'activity.json').get('worker') is not None:
        raise ValueError('Source review process exited without finalized ledger')
    report=read(root/'report.json')
    if report['binding']!=cfg['source_alternative_review_manifest_sha256'] or report['target_predictions']!=0:
        raise ValueError('Source route review used unbound or target predictions')
    if report['phase']=='matched_source_gate_passed':return 'prefer_new_expert_integration'
    if report['phase'] not in ('not_started_standalone_source_screen_failed',
            'not_started_source_recheck_or_adequacy_failed','matched_source_gate_failed'):
        raise ValueError('Unknown scientific source-route decision')
    return 'continue_existing_memory_candidate'


def run(cfg,root,snapshot,config,proof):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only coordinator required')
    root.mkdir(parents=True,exist_ok=True);lock=(root/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cpu=read(proof)
    if not cpu.get('passed') or cpu.get('cuda_initialized') is not False:raise ValueError('Related CPU proof required')
    files=cpu['code_sha256']
    for name,value in files.items():
        if sha(snapshot/name)!=value:raise ValueError('CPU-tested runtime changed: '+name)
    stopped=[False];children=[]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    def check():
        if stopped[0]:raise InterruptedError('Connected evaluation stopped cooperatively')
    def state(phase,**fields):atomic_json(root/'status.json',dict(state='running',phase=phase,goal_achieved=False,**fields))
    source=Path(cfg['source_output']);handles=source_handles(source)
    if sha(source/'manifest.json')!=cfg['source_training_manifest_sha256']:raise ValueError('Wrong source experiment')
    try:
        seal(root/'deployment_source.json',dict(snapshot=str(snapshot),evaluation_config=str(config),
            evaluation_config_sha256=sha(config),cpu_proof_sha256=sha(proof)))
        while True:
            check();decision=route_review(cfg)
            if decision!='waiting':break
            state('waiting_for_verified_source_expert_review',target_inputs_prepared=False);time.sleep(10)
        review=Path(cfg['source_alternative_review_output'])
        seal(root/'source_route_decision.json',dict(decision=decision,report_sha256=sha(review/'report.json'),
            target_results_used=False,preference='integrate_standalone_expert_only_after_source_gates'))
        if decision=='prefer_new_expert_integration':
            atomic_json(root/'status.json',dict(state='deferred',phase='source_approved_new_expert_requires_Memory_integration',
                goal_achieved=False,target_predictions=0));return
        while not source_ready(source):
            check();states={n:read(source/n) for n in ('status.json','ufg_status.json') if (source/n).exists()}
            if any(v.get('state')=='failed' for v in states.values()):raise ValueError('Source failure requires audit')
            live=[h for h in handles if alive(h)]
            if not live:raise ValueError('Incomplete source lock and no verified live source handle')
            state('waiting_for_verified_source_jobs',live_handles=[dict(pid=h['pid'],start_ticks=h['start_ticks']) for h in live],
                source_states=states,target_inputs_prepared=False)
            time.sleep(10)
        while any(alive(h) for h in handles):check();state('waiting_for_source_process_exit');time.sleep(1)
        seal(root/'source_exit_audit.json',dict(handles=[dict(pid=h['pid'],start_ticks=h['start_ticks'],exited=True) for h in handles],
            source_activity_worker=read(source/'activity.json')['worker']))
        locked=adequate_deployment(cfg);seal(root/'deployment_before_confirmation.json',locked)
        for scope,key in (('source_confirmation','source_confirmation_output'),('targets','target_output')):
            check();output=Path(cfg[key])
            if (output/'final_status.json').exists() and read(output/'final_status.json')['state']=='complete':continue
            state('preparing_'+scope);prepare_scope(cfg,output,scope,locked)
            seal(output/'cpu_preflight.json',dict(passed=True,binding=sha(output/'manifest.json'),
                proof_sha256=sha(proof),code_sha256=files,cuda_initialized=False))
            if not (output/'ufg_activity.json').exists():atomic_json(output/'ufg_activity.json',dict(spent_seconds=0.,worker=None))
            for ufg in (True,False):
                handle=launch(snapshot,config,cfg,output,ufg)
                if handle is not None:children.append(handle)
            while True:
                check();live=[h for h in children if alive(h)]
                states={n:read(output/n) for n in ('status.json','ufg_status.json') if (output/n).exists()}
                if any(v.get('state')=='failed' for v in states.values()):raise ValueError('Frozen evaluation failed; preserve evidence')
                if not live:break
                state('predicting_'+scope,prediction_states=states);time.sleep(10)
            children=[]
            if not all((output/f'{name}_complete.json').exists() for name in ('native','enhanced','fusion','ufg')):
                raise ValueError('Incomplete prediction groups after worker exit')
            if any(read(output/f).get('worker') is not None for f in ('activity.json','ufg_activity.json')):
                raise ValueError('A prediction ledger still owns a live worker')
            state('scoring_'+scope)
            subprocess.run([sys.executable,'-m','gofa_f.dual_memory_revalidation','--config',str(config),
                '--output',str(output),'--summarize'],check=True,cwd=snapshot,
                env=dict(os.environ,PYTHONPATH=str(snapshot/'src'),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
        report=cost_report(source,cfg,locked);seal(root/'report.json',report);atomic_json(root/'status.json',report)
        from gofa_f.dual_memory_revalidation_delivery import export
        try:
            export(cfg,root,Path(cfg['delivery_output']))
            atomic_json(root/'delivery_status.json',dict(state='complete',output=cfg['delivery_output']))
        except BaseException as exc:
            atomic_json(root/'delivery_status.json',dict(state='failed',error=repr(exc),new_predictions=0))
            raise
    finally:
        for handle in children:
            if alive(handle):os.kill(handle['pid'],signal.SIGTERM)
        until=time.monotonic()+60
        while any(alive(h) for h in children) and time.monotonic()<until:time.sleep(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--snapshot',required=True);parser.add_argument('--cpu-proof',required=True);args=parser.parse_args()
    cfg=configuration(args.config);root=Path(args.output)
    try:run(cfg,root,Path(args.snapshot),Path(args.config),Path(args.cpu_proof))
    except InterruptedError:
        atomic_json(root/'status.json',dict(state='paused',phase='evaluation_pipeline',goal_achieved=False));raise SystemExit(75)
    except BaseException as exc:
        atomic_json(root/f'failure-{time.time_ns()}.json',dict(error=repr(exc)))
        if not (root/'report.json').exists():atomic_json(root/'status.json',dict(state='failed',error=repr(exc),goal_achieved=False))
        raise


if __name__=='__main__':main()
