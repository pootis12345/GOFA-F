"""Fixed source-only TSGFM adaptation; no target or LLM training entry point."""
import argparse
from copy import deepcopy
import fcntl
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
import yaml

from gofa_f.dual_memory_data import GROUPS, SHAPES
from gofa_f.dual_memory_fitting import fit, read
from gofa_f.dual_memory_policy import DOMAINS, digest, summarize, rng_state, restore_rng
from gofa_f.dual_memory_store import SourceStore
from gofa_f.dual_memory_warmup import cpu_only
from gofa_f.generative_data import sha
from gofa_f.prefix_cache import atomic_json
from gofa_f.tsgfm_expert import TSGFMEngine, UPSTREAM_COMMIT

PROTOCOL = 'stage61-independent-tsgfm-source-v1'
FILES = ('tsgfm_expert.py','tsgfm_source.py','dual_memory_data.py','dual_memory_store.py',
    'dual_memory_fitting.py','dual_memory_engines.py','dual_memory_policy.py',
    'dual_memory_warmup.py','dual_memory_model.py','generative_data.py','prefix_cache.py')
RECIPE = dict(max_updates=10000,effective_batch=16,microbatch=1,learning_rate=.001,
    warmup_updates=50,interface_only_updates=0,weight_decay=0.,gradient_clip=1.,
    validation_every=250,early_min_updates=2000,early_patience=8,significant_improvement=.001)


def configuration(path):
    cfg = yaml.safe_load(Path(path).read_text().replace('${GOFA_WORKSPACE}',os.environ['GOFA_WORKSPACE']))
    if cfg.get('protocol') != PROTOCOL or cfg.get('training') != RECIPE:
        raise ValueError('Fixed source TSGFM recipe required')
    if cfg.get('target_training_enabled') is not False or cfg.get('target_evaluation_enabled') is not False:
        raise ValueError('No target training/evaluation in the independent source expert')
    if cfg.get('model') != dict(width=128,message_iterations=1,maximum_history=48,
        physical_subgraph=True,observed_origin_scaling=True,shared_multistep_head=True):
        raise ValueError('The preregistered adaptation cannot be searched during a run')
    if cfg.get('resources') != dict(cuda_enabled=False,cpu_threads=2,deadline=None,no_new_llm_training=True):
        raise ValueError('CPU-only execution without a new LLM job required')
    return cfg


def runtime():
    return {name:sha(Path(__file__).parent/name) for name in FILES}


def prepare(cfg,output):
    cpu_only(); parent=Path(cfg['parent_source_output'])
    if sha(parent/'manifest.json') != cfg['parent_source_manifest_sha256']:
        raise ValueError('Source lineage changed')
    old=read(parent/'manifest.json')
    if read(parent/'status.json')['state']!='complete' or read(parent/'activity.json')['worker'] is not None:
        raise ValueError('Exited, frozen parent source required')
    for name,value in old['code'].items():
        if sha(Path(__file__).parent/name)!=value:raise ValueError('Audited source/input core changed: '+name)
    m=deepcopy(old);m['configuration']['training']=cfg['training']
    m['configuration']['queries']['initial_training_per_view']=8192
    m['tsgfm_configuration']=cfg;m['tsgfm_runtime_code']=runtime()
    m['source_cache_index_sha256']=sha(parent/'source_inputs/index.json')
    m['upstream']=dict(repository='https://github.com/HamidL/TSGFM',revision=UPSTREAM_COMMIT,
        equation_reference='https://dl.ifip.org/db/conf/cnsm/cnsm2025/1571173306.pdf',
        official_code_bundled=False,exact_paper_replication=False,
        adaptations=['causal_per_origin_scaling','historical_missing_mask','physical_induced_edges',
            'same_31_real_node_roster','one_shared_direct_multistep_residual_head'])
    m['source_confirmation_labels_opened']=False;m['target_labels_opened']=False
    m['screening_scope']='included_source_development_not_zero_shot_evidence'
    m['screening_rule']=dict(maximum_each_domain_ratio=1.02,maximum_geometric_domain_ratio=.99,
        comparator='locked_unchanged_official_width128_UFG',no_memory_integration_from_this_entry=True)
    costs=Path(cfg['inherited_costs_file'])
    if sha(costs)!=cfg['inherited_costs_sha256']:raise ValueError('Inherited cost evidence changed')
    m['inherited_prior_costs']=read(costs);m['inherited_prior_costs_sha256']=sha(costs)
    m['parallel_stage60_costs_included_here']=False
    # Only source train/dev are available to this process. The inherited
    # confirmation identities remain unopened and are NOT advertised as fresh.
    output.mkdir(parents=True,exist_ok=True)
    existing=read(output/'manifest.json')
    if existing is not None and existing!=m:raise ValueError('Independent expert manifest is immutable')
    atomic_json(output/'manifest.json',m)
    if not (output/'status.json').exists():
        atomic_json(output/'status.json',dict(state='prepared',phase='source_only_expert',goal_achieved=False,
            target_predictions=0,source_confirmation_opened=False))
    return m


def ufg_baseline(parent,manifest,store):
    """Reuse source predictions only, checking actual source inputs and masks."""
    records=[];proof=[]
    for length,horizon in SHAPES:
        shape=f'{length}-{horizon}';root=Path(parent)/('ufg-'+shape)
        selected=read(root/'selection.json')
        if selected['target_labels_used'] or sha(root/'best.pt')!=selected['checkpoint_sha256']:
            raise ValueError('Official source UFG selection changed')
        paths=sorted((root/'evaluations'/f"update-{selected['best_update']:06d}").glob('*.json'))
        expected={digest(q):q for name in GROUPS for q in manifest['queries'][name][shape]['development']}
        seen=set()
        for path in paths:
            r=read(path);key=digest(r['query'])
            if key not in expected or key in seen or r['identity']['checkpoint_sha']!=selected['model_sha256']:
                raise ValueError('UFG source prediction/query/checkpoint mismatch')
            item,y,mask=store.get(expected[key],'development')
            if r['input_sha256']!=item.identity or not np.array_equal(r['target'],y[0].numpy()) or \
                not np.array_equal(r['observed'],mask[0].numpy()):
                raise ValueError('UFG source truth/input differs from the shared source cache')
            records.append(r);seen.add(key);proof.append(dict(file=str(path),sha256=sha(path)))
        if seen!=set(expected):raise ValueError('Incomplete UFG source baseline')
    return dict(records=records,metrics=summarize(records,manifest['training_unit_normalizers']),
        source_prediction_hashes=proof,new_ufg_training_updates=0,
        inherited_training_cpu=read(Path(parent)/'ufg_activity.json'),
        short_shape_stop='maximum_updates_not_convergence_claim',
        same_all_node_label_eligibility=True,same_history_rosters=True)


def screen(expert,baseline):
    ratios={name:max(expert['views'][name]['macro_mae'],1e-12)/max(baseline['views'][name]['macro_mae'],1e-12)
        for name in GROUPS}
    domains={d:float(np.exp(np.mean([np.log(max(ratios[n],1e-12)) for n in GROUPS if GROUPS[n]==d])))
        for d in DOMAINS}
    aggregate=float(np.exp(np.mean(np.log(list(domains.values())))))
    passed=max(domains.values())<=1.02 and aggregate<=.99
    return dict(passed=bool(passed),view_ratios_to_ufg=ratios,domain_ratios_to_ufg=domains,
        geometric_domain_ratio=aggregate,scope='included_source_development',
        zero_shot_superiority_proven=False,target_results_read=False)


def profile(engine,manifest,check):
    """Finite real-roster benchmark, no optimizer, restore RNG and gradients."""
    before=rng_state(False);rows=[];trials=[]
    for source in GROUPS:
        for length,horizon in SHAPES:
            rows.append(manifest['queries'][source][f'{length}-{horizon}']['train'][0])
    try:
        for row in rows:
            check();engine.model.zero_grad(set_to_none=True)
            item,_,_=engine.store.get(row,'train');start=time.monotonic()
            loss=engine.loss([row]);loss.backward()
            if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all()
                for p in engine.model.parameters()):raise FloatingPointError('Invalid real-source expert gradient')
            trials.append(dict(source=row['source'],length=row['length'],horizon=row['horizon'],
                real_nodes=len(item.node_ids),forward_backward_seconds=time.monotonic()-start,
                finite_loss=float(loss.detach())))
    finally:
        engine.model.zero_grad(set_to_none=True);restore_rng(before)
    return dict(trials=trials,optimizer_updates=0,model_parameters=sum(p.numel() for p in engine.model.parameters()),
        mean_query_seconds=float(np.mean([v['forward_backward_seconds'] for v in trials])),
        cuda_initialized=torch.cuda.is_initialized(),real_nodes_truncated=False)


def run(cfg,output):
    cpu_only();lock=(output/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    m=read(output/'manifest.json');binding=sha(output/'manifest.json')
    if m['tsgfm_configuration']!=cfg or m['tsgfm_runtime_code']!=runtime():raise ValueError('Source runtime/configuration changed')
    proof=read(output/'cpu_preflight.json',{})
    if not proof.get('passed') or proof.get('manifest_sha256')!=binding or proof.get('runtime_code')!=runtime():
        raise ValueError('Related CPU proof must bind the prepared model/source runtime')
    prior_status=read(output/'status.json',{})
    if prior_status.get('state')=='complete':return
    if prior_status.get('state')=='failed':raise ValueError('Audit implementation failure before any restart')
    parent=Path(cfg['parent_source_output'])
    if sha(parent/'source_inputs/index.json')!=m['source_cache_index_sha256']:raise ValueError('Shared source cache changed')
    stopped=[False]
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    from gofa_f.idle_launcher import process_identity
    worker=dict(pid=os.getpid(),start_ticks=process_identity(os.getpid()))
    initial=read(output/'activity.json',{}).get('spent_cpu_residency_seconds',0.)
    start=time.monotonic();engine=None;visible={}
    def check():
        if stopped[0]:raise InterruptedError('Cooperative independent-expert pause')
    def progress(**fields):
        if fields.get('phase')!=visible.get('phase'):visible.clear()
        visible.update(fields)
        atomic_json(output/'status.json',dict(state='running',goal_achieved=False,
            target_predictions=0,source_confirmation_opened=False,model='TSGFM adaptation',**visible))
        atomic_json(output/'activity.json',dict(worker=worker,spent_cpu_residency_seconds=initial+time.monotonic()-start,
            inherited_gpu_seconds=m['inherited_prior_costs']['cumulative_gpu_seconds'],new_gpu_seconds=0,
            parallel_stage60_costs_included=False,cuda_initialized=False,threads=2))
    try:
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        store=SourceStore(m,parent/'source_inputs');engine=TSGFMEngine(m,store)
        if not (output/'profile.json').exists():
            progress(phase='real_source_throughput');atomic_json(output/'profile.json',profile(engine,m,check))
        if not (output/'ufg_baseline.json').exists():
            progress(phase='source_baseline_reuse');atomic_json(output/'ufg_baseline.json',ufg_baseline(parent,m,store))
        best=fit(engine,m,cfg,output/'all','all',None,digest([binding,'independent-tsgfm','all']),check,progress)
        result=screen(best['source_metrics'],read(output/'ufg_baseline.json')['metrics'])
        selected=read(output/'all/selection.json')
        atomic_json(output/'report.json',dict(screen=result,selection=selected,goal_achieved=False,
            shared_expert_checkpoint_sha256=selected['checkpoint_sha256'],runtime_code=runtime(),
            source_confirmation_run=False,target_predictions=0,memory_integration_run=False,
            next_step='matched_source_group_crossfit_before_Memory' if result['passed'] else 'seal_source_screen_failure',
            inherited_fitter_checkpoint_engine_tag='memory_is_legacy_internal_tag_actual_model_is_tsgfm'))
        atomic_json(output/'status.json',dict(state='complete',phase='source_screen_passed' if result['passed'] else 'source_screen_failed',
            goal_achieved=False,target_predictions=0,best_update=selected['best_update'],stopped_update=selected['stopped_update'],
            source_screen=result,eta_seconds=None))
    except InterruptedError:
        atomic_json(output/'status.json',dict(state='paused',phase='source_expert',goal_achieved=False));raise SystemExit(75)
    except BaseException as exc:
        atomic_json(output/f'failure-{time.time_ns()}.json',dict(error=repr(exc),binding=binding))
        atomic_json(output/'status.json',dict(state='failed',error=repr(exc),goal_achieved=False));raise
    finally:
        if engine is not None:engine.close()
        atomic_json(output/'activity.json',dict(worker=None,spent_cpu_residency_seconds=initial+time.monotonic()-start,
            inherited_gpu_seconds=m['inherited_prior_costs']['cumulative_gpu_seconds'],new_gpu_seconds=0,
            parallel_stage60_costs_included=False,cuda_initialized=False,threads=2,
            peak_rss_kib=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--progress',action='store_true');args=parser.parse_args()
    cfg=configuration(args.config);output=Path(args.output)
    if args.progress:
        status=read(output/'status.json',{});activity=read(output/'activity.json',{})
        curves=output/'all/curve.jsonl';last=json.loads(curves.read_text().splitlines()[-1]) if curves.exists() else {}
        terminal=status.get('state') in ('complete','failed','paused')
        update=status.get('updates',last.get('update',0))
        estimate=None if terminal or update<10 else \
            (RECIPE['max_updates']-update)*last.get('active_seconds',0)/max(last.get('update',1),1)
        print(json.dumps(dict(status=status,activity=activity,
            eta_this_source_fit_seconds=estimate,overall_goal_eta_unknown=True,
            eta_note='Includes observed checkpoint/validation overhead; excludes future crossfits, Memory integration and target evaluation'),indent=2))
    elif args.prepare:prepare(cfg,output)
    else:run(cfg,output)


if __name__=='__main__':main()
