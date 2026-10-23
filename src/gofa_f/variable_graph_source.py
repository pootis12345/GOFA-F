"""Fresh source-only multi-length expert/UFG training with exact recovery."""
import argparse
from collections import defaultdict
import fcntl
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from torch.nn import functional as F
import yaml

from gofa_f.dual_memory_fitting import fit,read,evaluate
from gofa_f.dual_memory_engines import model_digest
from gofa_f.dual_memory_policy import query_loss,digest
from gofa_f.dual_memory_warmup import cpu_only
from gofa_f.generative_data import sha
from gofa_f.prefix_cache import atomic_json
from gofa_f.variable_graph_data import VariableStore,VariableSchedule,prepare_variable,length_balanced_summary
from gofa_f.variable_graph_expert import VariableGraphExpert,pack_items
from gofa_f.ufg_memory_roster import pack_roster
from gofa_f.ufg_subgraph import SubgraphInput

PROTOCOL='stage66-variable-graph-source-v1'
RECIPE=dict(max_updates=40000,effective_batch=32,microbatch=4,learning_rate=.001,warmup_updates=50,
    interface_only_updates=0,weight_decay=0.,gradient_clip=5.,validation_every=500,
    early_min_updates=4000,early_patience=8,significant_improvement=.001)
FILES=('variable_graph_source.py','variable_graph_data.py','variable_graph_expert.py','variable_ufg.py',
    'variable_graph_evaluation.py','variable_graph_pipeline.py',
    'dual_memory_fitting.py','dual_memory_policy.py','dual_memory_store.py','dual_memory_data.py',
    'dual_memory_prepare.py','dual_memory_warmup.py','dual_memory_engines.py',
    'generative_data.py','graph_memory_data.py','prefix_cache.py','ufg.py','ufg_subgraph.py','ufg_memory_roster.py')


def runtime():return {n:sha(Path(__file__).parent/n) for n in FILES}


def configuration(path):
    cfg=yaml.safe_load(Path(path).read_text().replace('${GOFA_WORKSPACE}',os.environ['GOFA_WORKSPACE']))
    if cfg['protocol']!=PROTOCOL or cfg['training']!=RECIPE or cfg['target_training_enabled']:
        raise ValueError('One fixed source-only multi-length recipe required')
    if cfg['expert']!=dict(width=64,patch=4,blocks=3,heads=4) or cfg['ufg']!=dict(width=128,canonical_history=48,maximum_horizon=144,single_parameter_set=True):
        raise ValueError('No architecture or shape-specialized checkpoint search')
    if cfg['resources']!=dict(cuda_enabled=False,cpu_threads=2,deadline=None):raise ValueError('CPU two-thread fixed resource policy')
    return cfg


class ExpertEngine:
    def __init__(self,cfg,m,store,kind):
        self.store,self.manifest,self.kind=store,m,kind;self.cuda=False;self.microbatch=4
        if kind=='expert':self.model=VariableGraphExpert(**cfg['expert'])
        elif kind=='ufg':
            from gofa_f.variable_ufg import VariableRosterUFG
            self.model=VariableRosterUFG(cfg['ufg_root'])
        else:raise ValueError('Expert or one shared UFG')
        self.state_model=self.model

    def warmup(self,enabled):pass

    def predictions(self,rows,split,role='fit'):
        values=[self.store.get(r,split,role) for r in rows];items=[v[0] for v in values]
        h=rows[0]['horizon']
        if len({(r['length'],r['horizon']) for r in rows})!=1:raise ValueError('Execution groups must share shape')
        if self.kind=='expert':
            packed=pack_items(items)
            pred=self.model(**packed,interval=torch.tensor([self.store.interval(r) for r in rows]),horizon=h)
        else:
            p=[pack_roster(i) for i in items]
            packed=SubgraphInput(*(torch.cat([getattr(v,k) for v in p],0) for k in
                ('normalized','observed','minimum','extent','present')),31)
            pred=self.model(packed,h,all_nodes=True).transpose(1,2)
        return pred,values

    def loss(self,rows):
        groups=defaultdict(list)
        for row in rows:groups[(row['length'],row['horizon'])].append(row)
        loss=[]
        for own in groups.values():
            pred,values=self.predictions(own,'train')
            for j,(row,(item,target,mask)) in enumerate(zip(own,values)):
                loss.append(query_loss(pred[j,:len(item.node_ids)],target,mask,item.present,
                    self.manifest['training_unit_normalizers'][row['source']]))
        return sum(loss)

    @torch.no_grad()
    def predict(self,row,split,role='fit'):
        pred,values=self.predictions([row],split,role);item,y,mask=values[0]
        return pred[0,0].numpy(),y[0].numpy(),mask[0].numpy(),item.identity

    def close(self):self.model=None


def prepare(cfg,root):
    cpu_only();m=prepare_variable(cfg,root,runtime())
    if not (root/'status.json').exists():atomic_json(root/'status.json',dict(state='prepared',goal_achieved=False))
    return m


def run(cfg,root,kind):
    from gofa_f.idle_launcher import process_identity
    cpu_only();m=read(root/'manifest.json');binding=sha(root/'manifest.json')
    if m['variable_configuration']!=cfg or m['variable_runtime']!=runtime():raise ValueError('Exact tested source runtime required')
    proof=read(root/'cpu_preflight.json',{})
    if not proof.get('passed') or proof.get('manifest_sha256')!=binding or proof.get('runtime_code')!=runtime():
        raise ValueError('Exact source-bound CPU acceptance required')
    output=root/kind;output.mkdir(parents=True,exist_ok=True)
    lock=(output/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prior=read(output/'status.json',{})
    if prior.get('state')=='complete':return
    if prior.get('state')=='failed':raise ValueError('Audit failure before fresh-version exact recovery')
    stop=[False]
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    worker=dict(pid=os.getpid(),start_ticks=process_identity(os.getpid()));start=time.monotonic()
    elapsed=read(output/'activity.json',{}).get('spent_cpu_residency_seconds',0.);visible={};engine=None
    def check():
        if stop[0]:raise InterruptedError('Cooperative exact partial-state stop')
    def progress(**fields):
        visible.update(fields)
        atomic_json(output/'status.json',dict(state='running',model=kind,target_predictions=0,goal_achieved=False,**visible))
        atomic_json(output/'activity.json',dict(worker=worker,spent_cpu_residency_seconds=elapsed+time.monotonic()-start,
            new_gpu_seconds=0,cuda_initialized=False,threads=2))
    try:
        progress(phase='source_cache')
        store=VariableStore(m,root/'source_inputs')
        if not (root/'source_inputs/index.json').exists():store.build(check,progress)
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        engine=ExpertEngine(cfg,m,store,kind)
        fit(engine,m,cfg,output/'all','all',None,digest([binding,kind,'all']),check,progress,
            schedule_factory=VariableSchedule,summarizer=length_balanced_summary)
        s=read(output/'all/selection.json')
        atomic_json(output/'source_lock.json',dict(binding=binding,selection=s,source_only=True,
            one_parameter_set=True,shape_specific_weights=False,checkpoint_path=str(output/'all/best.pt'),
            source_adequacy=s['stop_reason']=='source_early_stop',target_labels_used=False))
        # Only the already locked source-best can open the unused source confirmation windows.
        store.confirmation=True;rows=[q for d in m['queries'].values() for v in d.values() for q in v['confirmation']]
        model=engine.state_model;model.eval();before=model_digest(model)
        records=evaluate(engine,rows,'confirmation',output/'confirmation',binding,before,check,progress)
        if model_digest(model)!=before:raise ValueError('Confirmation changed weights')
        atomic_json(output/'confirmation.json',dict(binding=binding,model_sha256=before,
            metrics=length_balanced_summary(records,m['training_unit_normalizers']),used_for_selection=False))
        atomic_json(output/'status.json',dict(state='complete',phase='source_locked_and_length_confirmation_complete',
            model=kind,best_update=s['best_update'],stopped_update=s['stopped_update'],source_adequacy=s['stop_reason']=='source_early_stop',
            target_predictions=0,goal_achieved=False,eta_seconds=None))
    except InterruptedError:
        atomic_json(output/'status.json',dict(state='paused',model=kind,goal_achieved=False));raise
    except BaseException as exc:
        atomic_json(output/f'failure-{time.time_ns()}.json',dict(error=repr(exc),binding=binding))
        atomic_json(output/'status.json',dict(state='failed',model=kind,error=repr(exc),goal_achieved=False));raise
    finally:
        if engine is not None:engine.close()
        atomic_json(output/'activity.json',dict(worker=None,spent_cpu_residency_seconds=elapsed+time.monotonic()-start,
            new_gpu_seconds=0,cuda_initialized=False,threads=2,
            peak_rss_kib=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--model',choices=('expert','ufg'))
    p.add_argument('--progress',action='store_true');p.add_argument('--watch',action='store_true');a=p.parse_args()
    cfg=configuration(a.config);root=Path(a.output)
    if a.prepare:prepare(cfg,root)
    elif a.progress or a.watch:
        while True:
            pipeline=read(root/'pipeline_status.json',{});targets=read(Path(cfg['evaluation']['output'])/'status.json',{})
            print('Pipeline',pipeline.get('state','not_started'),pipeline.get('phase',''),
                'model',pipeline.get('model',''),
                'queries',pipeline.get('queries',''), '/',pipeline.get('total',''),
                'targets',targets.get('state','not_started'),flush=True)
            for kind in ('expert','ufg'):
                s=read(root/kind/'status.json',{});end=s.get('state') in ('complete','failed','paused')
                eta=None if end else s.get('eta_current_fit_seconds')
                print('VariableGraph',kind,s.get('state','not_started'),s.get('phase',''),
                    'updates',s.get('updates',s.get('stopped_update',0)),'best',s.get('selection',{}).get('best_update',s.get('best_update')),
                    'ETA','n/a' if end else 'unknown' if eta is None else f'{eta/3600:.2f}h to cap; early stop may shorten',flush=True)
            if not a.watch or pipeline.get('state') in ('complete','failed','paused'):break
            time.sleep(5)
    elif a.model:run(cfg,root,a.model)
    else:p.error('--prepare, --model, or --progress required')


if __name__=='__main__':main()
