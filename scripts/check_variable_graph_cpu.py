"""Source-only real-data acceptance and throughput; never produce target scores."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from gofa_f.dual_memory_engines import model_digest
from gofa_f.dual_memory_fitting import read,fit
from gofa_f.dual_memory_policy import accumulate
from gofa_f.dual_memory_warmup import cpu_only
from gofa_f.generative_data import sha
from gofa_f.prefix_cache import atomic_json
from gofa_f.variable_graph_data import VariableStore,VariableSchedule,length_balanced_summary,HELD_SHAPES
from gofa_f.variable_graph_source import configuration,runtime,ExpertEngine
from gofa_f.variable_graph_evaluation import seal_target_plan


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--proof',required=True);a=p.parse_args();cpu_only();cfg=configuration(a.config);root=Path(a.output)
    m=read(root/'manifest.json');binding=sha(root/'manifest.json')
    if m['variable_runtime']!=runtime():raise ValueError('Actual QA source bytes changed')
    started=time.monotonic();store=VariableStore(m,root/'qa-history-cache');schedule=VariableSchedule(m['queries'])
    rows=[]
    for _ in range(32):q,state=schedule.peek();schedule.commit(state);rows.append(q)
    if len({(q['length'],q['horizon']) for q in rows})!=6:raise ValueError('Mixed real-data length batch required')
    results={}
    for kind in ('expert','ufg'):
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        engine=ExpertEngine(cfg,m,store,kind);initial=model_digest(engine.model);before=time.monotonic()
        # Exactly the production effective batch/microbatch reduction, no Adam update.
        loss=0.
        for i in range(0,len(rows),4):
            value=engine.loss(rows[i:i+4])/len(rows);accumulate(value,list(engine.model.parameters()));loss+=float(value.detach())
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in engine.model.parameters() if p.requires_grad):
            raise ValueError('Every effective source branch must receive finite gradients')
        result=dict(loss=loss,actual_queries=len(rows),backward_seconds=time.monotonic()-before,
            model_initial_sha256=initial,trainable_numel=sum(p.numel() for p in engine.model.parameters() if p.requires_grad),
            trainable_real_dof=sum(p.numel()*(2 if p.is_complex() else 1) for p in engine.model.parameters() if p.requires_grad))
        engine.model.zero_grad(set_to_none=True)
        # Exercise the real generic fitter, all lengths, source scoring and save/load.
        tiny=deepcopy(m);tiny['configuration']['training'].update(max_updates=2,validation_every=1,early_min_updates=2)
        for d in tiny['queries'].values():
            for v in d.values():v['development']=v['development'][:1]
        tick=time.monotonic()
        fit(engine,tiny,cfg,root/('qa-fit-'+kind),'all',None,'QA_ONLY_NO_FORMAL_STATE',lambda:None,lambda **_:None,
            schedule_factory=VariableSchedule,summarizer=length_balanced_summary)
        selected=read(root/('qa-fit-'+kind)/'selection.json')
        if selected['stopped_update']!=2 or selected['target_labels_used']:raise ValueError('Real source fit audit failed')
        result.update(two_update_fit_seconds=time.monotonic()-tick,qa_fit_best=selected['best_update'])
        results[kind]=result;engine.close()
    # Historical windows/eligible IDs only, sealed before any formal optimizer update.
    target_plan=seal_target_plan(cfg,root)
    seen={tuple(v) for v in m['training_shapes']}
    if seen&{tuple(v) for v in m['held_length_pairs']}:raise ValueError('Held length pair was trained')
    proof=dict(passed=True,manifest_sha256=binding,runtime_code=runtime(),results=results,
        cuda_initialized=torch.cuda.is_initialized(),new_gpu_seconds=0,threads=2,
        elapsed_cpu_residency_seconds=time.monotonic()-started,formal_optimizer_updates=0,
        source_only_QA_two_update_states_not_reused_for_training=True,
        target_plan_sha256=sha(root/'target_plan.json'),target_primary_queries=len(target_plan['primary']),
        target_extra_length_queries=len(target_plan['supplement']))
    if proof['cuda_initialized']:raise ValueError('QA initialized CUDA')
    atomic_json(Path(a.proof),proof);print(proof,flush=True)


if __name__=='__main__':main()
