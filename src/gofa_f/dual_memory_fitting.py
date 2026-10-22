"""Finite source fitting with atomic query predictions and partial accumulation."""
from copy import copy,deepcopy
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from gofa_f.dual_memory_data import GROUPS
from gofa_f.dual_memory_engines import model_digest
from gofa_f.dual_memory_policy import (SourceSchedule,accumulate,capture,digest,restore,
    rng_state,restore_rng,summarize,update_selection,fit_groups)
from gofa_f.dual_memory_store import atomic_torch
from gofa_f.generative_data import sha,Query
from gofa_f.prefix_cache import atomic_json


def read(path,default=None):return json.loads(Path(path).read_text()) if Path(path).exists() else default


def evaluate(engine,rows,split,root,binding,checkpoint_sha,check,progress,role='fit'):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);records=[]
    for index,row in enumerate(rows):
        check();path=root/f'{index:05d}.json'
        identity=dict(binding=binding,checkpoint_sha=checkpoint_sha,query=row,split=split)
        record=read(path)
        if record is None:
            started=time.monotonic()
            prediction,target,observed,input_sha=engine.predict(row,split,role)
            if not np.isfinite(prediction).all() or not np.asarray(observed).any():raise FloatingPointError('Invalid source forecast')
            record=dict(identity=identity,query=row,prediction=prediction.tolist(),target=target.tolist(),
                observed=observed.tolist(),input_sha256=input_sha)
            record['prediction_seconds']=time.monotonic()-started
            atomic_json(path,record)
        elif record['identity']!=identity:raise ValueError('Source prediction checkpoint/query binding changed')
        records.append(record)
        progress(phase='validation' if role=='fit' else 'out_of_fit',completed=index+1,total=len(rows))
    return records


def fit(engine,manifest,config,output,fit_id,depth,binding,check,progress,ufg=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    settings=manifest['configuration']['ufg'] if ufg else manifest['configuration']['training']
    maximum=settings['maximum_updates'] if ufg else settings['max_updates']
    shape=f'{engine.length}-{engine.horizon}' if ufg else None
    held=None if fit_id=='all' else fit_id[5:]
    own=fit_groups(fit_id)
    rows=[q for name in own for s,v in manifest['queries'][name].items() if shape is None or s==shape for q in v['development']]
    schedule=SourceSchedule(manifest['queries'],held_group=held,shape=shape,
        initial=manifest['configuration']['queries']['training_per_view'] if ufg else manifest['configuration']['queries']['initial_training_per_view'],expand=not ufg)
    initial=dict(updates=0,pending=0,loss_sum=0.,cursor=schedule.state,selection={},active_seconds=0.,
        completed=False,train_query_count=0,phases={},engine='ufg' if ufg else 'memory',fit_id=fit_id,depth=depth)
    parameters=list(engine.state_model.parameters())
    optimizer=(torch.optim.Adam if ufg else torch.optim.AdamW)(parameters,lr=settings['learning_rate'],
        **({} if ufg else dict(weight_decay=settings['weight_decay'])))
    state=initial;checkpoint=output/'last.pt';started=time.monotonic();base_seconds=0.
    if checkpoint.exists():
        state=restore(engine.state_model,optimizer,torch.load(checkpoint,map_location='cpu',weights_only=False),binding)
        schedule.commit(state['cursor']);base_seconds=state['active_seconds']
    def save():
        state['cursor']=deepcopy(schedule.state);state['active_seconds']=base_seconds+time.monotonic()-started
        atomic_torch(checkpoint,capture(engine.state_model,optimizer,state,binding,engine.cuda))
    def validate():
        validation_started=time.monotonic()
        engine.state_model.eval();key=model_digest(engine.state_model)
        records=evaluate(engine,rows,'development',output/'evaluations'/f"update-{state['updates']:06d}",binding,key,check,
            lambda **f:progress(**dict(f,fit=fit_id,depth=depth,updates=state['updates'],maximum=maximum)))
        state['validation_query_seconds']=sum(r.get('prediction_seconds',0.) for r in records)/max(len(records),1)
        summary=summarize(records,manifest['training_unit_normalizers'])
        score=summary['selection_score'];old=state['selection'].get('best_update')
        state['selection']=update_selection(state['selection'],score,state['updates'],
            settings['early_min_updates'],settings['early_patience'],.001)
        if old!=state['selection'].get('best_update'):
            atomic_torch(output/'best.pt',dict(binding=binding,model={n:p.detach().cpu().clone() for n,p in engine.state_model.state_dict().items()},
                update=state['updates'],model_sha256=key,source_metrics=summary))
        atomic_json(output/f"validation-{state['updates']:06d}.json",dict(update=state['updates'],metrics=summary,selection=state['selection'],model_sha256=key))
        state['validation_wall_seconds']=state.get('validation_wall_seconds',0.)+time.monotonic()-validation_started
        save();engine.state_model.train()
    if state['completed']:
        best=torch.load(output/'best.pt',map_location='cpu',weights_only=False)
        if best['binding']!=binding:raise ValueError('Locked source best binding changed')
        engine.state_model.load_state_dict(best['model']);return best
    try:
        if state['selection'].get('last_validated',-1)!=state['updates'] and state['updates']%settings['validation_every']==0:
            validate()
        while state['updates']<maximum and not state['selection'].get('stop',False):
            check()
            effective=settings['effective_batch']
            engine.warmup(not ufg and state['updates']<settings['interface_only_updates'])
            if state['pending']==effective:
                if not ufg:
                    rate=settings['learning_rate']*min(1.,(state['updates']+1)/settings['warmup_updates'])
                    for group in optimizer.param_groups:group['lr']=rate
                # Save the exact pre-step transaction before first Adam state
                # allocation and before newly unfrozen graph parameters allocate.
                if state['updates'] in (0,50):save()
                torch.nn.utils.clip_grad_norm_(parameters,5. if ufg else settings['gradient_clip'],error_if_nonfinite=True)
                optimizer.step();optimizer.zero_grad(set_to_none=True)
                state['updates']+=1;mean_loss=state['loss_sum'];state['pending']=0;state['loss_sum']=0.
                save()
                with (output/'curve.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(update=state['updates'],loss=mean_loss,active_seconds=state['active_seconds'],
                        training_queries=state['train_query_count'],cursor=schedule.state))+'\n')
                seconds_per_update=max(state.get('training_query_seconds',0.)/max(state['train_query_count'],1)*effective,
                    (state['active_seconds']-state.get('validation_wall_seconds',0.))/max(state['updates'],1))
                eta=None if state['updates']<10 else (maximum-state['updates'])*seconds_per_update+ \
                    ((maximum-state['updates']+settings['validation_every']-1)//settings['validation_every'])*len(rows)*state.get('validation_query_seconds',0.)
                progress(phase='training',fit=fit_id,depth=depth,updates=state['updates'],maximum=maximum,
                    pending=0,loss=mean_loss,active_seconds=state['active_seconds'],selection=state['selection'],
                    estimated_seconds_per_update=seconds_per_update,eta_current_fit_seconds=eta,
                    eta_excludes_unknown_resource_pauses=True)
                if state['updates']%settings['validation_every']==0:validate()
                continue
            count=min(engine.microbatch,effective-state['pending'])
            trial=copy(schedule);trial.commit(schedule.state)
            queries=[]
            for _ in range(count):
                row,next_state=trial.peek();trial.commit(next_state);queries.append(row)
            before=rng_state(engine.cuda)
            try:
                micro_started=time.monotonic()
                loss=engine.loss(queries)/effective
                accumulate(loss,parameters)
            except torch.OutOfMemoryError:
                restore_rng(before)
                if engine.cuda:torch.cuda.empty_cache()
                if engine.microbatch>1:engine.microbatch=1;continue
                # No roster truncation or untrained-state migration. Save exact
                # already committed gradients/cursor for a memory-strategy repair.
                save();raise
            schedule.commit(trial.state);state['pending']+=count;state['train_query_count']+=count
            state['training_query_seconds']=state.get('training_query_seconds',0.)+time.monotonic()-micro_started
            state['loss_sum']+=float(loss.detach())
            progress(phase='training',fit=fit_id,depth=depth,updates=state['updates'],maximum=maximum,
                pending=state['pending'],active_seconds=base_seconds+time.monotonic()-started)
        if state['selection'].get('last_validated')!=state['updates']:validate()
        state['completed']=True;state['stop_reason']='source_early_stop' if state['selection'].get('stop') else 'maximum_updates'
        save()
        best=torch.load(output/'best.pt',map_location='cpu',weights_only=False)
        if best['binding']!=binding:raise ValueError('Best checkpoint binding changed')
        atomic_json(output/'selection.json',dict(binding=binding,fit_id=fit_id,excluded_group=held,depth=depth,
            checkpoint_sha256=sha(output/'best.pt'),model_sha256=best['model_sha256'],best_update=best['update'],
            stopped_update=state['updates'],stop_reason=state['stop_reason'],source_metrics=best['source_metrics'],
            active_seconds=state['active_seconds'],held_source_used_for_checkpoint_selection=False,target_labels_used=False))
        engine.state_model.load_state_dict(best['model']);return best
    except BaseException:
        save();raise
