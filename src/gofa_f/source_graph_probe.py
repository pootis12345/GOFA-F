"""Independently trained source graph controls, with gated repeats/confirmation.

No target evaluation, pretrained GNN initialization, or ChatTime training entry.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import time

import numpy as np
import torch
import yaml

from gofa_f.generative_data import Series, sha
from gofa_f.graph_probe_data import (CONDITIONS, VIEWS, component_partition, controls,
    evaluation_queries, pack, select_training)
from gofa_f.lag_graph_probe import LagGraphProbe, state_digest
from gofa_f.prefix_cache import atomic_json

PROTOCOL = 'stage36-source-graph-probe-v1'
CODE_FILES = ('source_graph_probe.py','graph_probe_data.py','lag_graph_probe.py','generative_data.py')
STOP = [False]


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def immutable(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError('Immutable probe evidence differs: '+path.name)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, value)


def save_torch(path, value):
    part = Path(path).with_suffix('.part')
    torch.save(value, part)
    part.replace(path)


def configuration(path):
    text = Path(path).read_text()
    if '${GOFA_WORKSPACE}' in text:
        workspace = os.environ.get('GOFA_WORKSPACE')
        if not workspace:
            raise ValueError('Explicit GOFA_WORKSPACE required')
        text = text.replace('${GOFA_WORKSPACE}', workspace.replace('\\','/'))
    cfg = yaml.safe_load(text)
    expected_training = dict(queries_per_view=2048,max_updates=1000,effective_batch=64,
        learning_rate=.001,warmup_updates=20,weight_decay=.0001,gradient_clip=1.,
        validation_every=100,profile_updates=20)
    if cfg['protocol'] != PROTOCOL or cfg['seeds'] != [42,43,44] \
            or cfg['conditions'] != list(CONDITIONS) \
            or cfg['model'] != dict(width=32,rank=8,lags=[0,1,2,3],maximum_nodes=9,graph_blocks=2) \
            or cfg['training'] != expected_training \
            or cfg['gate'] != dict(minimum_view_improvement=.01,
                maximum_time_block_degradation=.02,paired_bootstrap_draws=2000) \
            or cfg['policy'] != dict(source_training_enabled=True,target_training_enabled=False,
                target_evaluation_enabled=False,neural_chattime_training_enabled=False,
                gpu_enabled=False,hourly_followup=False):
        raise ValueError('Fixed single-plan source probe configuration changed')
    return cfg


def code_hashes():
    return {name: sha(Path(__file__).parent/name) for name in CODE_FILES}


def source_series(m):
    result = {}
    for name in VIEWS:
        path = Path(m['source_directory'])/(name+'.npz')
        if sha(path) != m['source_hashes'][name] or sha(path.with_suffix('.json')) != m['metadata_hashes'][name]:
            raise ValueError('Audited source observations/topology changed')
        result[name] = Series(path)
    if result[VIEWS[0]].ids != result[VIEWS[1]].ids \
            or not np.array_equal(result[VIEWS[0]].external, result[VIEWS[1]].external):
        raise ValueError('Traffic views must share exact topology and node mapping')
    return result


def prepare(cfg, output):
    parent, prior = Path(cfg['source_output']), Path(cfg['prior_screen_output'])
    if sha(parent/'manifest.json') != cfg['source_manifest_sha256'] \
            or sha(prior/'manifest.json') != cfg['prior_screen_manifest_sha256']:
        raise ValueError('Locked source/query lineage changed')
    sm, old = read(parent/'manifest.json'), read(prior/'manifest.json')
    m = dict(configuration=cfg,code=code_hashes(),
        source_directory=str(Path(sm['configuration']['data']['directory'])/'sources'),
        source_hashes={v:sm['source_hashes'][v] for v in VIEWS},
        metadata_hashes={v:sm['metadata_hashes'][v] for v in VIEWS})
    series = source_series(m)
    partition = component_partition(series[VIEWS[0]].external, series[VIEWS[0]].ids)
    if tuple(len(partition[k]) for k in ('train','validation','confirmation')) != (257,18,32):
        raise ValueError('Pinned PEMS04 disconnected-region design changed')
    excluded = sm['pools']['confirmation']+sm['pools']['diagnostic'] \
        +[q for rows in old['queries'].values() for q in rows]
    queries = dict(train={name:select_training(sm['pools']['train'][name],partition['train'],
        cfg['training']['queries_per_view'],i) for i,name in enumerate(VIEWS)})
    for split in ('validation','confirmation'):
        queries[split] = evaluation_queries(series,partition[split],split,excluded)
    m.update(partition=partition,queries=queries,exclusion_sha256=digest(excluded),
        target_data_opened=False,pretrained_probe_weights_used=False,
        whole_new_traffic_dataset_test=False,source_region_identity_and_time_held_out=True,
        prior_source_development_exposure=True)
    output.mkdir(parents=True,exist_ok=True)
    immutable(output/'manifest.json',m)
    if not (output/'status.json').exists():
        atomic_json(output/'status.json',dict(state='prepared',stage=36,gpu_seconds=0.,hourly_followup=False))
    return m


def verify(cfg, output):
    m = read(output/'manifest.json')
    if not m or m['configuration'] != cfg or m['code'] != code_hashes():
        raise ValueError('Prepared probe configuration/code binding differs')
    return m


def cache_path(output, split, name):
    if split not in ('train','validation','confirmation') or name not in VIEWS:
        raise ValueError('Only designated source caches are permitted')
    return output/'cache'/f'{split}-{name}.npz'


def caches(output,m,charge):
    data = source_series(m)
    result = {s:{} for s in ('train','validation','confirmation')}
    for split in result:
        for name in VIEWS:
            path = cache_path(output,split,name)
            identity = digest(dict(manifest=sha(output/'manifest.json'),split=split,view=name))
            if path.exists():
                proof = read(path.with_suffix('.json'))
                if not proof or proof['binding'] != identity or proof['sha256'] != sha(path):
                    raise ValueError('Immutable source cache changed')
            else:
                rows=[]
                for i,r in enumerate(m['queries'][split][name]):
                    rows.append(pack(data[name],r,m['partition'][split],
                        labels=split!='confirmation',split=split))
                    if i%256==0:
                        charge()
                        atomic_json(output/'progress.json',dict(state='running',phase='causal_cache',
                            split=split,view=name,completed=i,total=len(m['queries'][split][name])))
                arrays={key:np.stack([r[key] for r in rows]) for key in rows[0]}
                path.parent.mkdir(parents=True,exist_ok=True)
                temporary=path.with_suffix('.part')
                with temporary.open('wb') as stream:
                    np.savez_compressed(stream,**arrays)
                temporary.replace(path)
                immutable(path.with_suffix('.json'),dict(binding=identity,sha256=sha(path),
                    queries=len(rows),labels_present=split!='confirmation'))
            with np.load(path,allow_pickle=False) as stream:
                result[split][name]={key:stream[key] for key in stream.files}
            a=result[split][name]
            if split=='confirmation' and any(k in a for k in ('y','raw_target','observed')):
                raise ValueError('Confirmation labels must not be in prepared input cache')
            used=set(a['nodes'][a['nodes']>=0].tolist())
            if not used <= set(m['partition'][split]):
                raise ValueError('Held-out source node entered another partition')
            charge()
    return result


def train_weights(a,records):
    totals={}
    for i,r in enumerate(records):
        v=r['query']['variable'];totals[v]=totals.get(v,0)+int(a['observed'][i].sum())
    weights=np.array([len(records)/(len(totals)*totals[r['query']['variable']]) for r in records],np.float32)
    persistence=np.broadcast_to(a['persistence'][:,None],a['raw_target'].shape)
    per_variable={}
    for i,r in enumerate(records):
        e=np.abs(persistence[i]-a['raw_target'][i])[a['observed'][i]]
        per_variable.setdefault(r['query']['variable'],[]).extend(e.tolist())
    norm=max(float(np.mean([np.mean(x) for x in per_variable.values()])),1e-12)
    return weights,norm


def tensors(a,indices,records,condition,weight=None,norm=1.):
    selected=[records[int(i)] for i in indices]
    adjacency=controls(a['adjacency'][indices],a['nodes'][indices],selected,condition)
    result=dict(x=torch.from_numpy(a['x'][indices]),adjacency=torch.from_numpy(adjacency),
        edge_features=torch.from_numpy(a['edge_features'][indices]),
        scale=torch.from_numpy(a['scale'][indices].astype(np.float32)))
    if weight is not None:
        result.update(observed=torch.from_numpy(a['observed'][indices]),
            delta=torch.from_numpy((a['raw_target'][indices]-a['persistence'][indices,None]).astype(np.float32)),
            weight=torch.from_numpy(weight[indices]/norm))
    return result


def loss_value(residual,b):
    error=(residual*b['scale'][:,None]-b['delta']).abs()*b['observed']
    if not b['observed'].any(1).all():
        raise ValueError('Empty source training label window')
    return (error.sum(1)*b['weight']).mean()


def metrics(prediction,a,records):
    variables={}
    steps=[[] for _ in range(prediction.shape[1])]
    for variable in sorted({r['query']['variable'] for r in records}):
        ix=np.array([i for i,r in enumerate(records) if r['query']['variable']==variable])
        valid=a['observed'][ix]
        error=prediction[ix]-a['raw_target'][ix]
        own=error[valid]
        if not len(own):raise ValueError('Empty source metric variable')
        variables[str(variable)]=dict(mae=float(np.abs(own).mean()),rmse=float(np.sqrt(np.square(own).mean())),
            valid_points=len(own),queries=len(ix))
        for h in range(prediction.shape[1]):
            if valid[:,h].any():steps[h].append(float(np.abs(error[:,h])[valid[:,h]].mean()))
    blocks={}
    for block in sorted({r['block'] for r in records}):
        ix=np.array([i for i,r in enumerate(records) if r['block']==block])
        own=[]
        for v in sorted({records[i]['query']['variable'] for i in ix}):
            sub=np.array([i for i in ix if records[i]['query']['variable']==v])
            own.append(float(np.abs(prediction[sub]-a['raw_target'][sub])[a['observed'][sub]].mean()))
        blocks[str(block)]=float(np.mean(own))
    return dict(macro_mae=float(np.mean([v['mae'] for v in variables.values()])),
        macro_rmse=float(np.mean([v['rmse'] for v in variables.values()])),variables=variables,
        time_blocks=blocks,step_macro_mae=[float(np.mean(v)) if v else None for v in steps])


@torch.no_grad()
def forecast(model,a,records,condition):
    model.eval()
    result=[]
    for start in range(0,len(records),64):
        indices=np.arange(start,min(start+64,len(records)))
        b=tensors(a,indices,records,condition)
        delta=model(b['x'],b['adjacency'],b['edge_features'],12,return_residual=True)[:,0].double().numpy()
        prediction=a['persistence'][indices,None]+delta*a['scale'][indices,None]
        if not np.isfinite(prediction).all():raise ValueError('Nonfinite probe forecast')
        result.append(prediction)
    return np.concatenate(result)


def model_for(cfg,seed):
    torch.manual_seed(seed)
    return LagGraphProbe(cfg['model']['width'],cfg['model']['rank'],tuple(cfg['model']['lags']))


def profile(cfg,output):
    if (output/'speed_profile.json').exists():return read(output/'speed_profile.json')
    model=model_for(cfg,4236)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    x=torch.randn(64,9,12,2);x[...,1]=1
    adjacency=torch.rand(64,9,9)>.7
    edge=torch.randn(64,9,9,2)
    start=time.monotonic()
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        y=model(x,adjacency,edge,return_residual=True)[:,0]
        loss=y.square().mean()+.1*(y-1).abs().mean()
        loss.backward();optimizer.step()
    seconds=(time.monotonic()-start)/20
    infer=time.monotonic()
    with torch.no_grad():model(x,adjacency,edge,return_residual=True)
    value=dict(synthetic_updates=20,discarded_state=True,seconds_per_update=seconds,
        seconds_per_64_forecasts=time.monotonic()-infer,screen_training_eta_seconds=3000*seconds,
        conditional_nine_model_training_eta_seconds=9000*seconds,gpu_seconds=0.)
    immutable(output/'speed_profile.json',value)
    print(json.dumps(dict(phase='profile',**value)),flush=True)
    return value


def next_indices(state,name,count,rng):
    own=state[name];result=[]
    while len(result)<count:
        take=min(count-len(result),len(own['order'])-own['position'])
        result.extend(own['order'][own['position']:own['position']+take]);own['position']+=take
        if own['position']==len(own['order']):
            own['order']=rng.permutation(len(own['order'])).tolist();own['position']=0;own['epochs']+=1
    return np.asarray(result,np.int64)


def fit(cfg,output,m,packed,seed,condition,charge,speed):
    folder=output/'models'/f'seed-{seed}'/condition
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'selection.json').exists():return read(folder/'selection.json')
    binding=digest(dict(manifest=sha(output/'manifest.json'),seed=seed,condition=condition))
    model=model_for(cfg,seed)
    initial_sha=state_digest(model)
    immutable(output/'initializations'/f'{seed}-{condition}.json',dict(seed=seed,parameter_sha256=initial_sha,
        parameters=sum(p.numel() for p in model.parameters())))
    for other in CONDITIONS:
        old=read(output/'initializations'/f'{seed}-{other}.json')
        if old and old['parameter_sha256']!=initial_sha:raise ValueError('Matched initialization differs')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    rng=np.random.default_rng(np.random.SeedSequence([seed,36,100]))
    cursor={name:dict(order=rng.permutation(len(packed['train'][name]['x'])).tolist(),position=0,epochs=0)
            for name in VIEWS}
    weights={name:train_weights(packed['train'][name],m['queries']['train'][name]) for name in VIEWS}
    baselines={name:metrics(np.broadcast_to(packed['validation'][name]['persistence'][:,None],
        packed['validation'][name]['raw_target'].shape),packed['validation'][name],m['queries']['validation'][name])['macro_mae']
        for name in VIEWS}
    update,best_score,best_update=0,None,None
    if (folder/'last.pt').exists():
        saved=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        if saved['binding']!=binding or saved['initial_sha256']!=initial_sha:raise ValueError('Resume binding differs')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state=saved['rng'];torch.set_rng_state(saved['torch_rng'])
        cursor=saved['cursor'];update=saved['update'];best_score=saved['best_score'];best_update=saved['best_update']
    def checkpoint():
        save_torch(folder/'last.pt',dict(binding=binding,initial_sha256=initial_sha,model=model.state_dict(),
            optimizer=optimizer.state_dict(),rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
            cursor=cursor,update=update,best_score=best_score,best_update=best_update))
    def validate():
        nonlocal best_score,best_update
        results={name:metrics(forecast(model,packed['validation'][name],m['queries']['validation'][name],condition),
            packed['validation'][name],m['queries']['validation'][name]) for name in VIEWS}
        score=float(np.exp(np.mean([np.log(max(results[name]['macro_mae'],1e-12)/max(baselines[name],1e-12)) for name in VIEWS])))
        value=dict(update=update,score=score,views=results,binding=binding)
        immutable(folder/'validations'/f'{update:06d}.json',value)
        if best_score is None or score<best_score:
            best_score,best_update=score,update
            save_torch(folder/'best.pt',dict(binding=binding,model=model.state_dict(),update=update,score=score))
            atomic_json(folder/'best_validation.json',value)
        charge()
    if best_score is None:
        validate();checkpoint()
    while update<cfg['training']['max_updates']:
        if STOP[0]:checkpoint();return None
        model.train();optimizer.zero_grad(set_to_none=True)
        batches=[]
        for name in VIEWS:
            ix=next_indices(cursor,name,32,rng)
            weight,norm=weights[name]
            batches.append(tensors(packed['train'][name],ix,m['queries']['train'][name],condition,weight,norm))
        batch={k:torch.cat([b[k] for b in batches]) for k in batches[0]}
        prediction=model(batch['x'],batch['adjacency'],batch['edge_features'],12,return_residual=True)[:,0]
        loss=loss_value(prediction,batch)
        if not loss.isfinite():raise ValueError('Nonfinite matched-source loss')
        loss.backward()
        if any(p.grad is not None and not p.grad.isfinite().all() for p in model.parameters()):
            raise ValueError('Nonfinite source gradient')
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        for group in optimizer.param_groups:group['lr']=.001*min(1.,(update+1)/20)
        optimizer.step();update+=1
        if update%100==0:validate()
        if update%50==0 or update==1:
            checkpoint();charge()
            point=dict(update=update,loss=float(loss.detach()),gradient_norm=float(norm),
                epochs={k:v['epochs']+v['position']/len(v['order']) for k,v in cursor.items()})
            immutable(folder/'training_points'/f'{update:06d}.json',point)
            atomic_json(folder/'training_progress.json',point)
            atomic_json(output/'progress.json',dict(state='running',phase='matched_source_training',seed=seed,
                condition=condition,updates=update,max_updates=1000,best_update=best_update,best_score=best_score,
                current_model_eta_seconds=(1000-update)*speed['seconds_per_update'],
                future_seeds_conditional=seed==42,gpu_seconds=0.))
            print(json.dumps(dict(seed=seed,condition=condition,update=update,loss=float(loss.detach()),best_score=best_score)),flush=True)
    value=read(folder/'best_validation.json')
    selected=model_for(cfg,seed)
    selected.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=False)['model'])
    selection=dict(binding=binding,seed=seed,condition=condition,completed_updates=update,
        selected_update=best_update,score=best_score,initial_sha256=initial_sha,
        checkpoint_sha256=sha(folder/'best.pt'),tensor_sha256=state_digest(selected),validation=value['views'])
    immutable(folder/'selection.json',selection)
    return selection


def gate(selections,cfg,repeated=False):
    ratios,issues={},[]
    seeds=sorted(selections)
    for name in VIEWS:
        ratios[name]={}
        for control in ('self','rewired'):
            real=np.array([selections[s]['physical']['validation'][name]['macro_mae'] for s in seeds])
            base=np.array([selections[s][control]['validation'][name]['macro_mae'] for s in seeds])
            ratio=float(real.mean()/base.mean()) if base.mean()>0 else (1. if real.mean()==0 else 1e30)
            quarter_ratios=[]
            for block in ('0','1','2','3'):
                a=np.mean([selections[s]['physical']['validation'][name]['time_blocks'][block] for s in seeds])
                b=np.mean([selections[s][control]['validation'][name]['time_blocks'][block] for s in seeds])
                quarter_ratios.append(float(a/b) if b>0 else (1. if a==0 else 1e30))
            result=dict(physical_to_control_macro_mae=ratio,per_seed_ratios=(real/np.maximum(base,1e-30)).tolist(),
                time_block_ratios=quarter_ratios,zero_error_control=bool(base.mean()==0))
            if ratio>1-cfg['gate']['minimum_view_improvement']:issues.append(name+':'+control+':less_than_1pct_gain')
            if max(quarter_ratios)>1+cfg['gate']['maximum_time_block_degradation']:
                issues.append(name+':'+control+':time_block_regression')
            if repeated:
                if not (real<base).all():issues.append(name+':'+control+':seed_direction_not_repeated')
                variables=sorted(selections[seeds[0]]['physical']['validation'][name]['variables'])
                a=np.array([[selections[s]['physical']['validation'][name]['variables'][v]['mae'] for v in variables] for s in seeds])
                b=np.array([[selections[s][control]['validation'][name]['variables'][v]['mae'] for v in variables] for s in seeds])
                rng=np.random.default_rng(4236)
                ix=rng.integers(len(variables),size=(2000,len(variables)))
                draws=a[:,ix].mean((0,2))/np.maximum(b[:,ix].mean((0,2)),1e-30)
                result['paired_node_cluster_ratio_95pct']=np.quantile(draws,[.025,.975]).tolist()
                if result['paired_node_cluster_ratio_95pct'][1]>=1:issues.append(name+':'+control+':interval_crosses_1')
            ratios[name][control]=result
    return dict(passed=not issues,seeds=seeds,comparisons=ratios,reasons=issues,
        independent_unseen_dataset_transfer_proven=False)


def confirmation(cfg,output,m,packed,selections,charge):
    repeat=read(output/'repeated_validation_gate.json')
    lock=read(output/'source_selection.json')
    if not repeat or not repeat['passed'] or not lock:
        raise ValueError('Repeated validation and locked checkpoints required before confirmation labels')
    if lock['manifest_sha256']!=sha(output/'manifest.json'):
        raise ValueError('Locked confirmation manifest differs')
    for seed in cfg['seeds']:
        for condition in CONDITIONS:
            path=output/'models'/f'seed-{seed}'/condition/'best.pt'
            if sha(path)!=lock['checkpoints'][str(seed)][condition]:raise ValueError('Locked model changed')
    data=source_series(m)
    arrays=copy.deepcopy(packed['confirmation'])
    for name in VIEWS:
        rows=m['queries']['confirmation'][name];d=data[name]
        target=np.stack([d.values[r['query']['origin']:r['query']['origin']+12,r['query']['variable']] for r in rows]).astype(np.float64)
        observed=np.stack([d.mask[r['query']['origin']:r['query']['origin']+12,r['query']['variable']] for r in rows]).astype(bool)&np.isfinite(target)
        arrays[name].update(raw_target=np.where(observed,target,0),observed=observed)
    results={}
    for seed in cfg['seeds']:
        results[seed]={}
        for condition in CONDITIONS:
            model=model_for(cfg,seed)
            folder=output/'models'/f'seed-{seed}'/condition
            saved=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(saved['model'])
            view_results={}
            for name in VIEWS:
                prediction=forecast(model,arrays[name],m['queries']['confirmation'][name],condition)
                view_results[name]=metrics(prediction,arrays[name],m['queries']['confirmation'][name])
                immutable(output/'confirmation'/f'{seed}-{condition}-{name}.json',dict(
                    checkpoint_sha256=sha(folder/'best.pt'),queries=m['queries']['confirmation'][name],
                    predictions=prediction.tolist(),observed=arrays[name]['observed'].tolist(),
                    target=arrays[name]['raw_target'].tolist(),metrics=view_results[name]))
                if condition=='physical':
                    wrong=forecast(model,arrays[name],m['queries']['confirmation'][name],'rewired')
                    immutable(output/'confirmation'/f'{seed}-physical-rewired-inference-{name}.json',
                        dict(metrics=metrics(wrong,arrays[name],m['queries']['confirmation'][name]),
                             diagnostic_only=True,does_not_replace_independently_trained_control=True))
                charge()
            results[seed][condition]=dict(validation=view_results)
    value=gate(results,cfg,repeated=True)
    immutable(output/'source_confirmation.json',value)
    return value


def run(cfg,output,charge):
    m=verify(cfg,output)
    proof=read(output/'cpu_preflight.json')
    if not proof or not proof['passed'] or proof['manifest_sha256']!=sha(output/'manifest.json'):
        raise ValueError('Bound full CUDA-disabled CPU regression required')
    if (output/'report.json').exists():return read(output/'report.json')
    packed=caches(output,m,charge)
    speed=profile(cfg,output);charge()
    selections={}
    initial=None;repeat=None;confirmed=None
    for seed in cfg['seeds']:
        selections[seed]={}
        for condition in CONDITIONS:
            selection=fit(cfg,output,m,packed,seed,condition,charge,speed)
            if selection is None or STOP[0]:
                atomic_json(output/'status.json',dict(state='paused',stage=36,gpu_seconds=0.))
                return None
            selections[seed][condition]=selection
        if seed==42:
            initial=gate(selections,cfg)
            immutable(output/'initial_validation_gate.json',initial)
            if not initial['passed']:break
    if initial['passed']:
        repeat=gate(selections,cfg,repeated=True)
        immutable(output/'repeated_validation_gate.json',repeat)
        if repeat['passed']:
            lock=dict(manifest_sha256=sha(output/'manifest.json'),
                checkpoints={str(s):{c:selections[s][c]['checkpoint_sha256'] for c in CONDITIONS} for s in selections})
            immutable(output/'source_selection.json',lock)
            confirmed=confirmation(cfg,output,m,packed,selections,charge)
    phase='initial_source_screen' if not initial['passed'] else 'repeat_source_screen' \
        if not repeat['passed'] else 'source_confirmation'
    passed=bool(confirmed and confirmed['passed'])
    report=dict(complete=True,stage=36,protocol=PROTOCOL,passed=passed,stop_phase=phase,
        initial_validation=initial,repeated_validation=repeat,source_confirmation=confirmed,
        selections={str(k):v for k,v in selections.items()},source_partition=m['partition'],
        profiling=speed,parameters=sum(p.numel() for p in model_for(cfg,42).parameters()),
        gpu_seconds=0.,target_forecasts=0,chattime_updates=0,hourly_followup=False,
        skipped=dict(extra_seeds=not initial['passed'],confirmation=confirmed is None,
            whole_source_dataset_transfer=True,chattime_pilot=True,ufg_comparison=True,six_target_evaluation=True),
        next_action='whole_source_graph_check_then_separately_registered_chattime_pilot' if passed
                    else 'seal_this_fixed_probe_without_llm_training',
        limitations=['single traffic dataset; disconnected source-region transfer only',
            'fixed small-model optimization budget does not prove convergence or graph impossibility',
            'graph-dependent roster shared across controls; sampler contribution not isolated',
            'prior source development exposure; not an independent blind source benchmark',
            'continuous diagnostic readout is not a numerical head attached to ChatTime'])
    immutable(output/'report.json',report)
    atomic_json(output/'status.json',dict(state='source_probe_passed' if passed else 'stopped_at_source_gate',
        stage=36,phase=phase,gpu_seconds=0.,target_forecasts=0,chattime_updates=0,hourly_followup=False))
    atomic_json(output/'progress.json',dict(state='complete',phase=phase,eta_seconds=None,
        conditional_stages_skipped=report['skipped']))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--mode',required=True,choices=('prepare','run','progress'))
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():
        raise ValueError('Explicit CUDA-disabled CPU execution required')
    torch.set_num_threads(2)
    cfg=configuration(args.config)
    args.output.mkdir(parents=True,exist_ok=True)
    if args.mode=='progress':
        for n in ('status.json','progress.json','activity.json'):
            if (args.output/n).exists():print(json.dumps({n:read(args.output/n)}))
        return
    import fcntl
    with (args.output/'run.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.mode=='prepare':prepare(cfg,args.output);return
        for signum in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):
            signal.signal(signum,lambda *_:STOP.__setitem__(0,True))
        path=args.output/'activity.json'
        activity=read(path,dict(cpu_active_wall_seconds=0.,gpu_seconds=0.))
        anchor=[time.monotonic()]
        def charge():
            now=time.monotonic();activity['cpu_active_wall_seconds']+=now-anchor[0];anchor[0]=now
            if torch.cuda.is_initialized():raise RuntimeError('Unexpected CUDA initialization')
            tick=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]
            activity.update(worker=dict(pid=os.getpid(),start_ticks=tick),gpu_seconds=0.,cuda_initialized=False,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
            atomic_json(path,activity)
        charge()
        try:
            value=run(cfg,args.output,charge)
            print(json.dumps(dict(complete=value is not None,passed=value['passed'] if value else None)),flush=True)
        except Exception as exc:
            immutable(args.output/f'failure-{time.time_ns()}.json',dict(error=repr(exc)))
            atomic_json(args.output/'status.json',dict(state='failed',stage=36,error=repr(exc)))
            raise
        finally:
            charge();activity['worker']=None;atomic_json(path,activity)


if __name__=='__main__':main()
