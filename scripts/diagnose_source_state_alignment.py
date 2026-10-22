"""Fixed Stage42 same-weight source interventions; no training or model selection."""
import argparse
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch

from gofa_f import source_graph_probe as p
from gofa_f.source_normalized_probe import normalized_runtime
from gofa_f.generative_data import sha
from gofa_f.lag_graph_probe import state_digest

PARENT_REPORT='b9508f08f6e65c9e5b6f71c06fccdfe4ebeb7e3cec533fc12f92adef96a04602'
CONDITIONS=('full','off','aligned4','opposed4','fixed4','shuffled_past')


def state_vectors(x):
    states=np.zeros((len(x),2),np.float64)
    for j in range(len(x)):
        valid=np.flatnonzero(x[j,:,1]>0)
        if not len(valid):continue
        states[j,0]=x[j,valid[-1],0]
        recent=valid[valid>=x.shape[1]-6]
        if len(recent)>1:
            t=recent.astype(np.float64);y=x[j,recent,0].astype(np.float64)
            states[j,1]=float(np.dot(t-t.mean(),y-y.mean())/np.square(t-t.mean()).sum())*6
    return states


def support(a,mode):
    if mode not in ('aligned4','opposed4','fixed4'):raise ValueError('Fixed source intervention required')
    result=np.zeros_like(a['adjacency'])
    for i,(x,adjacency,ids) in enumerate(zip(a['x'],a['adjacency'],a['nodes'])):
        present=(x[:,:,1]>0).any(1)&(ids>=0);states=state_vectors(x)
        for receiver in range(len(ids)):
            if not present[receiver]:continue
            candidates=[int(j) for j in np.flatnonzero(adjacency[receiver]&present) if j!=receiver]
            if mode=='fixed4':key=lambda j:str(ids[j])
            elif mode=='aligned4':key=lambda j:(float(np.square(states[j]-states[receiver]).sum()),str(ids[j]))
            else:key=lambda j:(-float(np.square(states[j]-states[receiver]).sum()),str(ids[j]))
            selected=sorted(candidates,key=key)[:4]
            result[i,receiver,selected]=True
    return result


def shuffle_past(a,rows):
    x=a['x'].copy()
    for i,(values,ids,row) in enumerate(zip(x,a['nodes'],rows)):
        for j in range(1,len(ids)):
            valid=np.flatnonzero(values[j,:,1]>0)
            if ids[j]<0 or len(valid)<3:continue
            earlier=valid[:-1]
            q=row['query']
            rng=np.random.default_rng(np.random.SeedSequence([42,42,q['variable'],q['origin'],int(ids[j])]))
            old=values[j,earlier,0].copy();values[j,earlier,0]=old[rng.permutation(len(old))]
    return dict(a,x=x)


def selfcheck(a,rows):
    modes={c:support(a,c) for c in ('aligned4','opposed4','fixed4')}
    future=dict(a,raw_target=np.full_like(a['raw_target'],1e20),observed=~a['observed'])
    for c,value in modes.items():
        assert np.array_equal(value,support(future,c)), 'Future labels affected support'
        assert not (value&~a['adjacency']).any() and not np.diagonal(value,axis1=1,axis2=2).any()
        assert np.array_equal(value.sum(-1),modes['fixed4'].sum(-1))
    permutation=np.arange(a['x'].shape[1])[::-1]
    other=dict(a,x=a['x'][:,permutation],nodes=a['nodes'][:,permutation],
        adjacency=a['adjacency'][:,permutation][:,:,permutation])
    for c,value in modes.items():
        assert np.array_equal(support(other,c),value[:,permutation][:,:,permutation])
    shuffled=shuffle_past(a,rows)
    assert np.array_equal(shuffled['x'][:,0],a['x'][:,0])
    assert np.array_equal(shuffled['x'][...,1],a['x'][...,1])
    assert np.array_equal(shuffled['x'],shuffle_past(a,rows)['x'])
    for before,after in zip(a['x'].reshape(-1,12,2),shuffled['x'].reshape(-1,12,2)):
        valid=np.flatnonzero(before[:,1]>0)
        if len(valid):
            assert after[valid[-1],0]==before[valid[-1],0]
            assert np.array_equal(np.sort(after[valid,0]),np.sort(before[valid,0]))
    return dict(future_independence=True,node_permutation=True,equal_retained_edge_counts=True,
        target_history_mask_scale_and_last_values_preserved=True,deterministic_history_shuffle=True)


def main():
    start=time.monotonic()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():
        raise ValueError('Explicit CUDA-disabled CPU execution required')
    torch.set_num_threads(2)
    parent=args.workspace/'outputs/stage40-source-normalized-probe-v1'
    assert sha(parent/'report.json')==PARENT_REPORT
    assert p.read(parent/'final_verification.json')['passed'] and p.read(parent/'activity.json')['worker'] is None
    assert not (parent/'source_confirmation.json').exists()
    if args.output.exists():raise ValueError('Do not overwrite this one-shot diagnostic')
    args.output.mkdir(parents=True)
    m=p.read(parent/'manifest.json');report=p.read(parent/'report.json');cfg=m['configuration']
    cache_root=Path(cfg['parent_output'])
    assert sha(cache_root/'manifest.json')==m['parent_manifest_sha256']
    binding=dict(stage=42,parent_report_sha256=PARENT_REPORT,parent_manifest_sha256=sha(parent/'manifest.json'),
        script_sha256=sha(Path(__file__)),checkpoint_sha256=report['selections']['42']['physical']['checkpoint_sha256'],
        conditions=CONDITIONS,retained_neighbors=4,training_updates=0,target_forecasts=0,confirmation_opened=False)
    p.immutable(args.output/'manifest.json',binding)
    checks={};results={};evidence={}
    with normalized_runtime():
        assert m['code']==__import__('gofa_f.source_neighborhood_probe',fromlist=['code_hashes']).code_hashes()
        path=parent/'models/seed-42/physical/best.pt';assert sha(path)==binding['checkpoint_sha256']
        model=p.model_for(cfg,42);model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'])
        model.eval();before=state_digest(model)
        for view in p.VIEWS:
            path=cache_root/'cache/physical'/f'validation-{view}.npz'
            assert sha(path)==p.read(path.with_suffix('.json'))['sha256']
            with np.load(path,allow_pickle=False) as f:a={k:f[k] for k in f.files}
            rows=m['queries']['validation'][view];checks[view]=selfcheck(a,rows);results[view]={};evidence[view]={}
            baseline=None
            for condition in CONDITIONS:
                transformed=shuffle_past(a,rows) if condition=='shuffled_past' else dict(a)
                if condition=='off':transformed['adjacency']=np.zeros_like(a['adjacency'])
                elif condition in ('aligned4','opposed4','fixed4'):transformed['adjacency']=support(a,condition)
                prediction=p.forecast(model,transformed,rows,'physical')
                if condition=='full':
                    baseline=prediction
                    assert abs(p.metrics(prediction,a,rows)['macro_mae']-
                        report['selections']['42']['physical']['validation'][view]['macro_mae'])<=1e-6
                if condition=='off':assert np.array_equal(prediction,p.forecast(model,a,rows,'self'))
                results[view][condition]=dict(metrics=p.metrics(prediction,a,rows),
                    relative_macro_mae=float(p.metrics(prediction,a,rows)['macro_mae']/results[view]['full']['metrics']['macro_mae'])
                        if condition!='full' else 1.,mean_target_edges=float(transformed['adjacency'][:,0].sum(1).mean()))
                evidence[view][condition]=dict(predictions=prediction.tolist(),
                    input_sha256=p.digest(dict(x=transformed['x'].tolist(),adjacency=transformed['adjacency'].tolist())))
            assert np.array_equal(baseline,p.forecast(model,a,rows,'physical'))
        assert state_digest(model)==before and not torch.cuda.is_initialized()
    p.immutable(args.output/'interventions.json',evidence)
    value=dict(stage=42,complete=True,binding=binding,source_validation_interventions=results,acceptance=checks,
        parameters_unchanged=True,baseline_restored_exactly=True,elapsed_wall_seconds=time.monotonic()-start,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,gpu_seconds=0.,training_updates=0,
        target_forecasts=0,hourly_followup=False,limitations=[
            'same-weight interventions are distribution shifts, not independently trained controls',
            'state refers to normalized last-position and six-step trend, not verified physical congestion',
            'origin-frozen support uses pre-origin history; no claim of within-history prefix-causal construction',
            'single exposed source validation; no source confirmation, target or ChatTime conclusions'])
    p.immutable(args.output/'report.json',value)
    print(__import__('json').dumps(dict(seconds=value['elapsed_wall_seconds'],
        mae={v:{c:d['metrics']['macro_mae'] for c,d in x.items()} for v,x in results.items()})))


if __name__=='__main__':main()
