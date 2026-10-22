"""Raw-unit variable-equal metrics and paired spatial/time block uncertainty."""
from collections import defaultdict,deque
import hashlib

import numpy as np

MODELS=('native','enhanced','fusion','ufg')


def macro_metrics(prediction,target,observed,variables):
    p=np.asarray(prediction,np.float64);y=np.asarray(target,np.float64);mask=np.asarray(observed,bool)
    variables=np.asarray(variables)
    if p.shape!=y.shape or mask.shape!=y.shape or not np.isfinite(p).all() or not mask.any(1).all():
        raise ValueError('Complete finite predictions and a common nonempty query mask required')
    results={};step_results={};total={k:0. for k in ('absolute','squared','true_absolute','count','mape_count')}
    for variable in np.unique(variables):
        own=variables==variable;valid=mask[own];truth=y[own];error=np.abs(p[own]-truth)
        count=int(valid.sum());absolute=float(error[valid].sum());squared=float(np.square(error)[valid].sum())
        denom=float(np.abs(truth)[valid].sum());near=valid&(np.abs(truth)>1e-5)
        mape=None if not near.any() else float((error[near]/np.abs(truth[near])).mean())
        results[str(variable)]=dict(mae=absolute/count,rmse=float(np.sqrt(squared/count)),mape=mape,
            wape=None if denom==0 else absolute/denom,observed_points=count,mape_valid_points=int(near.sum()),
            mape_valid_fraction=float(near.sum()/count),queries=int(own.sum()))
        total['absolute']+=absolute;total['squared']+=squared;total['true_absolute']+=denom
        total['count']+=count;total['mape_count']+=int(near.sum())
        for h in range(p.shape[1]):
            v=valid[:,h]
            if v.any():step_results.setdefault(h,[]).append(float(error[:,h][v].mean()))
    means={key:(float(np.mean([v[key] for v in results.values() if v[key] is not None]))
        if any(v[key] is not None for v in results.values()) else None) for key in ('mae','rmse','mape','wape')}
    return dict(macro=means,variables=len(results),queries=len(p),observed_points=int(total['count']),
        mape_unit='fraction',wape_unit='fraction',rmse_aggregation='mean_of_variable_rmse',
        mape_valid_fraction=total['mape_count']/total['count'],mape_threshold=1e-5,
        macro_mape_variables=sum(v['mape'] is not None for v in results.values()),
        macro_wape_variables=sum(v['wape'] is not None for v in results.values()),
        micro_mae=total['absolute']/total['count'],micro_rmse=float(np.sqrt(total['squared']/total['count'])),
        micro_wape=None if total['true_absolute']==0 else total['absolute']/total['true_absolute'],
        per_variable=results,per_step_macro_mae={str(h+1):float(np.mean(v)) for h,v in step_results.items()})


def strict_improvement(actual,baseline):
    return bool(actual<baseline-max(1e-12,1e-9*baseline))


def spatial_blocks(adjacency,ids,maximum=16):
    """Fixed graph-local BFS blocks, independent of values/forecast errors.

    Undirected view is used only for uncertainty blocks; inference retains the
    original directed adjacency. No coordinates/value-dependent clustering fit.
    """
    a=np.asarray(adjacency)>0;a=a|a.T;remaining=set(range(len(a)));groups={};group=0
    while remaining:
        root=min(remaining,key=lambda i:str(ids[i]));queue=deque([root]);selected=[]
        while queue and len(selected)<maximum:
            node=queue.popleft()
            if node not in remaining:continue
            remaining.remove(node);selected.append(node);groups[node]=group
            neighbors=sorted((int(i) for i in np.flatnonzero(a[node]) if int(i) in remaining),key=lambda i:str(ids[i]))
            queue.extend(neighbors)
        group+=1
    return groups


def paired_block_intervals(predictions,target,observed,variables,spatial,time_blocks,seed=4259,repetitions=2000):
    """Resample spatial blocks and seven-day blocks jointly; keep pairing.

    Within each draw first recompute each variable's error over resampled time,
    then average variables with their spatial-block multiplicity. One spatial
    cluster cannot identify spatial uncertainty: also expose time-only intervals
    and mark that limitation instead of presenting a degenerate spatial result.
    """
    models=list(predictions);y=np.asarray(target,np.float64);mask=np.asarray(observed,bool)
    variables=np.asarray(variables);spatial=np.asarray(spatial);time_blocks=np.asarray(time_blocks)
    vs=np.unique(variables);ts=np.unique(time_blocks);ss=np.unique(spatial)
    vi={v:i for i,v in enumerate(vs)};ti={v:i for i,v in enumerate(ts)};si={v:i for i,v in enumerate(ss)}
    error=np.zeros((len(models),len(vs),len(ts)));count=np.zeros((len(vs),len(ts)))
    vspace=np.zeros(len(vs),int)
    for i,(v,t,s) in enumerate(zip(variables,time_blocks,spatial)):
        a,b=vi[v],ti[t];vspace[a]=si[s];count[a,b]+=mask[i].sum()
        for j,name in enumerate(models):error[j,a,b]+=np.abs(np.asarray(predictions[name])[i]-y[i])[mask[i]].sum()
    rng=np.random.default_rng(seed)
    time_weight=rng.multinomial(len(ts),np.full(len(ts),1/len(ts)),size=repetitions)
    space_weight=rng.multinomial(len(ss),np.full(len(ss),1/len(ss)),size=repetitions)
    point_counts=time_weight@count.T
    weighted_error=np.einsum('mvt,bt->bmv',error,time_weight)
    variable_error=np.divide(weighted_error,point_counts[:,None],out=np.zeros_like(weighted_error),where=point_counts[:,None]>0)
    available=point_counts>0
    def aggregate(weights):
        weights=weights*available;denom=weights.sum(1)
        return np.divide((variable_error*weights[:,None]).sum(2),denom[:,None],out=np.full((repetitions,len(models)),np.nan),where=denom[:,None]>0)
    both=aggregate(space_weight[:,vspace]);time_only=aggregate(np.ones((repetitions,len(vs))))
    result={}
    for name,values in (('space_time',both),('time_only',time_only)):
        comparisons={}
        for baseline in ('native','ufg'):
            current=values[:,models.index('fusion')];base=values[:,models.index(baseline)]
            valid=np.isfinite(current)&np.isfinite(base)
            difference=current[valid]-base[valid]
            ratio=np.divide(current[valid],base[valid],out=np.full(valid.sum(),np.nan),where=base[valid]>0)-1
            ratio=ratio[np.isfinite(ratio)]
            comparisons[baseline]=dict(mae_difference_95=np.quantile(difference,[.025,.975]).tolist() if len(difference) else None,
                relative_change_95=np.quantile(ratio,[.025,.975]).tolist() if len(ratio) else None,
                valid_draws=int(valid.sum()))
        result[name]=None if name=='space_time' and len(ss)<2 else comparisons
    return dict(seed=seed,repetitions=repetitions,spatial_blocks=len(ss),time_blocks=len(ts),variables=len(vs),
        insufficient_spatial_clusters=len(ss)<2,few_time_blocks=len(ts)<5,intervals=result,
        inference_adjacency_unchanged=True,point_estimate_is_not_a_general_stability_proof=True)


def compare_dataset(records,predictions,spatial_by_variable,time_width_ns=7*24*3600*10**9,seed=4259):
    if set(predictions)!=set(MODELS):raise ValueError('All four model groups are required')
    queries=[r['query'] for r in records];variables=[q['variable'] for q in queries]
    target=np.asarray([r['target'] for r in records],np.float64);observed=np.asarray([r['observed'] for r in records],bool)
    times=np.asarray([r['origin_time_ns'] for r in records],np.int64)
    metrics={name:macro_metrics(prediction,target,observed,variables) for name,prediction in predictions.items()}
    improvement={}
    fusion=np.asarray(predictions['fusion']);actual=metrics['fusion']['macro']['mae']
    for baseline in ('native','ufg'):
        base=np.asarray(predictions[baseline]);mae=metrics[baseline]['macro']['mae']
        query_delta=(np.abs(fusion-target)*observed).sum(1)/observed.sum(1)-(np.abs(base-target)*observed).sum(1)/observed.sum(1)
        differences=[v['mae']-metrics[baseline]['per_variable'][k]['mae'] for k,v in metrics['fusion']['per_variable'].items()]
        improvement[baseline]=dict(strictly_improved=strict_improvement(actual,mae),mae_ratio=actual/mae if mae>0 else None,
            paired_queries=dict(improved=int((query_delta<0).sum()),tied=int((query_delta==0).sum()),degraded=int((query_delta>0).sum())),
            paired_variables=dict(improved=sum(d<0 for d in differences),tied=sum(d==0 for d in differences),degraded=sum(d>0 for d in differences)))
    uncertainty=paired_block_intervals(predictions,target,observed,variables,
        [spatial_by_variable[v] for v in variables],times//time_width_ns,seed=seed)
    return dict(metrics=metrics,comparison=improvement,passes_both=all(v['strictly_improved'] for v in improvement.values()),
        uncertainty=uncertainty,raw_units_are_not_pooled_across_datasets=True)
