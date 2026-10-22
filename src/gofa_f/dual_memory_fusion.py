"""One shared source-cross-fitted reliability gate. No dataset-ID parameters."""
from collections import defaultdict
import numpy as np
import torch

from gofa_f.dual_memory_data import GROUPS

FEATURES=('horizon_fraction','history_fraction','target_missing','neighbor_coverage',
          'normalized_history_std','signed_expert_disagreement','absolute_expert_disagreement','relative_step')


def features(item,native,enhanced):
    n=np.asarray(native,np.float64);g=np.asarray(enhanced,np.float64)
    if n.shape!=g.shape or n.ndim!=1 or not np.isfinite(n).all() or not np.isfinite(g).all():
        raise ValueError('Two finite raw-unit forecasts required')
    mask=item.observed[0].cpu().numpy();history=item.normalized[0].cpu().numpy()
    external=item.adjacency[0,1:].cpu().numpy()
    availability=float(item.present[1:].float().mean()) if len(item.present)>1 else 0.
    difference=np.arcsinh((g-n)/float(item.extent[0]))
    h=len(n)
    x=np.stack((np.full(h,h/24),np.full(h,item.history.shape[1]/48),np.full(h,1-mask.mean()),
        np.full(h,availability),np.full(h,history[mask].std() if mask.any() else 0.),difference,np.abs(difference),
        np.arange(1,h+1)/h),-1)
    # Hard fallback is prediction-time observable, not an error oracle.
    return x,bool(external.any())


def fit_gate(records,native_records,store,settings):
    native_by_key={r['query_key']:r for r in native_records}
    data=[];counts=defaultdict(int);variables=defaultdict(set);native_error=defaultdict(lambda:[0.,0])
    for r in records:
        q=r['query'];key=r['query_key'];n=native_by_key[key]
        if r['query']!=n['query'] or r['target']!=n['target'] or r['observed']!=n['observed']:
            raise ValueError('Cross-fitted Native/enhanced labels or queries differ')
        item,_,_=store.get(q,'development')
        x,available=features(item,n['prediction'],r['prediction'])
        mask=np.asarray(r['observed'],bool);y=np.asarray(r['target']);npred=np.asarray(n['prediction'])
        source=q['source'];variable=q['variable'];counts[(source,variable)]+=int(mask.sum());variables[source].add(variable)
        native_error[(source,variable)][0]+=float(np.abs(npred-y)[mask].sum());native_error[(source,variable)][1]+=int(mask.sum())
        data.append((q,x,available,npred,np.asarray(r['prediction']),y,mask))
    # Source weighting only; no source identity enters the shared gate features.
    arrays=[]
    for q,x,available,n,g,y,mask in data:
        source=q['source'];domain=GROUPS[source]
        unit=max(float(np.mean([native_error[(source,v)][0]/native_error[(source,v)][1] for v in variables[source]])),1e-12)
        weight=1/(3*(2 if domain=='traffic' else 1)*len(variables[source])*counts[(source,q['variable'])]*unit)
        arrays.append((x[mask],np.full(mask.sum(),available),n[mask],g[mask],y[mask],np.full(mask.sum(),weight)))
    x,available,n,g,y,weight=[torch.from_numpy(np.concatenate([a[i] for a in arrays])).double() for i in range(6)]
    parameter=torch.zeros(9,dtype=torch.float64,requires_grad=True)
    with torch.no_grad():parameter[-1]=settings['bias_initial']
    optimizer=torch.optim.Adam([parameter],lr=settings['learning_rate']);curve=[]
    for step in range(settings['steps']):
        optimizer.zero_grad();alpha=(x@parameter[:-1]+parameter[-1]).sigmoid()*available
        prediction=n+alpha*(g-n)
        loss=((prediction-y).abs()*weight).sum()+settings['penalty']*alpha.mean()
        loss.backward();optimizer.step()
        if not parameter.isfinite().all():raise FloatingPointError('Nonfinite shared gate')
        if step%100==0 or step+1==settings['steps']:curve.append(dict(step=step+1,loss=float(loss.detach())))
    return dict(parameters=parameter.detach().tolist(),features=list(FEATURES),curve=curve,
        queries=len(records),target_labels_used=False,source_predictions_cross_fitted=True,
        per_dataset_parameters=False,hard_fallback='no observed direct external neighbor',configuration=settings)


def blend(gate,item,native,enhanced):
    x,available=features(item,native,enhanced)
    if not available:return np.asarray(native,np.float64).copy(),np.zeros(len(native))
    p=np.asarray(gate['parameters'],np.float64);z=x@p[:-1]+p[-1]
    alpha=np.exp(-np.logaddexp(0.,-z))
    n=np.asarray(native,np.float64);g=np.asarray(enhanced,np.float64)
    return n+alpha*(g-n),alpha
