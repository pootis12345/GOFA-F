"""Stage32 finite dense wind training; raw-unit-aligned normalized L1 weights."""
from dataclasses import asdict

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

from gofa_f.autonomous_iteration_models import design
from gofa_f.generative_data import Query
from gofa_f.short_data import origins

RECIPES=tuple(f'{scale}_{family}_{graph}' for scale in ('raw','norm')
              for family in ('hgb','extra') for graph in ('local','neighbor'))


def features(data,query,recipe):
    if recipe not in RECIPES or query['source']!='sdwpf':
        raise ValueError('Fixed Stage32 wind recipe/source required')
    q=Query(**query)
    x,base,scale=design(data,query,'norm150' if recipe.startswith('norm') else 'season150')
    if recipe.endswith('neighbor'):
        if data.external is None:
            raise ValueError('Audited static wind coordinates graph required')
        neighbors=[j for j in np.flatnonzero(data.external[q.variable]>0) if j!=q.variable]
        neighbors=sorted(neighbors,key=lambda j:str(data.ids[j]))[:8]
        signals=[]
        for j in neighbors:
            values=np.asarray(data.values[q.origin-q.length:q.origin,j],np.float64)
            valid=np.asarray(data.mask[q.origin-q.length:q.origin,j],bool)&np.isfinite(values)
            ix=np.flatnonzero(valid)
            if not len(ix):
                continue
            recent=np.flatnonzero(valid[-12:])+q.length-12
            slope=0.
            if len(recent)>=2:
                t=recent.astype(np.float64)-recent.mean()
                slope=float(t@(values[recent]-values[recent].mean())/(t@t))
            signals.append([(values[ix[-1]]-base[0])/scale,slope*q.horizon/scale,
                values[ix].std(ddof=0)/scale,(values[ix].max()-values[ix].min())/scale,len(ix)/q.length])
        if signals:
            signals=np.asarray(signals)
            own=np.r_[signals.mean(0),signals.std(0),len(signals)/8]
        else:
            own=np.zeros(11)
        x=np.column_stack((x,np.repeat(own[None],q.horizon,axis=0))).astype(np.float32)
    if np.isinf(x).any():
        raise ValueError('Nonfinite causal wind features')
    return x,base,scale


def arrays(data,queries,recipe):
    xs,ys,variables,scales=[],[],[],[]
    for query in queries:
        q=Query(**query)
        if q.origin+q.horizon>int(.6*len(data.values)) \
                or data.segments[q.origin-q.length]!=data.segments[q.origin+q.horizon-1]:
            raise ValueError('First60 percent training labels only')
        x,base,scale=features(data,query,recipe)
        y=np.asarray(data.values[q.origin:q.origin+q.horizon,q.variable],np.float64)
        mask=np.asarray(data.mask[q.origin:q.origin+q.horizon,q.variable],bool)&np.isfinite(y)
        if not mask.any():
            raise ValueError('No training labels')
        xs.append(x[mask])
        ys.extend(((y-base)/scale)[mask])
        variables.extend([q.variable]*int(mask.sum()))
        scales.extend([scale]*int(mask.sum()))
    _,inverse,counts=np.unique(variables,return_inverse=True,return_counts=True)
    # |normalized residual error| * past scale equals original-unit absolute error.
    weights=np.asarray(scales,np.float64)/counts[inverse]
    weights/=weights.mean()
    return np.concatenate(xs),np.asarray(ys),weights


def estimator(recipe):
    if 'extra' in recipe:
        return ExtraTreesRegressor(n_estimators=128,max_depth=14,min_samples_leaf=32,
            criterion='squared_error',max_features=1.,random_state=42,n_jobs=2)
    return HistGradientBoostingRegressor(loss='absolute_error',max_iter=300,learning_rate=.05,
        max_leaf_nodes=31,max_depth=5,min_samples_leaf=64,l2_regularization=1.,max_bins=63,
        early_stopping=False,categorical_features=None,random_state=42)


def predict(model,data,queries,recipe):
    result=[]
    for q in queries:
        x,base,scale=features(data,q,recipe)
        if 'extra' in recipe:
            x=np.nan_to_num(x,nan=0.)
        p=base+scale*model.predict(x)
        if not np.isfinite(p).all():
            raise ValueError('Nonfinite wind prediction')
        result.append(p)
    return result


def fresh_confirmation(data,manifest):
    """Same64 target identities, unseen disjoint fifth-block origins, no error selection."""
    old=manifest['queries']['sdwpf']
    known=old+manifest['development_queries']['sdwpf']
    count=len(data.values)
    begin=int(.6*count)+(int(.8*count)-int(.6*count))*4//5
    end=int(.8*count)
    selected=[]
    for index,query in enumerate(old[64:]):
        q=Query(**query)
        pool=origins(data,q.variable,q.length,q.horizon,'validation')
        pool=pool[(pool>=begin)&(pool+q.horizon<=end)]
        for other in known+selected:
            if other['variable']==q.variable:
                pool=pool[np.abs(pool-other['origin'])>=q.horizon]
        if not len(pool):
            raise ValueError('No unseen disjoint confirmation for existing variable identity')
        rng=np.random.default_rng(np.random.SeedSequence([42,32,index,q.variable]))
        selected.append(asdict(Query('sdwpf',q.variable,int(rng.choice(pool)),48,24)))
    if len(selected)!=64:
        raise ValueError('Exactly64 fresh wind confirmation queries required')
    return selected
