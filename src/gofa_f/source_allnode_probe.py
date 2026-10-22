"""Stage52: fixed all-node source supervision, matched sparse-road controls."""
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import time

import numpy as np
import torch

from gofa_f import source_graph_probe as p
from gofa_f import source_neighborhood_probe as n
from gofa_f import source_sparse_normalized_probe as sparse
from gofa_f.generative_data import sha
from gofa_f.lag_graph_probe import state_digest
from gofa_f.prefix_cache import atomic_json

PROTOCOL='stage52-source-allnode-supervision-v1'
STAGE51_REPORT='de3a483d8ac1163503641085487f0c6d559dc01798d84d48586c9b63a7de3c58'
AUX_KEYS=('node_scale','node_delta','node_observed','node_weight')


def auxiliary_labels(data,a,records,allowed):
    """Only source-train labels; scales and presence depend exclusively on history."""
    qcount=len(records);shape=(qcount,9,12)
    scale=np.ones((qcount,9),np.float64);delta=np.zeros(shape,np.float64)
    observed=np.zeros(shape,bool);allowed=set(allowed)
    for i,row in enumerate(records):
        q=row['query'];origin=q['origin']
        if q['length']!=12 or q['horizon']!=12 or q.get('chunk',0) or origin<12 or origin+12>int(.6*len(data.values)):
            raise ValueError('Auxiliary labels require first60% fixed source training queries')
        if data.segments[origin-12]!=data.segments[origin+11]:raise ValueError('Auxiliary time discontinuity')
        nodes=a['nodes'][i];active=nodes>=0;ids=nodes[active]
        if not set(ids)<=allowed:raise ValueError('Auxiliary node outside training partition')
        past=data.values[origin-12:origin,ids].astype(np.float64).T
        mask=data.mask[origin-12:origin,ids].astype(bool).T & np.isfinite(past)
        present=mask.any(1)
        low=np.where(present,np.where(mask,past,np.inf).min(1),0.)
        high=np.where(present,np.where(mask,past,-np.inf).max(1),0.)
        own_scale=np.where(present & (high-low>10*np.finfo(np.float64).eps),high-low,1.)
        last=np.array([v[np.flatnonzero(m)[-1]] if m.any() else 0. for v,m in zip(past,mask)])
        truth=data.values[origin:origin+12,ids].astype(np.float64).T
        valid=data.mask[origin:origin+12,ids].astype(bool).T & np.isfinite(truth) & present[:,None]
        scale[i,active]=own_scale;observed[i,active]=valid
        delta[i,active]=np.where(valid,truth-last[:,None],0.)
        np.testing.assert_array_equal(a['x'][i,active,:,1].astype(bool),mask)
        np.testing.assert_array_equal(a['x'][i,active,:,0],np.where(mask,(past-low[:,None])/own_scale[:,None]-.5,0).astype(np.float32))
        if own_scale[0]!=a['scale'][i] or last[0]!=a['persistence'][i]:raise ValueError('Target history scale binding differs')
    counts={};error={}
    for i,nodes in enumerate(a['nodes']):
        for j,v in enumerate(nodes):
            if v<0:continue
            count=int(observed[i,j].sum())
            if count:
                counts[int(v)]=counts.get(int(v),0)+count
                error[int(v)]=error.get(int(v),0.)+float(np.abs(delta[i,j])[observed[i,j]].sum())
    if not counts or not observed.any((1,2)).all():raise ValueError('Empty all-node source loss')
    norm=max(float(np.mean([error[v]/count for v,count in counts.items()])),1e-12)
    weight=np.zeros((qcount,9),np.float64)
    for i,nodes in enumerate(a['nodes']):
        for j,v in enumerate(nodes):
            if int(v) in counts:weight[i,j]=qcount/(len(counts)*counts[int(v)]*norm)
    arrays=dict(node_scale=scale,node_delta=delta,node_observed=observed,node_weight=weight)
    proof=dict(queries=qcount,valid_points=int(observed.sum()),variables=len(counts),
        variable_valid_points={str(v):c for v,c in counts.items()},persistence_macro_mae=norm,
        point_weight='Q/(V*C_v)/training_allnode_persistence_macro_mae',future_input=False)
    return arrays,proof


def allnode_loss(residual,b):
    if residual.shape!=b['node_delta'].shape:raise ValueError('Complete B,N,H output required')
    error=(residual*b['node_scale'][...,None]-b['node_delta']).abs()
    return (error*b['node_observed']*b['node_weight'][...,None]).sum((1,2)).mean()


def training_batch(a,indices,records,condition):
    b=p.tensors(a,indices,records,condition)
    for key in AUX_KEYS:
        value=a[key][indices]
        b[key]=torch.from_numpy(value if key=='node_observed' else value.astype(np.float32))
    return b


def allnode_caches(output,m,charge):
    # sparse.reuse_self is disabled by the context before this function can run.
    result=sparse.reused_caches(output,m,charge);data=p.source_series(m);evidence={}
    for view in p.VIEWS:
        base=result['physical']['train'][view]
        arrays,proof=auxiliary_labels(data[view],base,m['queries']['train'][view],m['partition']['train'])
        path=output/'allnode'/f'train-{view}.npz';path.parent.mkdir(parents=True,exist_ok=True)
        content={k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in arrays.items()}
        binding=p.digest(dict(manifest=sha(output/'manifest.json'),arrays=content,proof=proof))
        if path.exists():
            old=p.read(path.with_suffix('.json'))
            if old['binding']!=binding or old['sha256']!=sha(path):raise ValueError('Auxiliary cache changed')
            with np.load(path,allow_pickle=False) as saved:
                for key in AUX_KEYS:np.testing.assert_array_equal(saved[key],arrays[key])
        else:
            temporary=path.with_suffix('.part')
            with temporary.open('wb') as stream:np.savez_compressed(stream,**arrays)
            temporary.replace(path)
            p.immutable(path.with_suffix('.json'),dict(binding=binding,sha256=sha(path),arrays=content,proof=proof))
        for condition in p.CONDITIONS:result[condition]['train'][view]=dict(result[condition]['train'][view],**arrays)
        evidence[view]=dict(binding=binding,sha256=sha(path),arrays=content,proof=proof);charge()
    for condition in p.CONDITIONS:
        for split in ('validation','confirmation'):
            if any(set(AUX_KEYS)&set(a) for a in result[condition][split].values()):raise ValueError('Auxiliary evaluation labels forbidden')
    p.immutable(output/'allnode_binding.json',dict(manifest_sha256=sha(output/'manifest.json'),views=evidence,
        all_controls_same_auxiliary_arrays=True,all_models_fresh=True,auxiliary_evaluation_labels=False))
    return result


@contextmanager
def allnode_runtime():
    with sparse.sparse_runtime():
        old=(p.fit,n.PROTOCOL,n.code_hashes,n.caches,n.stage_number,sparse.reuse_self,n.run)
        _,_,hashes,_,number,_,run=old
        p.fit=fit_allnode;n.PROTOCOL=PROTOCOL;n.caches=allnode_caches
        sparse.reuse_self=lambda *_:None
        n.stage_number=lambda cfg:52 if cfg['protocol']==PROTOCOL else number(cfg)
        def code_hashes():
            parent=Path(os.environ['GOFA_WORKSPACE'])/'outputs/stage51-source-sparse-normalized-v1'
            if sha(parent/'report.json')!=STAGE51_REPORT or p.read(parent/'report.json')['passed'] or not p.read(parent/'final_verification.json')['passed'] or p.read(parent/'activity.json')['worker'] is not None:
                raise ValueError('Verified stopped failed Stage51 lineage required')
            return dict(hashes(),source_allnode_probe=sha(Path(__file__)),failed_stage51_report=STAGE51_REPORT)
        n.code_hashes=code_hashes
        try:yield
        finally:p.fit,n.PROTOCOL,n.code_hashes,n.caches,n.stage_number,sparse.reuse_self,n.run=old


def fit_allnode(cfg,output,m,packed,seed,condition,charge,speed):
    folder=output/'models'/f'seed-{seed}'/condition
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'selection.json').exists():return p.read(folder/'selection.json')
    binding=p.digest(dict(manifest=sha(output/'manifest.json'),seed=seed,condition=condition))
    model=p.model_for(cfg,seed)
    initial_sha=state_digest(model)
    p.immutable(output/'initializations'/f'{seed}-{condition}.json',dict(seed=seed,parameter_sha256=initial_sha,
        parameters=sum(p.numel() for p in model.parameters())))
    for other in p.CONDITIONS:
        old=p.read(output/'initializations'/f'{seed}-{other}.json')
        if old and old['parameter_sha256']!=initial_sha:raise ValueError('Matched initialization differs')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    rng=np.random.default_rng(np.random.SeedSequence([seed,36,100]))
    cursor={name:dict(order=rng.permutation(len(packed['train'][name]['x'])).tolist(),position=0,epochs=0)
            for name in p.VIEWS}
    baselines={name:p.metrics(np.broadcast_to(packed['validation'][name]['persistence'][:,None],
        packed['validation'][name]['raw_target'].shape),packed['validation'][name],m['queries']['validation'][name])['macro_mae']
        for name in p.VIEWS}
    update,best_score,best_update=0,None,None
    if (folder/'last.pt').exists():
        saved=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        if saved['binding']!=binding or saved['initial_sha256']!=initial_sha:raise ValueError('Resume binding differs')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state=saved['rng'];torch.set_rng_state(saved['torch_rng'])
        cursor=saved['cursor'];update=saved['update'];best_score=saved['best_score'];best_update=saved['best_update']
    def checkpoint():
        p.save_torch(folder/'last.pt',dict(binding=binding,initial_sha256=initial_sha,model=model.state_dict(),
            optimizer=optimizer.state_dict(),rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
            cursor=cursor,update=update,best_score=best_score,best_update=best_update))
    def validate():
        nonlocal best_score,best_update
        results={name:p.metrics(p.forecast(model,packed['validation'][name],m['queries']['validation'][name],condition),
            packed['validation'][name],m['queries']['validation'][name]) for name in p.VIEWS}
        score=float(np.exp(np.mean([np.log(max(results[name]['macro_mae'],1e-12)/max(baselines[name],1e-12)) for name in p.VIEWS])))
        value=dict(update=update,score=score,views=results,binding=binding)
        p.immutable(folder/'validations'/f'{update:06d}.json',value)
        if best_score is None or score<best_score:
            best_score,best_update=score,update
            p.save_torch(folder/'best.pt',dict(binding=binding,model=model.state_dict(),update=update,score=score))
            atomic_json(folder/'best_validation.json',value)
        charge()
    if best_score is None:
        validate();checkpoint()
    while update<cfg['training']['max_updates']:
        if time.time()>=float(os.environ['GOFA_AUTOMATION_DEADLINE_UNIX'])-65:p.STOP[0]=True
        if p.STOP[0]:checkpoint();return None
        model.train();optimizer.zero_grad(set_to_none=True)
        batches=[]
        for name in p.VIEWS:
            ix=p.next_indices(cursor,name,32,rng)
            batches.append(training_batch(packed['train'][name],ix,m['queries']['train'][name],condition))
        batch={k:torch.cat([b[k] for b in batches]) for k in batches[0]}
        prediction=model(batch['x'],batch['adjacency'],batch['edge_features'],12,return_residual=True)
        loss=allnode_loss(prediction,batch)
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
            p.immutable(folder/'training_points'/f'{update:06d}.json',point)
            atomic_json(folder/'training_progress.json',point)
            atomic_json(output/'progress.json',dict(state='running',phase='matched_source_training',seed=seed,
                condition=condition,updates=update,max_updates=1000,best_update=best_update,best_score=best_score,
                current_model_eta_seconds=(1000-update)*speed['seconds_per_update'],
                future_seeds_conditional=seed==42,gpu_seconds=0.))
            print(p.json.dumps(dict(seed=seed,condition=condition,update=update,loss=float(loss.detach()),best_score=best_score)),flush=True)
    value=p.read(folder/'best_validation.json')
    selected=p.model_for(cfg,seed)
    selected.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=False)['model'])
    selection=dict(binding=binding,seed=seed,condition=condition,completed_updates=update,
        selected_update=best_update,score=best_score,initial_sha256=initial_sha,
        checkpoint_sha256=sha(folder/'best.pt'),tensor_sha256=state_digest(selected),validation=value['views'])
    p.immutable(folder/'selection.json',selection)
    return selection

if __name__=='__main__':
    with allnode_runtime():n.main()
