"""Stage53: prediction-origin-causal cross-node attention over all observed history."""
from contextlib import contextmanager
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from gofa_f import source_graph_probe as p
from gofa_f import source_neighborhood_probe as n
from gofa_f import source_allnode_probe as allnode
from gofa_f import source_normalized_probe as bounded
from gofa_f.generative_data import sha
from gofa_f.lag_graph_probe import state_digest
from gofa_f.prefix_cache import atomic_json

PROTOCOL='stage53-source-origin-history-v1'
STAGE52_REPORT='13f371dec18001681cad1ff99dc7647fc1a577922ee4ef1f21e9b62cccb26011'


class OriginHistoryExchange(nn.Module):
    """Every key/value is before the immutable prediction origin, including t'>t.

    This is an encoder-style history operation, NOT a prefix-causal LM cache.
    No target/future slots are accepted; the source cache supplies exactly L=12.
    """
    def __init__(self,original):
        super().__init__();self.rank=original.rank
        for name in ('norm','query','key','value','edge_score','edge_value','gate','up'):
            setattr(self,name,getattr(original,name))

    def forward(self,hidden,observed,adjacency,edge_features):
        batch,nodes,length,_=hidden.shape
        if length!=12 or observed.shape!=(batch,nodes,12):raise ValueError('Exactly12 origin-frozen past slots required')
        z=self.norm(hidden)
        q=self.query(z).permute(0,2,1,3)
        keys=self.key(z);values=self.value(z)
        cross=adjacency.bool() & ~torch.eye(nodes,device=hidden.device,dtype=torch.bool)[None]
        present=observed.any(-1);cross=cross & present[:,:,None] & present[:,None,:]
        support=cross[:,None,:,:,None] & observed[:,None,None,:,:]
        score=torch.einsum('btir,bjkr->btijk',q,keys)/math.sqrt(self.rank)
        score=score+self.edge_score(edge_features).squeeze(-1)[:,None,:,:,None]
        empty=~support.flatten(-2).any(-1,keepdim=True)
        flat=score.masked_fill(~support,-torch.inf).flatten(-2)
        flat=torch.where(empty,torch.zeros_like(flat),flat)
        attention=(flat.softmax(-1)*support.flatten(-2)).reshape(batch,length,nodes,nodes,length)
        context=torch.einsum('btijk,bjkd->btid',attention,values)
        context=context+torch.einsum('btij,bijd->btid',attention.sum(-1),self.edge_value(edge_features))
        delta=self.up(F.gelu(context))*self.gate(z.permute(0,2,1,3)).sigmoid()
        delta=delta.masked_fill(empty,0.).permute(0,2,1,3)
        return (hidden+delta)*present[:,:,None,None]


def add_origin_history(model):
    if getattr(model,'_stage53_origin_history',False):raise ValueError('Origin-history operation already installed')
    blocks=[]
    for old in model.exchanges:
        block=OriginHistoryExchange(old)
        block.register_forward_hook(bounded.bounded_exchange)
        blocks.append(block)
    model.exchanges=nn.ModuleList(blocks);model._stage53_origin_history=True
    return model


@contextmanager
def origin_history_runtime():
    with allnode.allnode_runtime():
        old=(p.model_for,p.fit,n.PROTOCOL,n.stage_number,n.code_hashes,n.SEMANTICS)
        factory,_,_,number,hashes,semantics=old
        p.model_for=lambda cfg,seed:add_origin_history(factory(cfg,seed))
        p.fit=fit_origin_allnode;n.PROTOCOL=PROTOCOL
        n.stage_number=lambda cfg:53 if cfg['protocol']==PROTOCOL else number(cfg)
        n.SEMANTICS={c:v+'_origin_causal_full12_history' for c,v in semantics.items()}
        def code_hashes():
            parent=Path(os.environ['GOFA_WORKSPACE'])/'outputs/stage52-source-allnode-supervision-v1'
            if sha(parent/'report.json')!=STAGE52_REPORT or p.read(parent/'report.json')['passed'] or not p.read(parent/'final_verification.json')['passed'] or p.read(parent/'activity.json')['worker'] is not None:
                raise ValueError('Sealed stopped failed Stage52 lineage required')
            return dict(hashes(),source_origin_history_probe=sha(Path(__file__)),failed_stage52_report=STAGE52_REPORT)
        n.code_hashes=code_hashes
        try:yield
        finally:p.model_for,p.fit,n.PROTOCOL,n.stage_number,n.code_hashes,n.SEMANTICS=old


def fit_origin_allnode(cfg,output,m,packed,seed,condition,charge,speed):
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
            batches.append(allnode.training_batch(packed['train'][name],ix,m['queries']['train'][name],condition))
        batch={k:torch.cat([b[k] for b in batches]) for k in batches[0]}
        prediction=model(batch['x'],batch['adjacency'],batch['edge_features'],12,return_residual=True)
        loss=allnode.allnode_loss(prediction,batch)
        if not loss.isfinite():raise ValueError('Nonfinite matched-source loss')
        loss.backward()
        if any(p.grad is not None and not p.grad.isfinite().all() for p in model.parameters()):
            raise ValueError('Nonfinite source gradient')
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        for group in optimizer.param_groups:group['lr']=.001*min(1.,(update+1)/20)
        optimizer.step();update+=1
        if update%100==0:validate()
        checkpoint();charge()
        if update%50==0 or update==1:
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
    with origin_history_runtime():n.main()
