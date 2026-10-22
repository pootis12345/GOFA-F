"""Memory and official-UFG execution adapters; same serialized source rosters."""
import gc
import hashlib
from pathlib import Path

import numpy as np
import torch

from gofa_f.dual_memory_policy import query_loss


class MemoryEngine:
    def __init__(self,native,tokenizer,codec,manifest,store,cache,depth):
        from gofa_f.dual_memory_model import DualMemoryGOFA
        self.original_forwards={module:module.forward for module in native.modules()
            if isinstance(module,torch.nn.Linear) or type(module).__name__=='LlamaRMSNorm'}
        self.model=DualMemoryGOFA(native,depth=depth,weight_binding=manifest['weights_sha256']).to('cuda:0')
        self.tokenizer,self.codec,self.manifest,self.store,self.cache=tokenizer,codec,manifest,store,cache
        self.metadata={};self.microbatch=1;self.cuda=True;self.state_model=self.model.adapter

    def warmup(self,enabled):
        for parameter in self.model.adapter.exchanges.parameters():parameter.requires_grad_(not enabled)

    def input(self,row,split,role='fit'):
        from gofa_f.dual_memory_run import make_tokens,metadata_tokens
        item,target,mask=self.store.get(row,split,role)
        item=item.to('cuda:0');tokens=make_tokens(self.codec,item,'cuda:0')
        shape=f"{row['length']}-{row['horizon']}";text=self.manifest['texts'][row['source']][shape]
        if text not in self.metadata:
            self.metadata[text]=self.model.metadata_vector(metadata_tokens(self.tokenizer,text,'cuda:0')).detach()
        prefix=self.cache.get(self.model,tokens)
        return item,tokens,self.metadata[text],prefix,target.to('cuda:0'),mask.to('cuda:0')

    def loss(self,rows):
        if len(rows)!=1:raise ValueError('Memory physical roster microbatch is one whole graph')
        row=rows[0];item,tokens,metadata,prefix,target,mask=self.input(row,'train')
        prediction,_=self.model.enhanced(tokens,item,metadata,prefix=prefix,optimized=True,checkpoint_tail=True)
        return query_loss(prediction,target,mask,item.present,self.manifest['training_unit_normalizers'][row['source']])

    @torch.no_grad()
    def predict(self,row,split,role='fit'):
        item,tokens,metadata,prefix,target,mask=self.input(row,split,role)
        prediction,_=self.model.enhanced(tokens,item,metadata,prefix=prefix,optimized=True)
        return prediction[0].cpu().numpy(),target[0].cpu().numpy(),mask[0].cpu().numpy(),item.identity

    def close(self):
        self.model.zero_grad(set_to_none=True)
        for module,forward in self.original_forwards.items():module.forward=forward
        self.metadata.clear();self.model=None;gc.collect();torch.cuda.empty_cache()


class UFGEngine:
    def __init__(self,root,length,horizon,manifest,store,microbatch=64):
        from gofa_f.ufg_memory_roster import MemoryRosterUFG
        self.model=MemoryRosterUFG(root,length,horizon,'cpu')
        self.length,self.horizon=length,horizon;self.manifest,self.store=manifest,store
        self.state_model=self.model;self.microbatch=microbatch;self.cuda=False

    def warmup(self,enabled):pass

    def inputs(self,rows,split,role='fit'):
        from gofa_f.ufg_subgraph import pack_history,SubgraphInput
        values=[];targets=[];masks=[];identities=[]
        for row in rows:
            if (row['length'],row['horizon'])!=(self.length,self.horizon):raise ValueError('UFG shape-specific checkpoint mismatch')
            item,target,mask=self.store.get(row,split,role)
            values.append(pack_history(item.history.T[None],item.observed.T[None],maximum_nodes=31))
            targets.append(torch.nn.functional.pad(target,(0,0,0,31-len(item.node_ids))))
            masks.append(torch.nn.functional.pad(mask,(0,0,0,31-len(item.node_ids))))
            identities.append(item.identity)
        packed=SubgraphInput(*(torch.cat([getattr(v,key) for v in values],0) for key in
            ('normalized','observed','minimum','extent','present')),31)
        return packed,torch.stack(targets),torch.stack(masks),identities

    def loss(self,rows):
        packed,target,mask,_=self.inputs(rows,'train')
        prediction=self.model(packed,all_nodes=True).transpose(1,2)
        return sum(query_loss(prediction[i],target[i],mask[i],packed.present[i],
            self.manifest['training_unit_normalizers'][row['source']]) for i,row in enumerate(rows))

    @torch.no_grad()
    def predict(self,row,split,role='fit'):
        packed,target,mask,identities=self.inputs([row],split,role)
        prediction=self.model(packed,all_nodes=True)
        return prediction[0,:,0].numpy(),target[0,0].numpy(),mask[0,0].numpy(),identities[0]

    def close(self):self.model=None;gc.collect()


def model_digest(model):
    value=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        value.update(name.encode());value.update(str((tensor.dtype,tuple(tensor.shape))).encode())
        value.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return value.hexdigest()
