"""Graph-only latent innovation; matched temporal control retains its own read."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from gofa_f.graph_memory import GraphMemoryExchange
from gofa_f.graph_memory_model import GraphMemoryGOFA


class CrossInnovationExchange(GraphMemoryExchange):
    innovation=True

    def read(self,hidden,times,receivers,bank,graph):
        if not self.innovation:return super().read(hidden,times,receivers,bank,graph)
        if hidden.ndim!=2 or times.shape!=(len(hidden),) or receivers.shape!=(len(hidden),):
            raise ValueError('Explicit query/time/receiver axes required')
        if not len(hidden):return hidden.float()
        if bank.nodes!=len(graph.base) or bank.length!=len(graph.history):raise ValueError('Memory axes rebound')
        parts=[]
        with torch.autocast(hidden.device.type,enabled=False):
            for start in range(0,len(hidden),128):
                own=hidden[start:start+128].float();count=len(own);full=F.pad(own,(0,0,0,128-count))
                clock=F.pad(times[start:start+128].float(),(0,128-count))
                receiver=F.pad(receivers[start:start+128].long(),(0,128-count))
                z=F.linear(F.layer_norm(full,self.norm.normalized_shape,None,None,self.norm.eps),self.down.weight)
                query=F.linear(z,self.query.weight)+F.linear(torch.stack((clock,clock.square()),-1),self.time_query.weight)
                score=F.linear(query,bank.keys)/math.sqrt(self.attention_rank)
                weights=graph.base[receiver][:,bank.node].float()
                support=(weights>0)&bank.observed[None]&graph.observed.any(0)[receiver,None]
                edge=graph.edges[receiver][:,bank.node].float()
                relative=clock[:,None]-bank.time[None]
                score=score+weights.clamp_min(torch.finfo(torch.float32).tiny).log()+(edge*self.edge_score.weight[0]).sum(-1)
                score=score+self.time_score[0]*relative+self.time_score[1]*relative.square()
                own_support=support&(bank.node[None]==receiver[:,None])
                def attention(mask):
                    empty=~mask.any(-1,keepdim=True)
                    masked=score.masked_fill(~mask,-torch.inf)
                    return torch.where(empty,torch.zeros_like(masked),masked).softmax(-1)*mask.float()
                # Subtract before the value projection: identical self supports
                # cancel exactly, while both branches keep the same geometry.
                innovation=attention(support)-attention(own_support)
                message=F.linear(innovation,bank.values.T)
                delta=F.linear(F.gelu(message),self.up.weight)*F.linear(z,self.gate.weight,self.gate.bias).sigmoid()
                base_norm=full.norm(dim=-1,keepdim=True);delta_norm=delta.norm(dim=-1,keepdim=True)
                delta=delta*(base_norm/(base_norm+delta_norm).clamp_min(torch.finfo(torch.float32).tiny))
                parts.append(delta[:count])
        return torch.cat(parts)


class CrossInnovationAdapter(nn.Module):
    def __init__(self,width):
        super().__init__();self.exchanges=nn.ModuleList([CrossInnovationExchange(width) for _ in range(2)])


class CrossInnovationGOFA(GraphMemoryGOFA):
    def __init__(self,native,codec,stable_fp32=False):
        super().__init__(native,codec,stable_fp32)
        self.adapter=CrossInnovationAdapter(native.config.hidden_size).float()
        self.set_condition('physical')

    def set_condition(self,condition):
        if condition not in ('native','physical','self','rewired'):raise ValueError('Fixed innovation control required')
        for block in self.adapter.exchanges:block.innovation=condition!='self'
        self._memory_architecture='cross_innovation_slot_v1' if condition!='self' else 'rich_temporal_slot_v1'
        self.cache_generation+=1
