"""Origin-frozen all-node LLM memories, read at each forecast numeric slot.

This adapter returns hidden residuals, never a continuous numerical forecast.
Fixed128-row FP32 projection geometry is shared by full and incremental reads.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


def linear(module,x):
    shape=x.shape;flat=x.reshape(-1,shape[-1]);parts=[]
    for start in range(0,len(flat),128):
        own=flat[start:start+128];padded=F.pad(own,(0,0,0,128-len(own)))
        parts.append(F.linear(padded,module.weight,module.bias)[:len(own)])
    return torch.cat(parts).reshape(*shape[:-1],module.out_features)


def normalize(module,x):
    shape=x.shape;flat=x.reshape(-1,shape[-1]);parts=[]
    for start in range(0,len(flat),128):
        own=flat[start:start+128];padded=F.pad(own,(0,0,0,128-len(own)))
        parts.append(F.layer_norm(padded,module.normalized_shape,None,None,module.eps)[:len(own)])
    return torch.cat(parts).reshape(shape)


@dataclass
class MemoryBank:
    keys: torch.Tensor
    values: torch.Tensor
    observed: torch.Tensor
    node: torch.Tensor
    time: torch.Tensor
    length: int
    nodes: int


class GraphMemoryExchange(nn.Module):
    def __init__(self,width,rank=64,attention_rank=16):
        super().__init__();self.rank=rank;self.attention_rank=attention_rank
        self.norm=nn.LayerNorm(width,elementwise_affine=False)
        self.down=nn.Linear(width,rank,bias=False)
        self.query=nn.Linear(rank,attention_rank,bias=False)
        self.key=nn.Linear(rank,attention_rank,bias=False)
        self.value=nn.Linear(rank,rank,bias=False)
        self.edge_score=nn.Linear(2,1,bias=False)
        self.time_query=nn.Linear(2,attention_rank,bias=False)
        self.time_score=nn.Parameter(torch.zeros(2))
        self.gate=nn.Linear(rank,1)
        self.up=nn.Linear(rank,width,bias=False)
        nn.init.zeros_(self.up.weight);nn.init.zeros_(self.time_query.weight)
        nn.init.zeros_(self.gate.weight);nn.init.zeros_(self.gate.bias)

    def build(self,hidden,graph):
        nodes,length,_=hidden.shape
        if graph.history.shape!=(length,nodes):raise ValueError('Original origin axes changed')
        with torch.autocast(hidden.device.type,enabled=False):
            z=linear(self.down,normalize(self.norm,hidden.float()))
            keys=linear(self.key,z).reshape(nodes*length,self.attention_rank)
            values=linear(self.value,z).reshape(nodes*length,self.rank)
            observed=graph.observed.T.reshape(-1).bool()
            node=torch.arange(nodes,device=hidden.device).repeat_interleave(length)
            times=torch.arange(1-length,1,device=hidden.device,dtype=torch.float32).repeat(nodes)/length
            return MemoryBank(keys,values,observed,node,times,length,nodes)

    def read(self,hidden,times,receivers,bank,graph):
        """Read one fixed graph memory for arbitrary causal query-token positions."""
        if hidden.ndim!=2 or times.shape!=(len(hidden),) or receivers.shape!=(len(hidden),):
            raise ValueError('Explicit query rows/time/receiver coordinates required')
        if not len(hidden):return hidden.float()
        device=hidden.device
        if bank.nodes!=len(graph.base) or bank.length!=len(graph.history):raise ValueError('Memory axes rebound')
        parts=[]
        with torch.autocast(device.type,enabled=False):
            for start in range(0,len(hidden),128):
                own=hidden[start:start+128].float();count=len(own)
                full=F.pad(own,(0,0,0,128-count))
                clock=F.pad(times[start:start+128].float(),(0,128-count))
                receiver=F.pad(receivers[start:start+128].long(),(0,128-count))
                z=F.linear(F.layer_norm(full,self.norm.normalized_shape,None,None,self.norm.eps),self.down.weight)
                q=F.linear(z,self.query.weight)+F.linear(torch.stack((clock,clock.square()),-1),self.time_query.weight)
                score=F.linear(q,bank.keys)/math.sqrt(self.attention_rank)
                weights=graph.base[receiver][:,bank.node].float()
                receiver_present=graph.observed.any(0)[receiver]
                support=(weights>0)&bank.observed[None]&receiver_present[:,None]
                edge=graph.edges[receiver][:,bank.node].float()
                edge_score=(edge*self.edge_score.weight[0]).sum(-1)
                relative=clock[:,None]-bank.time[None]
                score=score+weights.clamp_min(torch.finfo(torch.float32).tiny).log()+edge_score
                score=score+self.time_score[0]*relative+self.time_score[1]*relative.square()
                empty=~support.any(-1,keepdim=True)
                score=score.masked_fill(~support,-torch.inf)
                attention=torch.where(empty,torch.zeros_like(score),score).softmax(-1)*support.float()
                message=F.linear(attention,bank.values.T)
                delta=F.linear(F.gelu(message),self.up.weight)*F.linear(z,self.gate.weight,self.gate.bias).sigmoid()
                delta=delta.masked_fill(empty,0.)
                base_norm=full.norm(dim=-1,keepdim=True);delta_norm=delta.norm(dim=-1,keepdim=True)
                delta=delta*(base_norm/(base_norm+delta_norm).clamp_min(torch.finfo(torch.float32).tiny))
                parts.append(delta[:count])
        return torch.cat(parts)

    def history_delta(self,hidden,bank,graph):
        nodes,length,width=hidden.shape
        receivers=torch.arange(nodes,device=hidden.device).repeat_interleave(length)
        return self.read(hidden.reshape(-1,width),bank.time,receivers,bank,graph).reshape_as(hidden)


class GraphMemoryAdapter(nn.Module):
    def __init__(self,width):
        super().__init__();self.exchanges=nn.ModuleList([GraphMemoryExchange(width) for _ in range(2)])
