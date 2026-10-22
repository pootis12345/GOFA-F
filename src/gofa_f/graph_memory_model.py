"""Shared frozen ChatTime for all nodes; final-two-block graph-memory exchange."""
from dataclasses import dataclass
import hashlib

import numpy as np
import torch

from gofa_f.allnode_model import AllNodeGOFA,OriginContext
from gofa_f.graph_memory import GraphMemoryAdapter
from gofa_f.generative_kernel import stable_attention


@dataclass
class GraphMemoryContext(OriginContext):
    memories: tuple
    graph: object


class GraphMemoryGOFA(AllNodeGOFA):
    def __init__(self,native,codec,stable_fp32=False):
        super().__init__(native,stable_fp32)
        self.adapter=GraphMemoryAdapter(native.config.hidden_size).float()
        numeric=torch.zeros(native.config.vocab_size,dtype=torch.bool)
        numeric[list(codec.finite_ids)]=True
        self.register_buffer('_finite_numeric',numeric,persistent=False)
        self._memory_architecture='origin_graph_memory_slot_read_v1'

    def _hook(self,slot):
        def inject(module,args,kwargs):
            if self._active is None:return args,kwargs
            hidden=args[0] if args else kwargs['hidden_states']
            tokens,contexts,offset=self._active
            positions=torch.arange(offset,offset+hidden.shape[1],device=hidden.device)
            all_positions=torch.arange(tokens.full_ids.shape[1],device=hidden.device)
            answer=(all_positions[None]>tokens.prompt_ends[:,None])&tokens.attention_mask
            emitted=(self._finite_numeric[tokens.full_ids]&answer).long().cumsum(-1)
            changed=[]
            for row,context in enumerate(contexts):
                if not isinstance(context,GraphMemoryContext):raise ValueError('Graph-memory context required')
                h=hidden[row]
                numeric=tokens.value_positions[row][:context.length]-offset
                valid=(numeric>=0)&(numeric<hidden.shape[1])
                if valid.any():
                    where=numeric[valid];h=h.index_copy(0,where,h[where]+context.deltas[slot][valid].to(h.dtype))
                selected=(positions>=tokens.prompt_ends[row])&tokens.attention_mask[row,positions]
                if selected.any():
                    where=torch.nonzero(selected).flatten()
                    clock=emitted[row,positions[where]].float()/context.length
                    receiver=torch.zeros(len(where),device=hidden.device,dtype=torch.long)
                    delta=self.adapter.exchanges[slot].read(h[where],clock,receiver,context.memories[slot],context.graph)
                    h=h.index_copy(0,where,h[where]+delta.to(h.dtype))
                changed.append(h)
            result=torch.stack(changed)
            return ((result,*args[1:]),kwargs) if args else (args,dict(kwargs,hidden_states=result))
        return inject

    def encode_origins(self,graphs,codec,cache=None,check=lambda:None):
        result=[];device=next(self.native.parameters()).device
        for item in graphs:
            check();graph=item.graph
            histories=np.where(graph.observed.cpu().numpy(),graph.history.cpu().numpy(),np.nan).T
            tokens=codec.batch([codec.record(row) for row in histories]).to(device);graph=graph.to(device)
            hidden,_=self.cached_prefix(tokens,cache);native=self.native.model
            positions=torch.arange(hidden.shape[1],device=device);ids=self.positions(tokens.attention_mask,hidden.shape[1])
            causal=native._update_causal_mask(tokens.attention_mask,hidden,positions,None,False)
            rotary=native.rotary_emb(hidden,ids);deltas=[];banks=[]
            with stable_attention(self,positions,tokens.attention_mask,tokens.prompt_ends):
                for slot,index in enumerate(self.graph_layers):
                    check();selected=torch.stack([hidden[i].index_select(0,p) for i,p in enumerate(tokens.value_positions)])
                    block=self.adapter.exchanges[slot];bank=block.build(selected,graph)
                    delta=block.history_delta(selected,bank,graph);banks.append(bank);deltas.append(delta[0])
                    hidden=torch.stack([hidden[i].index_copy(0,p,hidden[i].index_select(0,p)+delta[i].to(hidden.dtype))
                        for i,p in enumerate(tokens.value_positions)])
                    if slot==0:
                        hidden=native.layers[index](hidden,attention_mask=causal,position_ids=ids,cache_position=positions,
                            position_embeddings=rotary,use_cache=False,output_attentions=False)[0]
                    else:
                        with torch.no_grad():
                            end=native.layers[index](hidden,attention_mask=causal,position_ids=ids,cache_position=positions,
                                position_embeddings=rotary,use_cache=False,output_attentions=False)[0];native.norm(end)
            self.origin_executions.append(dict(nodes=len(histories),layers=len(native.layers),history=len(graph.history)))
            self.origin_executions=self.origin_executions[-32:]
            identity=hashlib.sha256((item.identity+':'+self._memory_architecture).encode()).hexdigest()
            result.append(GraphMemoryContext(identity,self.parameter_version,self.cache_generation,tuple(deltas),
                len(graph.history),len(histories),tuple(banks),graph))
        return result
