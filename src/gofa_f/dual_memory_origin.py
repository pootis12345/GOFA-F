"""Read-only Native prompt KV and actual shared pre-graph root history.

This adds no graph hook, token, parameter, sampling draw or optimizer. Eight
trajectories clone the unconditioned prompt KV independently. The enhanced root
uses the exact pre-graph hidden state captured from that Native prefill.
"""
import weakref

import torch
from torch import nn
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from gofa_f.allnode_codec import AllNodeTokens
from gofa_f.dual_memory_model import FrozenHistory,ProtectedKV
from gofa_f.generative_kernel import stable_attention


def select_rows(tokens,indices,crop=False):
    indices=list(indices)
    size=max(int(tokens.prompt_ends[i])+1 for i in indices) if crop else tokens.input_ids.shape[1]
    return AllNodeTokens(tokens.input_ids[indices,:size],tokens.attention_mask[indices,:size],tokens.labels[indices,:size],
        tokens.prompt_ends[indices],tuple(tokens.value_positions[i] for i in indices),
        tuple(tokens.prompt_hashes[i] for i in indices),tokens.full_ids[indices,:size])


class NativeOrigin(nn.Module):
    def __init__(self,owner,root_tokens,first_separator=()):
        super().__init__();object.__setattr__(self,'_owner',weakref.ref(owner))
        if len(root_tokens.input_ids)!=1 or root_tokens.input_ids.shape[1]!=int(root_tokens.prompt_ends[0])+1:
            raise ValueError('One unpadded official origin prompt required')
        if any(p.requires_grad for p in owner.native.parameters()):raise ValueError('Native backbone must be frozen')
        self.plain=root_tokens
        if first_separator:
            root_tokens=root_tokens.append(torch.tensor(first_separator,device=root_tokens.input_ids.device)[None])
            root_tokens.input_ids=root_tokens.full_ids
        self.tokens=root_tokens
        self.logits=None;self.kv=None;self.root_hidden=None

    @property
    def native(self):return self._owner().native

    @torch.no_grad()
    def prepare(self):
        if self.logits is not None:return
        owner=self._owner();tokens=self.tokens
        if owner._kernel_active:raise RuntimeError('Native origin execution overlaps graph execution')
        positions=torch.arange(tokens.input_ids.shape[1],device=tokens.input_ids.device)
        ids=(tokens.attention_mask.long().cumsum(-1)-1).clamp_min(0)
        with stable_attention(owner,positions,tokens.attention_mask,tokens.prompt_ends):
            out=owner.native(input_ids=tokens.input_ids,attention_mask=tokens.attention_mask,position_ids=ids,
                past_key_values=DynamicCache(),use_cache=True,output_hidden_states=True,return_dict=True)
        self.logits=out.logits.detach();self.kv=out.past_key_values
        self.root_hidden=out.hidden_states[owner.prefix_layers][:,:self.plain.input_ids.shape[1]].detach()
        if not isinstance(self.kv,DynamicCache):raise ValueError('Native prompt returned an unexpected KV representation')

    def _clone(self):
        result=DynamicCache()
        for index,(key,value) in enumerate(zip(self.kv.key_cache,self.kv.value_cache)):
            result.update(key.detach().clone(),value.detach().clone(),index)
        return result

    def forward(self,tokens,contexts=None,cache=None,use_cache=False):
        owner=self._owner()
        if contexts is not None:raise ValueError('Protected Native cannot read graph contexts')
        if cache is not None or not use_cache:
            return owner.protected_forward(tokens,cache,use_cache)
        if not torch.equal(tokens.full_ids,self.tokens.full_ids) or \
                not torch.equal(tokens.attention_mask,self.tokens.attention_mask) or \
                not torch.equal(tokens.prompt_ends,self.tokens.prompt_ends):
            raise ValueError('A Native origin cache cannot serve another prompt')
        self.prepare()
        return CausalLMOutputWithPast(logits=self.logits,past_key_values=ProtectedKV(self._clone(),
            tokens.full_ids.detach().clone(),tokens.attention_mask.detach().clone(),tokens.prompt_ends.detach().clone()))


class SharedPrefixBuilder:
    """HistoryCache producer; only pure frozen history, never graph Memory/KV."""
    def __init__(self,owner,origin):self.owner,self.origin=owner,origin

    def input_binding(self,tokens):return self.owner.input_binding(tokens)

    @torch.no_grad()
    def frozen_history(self,tokens):
        self.origin.prepare();root=select_rows(tokens,[0],crop=True)
        if not torch.equal(root.input_ids,self.origin.plain.input_ids):raise ValueError('Enhanced and Native root inputs differ')
        shape=(len(tokens.input_ids),tokens.input_ids.shape[1],self.origin.root_hidden.shape[-1])
        hidden=self.origin.root_hidden.new_zeros(shape)
        hidden[0,:self.origin.root_hidden.shape[1]]=self.origin.root_hidden[0]
        if len(tokens.input_ids)>1:
            auxiliary=self.owner.frozen_history(select_rows(tokens,range(1,len(tokens.input_ids))))
            hidden[1:]=auxiliary.hidden
        return FrozenHistory(hidden,tokens.attention_mask.detach().clone(),tokens.prompt_ends.detach().clone(),
            self.owner.input_binding(tokens))
