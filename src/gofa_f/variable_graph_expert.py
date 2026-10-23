"""Length-independent physical graph forecaster; no language model weights.

Each node retains its masked historical patch sequence. Temporal encoding is
causal within history; spatial attention reads incoming edges at past lags.
Forecast queries attend independently to history, never to one another.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def time_features(seconds):
    """Fixed physical-time basis, independent of L, H, node or dataset IDs."""
    hours=seconds.float()/3600
    phase=hours[...,None]*hours.new_tensor([2*math.pi,2*math.pi/24,2*math.pi/168])
    return torch.cat((torch.sign(hours)[...,None]*torch.log1p(hours.abs())[...,None],
                      phase.sin(),phase.cos()),-1)


def safe_softmax(scores,allowed,dim=-1):
    valid=allowed.any(dim=dim,keepdim=True)
    filled=scores.masked_fill(~allowed,float('-inf'))
    filled=torch.where(valid,filled,torch.zeros_like(filled))
    return torch.softmax(filled,dim=dim).masked_fill(~allowed,0)


class SpaceTimeBlock(nn.Module):
    def __init__(self,width=64,heads=4,lags=(0,1,2,4,8)):
        super().__init__();self.width=width;self.heads=heads;self.lags=lags
        self.temporal=nn.MultiheadAttention(width,heads,dropout=0,batch_first=True)
        self.tnorm=nn.LayerNorm(width);self.snorm=nn.LayerNorm(width)
        self.qkv=nn.Linear(width,width*3);self.edge=nn.Sequential(nn.Linear(4,width),nn.SiLU(),nn.Linear(width,heads))
        self.lag_time=nn.Linear(7,heads,bias=False);self.mix=nn.Linear(width,width)
        self.ffnorm=nn.LayerNorm(width);self.ff=nn.Sequential(nn.Linear(width,width*2),nn.GELU(),nn.Linear(width*2,width))

    def forward(self,x,observed,adjacency,edges,interval,ends):
        b,n,p,d=x.shape;h=self.heads;c=d//h
        z=self.tnorm(x).reshape(b*n,p,d);valid=observed.reshape(b*n,p)
        # Empty auxiliary histories get a dummy zero key; their outputs stay masked.
        causal=torch.ones(p,p,device=x.device,dtype=torch.bool).triu(1)
        # An early all-missing prefix must not make attention softmax all -inf.
        # A zero sentinel at time0 is always allowed and carries no observation.
        sentinel=z.new_zeros(b*n,1,d)
        kv=torch.cat((sentinel,z),1);key_mask=torch.cat((torch.zeros_like(valid[:,:1]),~valid),1)
        causal=torch.cat((torch.zeros(p,1,device=x.device,dtype=torch.bool),causal),1)
        temporal=self.temporal(z,kv,kv,key_padding_mask=key_mask,attn_mask=causal,need_weights=False)[0]
        x=(x+temporal.reshape(b,n,p,d)).masked_fill(~observed[...,None],0)
        q,k,v=self.qkv(self.snorm(x)).chunk(3,-1)
        q=q.reshape(b,n,p,h,c);k=k.reshape(b,n,p,h,c);v=v.reshape(b,n,p,h,c)
        keys=[];values=[];masks=[]
        for lag in self.lags:
            if lag>=p:
                keys.append(k.new_zeros(k.shape));values.append(v.new_zeros(v.shape));masks.append(observed.new_zeros(observed.shape))
            elif lag:
                keys.append(F.pad(k[:,:,:-lag],(0,0,0,0,lag,0)))
                values.append(F.pad(v[:,:,:-lag],(0,0,0,0,lag,0)))
                masks.append(F.pad(observed[:,:,:-lag],(lag,0)))
            else:keys.append(k);values.append(v);masks.append(observed)
        # B,receiver,patch,head,sender,lag. Sparse topology is applied before normalization.
        ks=torch.stack(keys,-2);vs=torch.stack(values,-2)
        scores=torch.einsum('biphd,bjphld->biphjl',q,ks)/math.sqrt(c)
        scores=scores+self.edge(edges).permute(0,1,3,2)[:,:,None,:,:,None]
        previous=torch.stack([F.pad(ends[:-lag],(lag,0)) if 0<lag<p else
            torch.zeros_like(ends) if lag>=p else ends for lag in self.lags],-1)
        # The last partial patch may be shorter: use actual endpoints, not lag*patch.
        lag_seconds=interval[:,None,None]*(ends[:,None]-previous)[None]
        lag_bias=self.lag_time(time_features(-lag_seconds)).permute(0,1,3,2)
        scores=scores+lag_bias[:,None,:,:,None,:]
        allowed=adjacency[:,:,None,None,:,None]&torch.stack(masks,-1).permute(0,2,1,3)[:,None,:,None]
        probability=safe_softmax(scores.flatten(-2),allowed.expand(-1,-1,-1,h,-1,-1).flatten(-2)).reshape(scores.shape)
        message=torch.einsum('biphjl,bjphld->biphd',probability,vs).reshape(b,n,p,d)
        x=(x+self.mix(message)).masked_fill(~observed[...,None],0)
        return (x+self.ff(self.ffnorm(x))).masked_fill(~observed[...,None],0)


class VariableGraphExpert(nn.Module):
    def __init__(self,width=64,patch=4,blocks=3,heads=4):
        super().__init__();self.width=width;self.patch=patch
        self.patch_projection=nn.Linear(2*patch,width)
        self.history_time=nn.Linear(7,width);self.interval=nn.Linear(7,width)
        self.blocks=nn.ModuleList(SpaceTimeBlock(width,heads) for _ in range(blocks))
        self.query=nn.Sequential(nn.Linear(14,width),nn.SiLU(),nn.Linear(width,width))
        self.read=nn.MultiheadAttention(width,heads,dropout=0,batch_first=True)
        self.final=nn.Sequential(nn.LayerNorm(width*2),nn.Linear(width*2,width),nn.SiLU(),nn.Linear(width,1))
        nn.init.normal_(self.final[-1].weight,std=.001);nn.init.zeros_(self.final[-1].bias)

    def forward(self,normalized,observed,minimum,extent,last,adjacency,edges,interval,horizon):
        if normalized.ndim!=3 or observed.shape!=normalized.shape:
            raise ValueError('Complete B,N,L history and observation mask required')
        b,n,length=normalized.shape
        if not 10<=length<=288 or not 1<=horizon<=144 or n>31 or n<1:
            raise ValueError('Declared L10..288/H1..144 and full at-most31 real nodes')
        if interval.shape!=(b,) or not torch.isfinite(interval).all() or not (interval>0).all():
            raise ValueError('Positive actual historical sampling intervals required')
        if not observed[:,0].any(-1).all():raise ValueError('Observed target history required')
        if adjacency.shape!=(b,n,n) or edges.shape!=(b,n,n,4):raise ValueError('Receiver-row physical topology required')
        if not torch.isfinite(normalized.masked_select(observed)).all() or not (extent>0).all():
            raise ValueError('Finite observations and positive origin scales required')
        pad=(-length)%self.patch
        # Right padding retains every original value; partial-patch masks stay explicit.
        values=F.pad(normalized.float().masked_fill(~observed,0),(0,pad))
        masks=F.pad(observed,(0,pad));p=values.shape[-1]//self.patch
        z=torch.cat((values.reshape(b,n,p,self.patch),masks.reshape(b,n,p,self.patch).float()),-1)
        valid=masks.reshape(b,n,p,self.patch).any(-1)
        # The final partial patch ends at the last actual observation position.
        ends=torch.arange(1,p+1,device=z.device).mul(self.patch).clamp_max(length)-length
        seconds=interval[:,None]*ends[None]
        z=self.patch_projection(z)+self.history_time(time_features(seconds))[:,None]
        z=z+self.interval(time_features(interval))[:,None,None]
        z=z.masked_fill(~valid[...,None],0)
        for block in self.blocks:z=block(z,valid,adjacency.bool(),edges.float(),interval,ends)
        future=interval[:,None]*torch.arange(1,horizon+1,device=z.device)[None]
        q=self.query(torch.cat((time_features(future),time_features(interval)[:,None].expand(-1,horizon,-1)),-1))
        q=q[:,None].expand(-1,n,-1,-1).reshape(b*n,horizon,self.width)
        # No future-query self-attention or h/H: overlapping forecasts are consistent.
        keys=z.reshape(b*n,p,self.width);mask=~valid.reshape(b*n,p)
        keys=torch.cat((keys.new_zeros(b*n,1,self.width),keys),1)
        mask=torch.cat((torch.zeros(b*n,1,dtype=torch.bool,device=z.device),mask),1)
        read=self.read(q,keys,keys,key_padding_mask=mask,need_weights=False)[0]
        residual=self.final(torch.cat((q,read),-1)).reshape(b,n,horizon)
        pred=last.double()[...,None]+extent.double()[...,None]*residual.double()
        if not torch.isfinite(pred).all():raise FloatingPointError('Nonfinite expert prediction')
        return pred


def pack_items(items):
    if len({i.history.shape[-1] for i in items})!=1:raise ValueError('Group only equal L for batched execution')
    n=max(len(i.node_ids) for i in items);out={}
    for name in ('normalized','observed','minimum','extent','last','adjacency','edges'):
        arrays=[]
        for item in items:
            x=getattr(item,name);extra=n-len(item.node_ids)
            if name in ('normalized','observed'):x=F.pad(x,(0,0,0,extra))
            elif name=='adjacency':x=F.pad(x,(0,extra,0,extra))
            elif name=='edges':x=F.pad(x,(0,0,0,extra,0,extra))
            else:x=F.pad(x,(0,extra),value=1. if name=='extent' else 0.)
            arrays.append(x)
        out[name]=torch.stack(arrays)
    return out
