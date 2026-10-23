"""One UFG parameter set for both history lengths and forecast horizons.

Canonical parameters use L48/H24, width128 and31 roster roles. Short history
is never padded/truncated/resampled: the spectral filters and input-indexed
linear weights are deterministically evaluated on its actual length. No
shape-specific parameters, optimizers, model copies or checkpoint choices.
"""
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F

from gofa_f.ufg import UFGTime
from gofa_f.ufg_memory_roster import pack_roster
from gofa_f.ufg_subgraph import pack_history,SubgraphInput
from gofa_f.dual_memory_policy import query_loss


def history_weight(weight,length):
    """Shared temporal basis; preserve constant response for12/48 bins."""
    if length==weight.shape[-1]:return weight
    if (weight.shape[-1],length)!=(48,12):raise ValueError('Only locked12/48 history grids')
    return F.adaptive_avg_pool1d(weight[None],length)[0]*(weight.shape[-1]/length)


def frequency_filter(weight,frequencies):
    if frequencies==25:return weight
    if frequencies!=9:raise ValueError('Only pinned16/48-point FFT grids')
    return F.interpolate(weight.reshape(31,1,25),size=frequencies,mode='linear',align_corners=True).reshape(-1,1)


def spectral_linear(weight,bias,frequencies,width=128):
    """Resize frequency axes only; never mix or resize learned channels."""
    if frequencies==25:return weight,bias
    if frequencies!=9 or weight.shape!=(25,width*25):raise ValueError('Shared canonical complex projection required')
    z=weight.reshape(25,width,25).permute(1,0,2)
    real=F.interpolate(z.real[None],size=(frequencies,frequencies),mode='bilinear',align_corners=True)[0]
    imag=F.interpolate(z.imag[None],size=(frequencies,frequencies),mode='bilinear',align_corners=True)[0]
    w=torch.complex(real,imag).permute(1,0,2).reshape(frequencies,width*frequencies)*(25/frequencies)
    def resize(v):return F.interpolate(v[None,None],size=frequencies,mode='linear',align_corners=True)[0,0]
    return w,torch.complex(resize(bias.real),resize(bias.imag))


class SharedRosterUFG(nn.Module):
    def __init__(self,root,device='cpu'):
        super().__init__()
        self.core=UFGTime(root,48,24,31,device=device,width=128)
        self.provenance=dict(self.core.provenance,cross_shape_single_checkpoint=True,
            shape_specific_parameters=False,canonical_history=48,maximum_horizon=24,
            history_padding_or_resampling=False,short_forecast='first12_of_shared24_outputs',
            parameter_adaptation='parameter_free_temporal_bin_and_frequency_grid_maps',
            query_graphs='disjoint_batched_knn_not_cross_query_edges',
            exact_official_forward_only_canonical_single_query=True,
            real_node_cap=31,absent_roles='masked_zero_trailing_nodes_not_real_observations')

    def normalized_forecast(self,normalized,observed):
        x=normalized.float().masked_fill(~observed,0)
        if x.ndim!=3 or x.shape[1] not in (12,48) or x.shape[2]!=31:
            raise ValueError('One shared model accepts complete12/48 by31-role histories')
        base=self.core.official;b,length,n=x.shape;signal=max(16,length);freq=signal//2+1
        _,trend=base.decompsition(x)
        trend=F.linear(trend.permute(0,2,1),history_weight(base.Linear_Trend.weight,length),base.Linear_Trend.bias)
        z=torch.fft.rfft(x,n=signal,dim=1,norm='ortho').permute(0,2,1).contiguous().reshape(b,n*freq,1)
        # DGL's3-D kNN input explicitly separates queries. Different temporal
        # origins must not exchange observations through a training batch.
        laplacian=base.construct_laplacian(torch.cat((z.real,z.imag),-1),base.k,num_nodes=b*n*freq)
        operators=base.get_operator(laplacian,base.approx,base.s,base.J,base.lev,base.device)
        flat=z.reshape(b*n*freq,1)
        mixed=sum(frequency_filter(base.filters[i],freq).repeat(b,1)*base.conv1_list[i](
            flat,operators[i].indices(),operators[i].val) for i in range(base.lev+1))
        mixed=mixed.reshape(b,n,freq,-1).permute(0,1,3,2).reshape(b,n,-1)
        weight,bias=spectral_linear(base.clin.weight,base.clin.bias,freq)
        time=torch.fft.irfft(F.linear(mixed,weight,bias),n=length,dim=-1,norm='ortho')
        time=base.act_real(base.isn(time))
        time=F.linear(time,history_weight(base.lin2.weight,length),base.lin2.bias)
        time=base.act_real(base.isn(time));time=base.lin3(time)
        return base.lin4(torch.cat((time,trend),-1)).transpose(1,2).contiguous()

    def forward(self,packed,horizon,all_nodes=False):
        length=packed.normalized.shape[1]
        if (length,horizon) not in ((12,12),(48,24)):raise ValueError('Only the two registered forecasting tasks')
        pred=self.normalized_forecast(packed.normalized,packed.observed)[:,:horizon]
        raw=(pred.double()+.5)*packed.extent[:,None]+packed.minimum[:,None]
        if not torch.isfinite(raw).all():raise FloatingPointError('Nonfinite shared UFG forecast')
        return raw[:,:,:packed.real_nodes] if all_nodes else raw[:,:,0]


class SharedUFGEngine:
    def __init__(self,root,manifest,store):
        self.model=self.state_model=SharedRosterUFG(root)
        self.manifest,self.store=manifest,store;self.cuda=False;self.microbatch=64

    def warmup(self,enabled):pass

    def loss(self,rows):
        groups=defaultdict(list)
        for row in rows:groups[(row['length'],row['horizon'])].append(row)
        losses=[]
        for (length,horizon),own in groups.items():
            values=[];targets=[];masks=[]
            for row in own:
                item,target,mask=self.store.get(row,'train')
                values.append(pack_roster(item))
                targets.append(F.pad(target,(0,0,0,31-len(item.node_ids))))
                masks.append(F.pad(mask,(0,0,0,31-len(item.node_ids))))
            packed=SubgraphInput(*(torch.cat([getattr(v,k) for v in values],0) for k in
                ('normalized','observed','minimum','extent','present')),31)
            pred=self.model(packed,horizon,all_nodes=True).transpose(1,2)
            losses.extend(query_loss(pred[i],targets[i],masks[i],packed.present[i],
                self.manifest['training_unit_normalizers'][row['source']]) for i,row in enumerate(own))
        return sum(losses)

    @torch.no_grad()
    def predict(self,row,split,role='fit'):
        item,target,mask=self.store.get(row,split,role)
        prediction=self.model(pack_roster(item),row['horizon'])[0]
        return prediction.numpy(),target[0].numpy(),mask[0].numpy(),item.identity

    def close(self):self.model=None
