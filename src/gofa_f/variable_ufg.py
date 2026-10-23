"""ONE canonical L48/H144 UFG parameter set for the declared variable range.

No per-shape state. Actual observations enter the actual-length FFT unchanged;
only shared learned temporal/frequency bases are evaluated at another grid.
"""
import torch
from torch import nn
from torch.nn import functional as F

from gofa_f.ufg import UFGTime


def temporal_weight(weight,length):
    old=weight.shape[-1]
    if length==old:return weight
    # Exact overlap of normalized temporal bins. This also preserves constants
    # when L is not a divisor of48, unlike integer-only pooling shortcuts.
    left=torch.arange(length,device=weight.device,dtype=weight.dtype)/length
    right=(torch.arange(length,device=weight.device,dtype=weight.dtype)+1)/length
    a=torch.arange(old,device=weight.device,dtype=weight.dtype)/old
    b=(torch.arange(old,device=weight.device,dtype=weight.dtype)+1)/old
    overlap=(torch.minimum(b[:,None],right[None])-torch.maximum(a[:,None],left[None])).clamp_min(0)*old
    return weight@overlap


def resize_filter(weight,f):
    if f==25:return weight
    return F.interpolate(weight.reshape(31,1,25),size=f,mode='linear',align_corners=True).reshape(-1,1)


def resize_complex(weight,bias,f,width):
    if f==25:return weight,bias
    x=weight.reshape(25,width,25).permute(1,0,2)
    real=F.interpolate(x.real[None],size=(f,f),mode='bilinear',align_corners=True)[0]
    imag=F.interpolate(x.imag[None],size=(f,f),mode='bilinear',align_corners=True)[0]
    w=torch.complex(real,imag).permute(1,0,2).reshape(f,width*f)*(25/f)
    def resize(v):return F.interpolate(v[None,None],size=f,mode='linear',align_corners=True)[0,0]
    return w,torch.complex(resize(bias.real),resize(bias.imag))


class VariableRosterUFG(nn.Module):
    def __init__(self,root,device='cpu'):
        super().__init__();self.core=UFGTime(root,48,144,31,device=device,width=128)
        self.provenance=dict(self.core.provenance,one_parameter_set=True,shape_specific_parameters=False,
            canonical_history=48,maximum_horizon=144,forecast='prefix_of_one_shared144_head',
            adaptation='exact_temporal_bin_overlap; frequency_basis_interpolation',
            input_values_resampled=False,query_graphs='disjoint',length_range=[10,288],horizon_range=[1,144])

    def normalized_forecast(self,normalized,observed):
        x=normalized.float().masked_fill(~observed,0)
        b,length,n=x.shape
        if n!=31 or not 10<=length<=288:raise ValueError('Complete actual-length history and31 masked roles required')
        base=self.core.official;signal=max(16,length);freq=signal//2+1
        _,trend=base.decompsition(x)
        trend=F.linear(trend.permute(0,2,1),temporal_weight(base.Linear_Trend.weight,length),base.Linear_Trend.bias)
        z=torch.fft.rfft(x,n=signal,dim=1,norm='ortho').permute(0,2,1).contiguous().reshape(b,n*freq,1)
        laplacian=base.construct_laplacian(torch.cat((z.real,z.imag),-1),base.k,num_nodes=b*n*freq)
        ops=base.get_operator(laplacian,base.approx,base.s,base.J,base.lev,base.device)
        flat=z.reshape(b*n*freq,1)
        mixed=sum(resize_filter(base.filters[i],freq).repeat(b,1)*base.conv1_list[i](flat,ops[i].indices(),ops[i].val)
                  for i in range(base.lev+1))
        mixed=mixed.reshape(b,n,freq,-1).permute(0,1,3,2).reshape(b,n,-1)
        w,bias=resize_complex(base.clin.weight,base.clin.bias,freq,128)
        time=torch.fft.irfft(F.linear(mixed,w,bias),n=length,dim=-1,norm='ortho')
        time=base.act_real(base.isn(time))
        time=F.linear(time,temporal_weight(base.lin2.weight,length),base.lin2.bias)
        time=base.lin3(base.act_real(base.isn(time)))
        return base.lin4(torch.cat((time,trend),-1)).transpose(1,2).contiguous()

    def forward(self,packed,horizon,all_nodes=False):
        if not 1<=horizon<=144:raise ValueError('Shared H1..144 required')
        z=self.normalized_forecast(packed.normalized,packed.observed)[:,:horizon]
        raw=(z.double()+.5)*packed.extent[:,None]+packed.minimum[:,None]
        if not torch.isfinite(raw).all():raise FloatingPointError('Nonfinite variable UFG')
        return raw[:,:,:packed.real_nodes] if all_nodes else raw[:,:,0]
