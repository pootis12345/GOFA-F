"""GPU execution of unchanged shared source models; historical UFG graphs stay CPU-exact."""
from collections import defaultdict
from types import SimpleNamespace

import torch

from gofa_f.dual_memory_policy import query_loss
from gofa_f.ufg_memory_roster import pack_roster
from gofa_f.ufg_subgraph import SubgraphInput
from gofa_f.variable_graph_expert import pack_items


def move_pack(value,device):
    return SubgraphInput(*(getattr(value,k).to(device) for k in
        ('normalized','observed','minimum','extent','present')),value.real_nodes)


_INTERPOLATION_BASES={}


def deterministic_interpolation():
    """Evaluate the unchanged linear frequency basis with deterministic GEMMs.

    PyTorch2.1 CUDA interpolation backward uses nondeterministic atomics.
    The CPU-generated interpolation matrix is parameter-free; learned values
    and their gradients stay on CUDA. CPU and unrelated modes are untouched.
    """
    from torch.nn import functional as functional
    original=functional.interpolate
    if getattr(original,'fixed_linear_basis',False):return
    pool=functional.adaptive_avg_pool1d
    def fixed_pool(x,output_size):
        size=output_size if isinstance(output_size,int) else output_size[0]
        if x.device.type=='cuda' and x.shape[-1]%size==0:
            return x.reshape(*x.shape[:-1],size,x.shape[-1]//size).mean(-1)
        return pool(x,output_size)
    functional.adaptive_avg_pool1d=fixed_pool
    def basis(old,new,dtype,device):
        key=(old,new,dtype,str(device))
        if key not in _INTERPOLATION_BASES:
            eye=torch.eye(old,dtype=dtype).reshape(old,1,old)
            matrix=original(eye,size=new,mode='linear',align_corners=True)[:,0].T.contiguous()
            _INTERPOLATION_BASES[key]=matrix.to(device)
        return _INTERPOLATION_BASES[key]
    def interpolate(x,size=None,scale_factor=None,mode='nearest',align_corners=None,**kwargs):
        if x.device.type!='cuda' or scale_factor is not None or align_corners is not True or mode not in ('linear','bilinear'):
            return original(x,size=size,scale_factor=scale_factor,mode=mode,align_corners=align_corners,**kwargs)
        if mode=='linear':
            new=size if isinstance(size,int) else size[0]
            return x@basis(x.shape[-1],new,x.dtype,x.device).T
        height,width=(size,size) if isinstance(size,int) else size
        return basis(x.shape[-2],height,x.dtype,x.device)@x@basis(x.shape[-1],width,x.dtype,x.device).T
    interpolate.fixed_linear_basis=True
    functional.interpolate=interpolate


class DeviceEngine:
    def __init__(self,base,kind,device):
        self.kind=kind;self.store=base.store;self.manifest=base.manifest;self.device=torch.device(device)
        self.model=self.state_model=base.model.to(self.device);self.cuda=self.device.type=='cuda';self.microbatch=base.microbatch
        self._operators=None
        if kind=='expert' and self.cuda:
            from gofa_f import variable_graph_expert as implementation
            current=implementation.time_features
            if not getattr(current,'fixed_cpu_metadata',False):
                def fixed_time_features(seconds):
                    if seconds.device.type=='cuda':return current(seconds.detach().cpu()).to(seconds.device)
                    return current(seconds)
                fixed_time_features.fixed_cpu_metadata=True
                implementation.time_features=fixed_time_features
        if kind!='expert' and self.cuda:
            deterministic_interpolation()
            core=self.model.core.official
            self._construct=type(core).construct_laplacian;self._get_operator=type(core).get_operator
            self._graph_parameters=(core.k,core.approx,core.s,core.J,core.lev)
            # No learned CPU copy, no additional model parameters. Only the
            # already accepted historical kNN/degree/Chebyshev calculations.
            core.construct_laplacian=lambda *args,**kwargs:None
            core.get_operator=lambda *args,**kwargs:self._operators
            core.device=str(self.device)

    def warmup(self,enabled):pass

    def _historical_operators(self,packed):
        if self.kind=='expert' or not self.cuda:return
        x=packed.normalized.float().masked_fill(~packed.observed,0);b,length,n=x.shape
        freq=max(16,length)//2+1
        z=torch.fft.rfft(x,n=max(16,length),dim=1,norm='ortho').permute(0,2,1).contiguous().reshape(b,n*freq,1)
        k,approx,s,j,level=self._graph_parameters;reference=SimpleNamespace(device='cpu')
        laplacian=self._construct(reference,torch.cat((z.real,z.imag),-1),k,num_nodes=b*n*freq)
        operators=self._get_operator(reference,laplacian,approx,s,j,level,'cpu')
        self._operators=[v.to(self.device) for v in operators]

    def predictions(self,rows,split,role='fit'):
        values=[self.store.get(r,split,role) for r in rows];items=[v[0] for v in values]
        h=rows[0]['horizon']
        if len({(r['length'],r['horizon']) for r in rows})!=1:raise ValueError('Shape-homogeneous execution required')
        if self.kind=='expert':
            packed={k:v.to(self.device) for k,v in pack_items(items).items()}
            prediction=self.model(**packed,interval=torch.tensor([self.store.interval(r) for r in rows],device=self.device),horizon=h)
        else:
            own=[pack_roster(i) for i in items]
            cpu=SubgraphInput(*(torch.cat([getattr(v,k) for v in own],0) for k in
                ('normalized','observed','minimum','extent','present')),31)
            self._historical_operators(cpu)
            prediction=self.model(move_pack(cpu,self.device),h,all_nodes=True).transpose(1,2)
        return prediction,values

    def loss(self,rows):
        groups=defaultdict(list)
        for row in rows:groups[(row['length'],row['horizon'])].append(row)
        losses=[]
        try:
            for own in groups.values():
                pred,values=self.predictions(own,'train')
                for i,(row,(item,target,mask)) in enumerate(zip(own,values)):
                    losses.append(query_loss(pred[i,:len(item.node_ids)],target.to(self.device),mask.to(self.device),
                        item.present.to(self.device),self.manifest['training_unit_normalizers'][row['source']]))
        except RuntimeError as exc:
            if self.cuda and ('out of memory' in str(exc).lower() or 'cudaerrormemoryallocation' in str(exc).lower()):
                raise torch.OutOfMemoryError(str(exc)) from exc
            raise
        return sum(losses)

    @torch.no_grad()
    def predict(self,row,split,role='fit'):
        pred,values=self.predictions([row],split,role);item,target,mask=values[0]
        return pred[0,0].cpu().numpy(),target[0].numpy(),mask[0].numpy(),item.identity

    def close(self):
        self._operators=None;self.model=self.state_model=None
        if self.cuda:torch.cuda.empty_cache()
