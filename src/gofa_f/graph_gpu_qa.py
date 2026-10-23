"""Device migration acceptance; independent of final target scores."""
from copy import deepcopy
from pathlib import Path
import random
import time

import numpy as np
import torch

from gofa_f.dual_memory_fitting import read
from gofa_f.dual_memory_policy import accumulate,capture,restore,SourceSchedule
from gofa_f.dual_memory_store import atomic_torch
from gofa_f.dual_memory_engines import model_digest
from gofa_f.generative_data import sha
from gofa_f.prefix_cache import atomic_json
from gofa_f.variable_graph_data import VariableSchedule


def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a) is type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def replay_close(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-6);return
    if isinstance(a,dict):
        if a.keys()!=b.keys():raise ValueError('Replay fields changed')
        for key in a:replay_close(a[key],b[key])
    elif isinstance(a,(tuple,list)):
        if type(a) is not type(b) or len(a)!=len(b):raise ValueError('Replay layout changed')
        for x,y in zip(a,b):replay_close(x,y)
    elif not equal(a,b):raise ValueError('Replay metadata changed')


def acceptance(root,case,cpu_only=False):
    from gofa_f.graph_gpu_execution import construct,source_settings,gpu_environment,runtime
    torch.set_num_threads(2);source,work,m,cfg=source_settings(root,case);began=time.monotonic()
    if not cpu_only:gpu_environment()
    else:
        if torch.cuda.is_initialized():raise ValueError('CPU regression initialized CUDA')
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
    original=construct(root,case,'cpu',base_only=True)
    path=work/'all/last.pt';saved=torch.load(path,map_location='cpu',weights_only=False) if path.exists() else None
    if saved:original.model.load_state_dict(saved['model'])
    device='cpu' if cpu_only else 'cuda';candidate=construct(root,case,device);candidate.model.load_state_dict(original.model.state_dict())
    original.model.eval();candidate.model.eval()
    rows=[v['train'][0] for d in m['queries'].values() for v in d.values() if v['train']]
    predictions=[]
    for row in rows:
        expected,_,_,_=original.predict(row,'train');actual,_,_,_=candidate.predict(row,'train')
        item,_,_=original.store.get(row,'train');scale=float(item.extent[0]);last=float(item.last[0])
        # Cross-device FP32 arithmetic is checked in each origin's dimensionless
        # numerical scale. Raw-unit errors remain in evidence, including the
        # prior diagnostic failures; no raw bitwise trajectory claim is made.
        cross_close=np.allclose((expected-last)/scale,(actual-last)/scale,rtol=1e-6,atol=1e-6)
        if cpu_only:torch.testing.assert_close(torch.as_tensor(expected),torch.as_tensor(actual),rtol=1e-6,atol=1e-6)
        else:
            repeated,_,_,_=candidate.predict(row,'train')
            torch.testing.assert_close(torch.as_tensor(actual),torch.as_tensor(repeated),rtol=1e-6,atol=1e-6)
        predictions.append(dict(query=row,max_raw_abs_error=float(np.abs(expected-actual).max()),
            max_origin_normalized_abs_error=float((np.abs(expected-actual)/scale).max()),cross_device_1e6_diagnostic_passed=bool(cross_close)))
    original.model.train();candidate.model.train()
    loss=original.loss(rows)/len(rows);reference=torch.autograd.grad(loss,[p for p in original.model.parameters() if p.requires_grad])
    other=candidate.loss(rows)/len(rows);gradients=torch.autograd.grad(other,[p for p in candidate.model.parameters() if p.requires_grad])
    grad_max=0.;cross_gradient=True
    for a,b in zip(reference,gradients):
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():raise ValueError('Nonfinite gradients')
        cross_gradient=cross_gradient and torch.allclose(a,b.cpu(),rtol=1e-6,atol=1e-6)
        if cpu_only:torch.testing.assert_close(a,b.cpu(),rtol=1e-6,atol=1e-6)
        grad_max=max(grad_max,float((a-b.cpu()).abs().max()))
    proof=dict(passed=True,case=case,execution_manifest_sha256=sha(root/'execution_manifest.json'),runtime=runtime(),
        original_model_sha256=model_digest(original.model),predictions=predictions,gradient_max_abs_error=grad_max,
        origin_normalized_forward_rtol=1e-6,origin_normalized_forward_atol=1e-6,gradient_rtol=1e-6,gradient_atol=1e-6,
        raw_CPU_GPU_bitwise_or_unscaled_1e6_equivalence_claimed=False,new_target_predictions=0,target_scores_read=False)
    proof.update(cross_device_forward_1e6_diagnostic_passed=all(v['cross_device_1e6_diagnostic_passed'] for v in predictions),
        cross_device_gradient_1e6_diagnostic_passed=bool(cross_gradient),
        CPU_wrapper_1e6_required=cpu_only,within_GPU_repeated_forward_1e6_required=not cpu_only)
    if not cpu_only:
        parameters=list(candidate.model.parameters());optimizer=torch.optim.AdamW(parameters,lr=cfg['training']['learning_rate'],weight_decay=cfg['training']['weight_decay'])
        if saved:
            state=restore(candidate.model,optimizer,saved,saved['binding']);binding=saved['binding']
            transferred=capture(candidate.model,optimizer,state,binding,True)
            for key in ('model','optimizer','gradients','state'):
                if not equal(saved[key],transferred[key]):raise ValueError('CPU to GPU state transfer changed '+key)
            for key in ('python','numpy','torch'):
                if not equal(saved['rng'][key],transferred['rng'][key]):raise ValueError('CPU RNG changed during transfer')
        else:
            state=dict(updates=0,pending=0,loss_sum=0.,cursor=None,train_query_count=0);binding='QA_ONLY_NOT_FORMAL'
        if case=='primary_ufg':schedule=SourceSchedule(m['queries'],initial=8192,expand=True)
        else:schedule=VariableSchedule(m['queries'])
        if state['cursor'] is not None:schedule.commit(state['cursor'])
        effective=cfg['training']['effective_batch']
        def micro(limit):
            while state['pending']<limit:
                count=min(4,limit-state['pending']);queries=[]
                for _ in range(count):q,next_state=schedule.peek();schedule.commit(next_state);queries.append(q)
                loss=candidate.loss(queries)/effective;accumulate(loss,parameters);state['pending']+=count
                state['loss_sum']+=float(loss.detach());state['train_query_count']+=count
                state['cursor']=deepcopy(schedule.state)
        if not saved:
            micro(effective);torch.nn.utils.clip_grad_norm_(parameters,5,error_if_nonfinite=True);optimizer.step();optimizer.zero_grad(set_to_none=True)
            state['updates']=1;state['pending']=0;state['loss_sum']=0.
        micro(min(effective,max(state['pending'],4)))
        partial=capture(candidate.model,optimizer,state,binding,True)
        folder=root/'qa_partial';folder.mkdir(exist_ok=True);atomic_torch(folder/(case+'.pt'),partial)
        def finish():
            micro(effective);torch.nn.utils.clip_grad_norm_(parameters,5,error_if_nonfinite=True);optimizer.step();optimizer.zero_grad(set_to_none=True)
            state['updates']+=1;state['pending']=0;state['loss_sum']=0.
            return capture(candidate.model,optimizer,state,binding,True)
        expected=finish();state=restore(candidate.model,optimizer,partial,binding);schedule.commit(state['cursor'])
        after_restore=capture(candidate.model,optimizer,state,binding,True)
        if not equal(after_restore,partial):raise ValueError('GPU partial snapshot was not exactly restored')
        actual=finish()
        replay_close(expected['optimizer'],actual['optimizer'])
        # State transfer is bitwise; repeated CUDA arithmetic uses the unchanged
        # FP32 tolerance and is reported separately from the transfer itself.
        for key in ('model','gradients'):
            for name,value in expected[key].items():
                if value is None:
                    if actual[key][name] is not None:raise ValueError('Gradient presence changed')
                else:torch.testing.assert_close(value,actual[key][name],rtol=1e-6,atol=1e-6)
        if not equal(expected['state'],actual['state']) or not equal(expected['rng'],actual['rng']):raise ValueError('CUDA RNG or cursor changed')
        proof.update(exact_state_transfer=bool(saved),exact_GPU_partial_snapshot_restore=True,
            GPU_replay_parameter_tolerance=1e-6,QA_states_not_used_for_formal_training=True,
            starting_cpu_update=saved['state']['updates'] if saved else None,
            starting_cpu_pending=saved['state']['pending'] if saved else None,
            peak_torch_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
    proof['elapsed_cpu_residency_seconds']=time.monotonic()-began
    folder=root/('cpu_acceptance' if cpu_only else 'gpu_acceptance');folder.mkdir(exist_ok=True)
    atomic_json(folder/(case+'.json'),proof);print({k:v for k,v in proof.items() if k not in ('runtime','predictions')},flush=True)
