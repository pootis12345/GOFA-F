"""Exact real-source partial recovery in isolated copies, never resume live jobs."""
import argparse
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
import torch

from gofa_f.dual_memory_engines import model_digest
from gofa_f.dual_memory_policy import accumulate,capture,restore
from gofa_f.dual_memory_store import atomic_torch
from gofa_f.dual_memory_warmup import cpu_only
from gofa_f.generative_data import sha
from gofa_f.prefix_cache import atomic_json
from gofa_f.variable_graph_data import VariableStore,VariableSchedule
from gofa_f.variable_graph_source import ExpertEngine,configuration,runtime


def exactly(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(exactly(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(exactly(x,y) for x,y in zip(a,b))
    return a==b


def check_case(cfg,manifest,store,kind,checkpoint,output):
    # Read one atomic generation: the live path may be replaced by the next update.
    payload=Path(checkpoint).read_bytes();saved=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False)
    initial_update=saved['state']['updates'];began=time.monotonic();binding=saved['binding']
    first=ExpertEngine(cfg,manifest,store,kind);optimizer=torch.optim.AdamW(first.model.parameters(),lr=cfg['training']['learning_rate'])
    state=restore(first.model,optimizer,saved,binding);start_digest=model_digest(first.model)
    if not initial_update or state['pending']:raise ValueError('Nonzero committed checkpoint required')
    schedule=VariableSchedule(manifest['queries']);schedule.commit(state['cursor']);first.model.train()
    effective=cfg['training']['effective_batch'];micro=cfg['training']['microbatch']
    def forward(engine,optim,state,order,limit):
        while state['pending']<limit:
            rows=[];count=min(micro,limit-state['pending'])
            for _ in range(count):
                row,next_state=order.peek();order.commit(next_state);rows.append(row)
            loss=engine.loss(rows)/effective;accumulate(loss,list(engine.model.parameters()))
            state['pending']+=count;state['loss_sum']+=float(loss.detach());state['train_query_count']+=count
            state['cursor']=deepcopy(order.state)
    def finish(engine,optim,state,order):
        forward(engine,optim,state,order,effective)
        for group in optim.param_groups:group['lr']=cfg['training']['learning_rate']*min(1.,(state['updates']+1)/cfg['training']['warmup_updates'])
        torch.nn.utils.clip_grad_norm_(list(engine.model.parameters()),cfg['training']['gradient_clip'],error_if_nonfinite=True)
        optim.step();optim.zero_grad(set_to_none=True);state['updates']+=1;state['pending']=0;state['loss_sum']=0.
        return capture(engine.model,optim,state,binding)
    forward(first,optimizer,state,schedule,12)
    partial=capture(first.model,optimizer,state,binding);path=output/(kind+'-partial.pt');atomic_torch(path,partial)
    if partial['state']['pending']!=12 or not any(x is not None for x in partial['gradients'].values()):raise ValueError('Actual partial gradients required')
    expected=finish(first,optimizer,state,schedule)
    second=ExpertEngine(cfg,manifest,store,kind);other=torch.optim.AdamW(second.model.parameters(),lr=cfg['training']['learning_rate'])
    value=torch.load(path,map_location='cpu',weights_only=False);recovered=restore(second.model,other,value,binding)
    order=VariableSchedule(manifest['queries']);order.commit(recovered['cursor']);second.model.train()
    actual=finish(second,other,recovered,order)
    if not exactly(expected,actual):raise ValueError('Real-source partial recovery is not bitwise exact')
    result=dict(passed=True,initial_update=initial_update,source_checkpoint_sha256=hashlib.sha256(payload).hexdigest(),
        initial_model_sha256=start_digest,partial_queries=12,completed_effective_batch=effective,
        resumed_update=recovered['updates'],final_model_sha256=model_digest(second.model),
        exactly_equal=['model','optimizer','gradients','state_and_cursor','python_numpy_torch_rng'],
        partial_checkpoint_sha256=sha(path),elapsed_cpu_residency_seconds=time.monotonic()-began)
    first.close();second.close();return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--source',required=True)
    p.add_argument('--ufg-qa-checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args();cpu_only()
    root=Path(a.source);out=Path(a.output)
    if out.exists():raise ValueError('Preserve the previous recovery proof; do not overwrite')
    out.mkdir(parents=True);cfg=configuration(a.config);manifest=json.loads((root/'manifest.json').read_text())
    if manifest['variable_runtime']!=runtime() or manifest['variable_configuration']!=cfg:raise ValueError('Formal source runtime changed')
    store=VariableStore(manifest,root/'source_inputs')
    if not (root/'source_inputs/index.json').exists():raise ValueError('Use only completed read-only source inputs')
    results={}
    results['trained_expert']=check_case(cfg,manifest,store,'expert',root/'expert/all/last.pt',out)
    results['nonzero_ufg_QA_only']=check_case(cfg,manifest,store,'ufg',Path(a.ufg_qa_checkpoint),out)
    proof=dict(passed=True,source_manifest_sha256=sha(root/'manifest.json'),runtime_code=runtime(),results=results,
        no_live_state_modified=True,no_training_restart=True,formal_state_reused_from_QA=False,
        cuda_initialized=torch.cuda.is_initialized(),new_gpu_seconds=0,threads=2,
        target_predictions=0,target_scores_read=False)
    if proof['cuda_initialized']:raise ValueError('Unexpected CUDA initialization')
    atomic_json(out/'proof.json',proof);print(json.dumps(proof,indent=2),flush=True)


if __name__=='__main__':main()
