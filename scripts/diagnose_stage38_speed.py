"""Fixed Stage39 CPU attribution; no optimizer, fitting, target or confirmation access."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch

from gofa_f import source_graph_probe as p
from gofa_f.lag_graph_probe import state_digest
from gofa_f.generative_data import sha

PARENT_REPORT='e38a95b55f20579f9467f0c42a3b6cf7b1df1dae06843c66f00742d21eeeffae'


@contextmanager
def disable_exchanges(model,indices):
    """Zero cross support in selected blocks; always remove diagnostic hooks."""
    handles=[]
    def zero_support(module,inputs):
        hidden,observed,adjacency,edge=inputs
        return hidden,observed,torch.zeros_like(adjacency),edge
    try:
        for i in indices:handles.append(model.exchanges[i].register_forward_pre_hook(zero_support))
        yield
    finally:
        for h in handles:h.remove()


def grouped_error(prediction,a,rows):
    result={}
    for first,last in ((0,4),(4,8),(8,12)):
        per_variable=[]
        for v in sorted({r['query']['variable'] for r in rows}):
            ix=[i for i,r in enumerate(rows) if r['query']['variable']==v]
            mask=a['observed'][ix,first:last]
            if mask.any():per_variable.append(float(np.abs(prediction[ix,first:last]-a['raw_target'][ix,first:last])[mask].mean()))
        result[f'{first+1}-{last}']=float(np.mean(per_variable))
    return result


def cosine(a,b):
    denominator=float(a.norm()*b.norm())
    return float(torch.dot(a,b)/denominator) if denominator>0 else None


def main():
    start=time.monotonic();started=time.time()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():
        raise ValueError('Explicit CUDA-disabled CPU execution required')
    torch.set_num_threads(2)
    parent=args.workspace/'outputs/stage38-source-neighborhood-probe-v1'
    assert sha(parent/'report.json')==PARENT_REPORT
    assert p.read(parent/'final_verification.json')['passed']
    assert p.read(parent/'activity.json')['worker'] is None
    assert not (parent/'source_confirmation.json').exists()
    m=p.read(parent/'manifest.json');cfg=m['configuration'];report=p.read(parent/'report.json')
    assert all(m['code'][n]==v for n,v in p.code_hashes().items())
    if args.output.exists():raise ValueError('One-shot diagnostic output already exists; do not overwrite')
    args.output.mkdir(parents=True)
    binding=dict(stage=39,parent_report_sha256=PARENT_REPORT,parent_manifest_sha256=sha(parent/'manifest.json'),
        script_sha256=sha(Path(__file__)),checkpoint_sha256=report['selections']['42']['physical']['checkpoint_sha256'],
        train_gradient_queries='16 equally spaced existing indices per view; no error selection',
        diagnostics=['saved_validation_curve','four fixed block ablations','random-roster substitution',
            'hidden delta and gate observation','fixed train-gradient cosine'],
        training_updates=0,target_forecasts=0,gpu_seconds=0.,confirmation_opened=False)
    p.immutable(args.output/'manifest.json',binding)
    arrays={}
    for c in ('physical','rewired'):
        for split in ('train','validation'):
            for view in p.VIEWS:
                path=parent/'cache'/c/f'{split}-{view}.npz'
                assert sha(path)==p.read(path.with_suffix('.json'))['sha256']
                with np.load(path) as f:arrays[c,split,view]={k:f[k] for k in f.files}
    saved=torch.load(parent/'models/seed-42/physical/best.pt',map_location='cpu',weights_only=False)
    assert sha(parent/'models/seed-42/physical/best.pt')==binding['checkpoint_sha256']
    model=p.model_for(cfg,42);model.load_state_dict(saved['model']);model.eval()
    initial=state_digest(model);results={};checks={};traces={}
    for view in p.VIEWS:
        a=arrays['physical','validation',view];rows=m['queries']['validation'][view]
        results[view]={}
        for label,blocks in (('both_on',()),('first_only',(1,)),('second_only',(0,)),('both_off',(0,1))):
            with disable_exchanges(model,blocks):prediction=p.forecast(model,a,rows,'physical')
            scores=p.metrics(prediction,a,rows)
            signed=[]
            for v in sorted({r['query']['variable'] for r in rows}):
                ix=[i for i,r in enumerate(rows) if r['query']['variable']==v]
                signed.append(float((prediction[ix]-a['raw_target'][ix])[a['observed'][ix]].mean()))
            results[view][label]=dict(metrics=scores,step_groups=grouped_error(prediction,a,rows),
                macro_signed_error=float(np.mean(signed)))
            p.immutable(args.output/f'{view}-{label}.json',dict(predictions=prediction.tolist(),metrics=scores))
            if label=='both_on':
                assert abs(scores['macro_mae']-report['selections']['42']['physical']['validation'][view]['macro_mae'])<=1e-6
                baseline=prediction.copy()
        # Intervention uses the physical model's SAME weights, not the independently trained random model.
        random_prediction=p.forecast(model,arrays['rewired','validation',view],rows,'physical')
        results[view]['random_roster_same_weights']=dict(metrics=p.metrics(random_prediction,a,rows),
            step_groups=grouped_error(random_prediction,a,rows))
        checks[view]=dict(restored_forecast_exact=bool(np.array_equal(baseline,p.forecast(model,a,rows,'physical'))))
        assert checks[view]['restored_forecast_exact']
        both_off=p.forecast(model,a,rows,'self')
        assert p.metrics(both_off,a,rows)==results[view]['both_off']['metrics']
        stats={}
        def observe(index):
            def hook(module,inputs,output):
                hidden,observed,_,_=inputs
                ratio=(output-hidden).norm(dim=-1)/hidden.norm(dim=-1).clamp_min(1e-12)
                valid=observed.bool()
                gates=module.gate(module.norm(hidden)).sigmoid().squeeze(-1)
                own=stats.setdefault(index,dict(all=[],target=[],gates=[]))
                own['all'].extend(ratio[valid].detach().tolist())
                own['target'].extend(ratio[:,0][valid[:,0]].detach().tolist())
                own['gates'].extend(gates[valid].detach().tolist())
            return hook
        handles=[block.register_forward_hook(observe(i)) for i,block in enumerate(model.exchanges)]
        try:p.forecast(model,a,rows,'physical')
        finally:
            for h in handles:h.remove()
        traces[view]={str(i):{k:dict(mean=float(np.mean(v)),quantiles=np.quantile(v,[.5,.9,.99]).tolist())
            for k,v in d.items()} for i,d in stats.items()}
    gradients={};losses={}
    names=[n for n,_ in model.named_parameters()]
    for view in p.VIEWS:
        a=arrays['physical','train',view];rows=m['queries']['train'][view]
        indices=np.linspace(0,len(rows)-1,16,dtype=np.int64)
        weight,norm=p.train_weights(a,rows)
        b=p.tensors(a,indices,rows,'physical',weight,norm)
        model.zero_grad(set_to_none=True)
        prediction=model(b['x'],b['adjacency'],b['edge_features'],12,return_residual=True)[:,0]
        loss=p.loss_value(prediction,b);loss.backward()
        own={n:par.grad.detach().clone().flatten() if par.grad is not None else torch.zeros_like(par).flatten()
            for n,par in model.named_parameters()}
        assert all(v.isfinite().all() for v in own.values())
        gradients[view]=own;losses[view]=dict(loss=float(loss.detach()),query_indices=indices.tolist())
    gradient_summary={}
    for label,selected in (('all',names),('graph',[n for n in names if n.startswith('exchanges.')]),
            ('temporal_readout',[n for n in names if not n.startswith('exchanges.')])):
        a,b=[torch.cat([gradients[v][n] for n in selected]) for v in p.VIEWS]
        gradient_summary[label]=dict(cosine=cosine(a,b),flow_norm=float(a.norm()),speed_norm=float(b.norm()))
    curves={}
    for c in p.CONDITIONS:
        points=[p.read(path) for path in sorted((parent/'models/seed-42'/c/'validations').glob('*.json'))]
        curves[c]=[dict(update=r['update'],score=r['score'],
            mae={v:r['views'][v]['macro_mae'] for v in p.VIEWS}) for r in points]
    assert state_digest(model)==initial and not torch.cuda.is_initialized()
    assert all(not b._forward_hooks and not b._forward_pre_hooks for b in model.exchanges)
    value=dict(complete=True,stage=39,binding=binding,same_weight_interventions=results,
        hidden_observations=traces,train_gradient_diagnostic=dict(details=losses,summary=gradient_summary),
        saved_validation_curves=curves,acceptance=checks,parameters_unchanged=True,
        interpretation_limits=['inference ablation is distribution shift, not independently trained control',
            'gradient cosine is a small fixed training sample at one checkpoint, not proof of general conflict',
            'no re-selection by per-view validation minima; old selections and gates unchanged',
            'direct horizon-conditioned source probe is not autoregressive ChatTime'],
        elapsed_wall_seconds=time.monotonic()-start,started_at_unix=started,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        gpu_seconds=0.,training_updates=0,target_forecasts=0,all_diagnostic_hooks_removed=True)
    p.immutable(args.output/'report.json',value)
    print(json.dumps(dict(complete=True,seconds=value['elapsed_wall_seconds'],
        interventions={v:{k:d['metrics']['macro_mae'] for k,d in x.items()} for v,x in results.items()},
        gradient_summary=gradient_summary,hidden_observations=traces)),flush=True)


if __name__=='__main__':main()
