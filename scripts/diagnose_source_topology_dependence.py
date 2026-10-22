"""Fixed-weight matched-input topology interventions; no fitting or selection.

Predeclared2stages x3existing seed42 models x3inference topologies x2views.
Only already-exposed source validation; no training/confirmation/target access.
"""
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch

from gofa_f import source_graph_probe as p
from gofa_f import source_neighborhood_probe as n
from gofa_f import source_allnode_probe as allnode
from gofa_f import source_origin_history_probe as full_history
from gofa_f import source_sparse_normalized_probe as sparse
from gofa_f.generative_data import sha
from gofa_f.lag_graph_probe import state_digest

STAGES={
    52:('stage52-source-allnode-supervision-v1','13f371dec18001681cad1ff99dc7647fc1a577922ee4ef1f21e9b62cccb26011'),
    53:('stage53-source-origin-history-v1','a8aad3dae897cf204b5ca990b8016280c9c2cc4a413607fd3554d8b1a85f53c7')}
DEADLINE=1792663670
TOPOLOGIES=('physical','rewired','self')


def check_deadline():
    if int(os.environ.get('GOFA_AUTOMATION_DEADLINE_UNIX','0'))!=DEADLINE or time.time()>=DEADLINE-65:
        raise RuntimeError('Original fixed safety deadline reached; no new inference')


def array_digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def paired(rows,a,real,control):
    ids=sorted({r['query']['variable'] for r in rows});left=[];right=[];query=[]
    for variable in ids:
        ix=[i for i,r in enumerate(rows) if r['query']['variable']==variable];valid=a['observed'][ix]
        left.append(float(np.abs(real[ix]-a['raw_target'][ix])[valid].mean()))
        right.append(float(np.abs(control[ix]-a['raw_target'][ix])[valid].mean()))
    for i in range(len(rows)):
        valid=a['observed'][i]
        query.append(float(np.abs(real[i]-a['raw_target'][i])[valid].mean()-np.abs(control[i]-a['raw_target'][i])[valid].mean()))
    left=np.array(left);right=np.array(right);rng=np.random.default_rng(4254)
    ix=rng.integers(len(ids),size=(2000,len(ids)))
    draws=left[ix].mean(1)/np.maximum(right[ix].mean(1),1e-30)
    difference=left-right
    return dict(physical_to_control_macro_mae=float(left.mean()/max(right.mean(),1e-30)),
        paired_variable_cluster_ratio_95pct=np.quantile(draws,[.025,.975]).tolist(),
        improved_variables=int((difference < -1e-12).sum()),tied_variables=int((np.abs(difference)<=1e-12).sum()),
        degraded_variables=int((difference>1e-12).sum()),
        improved_queries=int((np.array(query)<-1e-12).sum()),tied_queries=int((np.abs(query)<=1e-12).sum()),
        degraded_queries=int((np.array(query)>1e-12).sum()),query_mae_differences=query)


@torch.no_grad()
def main():
    check_deadline()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():raise ValueError('CPU-only diagnostic')
    torch.set_num_threads(2);w=Path(os.environ['GOFA_WORKSPACE']);o=w/'outputs/stage54-source-topology-audit-v1'
    o.mkdir(parents=True,exist_ok=True)
    if (o/'report.json').exists():raise ValueError('Sealed diagnostic must not be rerun')
    started=time.monotonic();binding={};models={}
    for stage,(name,expected) in STAGES.items():
        parent=w/'outputs'/name;report=p.read(parent/'report.json');m=p.read(parent/'manifest.json')
        if sha(parent/'report.json')!=expected or not report['complete'] or report['passed'] or not p.read(parent/'final_verification.json')['passed'] or p.read(parent/'activity.json')['worker'] is not None:
            raise ValueError('Verified stopped failed source parent required')
        models[stage]=(parent,m,report)
        binding[str(stage)]=dict(manifest_sha256=sha(parent/'manifest.json'),report_sha256=expected,
            selected_checkpoints={c:report['selections']['42'][c]['checkpoint_sha256'] for c in TOPOLOGIES})
    manifest=dict(protocol='stage54-fixed-weight-topology-audit-v1',script_sha256=sha(Path(__file__)),
        parent_bindings=binding,training_conditions=list(TOPOLOGIES),inference_topologies=list(TOPOLOGIES),
        views=list(p.VIEWS),seed=42,bootstrap_seed=4254,bootstrap_draws=2000,fit_enabled=False,
        confirmation_enabled=False,target_enabled=False,model_selection_enabled=False,deadline_unix=DEADLINE)
    # Seal all cases before the first model forward; no result-dependent intervention selection.
    p.immutable(o/'manifest.json',manifest);results={};unchanged={};replay={}
    root=w/'outputs/stage38-source-neighborhood-probe-v1'
    for stage,(parent,m,report) in models.items():
        context=allnode.allnode_runtime() if stage==52 else full_history.origin_history_runtime()
        with context:
            if m['code']!=n.code_hashes():raise ValueError('Immutable numerical source kernel changed')
            source=p.source_series(m);results[str(stage)]={};arrays={}
            for view in p.VIEWS:
                path=root/'cache/physical'/f'validation-{view}.npz';proof=p.read(path.with_suffix('.json'))
                if sha(path)!=proof['sha256']:raise ValueError('Frozen validation cache changed')
                with np.load(path,allow_pickle=False) as stored:a={k:stored[k] for k in stored.files}
                arrays[view]={}
                for topology in TOPOLOGIES:
                    adj=np.zeros_like(a['adjacency']) if topology=='self' else np.stack([
                        sparse.road_adjacency(source[view].external,nodes,topology=='rewired') for nodes in a['nodes']])
                    arrays[view][topology]=dict(a,adjacency=adj)
                for topology in ('physical','rewired'):
                    adj=arrays[view][topology]['adjacency'];base=arrays[view]['physical']['adjacency']
                    np.testing.assert_array_equal(adj[:,0].sum(1),base[:,0].sum(1))
                    np.testing.assert_array_equal(adj[:,:,0].sum(1),base[:,:,0].sum(1))
                    np.testing.assert_array_equal(adj.sum((1,2)),base.sum((1,2)))
                    for key in a:
                        if key!='adjacency':np.testing.assert_array_equal(arrays[view][topology][key],a[key])
                unchanged[f'{stage}-{view}']=dict(original_cache_sha256=proof['sha256'],
                    input_digest=p.digest({k:array_digest(v) for k,v in a.items() if k!='adjacency'}),
                    queries_sha256=p.digest(m['queries']['validation'][view]),
                    changed_queries=int((arrays[view]['physical']['adjacency']!=arrays[view]['rewired']['adjacency']).any((1,2)).sum()))
            for trained in TOPOLOGIES:
                selected=report['selections']['42'][trained];checkpoint=parent/'models/seed-42'/trained/'best.pt'
                if sha(checkpoint)!=selected['checkpoint_sha256']:raise ValueError('Checkpoint binding changed')
                model=p.model_for(m['configuration'],42)
                model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model'])
                before=state_digest(model)
                if before!=selected['tensor_sha256']:raise ValueError('Model state changed')
                result={};predictions={}
                for view in p.VIEWS:
                    rows=m['queries']['validation'][view];result[view]={};predictions[view]={}
                    for topology in TOPOLOGIES:
                        check_deadline();a=arrays[view][topology]
                        # Prepared adjacency is passed as physical to avoid a second hidden permutation.
                        prediction=p.forecast(model,a,rows,'physical')
                        predictions[view][topology]=prediction
                        result[view][topology]=p.metrics(prediction,a,rows)
                        if trained==topology:
                            difference=abs(result[view][topology]['macro_mae']-selected['validation'][view]['macro_mae'])
                            if difference>1e-6:raise ValueError('Frozen baseline did not replay')
                            replay[f'{stage}-{trained}-{view}']=difference
                    if trained=='self':
                        for topology in TOPOLOGIES:np.testing.assert_array_equal(predictions[view][topology],predictions[view]['self'])
                    for control in ('rewired','self'):
                        result[view]['physical_vs_'+control]=paired(rows,arrays[view]['physical'],predictions[view]['physical'],predictions[view][control])
                    evidence=dict(queries=rows,observed=arrays[view]['physical']['observed'],raw_target=arrays[view]['physical']['raw_target'],
                        predictions=predictions[view],metrics=result[view],checkpoint_sha256=sha(checkpoint))
                    # Convert only at the serialization boundary; all numerical work remains float64 metrics.
                    value=json.loads(json.dumps(evidence,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v))
                    p.immutable(o/'queries'/f'{stage}-{trained}-{view}.json',value)
                if state_digest(model)!=before or sha(checkpoint)!=selected['checkpoint_sha256'] or any(v.grad is not None for v in model.parameters()):
                    raise ValueError('Read-only/no-gradient intervention altered parameters')
                results[str(stage)][trained]=result
    if torch.cuda.is_initialized():raise ValueError('Unexpected CUDA context')
    value=dict(stage=54,complete=True,fit_updates=0,new_gpu_seconds=0.,target_forecasts=0,
        new_best_model_created=False,manifest_sha256=sha(o/'manifest.json'),comparisons=results,
        unchanged_inputs=unchanged,frozen_baseline_macro_mae_replay_differences=replay,
        cpu_internal_wall_seconds=time.monotonic()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        limitations=['exposed18-source-variable validation; no target/zero-shot generality proof',
            'inference topology interventions can be out of training distribution',
            'fixed-weight message usefulness is distinct from a fair retrained graph contribution comparison'])
    serial=json.loads(json.dumps(value,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v))
    p.immutable(o/'report.json',serial)
    print(json.dumps(dict(report_sha256=sha(o/'report.json'),cpu_seconds=value['cpu_internal_wall_seconds'],
        matrix={s:{c:{v:{t:result[v][t]['macro_mae'] for t in TOPOLOGIES} for v in p.VIEWS} for c,result in entries.items()} for s,entries in results.items()},
        physical_models={s:{v:{c:entries['physical'][v]['physical_vs_'+c]['physical_to_control_macro_mae'] for c in ('rewired','self')} for v in p.VIEWS} for s,entries in results.items()})))


if __name__=='__main__':main()
