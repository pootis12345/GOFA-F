"""Stage46 fixed read-only source audit, no fitting, candidates or target access.

Replay Stage40/45 selected models on all 72 queries/view. Report all paired
errors and past-only correlations/trends; audit original-road support and
shortest distances inside the physical rosters. No thresholds/candidate search.
"""
import argparse
import copy
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from gofa_f import source_graph_probe as p
from gofa_f import source_neighborhood_probe as n
from gofa_f import source_normalized_probe as bounded
from gofa_f.source_scale_encoding_probe import history_edges
from gofa_f.generative_data import sha

PARENTS={40:('stage40-source-normalized-probe-v1','b9508f08f6e65c9e5b6f71c06fccdfe4ebeb7e3cec533fc12f92adef96a04602'),
    45:('stage45-source-scale-encoding-probe-v1','589e0e03135d3feb9d5f92ed3848de399967894af8e0a274648b23af77696a2a')}


def history_statistics(a,i):
    x=a['x'][i];mask=x[:,:,1].astype(bool);value=x[:,:,0]
    slopes=[];correlations=[]
    for node in range(len(x)):
        if a['nodes'][i,node]<0:continue
        valid=mask[node,-6:];t=np.arange(6.)[valid];v=value[node,-6:][valid]
        slope=float(((t-t.mean())*(v-v.mean())).sum()/((t-t.mean())**2).sum()*6) if len(t)>1 else 0.
        slopes.append(slope)
        if not node:continue
        own=[]
        for lag in range(4):
            left=value[0,lag:];right=value[node,:12-lag]
            good=mask[0,lag:]&mask[node,:12-lag]
            if good.sum()<6:continue
            l=left[good].astype(float);r=right[good].astype(float)
            if l.std()>0 and r.std()>0:own.append(float(np.corrcoef(l,r)[0,1]))
        if own:correlations.append(max(own))
    return dict(target_recent_normalized_slope=slopes[0],
        mean_neighbor_recent_normalized_slope=float(np.mean(slopes[1:])) if len(slopes)>1 else None,
        mean_neighbor_max_signed_lag_correlation=float(np.mean(correlations)) if correlations else None)


def road_support(a,external):
    distances=[];direct=total=0;components=[];degrees=[]
    undirected=(external>0)|(external.T>0)
    for nodes in a['nodes']:
        nodes=nodes[nodes>=0];target=int(nodes[0]);components.append(len(nodes))
        adjacency=external[np.ix_(nodes,nodes)]>0
        direct+=int(adjacency.sum()-np.trace(adjacency));total+=len(nodes)*(len(nodes)-1)
        degrees.append(int((external[target]>0).sum()))
        distance={target:0};todo=deque([target])
        while todo:
            current=todo.popleft()
            for other in np.flatnonzero(undirected[current]):
                if int(other) not in distance:distance[int(other)]=distance[current]+1;todo.append(int(other))
        distances.extend([distance[int(x)] for x in nodes[1:]])
    return dict(dense_message_pairs=total,published_road_pairs=direct,
        fraction_dense_pairs_not_published_edges=1-direct/max(total,1),
        target_to_sampled_node_hops={str(d):distances.count(d) for d in sorted(set(distances))},
        roster_sizes=sorted(set(components)),target_published_degree_min=min(degrees),target_published_degree_max=max(degrees))


def main():
    start=time.monotonic();parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();w=args.workspace;o=args.output
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():raise ValueError('CUDA-disabled CPU only')
    torch.set_num_threads(2)
    if o.exists():raise ValueError('Do not overwrite audit results')
    manifests={};reports={};bindings={}
    for stage,(name,expected) in PARENTS.items():
        parent=w/'outputs'/name
        assert sha(parent/'report.json')==expected and p.read(parent/'activity.json')['worker'] is None
        assert p.read(parent/'final_verification.json')['passed'] and not (parent/'source_confirmation.json').exists()
        m=p.read(parent/'manifest.json');manifests[stage]=m;reports[stage]=p.read(parent/'report.json')
        for f in ('source_graph_probe.py','lag_graph_probe.py','generative_data.py'):
            assert m['code'][f]==sha(Path(p.__file__).with_name(f))
        assert m['code']['source_normalized_probe']==sha(Path(bounded.__file__))
        bindings[str(stage)]=dict(manifest_sha256=sha(parent/'manifest.json'),report_sha256=expected)
    assert manifests[40]['queries']==manifests[45]['queries'] and manifests[40]['partition']==manifests[45]['partition']
    m=manifests[40];root=w/'outputs/stage38-source-neighborhood-probe-v1';data=p.source_series(m)
    assert sha(root/'manifest.json')==p.read(w/'outputs'/PARENTS[40][0]/'cache_reuse.json')['parent_manifest_sha256']
    o.mkdir(parents=True)
    p.immutable(o/'manifest.json',dict(stage=46,script_sha256=sha(Path(__file__)),parents=bindings,
        operations=['all_query_paired_errors','past_lag_correlation_and_slope','published_road_support'],
        source_queries_sha256=p.digest(m['queries']['validation']),training_updates=0,target_forecasts=0,
        confirmation_inputs_or_labels_materialized=False))
    results={};packing={}
    for view in p.VIEWS:
        packing[view]={}
        for c in ('physical','rewired'):
            path=root/'cache'/c/f'validation-{view}.npz';proof=p.read(path.with_suffix('.json'))
            assert proof['sha256']==sha(path) and proof['binding']==p.digest(dict(manifest=sha(root/'manifest.json'),condition=c,split='validation',view=view))
            with np.load(path,allow_pickle=False) as f:packing[view][c]={k:f[k] for k in f.files}
    with bounded.normalized_runtime():
        for stage,(name,_) in PARENTS.items():
            results[str(stage)]={}
            for view in p.VIEWS:
                rows=m['queries']['validation'][view];predictions={};arrays={};models={}
                for c in p.CONDITIONS:
                    a=dict(packing[view]['physical' if c=='self' else c]);arrays[c]=a
                    if stage==45:a['edge_features']=history_edges(data[view],a,rows)
                    selected=reports[stage]['selections']['42'][c];path=w/'outputs'/name/'models/seed-42'/c/'best.pt'
                    assert sha(path)==selected['checkpoint_sha256']
                    model=p.model_for(manifests[stage]['configuration'],42)
                    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'])
                    predictions[c]=p.forecast(model,a,rows,c);models[c]=selected['checkpoint_sha256']
                    score=p.metrics(predictions[c],a,rows)
                    assert abs(score['macro_mae']-selected['validation'][view]['macro_mae'])<=1e-6
                records=[]
                for i,r in enumerate(rows):
                    truth=arrays['physical']['raw_target'][i];valid=arrays['physical']['observed'][i]
                    item=dict(query=r['query'],block=r['block'],valid_points=int(valid.sum()))
                    for c in p.CONDITIONS:
                        item[c]=dict(mae=float(np.abs(predictions[c][i]-truth)[valid].mean()),
                            prediction=predictions[c][i].tolist(),history=history_statistics(arrays[c],i))
                    item['physical_minus_random_mae']=item['physical']['mae']-item['rewired']['mae']
                    records.append(item)
                diff=np.array([r['physical_minus_random_mae'] for r in records])
                by_block={str(b):dict(query_count=sum(r['block']==b for r in records),
                    mean_difference=float(np.mean([r['physical_minus_random_mae'] for r in records if r['block']==b]))) for b in range(4)}
                variables=sorted({r['query']['variable'] for r in records})
                by_variable={str(v):float(np.mean([r['physical_minus_random_mae'] for r in records if r['query']['variable']==v])) for v in variables}
                results[str(stage)][view]=dict(checkpoints=models,all_queries=records,by_time_block=by_block,
                    by_variable_mean_difference=by_variable,physical_better_queries=int((diff<0).sum()),
                    tied_queries=int((diff==0).sum()),physical_worse_queries=int((diff>0).sum()),
                    physical_roster=road_support(arrays['physical'],data[view].external))
    report=dict(stage=46,complete=True,results=results,cpu_wall_seconds=time.monotonic()-start,
        gpu_seconds=0,training_updates=0,target_forecasts=0,confirmation_opened=False,cuda_initialized=torch.cuda.is_initialized(),
        limitations=['Already exposed source validation; retrospective diagnosis, no new scientific gate passed',
            'Past correlations and error associations do not establish causality; no feature or model selected',
            'Dense physical-roster probe is not direct published-road message passing'])
    p.immutable(o/'report.json',report)
    compact={stage:{v:{k:x for k,x in result.items() if k!='all_queries'} for v,result in views.items()} for stage,views in results.items()}
    summary=dict(report,results=compact,manifest_sha256=sha(o/'manifest.json'),report_sha256=sha(o/'report.json'))
    p.immutable(o/'summary.json',summary);print(json.dumps(summary))


if __name__=='__main__':main()

