"""Stage44 read-only scale-channel audit. No model, fitting or candidate selection.

Fixed before statistics: compare train/validation physical/random input packing,
pair scale/offset clipping and hypothetical coordinate reconstruction. Never use
confirmation windows. Reconstruction errors are NOT forecast errors/bounds.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from gofa_f import source_graph_probe as p
from gofa_f.generative_data import sha

PARENT_REPORT='29747e1d2b076c38d6980dbd4ff80f6c2b407f368626853121d7e8101d8ad3e0'


def audit(data,a,records):
    counts=defaultdict(int);errors=defaultdict(list);quarter=defaultdict(list);unclipped_max=0.
    for i,r in enumerate(records):
        q=r['query'];indices=a['nodes'][i];indices=indices[indices>=0]
        raw=data.values[q['origin']-12:q['origin']][:,indices].T.astype(np.float64)
        mask=data.mask[q['origin']-12:q['origin']][:,indices].T.astype(bool)&np.isfinite(raw)
        present=mask.any(1);low=np.where(present,np.where(mask,raw,np.inf).min(1),0.)
        high=np.where(present,np.where(mask,raw,-np.inf).max(1),0.)
        scale=np.where(present&(high-low>10*np.finfo(np.float64).eps),high-low,1.)
        normalized=np.where(mask,(raw-low[:,None])/scale[:,None]-.5,0.)
        np.testing.assert_array_equal(normalized.astype(np.float32),a['x'][i,:len(indices),:,0])
        assert a['low'][i]==low[0] and a['scale'][i]==scale[0]
        logratio=np.log(scale[None,:]/scale[:,None]);offset=(low[None,:]-low[:,None])/scale[:,None]
        edge=np.stack((np.clip(logratio,-5,5),np.clip(offset,-5,5)),-1).astype(np.float32)
        np.testing.assert_array_equal(edge,a['edge_features'][i,:len(indices),:len(indices)])
        pair=a['adjacency'][i,:len(indices),:len(indices)]&present[:,None]&present[None,:]
        clipped=(np.abs(logratio)>5)|(np.abs(offset)>5)
        counts['queries']+=1;counts['pairs']+=int(pair.sum())
        counts['logratio_clipped_pairs']+=int((pair&(np.abs(logratio)>5)).sum())
        counts['offset_clipped_pairs']+=int((pair&(np.abs(offset)>5)).sum())
        counts['either_clipped_pairs']+=int((pair&clipped).sum())
        counts['constant_target_histories']+=int(high[0]==low[0])
        counts['target_pairs']+=int(pair[0].sum())
        counts['target_clipped_pairs']+=int((pair&clipped)[0].sum())
        ratio=np.exp(logratio)
        mapped=(normalized[None,:,:]+.5)*ratio[:,:,None]+offset[:,:,None]-.5
        restored=low[:,None,None]+(mapped+.5)*scale[:,None,None]
        valid=pair[:,:,None]&mask[None,:,:]
        if valid.any():
            difference=np.abs(restored-raw[None,:,:])[valid]
            assert difference.max()<=1e-6
            unclipped_max=max(unclipped_max,float(difference.max()))
        mapped_clipped=(normalized[None,:,:]+.5)*np.exp(np.clip(logratio,-5,5))[:,:,None] \
            +np.clip(offset,-5,5)[:,:,None]-.5
        restored_clipped=low[:,None,None]+(mapped_clipped+.5)*scale[:,None,None]
        own=pair[0,:,None]&mask
        error=float(np.abs(restored_clipped[0]-raw)[own].mean()) if own.any() else 0.
        errors[q['variable']].append(error);quarter[r['block']].append(error)
        target=a['raw_target'][i];observed=a['observed'][i]
        counts['target_points']+=int(observed.sum())
        counts['future_outside_history_points']+=int((observed&((target<low[0])|(target>high[0]))).sum())
        counts['future_outside_nominal_scale_points']+=int((observed&((target<low[0])|(target>low[0]+scale[0]))).sum())
    return dict(counts=dict(counts),
        logratio_clipped_pair_fraction=counts['logratio_clipped_pairs']/max(1,counts['pairs']),
        offset_clipped_pair_fraction=counts['offset_clipped_pairs']/max(1,counts['pairs']),
        either_clipped_pair_fraction=counts['either_clipped_pairs']/max(1,counts['pairs']),
        target_clipped_pair_fraction=counts['target_clipped_pairs']/max(1,counts['target_pairs']),
        constant_target_history_fraction=counts['constant_target_histories']/counts['queries'],
        hypothetical_clipped_coordinate_reconstruction_macro_mae=float(np.mean([np.mean(v) for v in errors.values()])),
        per_variable_coordinate_reconstruction_mae={str(k):float(np.mean(v)) for k,v in errors.items()},
        untruncated_algebra_reconstruction_max_error=unclipped_max,
        future_outside_history_fraction=counts['future_outside_history_points']/max(1,counts['target_points']),
        future_outside_nominal_scale_fraction=counts['future_outside_nominal_scale_points']/max(1,counts['target_points']))


def main():
    start=time.monotonic();parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():
        raise ValueError('Explicit CUDA-disabled CPU audit required')
    torch.set_num_threads(2)
    parent=args.workspace/'outputs/stage43-source-state-attention-probe-v2'
    assert sha(parent/'report.json')==PARENT_REPORT and p.read(parent/'activity.json')['worker'] is None
    assert p.read(parent/'final_verification.json')['passed'] and not (parent/'source_confirmation.json').exists()
    m=p.read(parent/'manifest.json');root=args.workspace/'outputs/stage38-source-neighborhood-probe-v1'
    source=p.source_series(m);baseline=p.read(root/'manifest.json')
    assert m['queries']==baseline['queries'] and m['partition']==baseline['partition']
    assert all(m['code'][k]==v for k,v in p.code_hashes().items())
    if args.output.exists():raise ValueError('Do not overwrite read-only audit evidence')
    args.output.mkdir(parents=True)
    binding=dict(stage=44,script_sha256=sha(Path(__file__)),parent_report_sha256=PARENT_REPORT,
        source_cache_manifest_sha256=sha(root/'manifest.json'),scope=['train','validation'],
        operations=['replay_packing','audit_clipping','coordinate_algebra','label_range_statistics'],
        model_loaded=False,training_updates=0,target_forecasts=0,confirmation_windows_materialized=False)
    p.immutable(args.output/'manifest.json',binding)
    result={}
    for condition in ('physical','rewired'):
        result[condition]={}
        for split in ('train','validation'):
            result[condition][split]={}
            for view in p.VIEWS:
                path=root/'cache'/condition/f'{split}-{view}.npz'
                assert sha(path)==p.read(path.with_suffix('.json'))['sha256']
                with np.load(path,allow_pickle=False) as f:a={k:f[k] for k in f.files}
                result[condition][split][view]=audit(source[view],a,m['queries'][split][view])
    value=dict(stage=44,complete=True,binding=binding,results=result,packing_replayed_exactly=True,
        elapsed_wall_seconds=time.monotonic()-start,gpu_seconds=0.,training_updates=0,target_forecasts=0,
        limitations=['clipped coordinate reconstruction is hypothetical, not the model computation or forecast error',
            'clipping alone does not prove the cause of forecast failure or an unavoidable error lower bound',
            'continuous source readout has no finite-vocabulary range cap; future range exceedance is descriptive',
            'single exposed source; no confirmation, target or ChatTime performance claim'])
    p.immutable(args.output/'report.json',value)
    print(json.dumps(dict(seconds=value['elapsed_wall_seconds'],results=result)))


if __name__=='__main__':main()
