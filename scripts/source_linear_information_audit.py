"""Stage48 fixed source-supervised linear information audit; no neural training.

One registered ridge=1 on mean weighted squared residual error, 12 direct
outputs. Self versus self+physical/random neighbor past increments. No tuning,
confirmation, target data, new graph/LLM training or future-neighbor features.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from gofa_f import source_graph_probe as p
from gofa_f.generative_data import sha
from source_linear_information_core import features,fit_predict,self_test

PARENT_REPORT='719c682aa16e051c55462430b9c8a8aae48249fac7d510334476938de53398dc'


def main():
    start=time.monotonic();parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();w=args.workspace;o=args.output
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():raise ValueError('CUDA-disabled CPU only')
    torch.set_num_threads(2)
    parent=w/'outputs/stage47-source-distance-prior-probe-v1';root=w/'outputs/stage38-source-neighborhood-probe-v1'
    assert sha(parent/'report.json')==PARENT_REPORT and p.read(parent/'activity.json')['worker'] is None
    assert p.read(parent/'final_verification.json')['passed'] and not (parent/'source_confirmation.json').exists()
    m=p.read(parent/'manifest.json');old=p.read(root/'manifest.json')
    assert m['queries']==old['queries'] and m['partition']==old['partition']
    if o.exists():raise ValueError('No overwrite/restart of completed diagnostic')
    o.mkdir(parents=True)
    core=Path(__file__).with_name('source_linear_information_core.py')
    p.immutable(o/'manifest.json',dict(stage=48,script_sha256=sha(Path(__file__)),core_sha256=sha(core),
        parent_report_sha256=PARENT_REPORT,cache_manifest_sha256=sha(root/'manifest.json'),
        queries_sha256=p.digest({s:m['queries'][s] for s in ('train','validation')}),
        ridge=1.,objective='mean_query_weighted_squared_normalized_residual_error_plus_L2_unpenalized_intercept',
        conditions=['self','physical','random'],neural_training=False,source_supervision=True,
        feature_rule='24 self(history increment and mask); append24 neighbor(mean own-scale increment and valid coverage)',
        hyperparameter_search=False,confirmation_opened=False,target_opened=False))
    self_test();results={};bindings={}
    for view in p.VIEWS:
        arrays={};inputs={}
        for split in ('train','validation'):
            arrays[split]={};inputs[split]={}
            for c in ('physical','rewired'):
                path=root/'cache'/c/f'{split}-{view}.npz';proof=p.read(path.with_suffix('.json'))
                expected=p.digest(dict(manifest=sha(root/'manifest.json'),condition=c,split=split,view=view))
                assert proof['binding']==expected and sha(path)==proof['sha256']
                with np.load(path,allow_pickle=False) as f:a={k:f[k] for k in f.files}
                assert set(a['nodes'][a['nodes']>=0])<=set(m['partition'][split])
                arrays[split][c]=a;inputs[split][c]=features(a)
                bindings[f'{view}-{split}-{c}']=proof['sha256']
            for key in ('raw_target','observed','scale','persistence'):
                np.testing.assert_array_equal(arrays[split]['physical'][key],arrays[split]['rewired'][key])
            np.testing.assert_array_equal(inputs[split]['physical'][0],inputs[split]['rewired'][0])
        train=arrays['train']['physical'];valid=arrays['validation']['physical']
        rows=m['queries']['validation'][view];weight,_=p.train_weights(train,m['queries']['train'][view])
        residual=(train['raw_target']-train['persistence'][:,None])/train['scale'][:,None]
        results[view]={};predictions={}
        for condition,roster,column in [('self','physical',0),('physical','physical',1),('random','rewired',1)]:
            predicted,model=fit_predict(inputs['train'][roster][column],inputs['validation'][roster][column],
                residual,train['observed'],weight,ridge=1.)
            prediction=valid['persistence'][:,None]+predicted*valid['scale'][:,None]
            assert np.isfinite(prediction).all();predictions[condition]=prediction
            results[view][condition]=p.metrics(prediction,valid,rows)
            p.immutable(o/f'{view}-{condition}.json',dict(model=model,prediction=prediction.tolist(),
                queries=rows,metrics=results[view][condition],future_neighbor_values_used=False))
        comparison={};variables=sorted(results[view]['physical']['variables']);rng=np.random.default_rng(4248)
        indices=rng.integers(len(variables),size=(2000,len(variables)))
        for control in ('self','random'):
            real=np.array([results[view]['physical']['variables'][v]['mae'] for v in variables])
            base=np.array([results[view][control]['variables'][v]['mae'] for v in variables])
            ratio=real.mean()/max(base.mean(),1e-30)
            interval=np.quantile(real[indices].mean(1)/np.maximum(base[indices].mean(1),1e-30),[.025,.975])
            comparison[control]=dict(physical_to_control_macro_mae=float(ratio),
                physical_better_variables=int((real<base).sum()),worse_variables=int((real>base).sum()),
                variable_cluster_ratio_95pct=interval.tolist(),
                time_block_ratios={b:results[view]['physical']['time_blocks'][b]/max(value,1e-30)
                    for b,value in results[view][control]['time_blocks'].items()})
        results[view]['comparison']=comparison
    p.immutable(o/'cache_bindings.json',bindings)
    report=dict(stage=48,complete=True,source_supervised_closed_form_fit=True,neural_updates=0,gpu_seconds=0,
        results=results,cpu_wall_seconds=time.monotonic()-start,cuda_initialized=torch.cuda.is_initialized(),
        confirmation_opened=False,target_forecasts=0,hyperparameter_search=False,core_self_tests_passed=True,
        limitations=['Exposed source validation, single fixed linear representation and regularizer',
            'Physical/random matched capacity, self has fewer features; no causal or information impossibility proof',
            'Continuous CPU source diagnostic is not a ChatTime numerical head or an implemented GOFA improvement'])
    p.immutable(o/'report.json',report)
    summary=dict(report,results={v:{c:({k:x for k,x in score.items() if k!='variables'} if c!='comparison' else score)
        for c,score in own.items()} for v,own in results.items()},manifest_sha256=sha(o/'manifest.json'),report_sha256=sha(o/'report.json'))
    p.immutable(o/'summary.json',summary);print(json.dumps(summary))


if __name__=='__main__':main()
