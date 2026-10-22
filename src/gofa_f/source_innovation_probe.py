"""Stage49: origin-centered innovations and a fixed source-self residual baseline.

This small source diagnostic GNN + two frozen source ridge baselines is not a
new ChatTime head or a deployable shared GOFA candidate. No coefficient search.
"""
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from gofa_f import source_graph_probe as p
from gofa_f import source_neighborhood_probe as n
from gofa_f import source_normalized_probe as bounded
from gofa_f.source_innovation_helpers import centered_history,baseline_delta
from gofa_f.generative_data import sha

PROTOCOL='stage49-source-innovation-probe-v1'
PARENT_REPORT='b9508f08f6e65c9e5b6f71c06fccdfe4ebeb7e3cec533fc12f92adef96a04602'
BASELINE_REPORT='2fc9e431dc0eebf0c358c22b5b79421d76e4c4c313df52bf3497855600f7a036'


def frozen_models():
    root=Path(os.environ['GOFA_WORKSPACE'])/'outputs/stage48-source-linear-information-v1'
    if sha(root/'report.json')!=BASELINE_REPORT:raise ValueError('Frozen source baseline report changed')
    proof=p.read(root/'verification.json');manifest=p.read(root/'manifest.json')
    if not proof['passed'] or manifest['ridge']!=1 or not manifest['source_supervision']:
        raise ValueError('Verified fixed source-only baseline required')
    models={v:p.read(root/f'{v}-self.json')['model'] for v in p.VIEWS}
    for view,model in models.items():
        if p.digest(model)!=proof['model_sha256'][f'{view}-self']:
            raise ValueError('Frozen source-only coefficient binding changed')
    return models,manifest


def reused_caches(output,m,charge):
    parent=Path(m['configuration']['parent_output']);old=p.read(parent/'manifest.json')
    if sha(parent/'manifest.json')!=m['parent_manifest_sha256'] or old['queries']!=m['queries']:
        raise ValueError('Exact Stage40 lineage required')
    root=Path(old['configuration']['parent_output'])
    if sha(root/'manifest.json')!=p.read(parent/'cache_reuse.json')['parent_manifest_sha256']:
        raise ValueError('Immutable cache ancestry changed')
    models,baseline_manifest=frozen_models()
    query_hash=p.digest({s:m['queries'][s] for s in ('train','validation')})
    if baseline_manifest['queries_sha256']!=query_hash or baseline_manifest['cache_manifest_sha256']!=sha(root/'manifest.json'):
        raise ValueError('Source baseline training/query/cache provenance changed')
    proxy=dict(m,parent_manifest_sha256=sha(root/'manifest.json'),configuration=dict(m['configuration'],parent_output=str(root)))
    result=bounded.reuse_parent_caches(output,proxy,charge);proof={}
    for c in ('physical','rewired'):
        for split,views in result[c].items():
            for view,a in views.items():
                original=a['x'];a['baseline_delta_normalized']=baseline_delta(original,models[view])
                a['x']=centered_history(original).astype(original.dtype)
                proof[f'{c}-{split}-{view}']=dict(x_sha256=hashlib.sha256(a['x'].tobytes()).hexdigest(),
                    baseline_sha256=hashlib.sha256(a['baseline_delta_normalized'].tobytes()).hexdigest())
                charge()
    for split in result['physical']:
        for view in p.VIEWS:
            np.testing.assert_array_equal(result['physical'][split][view]['baseline_delta_normalized'],
                result['rewired'][split][view]['baseline_delta_normalized'])
    p.immutable(output/'innovation_binding.json',dict(manifest_sha256=sha(output/'manifest.json'),
        frozen_model_sha256={v:p.digest(model) for v,model in models.items()},
        changed_arrays=['x_value_channel'],added_arrays=['baseline_delta_normalized'],
        labels_and_scaling_unchanged=True,confirmation_labels_read=False,derived=proof))
    return result


def residual_training_batch(original,a,indices,records,condition,weight=None,norm=1.):
    batch=original(a,indices,records,condition,weight,norm)
    if weight is not None:
        delta=(a['baseline_delta_normalized'][indices]*a['scale'][indices,None]).astype(np.float32)
        batch['delta']=batch['delta']-torch.from_numpy(delta)
    return batch


@contextmanager
def innovation_runtime():
    with bounded.normalized_runtime():
        old=(p.tensors,p.forecast,n.PROTOCOL,n.PARENT_REPORT,n.code_hashes,n.caches)
        tensors,forecast,_,_,hashes,_=old;models,_=frozen_models()
        p.tensors=lambda *a,**k:residual_training_batch(tensors,*a,**k)
        def full_forecast(model,a,records,condition):
            return forecast(model,a,records,condition)+a['baseline_delta_normalized']*a['scale'][:,None]
        p.forecast=full_forecast;n.PROTOCOL=PROTOCOL;n.PARENT_REPORT=PARENT_REPORT
        n.code_hashes=lambda:dict(hashes(),source_innovation_probe=sha(Path(__file__)),
            source_innovation_helpers=sha(Path(__file__).with_name('source_innovation_helpers.py')),
            frozen_source48_models=p.digest(models),frozen_source48_report=BASELINE_REPORT)
        n.caches=reused_caches
        try:yield
        finally:p.tensors,p.forecast,n.PROTOCOL,n.PARENT_REPORT,n.code_hashes,n.caches=old


if __name__=='__main__':
    with innovation_runtime():n.main()
