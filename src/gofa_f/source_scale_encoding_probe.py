"""Stage45: fixed asinh edge encoding; Stage40 model/loss/budget unchanged."""
from contextlib import contextmanager
import hashlib
from pathlib import Path

import numpy as np

from gofa_f import source_graph_probe as p
from gofa_f import source_neighborhood_probe as n
from gofa_f import source_normalized_probe as bounded
from gofa_f.generative_data import sha

PROTOCOL='stage45-source-scale-encoding-probe-v1'
PARENT_REPORT='b9508f08f6e65c9e5b6f71c06fccdfe4ebeb7e3cec533fc12f92adef96a04602'


def encode_pairs(low,scale):
    low=np.asarray(low,dtype=np.float64);scale=np.asarray(scale,dtype=np.float64)
    if low.ndim!=1 or low.shape!=scale.shape or not np.isfinite(low).all() \
            or not np.isfinite(scale).all() or (scale<=0).any():
        raise ValueError('Finite node minima and positive independent scales required')
    raw=np.stack((np.log(scale[None,:])-np.log(scale[:,None]),
        (low[None,:]-low[:,None])/scale[:,None]),-1)
    result=np.arcsinh(raw).astype(np.float32)
    if not np.isfinite(result).all():raise ValueError('Nonfinite edge encoding')
    return result


def history_edges(data,a,records):
    """Read only the existing roster's pre-origin history, including confirmation."""
    result=np.zeros_like(a['edge_features'])
    for i,r in enumerate(records):
        q=r['query'];nodes=a['nodes'][i];nodes=nodes[nodes>=0]
        values=data.values[q['origin']-q['length']:q['origin']][:,nodes].T.astype(np.float64)
        mask=data.mask[q['origin']-q['length']:q['origin']][:,nodes].T.astype(bool)&np.isfinite(values)
        present=mask.any(1)
        low=np.where(present,np.where(mask,values,np.inf).min(1),0.)
        high=np.where(present,np.where(mask,values,-np.inf).max(1),0.)
        scale=np.where(present&(high-low>10*np.finfo(np.float64).eps),high-low,1.)
        expected=np.where(mask,(values-low[:,None])/scale[:,None]-.5,0.).astype(np.float32)
        np.testing.assert_array_equal(expected,a['x'][i,:len(nodes),:,0])
        np.testing.assert_array_equal(mask,a['x'][i,:len(nodes),:,1].astype(bool))
        if a['low'][i]!=low[0] or a['scale'][i]!=scale[0]:raise ValueError('Origin scale changed')
        old=np.stack((np.clip(np.log(scale[None,:]/scale[:,None]),-5,5),
            np.clip((low[None,:]-low[:,None])/scale[:,None],-5,5)),-1).astype(np.float32)
        np.testing.assert_array_equal(old,a['edge_features'][i,:len(nodes),:len(nodes)])
        result[i,:len(nodes),:len(nodes)]=encode_pairs(low,scale)
    return result


def reused_caches(output,m,charge):
    parent=Path(m['configuration']['parent_output']);old=p.read(parent/'manifest.json')
    if sha(parent/'manifest.json')!=m['parent_manifest_sha256'] or old['queries']!=m['queries']:
        raise ValueError('Exact Stage40 parent required')
    root=Path(old['configuration']['parent_output'])
    if sha(root/'manifest.json')!=p.read(parent/'cache_reuse.json')['parent_manifest_sha256']:
        raise ValueError('Immutable cache ancestry changed')
    proxy=dict(m,parent_manifest_sha256=sha(root/'manifest.json'),
        configuration=dict(m['configuration'],parent_output=str(root)))
    packed=bounded.reuse_parent_caches(output,proxy,charge);data=p.source_series(m);proof={}
    for c in ('physical','rewired'):
        for split,views in packed[c].items():
            for view,a in views.items():
                edge=history_edges(data[view],a,m['queries'][split][view])
                a['edge_features']=edge
                proof[f'{c}-{split}-{view}']=hashlib.sha256(edge.tobytes()).hexdigest()
                charge()
    p.immutable(output/'edge_encoding.json',dict(manifest_sha256=sha(output/'manifest.json'),
        formula=['asinh(log(scale_sender/scale_receiver))','asinh((low_sender-low_receiver)/scale_receiver)'],
        changed_arrays=['edge_features'],confirmation_labels_read=False,derived_float32_sha256=proof))
    return packed


@contextmanager
def scale_encoding_runtime():
    with bounded.normalized_runtime():
        old=(n.PROTOCOL,n.PARENT_REPORT,n.code_hashes,n.caches)
        _,_,hashes,_=old
        n.PROTOCOL=PROTOCOL;n.PARENT_REPORT=PARENT_REPORT
        n.code_hashes=lambda:dict(hashes(),source_scale_encoding_probe=sha(Path(__file__)))
        n.caches=reused_caches
        try:yield
        finally:n.PROTOCOL,n.PARENT_REPORT,n.code_hashes,n.caches=old


if __name__=='__main__':
    with scale_encoding_runtime():n.main()
