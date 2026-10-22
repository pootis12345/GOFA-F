"""Single Stage40 source experiment: smooth parameter-free graph residual bound."""
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from gofa_f import source_graph_probe as p
from gofa_f import source_neighborhood_probe as n
from gofa_f.generative_data import sha

PROTOCOL='stage40-source-normalized-probe-v1'
PARENT_REPORT='e38a95b55f20579f9467f0c42a3b6cf7b1df1dae06843c66f00742d21eeeffae'


def bounded_exchange(module,inputs,output):
    hidden,observed,adjacency,_=inputs
    # Exact old self-control forward AND backward; no additional cancellation.
    if not adjacency.bool().any():return output
    base=hidden*observed.any(-1)[:,:,None,None]
    delta=output-base
    magnitude=base.norm(dim=-1,keepdim=True)
    denominator=(magnitude+delta.norm(dim=-1,keepdim=True)).clamp_min(torch.finfo(hidden.dtype).tiny)
    return base+delta*(magnitude/denominator)


def add_normalization(model):
    if getattr(model,'_stage40_normalized',False):raise ValueError('Normalization already installed')
    for block in model.exchanges:block.register_forward_hook(bounded_exchange)
    model._stage40_normalized=True
    return model


def reuse_parent_caches(output,m,charge):
    parent=Path(m['configuration']['parent_output'])
    old=p.read(parent/'manifest.json')
    if sha(parent/'manifest.json')!=m['parent_manifest_sha256'] \
            or old['queries']!=m['queries'] or old['partition']!=m['partition']:
        raise ValueError('Exact parent cache/query binding required')
    result={c:{s:{} for s in ('train','validation','confirmation')} for c in p.CONDITIONS}
    evidence={}
    for c in ('physical','rewired'):
        for split in result[c]:
            for view in p.VIEWS:
                path=parent/'cache'/c/f'{split}-{view}.npz'
                proof=p.read(path.with_suffix('.json'))
                expected=p.digest(dict(manifest=sha(parent/'manifest.json'),condition=c,split=split,view=view))
                if proof['binding']!=expected or sha(path)!=proof['sha256']:
                    raise ValueError('Parent immutable cache changed')
                with np.load(path,allow_pickle=False) as f:a={k:f[k] for k in f.files}
                if split=='confirmation' and {'y','observed','raw_target'}&set(a):
                    raise ValueError('Confirmation label cache forbidden')
                if not set(a['nodes'][a['nodes']>=0])<=set(m['partition'][split]):
                    raise ValueError('Source node isolation changed')
                result[c][split][view]=a;evidence[f'{c}-{split}-{view}']=proof['sha256']
                charge()
    result['self']=result['physical']
    p.immutable(output/'cache_reuse.json',dict(parent_manifest_sha256=sha(parent/'manifest.json'),caches=evidence))
    return result


@contextmanager
def normalized_runtime():
    """Dedicated isolated source process; restore the imported execution state."""
    original=(p.model_for,n.PROTOCOL,n.PARENT_REPORT,n.code_hashes,n.caches)
    factory,_,_,hashes,_=original
    def model_for(cfg,seed):return add_normalization(factory(cfg,seed))
    p.model_for=model_for;n.PROTOCOL=PROTOCOL;n.PARENT_REPORT=PARENT_REPORT
    n.code_hashes=lambda:dict(hashes(),source_normalized_probe=sha(Path(__file__)))
    n.caches=reuse_parent_caches
    try:yield
    finally:p.model_for,n.PROTOCOL,n.PARENT_REPORT,n.code_hashes,n.caches=original


if __name__=='__main__':
    with normalized_runtime():n.main()
