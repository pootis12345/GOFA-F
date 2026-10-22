"""Stage51: bounded published-road messages; fixed matched topology controls."""
from contextlib import contextmanager
import copy
import hashlib
import os
from pathlib import Path
import time

import numpy as np
import torch

from gofa_f import source_graph_probe as p
from gofa_f import source_neighborhood_probe as n
from gofa_f import source_normalized_probe as bounded
from gofa_f.generative_data import sha

PROTOCOL='stage51-source-sparse-normalized-v1'
PARENT_REPORT='b9508f08f6e65c9e5b6f71c06fccdfe4ebeb7e3cec533fc12f92adef96a04602'
SEMANTICS=dict(physical='bounded_published_bidirectional_road_on_physical_roster',
    self='same_physical_roster_no_cross_messages',
    rewired='same_physical_roster_target_fixed_static_permuted_road')


def road_adjacency(external,nodes,permuted=False):
    nodes=np.asarray(nodes);count=int((nodes>=0).sum());a=np.zeros((len(nodes),len(nodes)),bool)
    a[:count,:count]=external[np.ix_(nodes[:count],nodes[:count])]>0;np.fill_diagonal(a,False)
    if permuted and count>2:
        rng=np.random.default_rng(np.random.SeedSequence([42,51,*nodes[:count].tolist()]))
        other=rng.permutation(np.arange(1,count));order=np.arange(len(nodes));order[other]=np.roll(other,1)
        a=a[order][:,order]
    return a


def reuse_self(output,m):
    """Exact self branch, same training function/input/query/optimizer/seed prefix."""
    proof_path=output/'self_reuse.json';parent=Path(m['configuration']['parent_output'])
    source=parent/'models/seed-42/self';target=output/'models/seed-42/self'
    report=p.read(parent/'report.json');selected=report['selections']['42']['self']
    if sha(source/'best.pt')!=selected['checkpoint_sha256'] or selected['completed_updates']!=1000:
        raise ValueError('Verified complete original self model required')
    if proof_path.exists():
        proof=p.read(proof_path)
        if proof['manifest_sha256']!=sha(output/'manifest.json') or proof['source_best_sha256']!=sha(source/'best.pt'):
            raise ValueError('Exact reused self binding changed')
        return
    if (target/'selection.json').exists():raise ValueError('Partial self import requires inspection')
    target.mkdir(parents=True,exist_ok=True)
    binding=p.digest(dict(manifest=sha(output/'manifest.json'),seed=42,condition='self'))
    for name in ('best.pt','last.pt'):
        value=torch.load(source/name,map_location='cpu',weights_only=False);value['binding']=binding
        p.save_torch(target/name,value)
    value=p.read(source/'best_validation.json');value['binding']=binding;p.immutable(target/'best_validation.json',value)
    selection=copy.deepcopy(selected);selection.update(binding=binding,checkpoint_sha256=sha(target/'best.pt'))
    p.immutable(target/'selection.json',selection)
    p.immutable(output/'initializations/42-self.json',p.read(parent/'initializations/42-self.json'))
    p.immutable(proof_path,dict(manifest_sha256=sha(output/'manifest.json'),source_best_sha256=sha(source/'best.pt'),
        source_last_sha256=sha(source/'last.pt'),source_tensor_sha256=selected['tensor_sha256'],
        inherited_updates=1000,new_self_training_updates=0,
        proof='unchanged model/loss/trainer/queries/inputs; self tensor mapping exactly uses original zero adjacency',
        inherited_parent_worker_seconds=p.read(parent/'activity.json')['cpu_active_wall_seconds']))


def reused_caches(output,m,charge):
    parent=Path(m['configuration']['parent_output']);old=p.read(parent/'manifest.json')
    if sha(parent/'manifest.json')!=m['parent_manifest_sha256'] or old['queries']!=m['queries']:
        raise ValueError('Exact Stage40 parent required')
    root=Path(old['configuration']['parent_output'])
    if sha(root/'manifest.json')!=p.read(parent/'cache_reuse.json')['parent_manifest_sha256']:
        raise ValueError('Immutable cache ancestor changed')
    proxy=dict(m,parent_manifest_sha256=sha(root/'manifest.json'),configuration=dict(m['configuration'],parent_output=str(root)))
    original=bounded.reuse_parent_caches(output,proxy,charge);data=p.source_series(m)
    result={c:{s:{} for s in ('train','validation','confirmation')} for c in p.CONDITIONS};proof={}
    for split in result['physical']:
        for view in p.VIEWS:
            a=original['physical'][split][view]
            for c in ('physical','rewired'):
                adjacency=np.stack([road_adjacency(data[view].external,nodes,c=='rewired') for nodes in a['nodes']])
                result[c][split][view]=dict(a,adjacency=adjacency)
                proof[f'{c}-{split}-{view}']=hashlib.sha256(adjacency.tobytes()).hexdigest()
            left=result['physical'][split][view]['adjacency'];right=result['rewired'][split][view]['adjacency']
            np.testing.assert_array_equal(left[:,0].sum(1),right[:,0].sum(1))
            np.testing.assert_array_equal(left[:,:,0].sum(1),right[:,:,0].sum(1))
            np.testing.assert_array_equal(left.sum((1,2)),right.sum((1,2)));charge()
    result['self']=result['physical']
    p.immutable(output/'sparse_binding.json',dict(manifest_sha256=sha(output/'manifest.json'),changed_arrays=['adjacency'],
        same_nodes_values_labels_and_scaling=True,static_permutation_target_degree_preserved=True,
        confirmation_labels_read=False,derived=proof))
    reuse_self(output,m);return result


@contextmanager
def sparse_runtime():
    deadline=float(os.environ.get('GOFA_AUTOMATION_DEADLINE_UNIX','0'))
    if deadline<=0:raise ValueError('Fixed authorized autonomous deadline required')
    with bounded.normalized_runtime():
        old=(p.tensors,p.loss_value,n.run,n.PROTOCOL,n.PARENT_REPORT,n.code_hashes,n.caches,n.SEMANTICS,p.STOP[0])
        tensors,loss,run,_,_,hashes,_,_,_=old
        p.tensors=lambda a,i,r,c,*args,**kwargs:tensors(a,i,r,'self' if c=='self' else 'physical',*args,**kwargs)
        def guarded_loss(*a,**k):
            if time.time()>=deadline-65:p.STOP[0]=True
            return loss(*a,**k)
        def guarded_run(cfg,output,charge):
            value=None
            if time.time()<deadline-65:value=run(cfg,output,charge)
            else:p.STOP[0]=True
            if p.STOP[0] and time.time()>=deadline-65:
                n.atomic_json(output/'status.json',dict(state='stopped_at_authorized_deadline',deadline_unix=deadline))
            return value
        p.loss_value=guarded_loss;n.run=guarded_run
        n.PROTOCOL=PROTOCOL;n.PARENT_REPORT=PARENT_REPORT;n.SEMANTICS=SEMANTICS
        n.code_hashes=lambda:dict(hashes(),source_sparse_normalized_probe=sha(Path(__file__)))
        n.caches=reused_caches
        try:yield
        finally:p.tensors,p.loss_value,n.run,n.PROTOCOL,n.PARENT_REPORT,n.code_hashes,n.caches,n.SEMANTICS,p.STOP[0]=old


if __name__=='__main__':
    with sparse_runtime():n.main()
