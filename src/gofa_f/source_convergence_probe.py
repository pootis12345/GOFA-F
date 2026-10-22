"""Stage50: one fixed 3000-update convergence control of the Stage40 recipe.

Source40 stays stopped; transplant the exact 1000-update state into a new
experiment, carrying historical best/reference evidence without reselection.
"""
from contextlib import contextmanager
import copy
import os
from pathlib import Path
import shutil

import torch
import yaml

from gofa_f import source_graph_probe as p
from gofa_f import source_neighborhood_probe as n
from gofa_f import source_normalized_probe as bounded
from gofa_f.generative_data import sha

PROTOCOL='stage50-source-convergence-probe-v1'
PARENT_REPORT='b9508f08f6e65c9e5b6f71c06fccdfe4ebeb7e3cec533fc12f92adef96a04602'


def configuration(path):
    cfg=yaml.safe_load(Path(path).read_text().replace('${GOFA_WORKSPACE}',os.environ['GOFA_WORKSPACE']))
    reference=yaml.safe_load((Path(__file__).parents[2]/'configs/stage40_source_normalized_probe.yaml').read_text()
        .replace('${GOFA_WORKSPACE}',os.environ['GOFA_WORKSPACE']))
    expected=copy.deepcopy(reference);expected.update(protocol=PROTOCOL,parent_report_sha256=PARENT_REPORT,
        parent_output=str(Path(os.environ['GOFA_WORKSPACE'])/'outputs/stage40-source-normalized-probe-v1'))
    expected['training']['max_updates']=3000
    if cfg!=expected:raise ValueError('Single fixed 3000-update convergence control required')
    return cfg


def transplant_state(source,destination,old_binding,new_binding):
    """Only the identity binding changes. No numerical state or training reset."""
    checkpoint=torch.load(source,map_location='cpu',weights_only=False)
    if checkpoint['binding']!=old_binding or checkpoint['update']!=1000:
        raise ValueError('Exact completed Stage40 prefix required')
    rebound=copy.deepcopy(checkpoint);rebound['binding']=new_binding
    p.save_torch(destination,rebound)
    return dict(source_checkpoint_sha256=sha(source),destination_checkpoint_sha256=sha(destination),
        original_binding=old_binding,new_binding=new_binding,update=1000,
        preserved_fields=sorted(set(checkpoint)-{'binding'}))


def import_prefix(output,m):
    parent=Path(m['configuration']['parent_output']);parent_hash=sha(parent/'manifest.json')
    if parent_hash!=m['parent_manifest_sha256'] or sha(parent/'report.json')!=PARENT_REPORT:
        raise ValueError('Frozen original prefix changed')
    proof_path=output/'continuation_binding.json'
    if proof_path.exists():
        proof=p.read(proof_path)
        if proof['manifest_sha256']!=sha(output/'manifest.json'):raise ValueError('Continuation binding changed')
        for c,record in proof['conditions'].items():
            if sha(parent/'models/seed-42'/c/'last.pt')!=record['source_checkpoint_sha256']:
                raise ValueError('Frozen source prefix changed')
        return
    report=p.read(parent/'report.json');records={}
    for c in p.CONDITIONS:
        source=parent/'models/seed-42'/c;target=output/'models/seed-42'/c
        if (target/'last.pt').exists():raise ValueError('Partial import requires explicit inspection')
        target.mkdir(parents=True,exist_ok=True)
        old_binding=p.digest(dict(manifest=parent_hash,seed=42,condition=c))
        new_binding=p.digest(dict(manifest=sha(output/'manifest.json'),seed=42,condition=c))
        records[c]=transplant_state(source/'last.pt',target/'last.pt',old_binding,new_binding)
        if sha(source/'best.pt')!=report['selections']['42'][c]['checkpoint_sha256']:
            raise ValueError('Historical best changed')
        best=torch.load(source/'best.pt',map_location='cpu',weights_only=False);best['binding']=new_binding
        p.save_torch(target/'best.pt',best)
        validation=p.read(source/'best_validation.json');validation['binding']=new_binding
        p.immutable(target/'best_validation.json',validation)
        records[c].update(source_best_sha256=sha(source/'best.pt'),source_best_validation_sha256=sha(source/'best_validation.json'),
            inherited_best_update=validation['update'])
    p.immutable(proof_path,dict(manifest_sha256=sha(output/'manifest.json'),conditions=records,
        inherited_stage40_worker_seconds=p.read(parent/'activity.json')['cpu_active_wall_seconds'],
        inherited_updates_per_condition=1000,new_maximum_updates_per_condition=3000,
        architecture_loss_lr_and_queries_changed=False,budget_changed=True))


def reused_caches(output,m,charge):
    parent=Path(m['configuration']['parent_output']);old=p.read(parent/'manifest.json')
    root=Path(old['configuration']['parent_output'])
    if sha(root/'manifest.json')!=p.read(parent/'cache_reuse.json')['parent_manifest_sha256']:
        raise ValueError('Immutable cache ancestry changed')
    proxy=dict(m,parent_manifest_sha256=sha(root/'manifest.json'),configuration=dict(m['configuration'],parent_output=str(root)))
    result=bounded.reuse_parent_caches(output,proxy,charge);import_prefix(output,m);return result


@contextmanager
def convergence_runtime():
    with bounded.normalized_runtime():
        old=(n.PROTOCOL,n.PARENT_REPORT,n.configuration,n.code_hashes,n.caches,p.atomic_json)
        _,_,_,hashes,_,atomic=old
        n.PROTOCOL=PROTOCOL;n.PARENT_REPORT=PARENT_REPORT;n.configuration=configuration
        n.code_hashes=lambda:dict(hashes(),source_convergence_probe=sha(Path(__file__)))
        n.caches=reused_caches
        def write(path,value):
            if Path(path).name=='progress.json' and value.get('phase')=='matched_source_training':
                value=dict(value,max_updates=3000,current_model_eta_seconds=None,
                    inherited_updates=1000 if value.get('seed',42)==42 else 0,
                    extra_updates=max(value['updates']-(1000 if value.get('seed',42)==42 else 0),0))
            return atomic(path,value)
        p.atomic_json=write
        try:yield
        finally:n.PROTOCOL,n.PARENT_REPORT,n.configuration,n.code_hashes,n.caches,p.atomic_json=old


if __name__=='__main__':
    with convergence_runtime():n.main()
