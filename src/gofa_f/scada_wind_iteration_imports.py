"""Rebind only unchanged Stage31 paths; never import the failed wind confirmation."""
from pathlib import Path

import numpy as np

from gofa_f.allnode_stage import read
from gofa_f.generative_data import sha
from gofa_f.graph_benchmark_data import TARGETS
from gofa_f.scada_wind_iteration import inference_binding,load_coefficients
from gofa_f.short_sampled_forecast import checked
from gofa_f.short_sampling_diagnostic import sample_seed
from gofa_f.six_calibration_cpu import read_shard
from gofa_f.six_calibration_state import immutable


def import_confirmation(cfg,output,manifest,inference=False):
    _,parameters,_=load_coefficients(cfg,output)
    root=Path(cfg['stage31_output'])
    old=read(root/'manifest.json')
    oldparams=read(root/'coefficients.json')
    for k in ['raw_hashes','metadata_hashes','unchanged_tensor_files','alphabet','weights_sha256']:
        if old[k]!=manifest[k]:
            raise ValueError('Inherited numerical/data binding differs')
    receipts={}
    for name in TARGETS:
        if name=='sdwpf':
            continue
        if old['queries'][name]!=manifest['queries'][name] \
                or oldparams['coefficients'][name]!=parameters['coefficients'][name]:
            raise ValueError('Only exactly unchanged dataset parameters and queries can be reused')
        for mode in ['native','gofa']:
            if mode=='gofa' and parameters['coefficients'][name]['full']['weights'].get('gofa',0)==0:
                continue
            label=f'confirm-{name}-{mode}'
            source=root/'evaluations'/label
            destination=output/'evaluations'/label
            oldbinding=inference_binding(old,cfg['stage31_manifest_sha256'],name,mode)
            newbinding=inference_binding(manifest,sha(output/'manifest.json'),name,mode)
            for i,q in enumerate(manifest['queries'][name][64:]):
                for sample in range(8):
                    relative=Path('trajectories')/f'{i:06d}-{sample:02d}.json'
                    p=source/relative
                    if not p.exists():
                        continue
                    value=checked(p,oldbinding,q,seed=sample_seed(q,sample),sample=sample)
                    if np.asarray(value['prediction']).shape!=(q['horizon'],) or not np.isfinite(value['prediction']).all():
                        raise ValueError('Invalid inherited path')
                    receipt=dict(original_path=str(p),original_sha256=sha(p),original_binding=oldbinding)
                    target=destination/relative
                    target.parent.mkdir(parents=True,exist_ok=True)
                    immutable(target,dict(value,binding=newbinding,reused_stage31=receipt))
                    receipts[str(target.relative_to(output))]=receipt
                p=source/f'{i:06d}.json'
                if p.exists():
                    value=read_shard(source,i,q,oldbinding,{})
                    receipt=dict(original_path=str(p),original_sha256=sha(p),original_binding=oldbinding)
                    destination.mkdir(parents=True,exist_ok=True)
                    immutable(destination/p.name,dict(value,binding=newbinding,reused_stage31=receipt))
                    receipts[str((destination/p.name).relative_to(output))]=receipt
            complete=read(source/'complete.json')
            if complete and complete['complete']:
                if complete['binding']!=oldbinding:
                    raise ValueError('Complete source binding changed')
                immutable(destination/'complete.json',dict(complete,binding=newbinding,
                    reused_stage31=dict(original_path=str(source/'complete.json'),original_sha256=sha(source/'complete.json'))))
    return immutable(output/'confirmation_imports.json',dict(binding=sha(output/'manifest.json'),
        records=receipts,failed_sdwpf_confirmation_imported=False,used_for_fitting=False,
        inherited_gpu_activity_seconds=read(root/'activity.json')['spent_seconds']))
