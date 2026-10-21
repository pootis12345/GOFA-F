"""Import only identical, needed Stage27 confirmation paths after the Stage30 seal."""
from pathlib import Path
import numpy as np
from gofa_f.allnode_stage import read
from gofa_f.generative_data import sha
from gofa_f.graph_benchmark_data import TARGETS
from gofa_f.short_sampled_forecast import checked
from gofa_f.short_sampling_diagnostic import sample_seed
from gofa_f.six_calibration_cpu import read_shard
from gofa_f.six_calibration_state import immutable, result_binding
from gofa_f.autonomous_iteration import inference_binding

def import_confirmation(cfg, output, manifest, inference=False):
    """Called only AFTER the parameter seal; old evidence remains untouched."""
    from gofa_f.autonomous_iteration import load_coefficients
    _, parameters, _ = load_coefficients(cfg, output)
    root = Path(cfg['stage27_output'])
    old_manifest = read(root/'manifest.json')
    for key in ('queries', 'raw_hashes', 'metadata_hashes', 'unchanged_tensor_files', 'alphabet', 'weights_sha256'):
        if old_manifest[key] != manifest[key]:
            raise ValueError('Partial confirmation numerical provenance differs')
    receipts = {}
    for name in TARGETS:
        for mode in ('native', 'gofa'):
            if mode == 'gofa' and parameters['coefficients'][name]['full']['weights'].get('gofa', 0.) == 0.:
                continue
            label = f'confirm-{name}-{mode}'
            source = root/'evaluations'/label
            destination = output/'evaluations'/label
            old_binding = result_binding(old_manifest, cfg['stage27_manifest_sha256'], 'confirm', name, mode)
            new_binding = inference_binding(manifest, sha(output/'manifest.json'), name, mode)
            for i, q in enumerate(manifest['queries'][name][64:]):
                for sample in range(8):
                    relative = Path('trajectories')/f'{i:06d}-{sample:02d}.json'
                    path = source/relative
                    if not path.exists():
                        continue
                    value = checked(path, old_binding, q, seed=sample_seed(q, sample), sample=sample)
                    if np.asarray(value['prediction']).shape != (q['horizon'],) or not np.isfinite(value['prediction']).all():
                        raise ValueError('Invalid imported trajectory')
                    target = destination/relative
                    receipt = dict(original_path=str(path), original_sha256=sha(path), original_binding=old_binding)
                    imported = dict(value, binding=new_binding, imported_evidence=receipt)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():
                        immutable(target, imported)
                    elif read(target) != imported:
                        raise ValueError('Imported trajectory rebound')
                    receipts[str(target.relative_to(output))] = receipt
                median = source/f'{i:06d}.json'
                if median.exists():
                    hashes = {}
                    value = read_shard(source, i, q, old_binding, hashes)
                    target = destination/median.name
                    receipt = dict(original_path=str(median), original_sha256=sha(median), original_binding=old_binding)
                    imported = dict(value, binding=new_binding, imported_evidence=receipt)
                    if not target.exists():
                        immutable(target, imported)
                    elif read(target) != imported:
                        raise ValueError('Imported median rebound')
                    receipts[str(target.relative_to(output))] = receipt
    return immutable(output/'confirmation_imports.json', dict(binding=sha(output/'manifest.json'),
        records=receipts, used_for_fitting=False, old_evidence_modified=False,
        inherited_activity_seconds=read(root/'activity.json')['spent_seconds']))
