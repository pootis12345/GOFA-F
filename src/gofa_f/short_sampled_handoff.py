"""CPU-only, one-time handoff after the existing update1500 review completes.

Do not interrupt or modify the source run. Completed samples and selection state
are migrated, not regenerated; each new GPU worker still uses idle admission.
"""
import argparse
import copy
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

from gofa_f.allnode_stage import read
from gofa_f.generative_data import SOURCES,sha
from gofa_f.idle_launcher import process_identity
from gofa_f.prefix_cache import atomic_json
from gofa_f.short_migration import assert_equal
from gofa_f.short_sampled_forecast import checked
from gofa_f.short_sampled_state import TENSOR_FILES,atomic_torch,exited
from gofa_f.short_sampling_diagnostic import SETTINGS,sample_seed


def audit(cfg,output):
    """Bind immutable inputs while the old worker is still running, without CUDA."""
    source=Path(cfg['source_output'])
    base=read(source/'manifest.json')
    if cfg['protocol']!='stage19-sampled-continuation-v2' or cfg['review_at_update'] is not None \
            or cfg['sampling']!=SETTINGS or cfg['training']!={'max_updates':4000} \
            or any(cfg['data'].values()) or cfg['resources']!=base['configuration']['resources'] \
            or sha(source/'manifest.json')!=cfg['source_manifest_sha256']:
        raise ValueError('Authorized continuation scope, resources or source manifest differs')
    launch=read(Path(str(source)+'.launch.json'))
    if launch['code_revision']!=cfg['source_revision']:
        raise ValueError('Source snapshot revision differs')
    code={p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    unchanged=TENSOR_FILES+('short_sampled_forecast.py','short_sampling_diagnostic.py')
    if any(code[n]!=base['code'][n] for n in unchanged):
        raise ValueError('Accepted tensor/training/sampling path changed')
    proof=read(source/'cpu_preflight.json')
    if not proof['passed'] or proof['manifest_sha256']!=cfg['source_manifest_sha256']:
        raise ValueError('Original CPU/numerical equivalence proof is not bound')
    if sha(source/'reference-001000.pt')!=cfg['reference_1000_sha256']:
        raise ValueError('Reference checkpoint changed')
    manifest=dict(protocol=cfg['protocol'],configuration=cfg,source_configuration=base['source_configuration'],
        source_manifest_sha256=cfg['source_manifest_sha256'],code=code,
        weights_sha256=base['weights_sha256'],weight_stats=base['weight_stats'],
        raw_hashes=base['raw_hashes'],epsilon_sha256=base['epsilon_sha256'],
        source_pool_sha256=sha(source/'pools/queries.json'),dependencies=base['dependencies'],
        unchanged_tensor_files={n:code[n] for n in unchanged},
        inherited_cpu_proof_sha256=sha(source/'cpu_preflight.json'),
        source_training_updates=1500,review_at_update=None,target_evaluation_enabled=False,
        authority='User: continue training after this update1500 sample8 validation; 4000 or original early stop')
    if read(output/'manifest.json',manifest)!=manifest:
        raise ValueError('Immutable handoff manifest changed')
    atomic_json(output/'manifest.json',manifest)
    verify_inputs(manifest,source)
    return manifest


def verify_inputs(manifest,source):
    if sha(source/'manifest.json')!=manifest['source_manifest_sha256'] \
            or sha(source/'pools/queries.json')!=manifest['source_pool_sha256'] \
            or sha(source/'pools/metric_epsilons.json')!=manifest['epsilon_sha256']:
        raise ValueError('Source manifest, queries or epsilons changed')
    root=Path(manifest['source_configuration']['data']['directory'])
    if any(sha(root/'sources'/(s+'.npz'))!=manifest['raw_hashes'][s] for s in SOURCES):
        raise ValueError('Source observations changed')
    weights=Path(manifest['source_configuration']['weights']['directory'])
    if sha(weights/'snapshot.json')!=manifest['weights_sha256'] or any(
            (weights/n).stat().st_size!=v['size'] or (weights/n).stat().st_mtime_ns!=v['mtime_ns']
            for n,v in manifest['weight_stats'].items()):
        raise ValueError('Official weight content verification no longer applies')


def after_review_payload(original,binding):
    """Keep trained/sampled state intact; clear only the superseded manual stop."""
    if original['state']['updates']!=1500 or original['state']['pending'] \
            or original['state']['stop_reason']!='review_at_1500' \
            or original['sampling_ledger']['last_validated_update']!=1500 \
            or original['sampling_ledger']['validated_updates']!=[1000,1500]:
        raise ValueError('Source has not completed the authorized sampled review')
    result=copy.deepcopy(original)
    result['binding']=binding
    result['state']['stop_reason']=None
    return result


def copy_evaluations(source,output,queries,old_binding,new_binding,weights,alphabet):
    """Validate all paths; alter only the manifest prefix in the copied binding."""
    hashes=hashlib.sha256()
    count=0
    for label,digest in weights.items():
        root=source/'evaluations'/label
        old=f'{old_binding}:sample8:{label}:{digest}:{alphabet}'
        new=f'{new_binding}:sample8:{label}:{digest}:{alphabet}'
        complete=read(root/'complete.json')
        if not complete or complete['binding']!=old or not complete['complete'] \
                or complete['sampling']!=SETTINGS or complete['queries']!=len(queries) \
                or len(complete['errors'])!=len(queries):
            raise ValueError('Source sample8 evaluation is incomplete or bound differently')
        expected={'complete.json'}
        for i,q in enumerate(queries):
            item=checked(root/f'{i:06d}.json',old,q)
            paths=[checked(root/'trajectories'/f'{i:06d}-{s:02d}.json',old,q,
                seed=sample_seed(q,s),sample=s) for s in range(8)]
            if item is None or any(p is None for p in paths) \
                    or item['mae']!=complete['errors'][i] \
                    or not np.array_equal(np.median([p['prediction'] for p in paths],axis=0),item['prediction']):
                raise ValueError('Median, path coverage, seed or recorded error differs')
            expected.add(f'{i:06d}.json')
            expected.update(f'trajectories/{i:06d}-{s:02d}.json' for s in range(8))
        if {str(p.relative_to(root)) for p in root.rglob('*.json')}!=expected:
            raise ValueError('Unexpected evaluation shard prevents exact handoff')
        for relative in sorted(expected):
            path=root/relative
            value=read(path)
            if value.get('binding')!=old:
                raise ValueError('Unexpected nested evaluation binding')
            value['binding']=new
            target=output/'evaluations'/label/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and target.read_bytes()!=_json_bytes(value):
                raise ValueError('Existing migrated shard differs')
            atomic_json(target,value)
            hashes.update((label+'/'+relative+':'+sha(path)+':'+sha(target)+'\n').encode())
            count+=1
    return dict(files=count,source_and_destination_digest=hashes.hexdigest(),
                predictions_tokens_seeds_errors_and_costs_unchanged=True)


def _json_bytes(value):
    # Match atomic_json exactly, including indentation and final newline.
    import json
    return (json.dumps(value,indent=2)+'\n').encode()


def migrate(cfg,output):
    manifest=audit(cfg,output)
    source=Path(cfg['source_output'])
    activity=exited(source)
    binding=sha(output/'manifest.json')
    if (output/'migration.json').exists():
        return verify_migration(cfg,output)
    original=torch.load(source/'last.pt',map_location='cpu',weights_only=False)
    if original['binding']!=cfg['source_manifest_sha256']:
        raise ValueError('Source checkpoint binding changed')
    value=after_review_payload(original,binding)
    pool=read(source/'pools/queries.json')
    if len(pool['validation'])!=2264 or set(pool)!={'train','validation'}:
        raise ValueError('Expected original source-only query budget')
    best=read(source/'sampled_best_validation.json')
    if original['sampling_ledger']['best_update']!=best['updates'] \
            or sha(source/'sampled_best.pt')!=best['checkpoint_sha256']:
        raise ValueError('Sampled best checkpoint/ledger mismatch')
    weights={'sampled-native':manifest['weights_sha256'],
        'sampled-update-001000':cfg['reference_1000_sha256'],
        'sampled-update-001500':sha(source/'candidate-001500.pt')}
    shard_audit=copy_evaluations(source,output,pool['validation'],cfg['source_manifest_sha256'],binding,
        weights,read(source/'cpu_preflight.json')['alphabet']['alphabet_sha256'])
    for name in ('reference-001000.pt','candidate-001500.pt','sampled_best.pt',
                 'sampled_best_validation.json','validation-001000.json','validation-001500.json',
                 'training_history.jsonl','cuda_memory.json','cache.json','gpu_identity.json','lifecycle.jsonl'):
        if (source/name).exists():
            shutil.copyfile(source/name,output/name)
    (output/'pools').mkdir(exist_ok=True)
    shutil.copyfile(source/'pools/queries.json',output/'pools/queries.json')
    shutil.copyfile(source/'pools/metric_epsilons.json',output/'pools/metric_epsilons.json')
    assert_equal(torch.load(output/'candidate-001500.pt',map_location='cpu',weights_only=False),original['adapter'])
    for u,path in ((1000,'reference-001000.pt'),(1500,'candidate-001500.pt')):
        if read(source/f'validation-{u:06d}.json')['checkpoint_sha256']!=sha(output/path):
            raise ValueError('Completed validation candidate hash differs')
    if (output/'last.pt').exists():
        assert_equal(torch.load(output/'last.pt',map_location='cpu',weights_only=False),value)
    else:
        atomic_torch(output/'last.pt',value)
    assert_equal(torch.load(output/'last.pt',map_location='cpu',weights_only=False),value)
    atomic_json(output/'activity.json',activity)  # Already includes prior diagnostics; never add them again.
    atomic_json(output/'migration.json',dict(state_restore='exact',source=str(source),
        source_checkpoint_sha256=sha(source/'last.pt'),source_activity_sha256=sha(source/'activity.json'),
        source_sampled_best_sha256=sha(source/'sampled_best.pt'),target_initial_sha256=sha(output/'last.pt'),
        updates=1500,pending=0,sampling_ledger_preserved=original['sampling_ledger'],
        cleared_only_stop_reason='review_at_1500',merged_activity=activity,costs_added_again=False,
        shards=shard_audit,old_processes_exited=True))
    return manifest


def verify_migration(cfg,output):
    manifest=read(output/'manifest.json')
    source=Path(cfg['source_output'])
    exited(source)
    migration=read(output/'migration.json')
    if manifest['configuration']!=cfg or manifest['code']!={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')} \
            or sha(source/'last.pt')!=migration['source_checkpoint_sha256'] \
            or sha(source/'activity.json')!=migration['source_activity_sha256'] \
            or sha(source/'sampled_best.pt')!=migration['source_sampled_best_sha256'] \
            or sha(output/'reference-001000.pt')!=cfg['reference_1000_sha256'] \
            or sha(output/'pools/queries.json')!=manifest['source_pool_sha256'] \
            or sha(output/'pools/metric_epsilons.json')!=manifest['epsilon_sha256']:
        raise ValueError('Continued source, code, checkpoint or pool lineage changed')
    verify_inputs(manifest,source)
    return manifest


def wait_ready(source):
    """Wait for normal completion, including the controller's final ledger flush."""
    launcher=read(source/'launcher.json',{})
    if launcher.get('state') in ('failed','incomplete','stopped_by_user') or (source/'failures.json').exists():
        raise ValueError('Prior run failed/stopped; refuse an unaudited training handoff')
    if launcher.get('state')!='complete':
        return False
    try:
        exited(source)
    except ValueError:
        return False
    return True


def handoff(cfg,output,snapshot,config):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():
        raise ValueError('Handoff must be CPU-only with CUDA disabled')
    lock=(output/'handoff.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    proof=read(output/'cpu_preflight.json',{})
    if not proof.get('passed') or proof['manifest_sha256']!=sha(output/'manifest.json'):
        raise ValueError('Bound full CPU preflight required before arming handoff')
    stopped=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    try:
        while not stopped[0] and not wait_ready(Path(cfg['source_output'])):
            atomic_json(output/'handoff.json',dict(state='waiting_for_sampled_1500_validation',
                pid=os.getpid(),start_ticks=process_identity(os.getpid()),updated_at=time.time(),
                no_cuda_context=True,source=cfg['source_output']))
            for _ in range(10):
                if stopped[0]:
                    break
                time.sleep(1)
        if stopped[0]:
            atomic_json(output/'handoff.json',dict(state='stopped_by_user'))
            return
        migrate(cfg,output)
        if (output/'launcher.json').exists() or Path(str(output)+'.launch.json').exists():
            raise ValueError('Destination already has a launcher; refuse duplicate deployment')
        command=[sys.executable,'-u','-m','gofa_f.short_launcher','--snapshot',str(snapshot),
            '--output',str(output),'--config',str(config),'--worker-mode','sampled_continuation']
        with Path(str(output)+'.log').open('a') as log:
            child=subprocess.Popen(command,cwd=snapshot,env=os.environ.copy(),stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(Path(str(output)+'.launch.json'),dict(pid=child.pid,start_ticks=process_identity(child.pid),
            command=command,snapshot=str(snapshot),output=str(output),started_at=time.time(),
            admission_policy='idle',review_at_update=None,maximum_updates=4000,no_target_evaluation=True))
        atomic_json(output/'handoff.json',dict(state='launcher_deployed',pid=child.pid,
            start_ticks=process_identity(child.pid),completed_at=time.time(),migration=read(output/'migration.json')))
    except Exception as exc:
        atomic_json(output/'handoff.json',dict(state='failed',error=str(exc),updated_at=time.time()))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--snapshot')
    parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args()
    cfg,output=yaml.safe_load(Path(args.config).read_text()),Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    audit(cfg,output)
    if not args.audit_only:
        handoff(cfg,output,Path(args.snapshot).resolve(),Path(args.config).resolve())


if __name__=='__main__':
    main()
