"""Frozen Stage60 confirmation/evaluation; existing numerical workers unchanged."""
import argparse
from collections import Counter
import os
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from gofa_f.dual_memory_data import GROUPS, SHAPES, SAMPLER
from gofa_f.dual_memory_eval_data import (TARGETS, read, seal, verify_deployment, deployment,
    write_inputs, target_text)
from gofa_f.dual_memory_eval_prepare import require_confirmation, EXECUTION, RUNTIME as OLD_RUNTIME
from gofa_f.generative_data import Query, Series, sha
from gofa_f.prefix_cache import atomic_json

PROTOCOL = 'stage60-dual-memory-evaluation-v1'
RUNTIME = tuple(dict.fromkeys((*OLD_RUNTIME, 'dual_memory_revalidation.py',
    'dual_memory_revalidation_pipeline.py', 'dual_memory_revalidation_delivery.py',
    'dual_memory_delivery.py', 'short_launcher.py')))


def configuration(path):
    cfg=yaml.safe_load(Path(path).read_text().replace('${GOFA_WORKSPACE}',os.environ['GOFA_WORKSPACE']))
    if cfg['protocol']!=PROTOCOL or cfg['training']!={'max_updates':0} or \
        cfg['data']!={'target_training_enabled':False,'target_evaluation_enabled':True}:
        raise ValueError('Only frozen Stage60 evaluation is permitted')
    if cfg['resources']!=dict(admission_policy='memory_cap_only',admission_samples=1,
        resource_check_seconds=10,memory_cap_gib=40,torch_cap_gib=38,
        activity_limit_seconds=None,wall_deadline=None,yield_on_foreign=False,leave_free_gib=0):
        raise ValueError('Evaluation resource policy changed')
    if cfg['sampling']!=dict(samples=8,top_k=100,temperature=1.,top_p=1.,seed=42,
        aggregate='pointwise_median',execution='serial_single_query_independent_kv'):
        raise ValueError('Protected Native sampling changed')
    if cfg['target_query_seed']!=60 or not cfg['prior_target_query_manifests']:
        raise ValueError('New predeclared target windows and prior exclusions required')
    if not cfg.get('ufg_adequacy_output') or not cfg.get('source_alternative_review_output'):
        raise ValueError('Adequate UFG and source-only expert-route review required')
    return cfg


def adequate_deployment(cfg):
    """Bind the stronger source-only baseline without editing the source fit."""
    from gofa_f.dual_memory_eval_pipeline import alive
    source=Path(cfg['source_output']);base=deployment(source,Path(cfg['accepted_snapshot']))
    root=Path(cfg['ufg_adequacy_output'])
    if sha(root/'manifest.json')!=cfg['ufg_adequacy_manifest_sha256']:
        raise ValueError('Official baseline adequacy manifest changed')
    state=read(root/'status.json');activity=read(root/'activity.json')
    if state.get('state')!='complete' or not state.get('source_adequacy') or activity.get('worker') is not None or \
        alive(read(Path(str(root)+'.launch.json'))):
        raise ValueError('Exited adequately stopped UFG required before evaluation')
    locked=read(root/'ufg_source_lock.json');m=read(root/'manifest.json');own=read(source/'manifest.json')
    if locked['binding']!=cfg['ufg_adequacy_manifest_sha256'] or not locked['both_shapes_stopped_by_source_criterion'] or locked['target_labels_used']:
        raise ValueError('Invalid adequately trained source UFG lock')
    for key in ('source_hashes','metadata_hashes','sampler','partitions','training_unit_normalizers'):
        if m[key]!=own[key]:raise ValueError('New baseline differs in source inputs or scales')
    for name in GROUPS:
        for shape in ('12-12','48-24'):
            for split in ('train','development'):
                if m['queries'][name][shape][split]!=own['queries'][name][shape][split]:
                    raise ValueError('Adequate baseline changed the source query budget')
    entries={}
    for shape in ('12-12','48-24'):
        item=locked['models'][shape];folder=Path(item['directory']);selected=read(folder/'selection.json')
        if selected!=item['selection'] or selected['target_labels_used'] or selected['stop_reason']!='source_early_stop' or \
            sha(folder/'best.pt')!=selected['checkpoint_sha256']:
            raise ValueError('Adequate baseline source selection changed')
        entries[shape]=dict(path=str(folder/'best.pt'),checkpoint_sha256=selected['checkpoint_sha256'],
            model_sha256=selected['model_sha256'],best_update=selected['best_update'],
            stopped_update=selected['stopped_update'],stop_reason=selected['stop_reason'])
    base['ufg']=entries
    base['ufg_source_adequacy']=dict(manifest_sha256=sha(root/'manifest.json'),
        lock_sha256=sha(root/'ufg_source_lock.json'),source_early_stop_both_shapes=True,
        unchanged_effective_official_model=True,global_optimum_not_claimed=True)
    verify_deployment(base)
    return base


def fresh_targets(data,name,reference,exclusions,seed=60):
    """IDs/time/eligibility only; preserve128 quotas, exclude ALL prior windows."""
    if len(reference)!=128 or {q['source'] for q in reference}!={name}:
        raise ValueError('Fixed128-query target identity budget required')
    shapes={(q['length'],q['horizon']) for q in reference}
    if len(shapes)!=1:raise ValueError('A target must have one fixed shape')
    length,horizon=next(iter(shapes));quota=Counter(q['variable'] for q in reference)
    variables=sorted(quota,key=lambda v:str(data.ids[v]));rows=[]
    for turn in range(max(quota.values())):
        for variable in variables:
            if turn>=quota[variable]:continue
            candidates=data.candidates(variable,length,horizon,'test').copy()
            for old in reference+exclusions+rows:
                if old['source']==name and old['variable']==variable:
                    candidates=candidates[(candidates+horizon<=old['origin'])|
                                          (candidates>=old['origin']+old['horizon'])]
            if not len(candidates):raise ValueError('Fresh target capacity exhausted; no substitution')
            bins=np.array_split(candidates,min(quota[variable],len(candidates)))
            block=bins[min(turn,len(bins)-1)]
            rng=np.random.default_rng(np.random.SeedSequence([42,seed,variable,turn]))
            rows.append(dict(source=name,variable=int(variable),origin=int(rng.choice(block)),
                             length=length,horizon=horizon,chunk=0))
    if Counter(q['variable'] for q in rows)!=quota:raise AssertionError('Variable quotas changed')
    return rows


def prepare_scope(cfg,root,scope,locked):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():
        raise ValueError('Evaluation preparation must be CPU-only')
    root=Path(root);verify_deployment(locked);source=Path(locked['source_directory'])
    if sha(source/'manifest.json')!=cfg['source_training_manifest_sha256']:
        raise ValueError('The predeclared source experiment changed')
    parent=read(source/'manifest.json');prior_hashes=[]
    if scope=='source_confirmation':
        from gofa_f.graph_memory_stage import load_sources
        datasets=load_sources(parent)
        rows=[q for name in GROUPS for l,h in SHAPES for q in parent['queries'][name][f'{l}-{h}']['confirmation']]
        if len(rows)!=256:raise ValueError('Fixed new source confirmation budget required')
        partition=parent['partitions'];texts=parent['texts'];certificate_sha=None
    elif scope=='targets':
        # Check this BEFORE opening even the old target query registries.
        certificate=Path(cfg['source_confirmation_output'])/'confirmation_certificate.json'
        require_confirmation(certificate,locked);certificate_sha=sha(certificate)
        old_path=Path(cfg['target_reference_manifest'])
        if sha(old_path)!=cfg['target_reference_manifest_sha256']:
            raise ValueError('Original target registry changed')
        old=read(old_path);exclusions=[]
        for entry in cfg['prior_target_query_manifests']:
            path=Path(entry['path'])
            if sha(path)!=entry['sha256']:raise ValueError('Prior target query registry changed')
            value=read(path)
            if value.get('scope')!='targets' or {q['source'] for q in value['queries']}!=set(TARGETS):
                raise ValueError('Wrong prior target registry scope')
            exclusions.extend(value['queries']);prior_hashes.append(entry['sha256'])
        datasets={};rows=[];partition=None;texts={}
        for name in TARGETS:
            path=Path(cfg['target_directory'])/(name+'.npz')
            if sha(path)!=old['raw_hashes'][name] or sha(path.with_suffix('.json'))!=old['metadata_hashes'][name]:
                raise ValueError('Audited target data changed')
            data=Series(path);datasets[name]=data
            own=fresh_targets(data,name,old['queries']['targets'][name],exclusions,cfg['target_query_seed'])
            for row in own:target_text(data,name,Query(**row))
            rows.extend(own)
        if len(rows)!=768:raise ValueError('Complete six-target budget required')
    else:raise ValueError('Unknown evaluation scope')
    info={name:dict(path=str(data.path),sha256=sha(data.path),metadata_sha256=sha(data.path.with_suffix('.json')),
        graph_direction='receiver_rows_sender_columns',proxy_graph=not name.startswith('pems'),
        observed_frequency_from_past_only=True) for name,data in datasets.items()}
    manifest=dict(protocol=cfg['protocol'],configuration=cfg,scope=scope,deployment=locked,
        queries=rows,datasets=info,source_texts=texts,sampler=SAMPLER,prediction_execution=EXECUTION,
        evaluation_runtime_sha256={name:sha(Path(__file__).parent/name) for name in RUNTIME},
        source_confirmation_certificate_sha256=certificate_sha,
        target_labels_used_for_fitting=False,old_target_development_exposure=True,
        query_choice_uses_prediction_errors=False,original_target_query_variables_and_quotas_preserved=True,
        prior_target_query_manifest_sha256=prior_hashes,target_query_seed=cfg['target_query_seed'],
        old_stage59_target_label_windows_excluded=scope=='targets',
        confirmation_is_report_only_not_model_reselection=True)
    write_inputs(root,manifest,datasets,partition)
    seal(root/'preparation.json',dict(complete=True,binding=sha(root/'manifest.json'),
        input_index_sha256=sha(root/'inputs/index.json'),queries=len(rows),future_values_serialized=False))
    return manifest


def main():
    from gofa_f.dual_memory_evaluation import run_gpu,run_ufg,summarize
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--ufg',action='store_true');parser.add_argument('--summarize',action='store_true')
    args=parser.parse_args();cfg=configuration(args.config);root=Path(args.output)
    try:
        if args.summarize:summarize(cfg,root)
        elif args.ufg:run_ufg(cfg,root)
        else:run_gpu(cfg,root)
    except InterruptedError as exc:
        atomic_json(root/('ufg_status.json' if args.ufg else 'status.json'),
            dict(state='resource_paused',phase='frozen_evaluation',error=str(exc)))
        raise SystemExit(75)
    except BaseException as exc:
        atomic_json(root/f'failure-{time.time_ns()}.json',dict(error=repr(exc),training_updates=0))
        atomic_json(root/('ufg_status.json' if args.ufg else 'status.json'),dict(state='failed',error=repr(exc),goal_achieved=False))
        raise


if __name__=='__main__':main()
