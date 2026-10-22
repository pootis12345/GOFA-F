"""Prepare unopened source confirmation, then targets after frozen confirmation."""
from pathlib import Path

import torch

from gofa_f.dual_memory_data import GROUPS,SHAPES,SAMPLER
from gofa_f.dual_memory_eval_data import (TARGETS,CORE,read,seal,verify_deployment,
    write_inputs,fresh_target_queries,target_text)
from gofa_f.generative_data import Series,sha

EXECUTION='protected_native_shared_origin_v1:optimized_memory_tail:independent_sample_kv'
RUNTIME=('dual_memory_origin.py','dual_memory_eval_data.py','dual_memory_eval_metrics.py',
    'dual_memory_eval_records.py','dual_memory_eval_prepare.py','dual_memory_evaluation.py','dual_memory_eval_pipeline.py')


def verify_runtime(manifest):
    for name,value in manifest['evaluation_runtime_sha256'].items():
        if sha(Path(__file__).parent/name)!=value:raise ValueError('Sealed evaluation runtime changed: '+name)
    verify_deployment(manifest['deployment'])


def require_confirmation(path,deployment):
    certificate=read(path)
    if not certificate.get('complete') or certificate['deployment']!=deployment or \
            not certificate['all_model_groups_complete'] or certificate['parameters_modified']:
        raise ValueError('Completed unchanged-model source confirmation required before target preparation')
    root=Path(certificate['output'])
    if sha(root/'report.json')!=certificate['report_sha256']:raise ValueError('Source confirmation report changed')
    if sha(root/'manifest.json')!=certificate['manifest_sha256']:raise ValueError('Source confirmation input binding changed')
    return certificate


def prepare_scope(cfg,root,scope,locked):
    if torch.cuda.is_initialized() or __import__('os').environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('Evaluation preparation is CPU-only')
    root=Path(root);verify_deployment(locked);source=Path(locked['source_directory']);parent=read(source/'manifest.json')
    if scope=='source_confirmation':
        from gofa_f.graph_memory_stage import load_sources
        datasets=load_sources(parent)
        rows=[q for name in GROUPS for l,h in SHAPES for q in parent['queries'][name][f'{l}-{h}']['confirmation']]
        partition=parent['partitions'];texts=parent['texts'];certificate_sha=None
        if len(rows)!=256:raise ValueError('The original sealed256source confirmation queries are required')
    elif scope=='targets':
        certificate=Path(cfg['source_confirmation_output'])/'confirmation_certificate.json'
        require_confirmation(certificate,locked);certificate_sha=sha(certificate)
        old_path=Path(cfg['target_reference_manifest'])
        if sha(old_path)!=cfg['target_reference_manifest_sha256']:raise ValueError('Original target registry changed')
        old=read(old_path);datasets={};rows=[];partition=None;texts={}
        for name in TARGETS:
            path=Path(cfg['target_directory'])/(name+'.npz')
            if sha(path)!=old['raw_hashes'][name] or sha(path.with_suffix('.json'))!=old['metadata_hashes'][name]:
                raise ValueError('Audited target observations changed')
            data=Series(path);datasets[name]=data
            fresh=fresh_target_queries(data,name,old['queries']['targets'][name])
            for row in fresh:
                from gofa_f.generative_data import Query
                target_text(data,name,Query(**row))
            rows.extend(fresh)
        if len(rows)!=768:raise ValueError('Fixed six-target128query budget changed')
    else:raise ValueError('Unknown evaluation scope')
    info={name:dict(path=str(data.path),sha256=sha(data.path),metadata_sha256=sha(data.path.with_suffix('.json')),
        graph_direction='receiver_rows_sender_columns',proxy_graph=not name.startswith('pems'),
        observed_frequency_from_past_only=True) for name,data in datasets.items()}
    manifest=dict(protocol=cfg['protocol'],configuration=cfg,scope=scope,deployment=locked,queries=rows,datasets=info,
        source_texts=texts,sampler=SAMPLER,prediction_execution=EXECUTION,
        evaluation_runtime_sha256={n:sha(Path(__file__).parent/n) for n in RUNTIME},
        source_confirmation_certificate_sha256=certificate_sha,target_labels_used_for_fitting=False,
        old_target_development_exposure=True,query_choice_uses_prediction_errors=False,
        original_target_query_variables_and_quotas_preserved=True if scope=='targets' else None,
        confirmation_is_report_only_not_model_reselection=True)
    write_inputs(root,manifest,datasets,partition)
    seal(root/'preparation.json',dict(complete=True,binding=sha(root/'manifest.json'),
        input_index_sha256=sha(root/'inputs/index.json'),queries=len(rows),future_values_serialized=False))
    return manifest
