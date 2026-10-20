"""Stage23 source-only warm-start training, selection and locked confirmation.

CPU modes never admit GPU work. The supervised worker may only forecast the four
sealed sources. There is intentionally no six-target evaluation entry point.
"""
import argparse
import json
import os
from pathlib import Path

import yaml


def configuration(path):
    text = os.path.expandvars(Path(path).read_text())
    if '${' in text:
        raise ValueError('Set GOFA_WORKSPACE to the local project root before running Stage23')
    return yaml.safe_load(text)


def report(output):
    from gofa_f.allnode_stage import read
    from gofa_f.graph_benchmark import atomic, sha
    output = Path(output)
    manifest = read(output/'manifest.json')
    if manifest is None:
        raise ValueError('No prepared source experiment')
    selected, confirm = read(output/'final_selection.json'), read(output/'source_confirmation.json')
    groups = {}
    for label in ('validation-native', 'validation-parent', 'validation-update-000250',
                  'validation-update-000500', 'validation-update-000750', 'validation-update-001000',
                  'confirmation-native', 'confirmation-parent', 'confirmation-selected',
                  'diagnostic-parent-real', 'diagnostic-parent-self', 'diagnostic-selected-real', 'diagnostic-selected-self'):
        value = read(output/(label+'-result.json'))
        if value is not None:
            groups[label] = dict(complete=value['complete'], queries=value['queries'],
                active_seconds=value['active_seconds'],
                views={k: v['metrics'] for k, v in value['views'].items()},
                result_sha256=sha(output/(label+'-result.json')))
    value = dict(protocol=manifest['protocol'], status=read(output/'status.json'),
        parent_adapter_sha256=manifest['configuration']['adapter_sha256'], parent_updates=3500,
        trainable_parameters=1060998, selection=selected, confirmation=confirm,
        selection_history=read(output/'selection_progress.json'), source_results=groups,
        activity=read(output/'activity.json'), cuda_memory=read(output/'cuda_memory.json'),
        cache=read(output/'cache.json'), target_predictions=0,
        hourly_followup_enabled=False, no_target_score_based_adjustments=True,
        limitations=['single seed', 'finite source confirmation samples',
            'native BF16 sample8 constrained protocol differs from original FP16 sampling implementation',
            'official backbone pretraining exposure cannot be independently established',
            'parent3500 previously exposed to original four source training domains',
            'source adaptation changes data/domain/learning-rate jointly; no isolated causal attribution'])
    atomic(output/'report.json', value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--audit-data', action='store_true')
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--report-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cfg, output = configuration(args.config), Path(args.output)
    from gofa_f.source_retraining_state import validate
    validate(cfg)
    if args.audit_data or args.prepare or args.report_only:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise RuntimeError('CPU mode requires CUDA_VISIBLE_DEVICES empty')
    if args.audit_data:
        from gofa_f.source_retraining_data import prepare_sources
        print(json.dumps(prepare_sources(cfg['data']['directory'], cfg['data']['original_pems04'])), flush=True)
    elif args.prepare:
        from gofa_f.source_retraining_state import prepare
        value = prepare(cfg, output)
        print(json.dumps(dict(prepared=True, views=list(value['source_hashes']),
                              validation_queries=len(value['pools']['validation']))), flush=True)
    elif args.report_only:
        from gofa_f.idle_launcher import process_identity
        from gofa_f.allnode_stage import read
        worker = (read(output/'activity.json', {}) or {}).get('worker') or {}
        supervisor = (read(output.with_suffix('.launch.json'), {}) or {})
        live_supervisor = bool(supervisor.get('pid') and
            process_identity(supervisor['pid']) == str(supervisor.get('start_ticks')))
        if worker or live_supervisor:
            raise RuntimeError('Final CPU summary requires verified supervisor/worker exit')
        print(json.dumps(report(output)), flush=True)
    else:
        from gofa_f.source_retraining_run import run
        run(cfg, output, args.resume)


if __name__ == '__main__':
    main()
