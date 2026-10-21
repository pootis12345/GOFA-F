"""Stage24 diagnostics, one fixed source adaptation and source-gated target evaluation."""
import argparse
import json
import os
from pathlib import Path

from gofa_f.allnode_stage import read
from gofa_f.graph_benchmark import atomic
from gofa_f.source_retraining import configuration


def report(output):
    output = Path(output)
    manifest = read(output/'manifest.json')
    if manifest is None:
        raise ValueError('Prepared Stage24 manifest required')
    sources = {p.name: read(p) for p in output.glob('*-result.json')}
    value = dict(protocol=manifest['protocol'], status=read(output/'status.json'),
        parent_adapter_sha256=manifest['configuration']['adapter_sha256'],
        source_diagnostic=read(output/'diagnostics/complete.json'),
        update100_gate=read(output/'update100_gate.json'), selection=read(output/'final_selection.json'),
        source_confirmation=read(output/'source_confirmation.json'), source_results=sources,
        targets=read(output/'target_results.json'), activity=read(output/'activity.json'),
        cuda_memory=read(output/'cuda_memory.json'), trainable_parameters=1060998,
        hourly_followup_enabled=False, target_label_fitting=False,
        limitations=['single seed; fixed small source and target samples',
            'target results previously seen: development reevaluation, not independent blind testing',
            'no target-result-based return to training or coefficient changes',
            'CDF loss cannot remove finite-alphabet range limits',
            'source diagnosis and loss changes do not isolate a single causal mechanism'])
    atomic(output/'report.json', value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--report-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cfg, output = configuration(args.config), Path(args.output)
    from gofa_f.robust_state import validate
    validate(cfg)
    if args.prepare or args.report_only:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise RuntimeError('CPU modes require CUDA_VISIBLE_DEVICES empty')
    if args.prepare:
        from gofa_f.robust_state import prepare
        m = prepare(cfg, output)
        print(json.dumps(dict(prepared=True, source_confirmation_queries=len(m['pools']['confirmation']))), flush=True)
    elif args.report_only:
        from gofa_f.idle_launcher import process_identity
        worker = (read(output/'activity.json', {}) or {}).get('worker')
        launcher = read(output.with_suffix('.launch.json'), {}) or {}
        if worker or launcher.get('pid') and process_identity(launcher['pid']) == str(launcher.get('start_ticks')):
            raise RuntimeError('Final CPU report requires verified worker and supervisor exit')
        print(json.dumps(report(output)), flush=True)
    else:
        from gofa_f.robust_run import run
        run(cfg, output, args.resume)


if __name__ == '__main__':
    main()
