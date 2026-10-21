"""Stage25 source-only weather correction, shared-checkpoint source-gated evaluation."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time

from gofa_f.allnode_stage import read
from gofa_f.graph_benchmark import atomic
from gofa_f.source_retraining import configuration


def report(output):
    output = Path(output)
    m = read(output/'manifest.json')
    value = dict(protocol=m['protocol'], status=read(output/'status.json'),
        parent_adapter_sha256=m['configuration']['adapter_sha256'],
        source_diagnostic=read(output/'diagnostics/complete.json'),
        fixed_parent_prefixes=read(output/'parent_prefixes/complete.json'),
        quick_gate=read(output/'quick_gate.json'), source_selection=read(output/'selection_progress.json'),
        final_selection=read(output/'final_selection.json'), source_confirmation=read(output/'source_confirmation.json'),
        targets=read(output/'target_results.json'), activity=read(output/'activity.json'),
        cuda_memory=read(output/'cuda_memory.json'), trainable_parameters=1060998,
        target_label_fitting=False, hourly_followup_enabled=False,
        limitations=['single training seed and fixed small source/target samples',
            'previous target scores seen: development reevaluation, not independent blind testing',
            'fixed parent greedy prefixes do not cover the complete sampled inference distribution',
            'KL replay is a soft constraint, not a guarantee of unchanged forecast MAE',
            'finite-alphabet range limits and unverified pretraining exposure remain'])
    atomic(output/'report.json', value)
    return value


def progress(output, watch=False):
    output = Path(output)
    def duration(v):
        return 'unknown' if v is None else str(timedelta(seconds=round(v)))
    while True:
        p, status, launcher = read(output/'progress.json', {}), read(output/'status.json', {}), read(output/'launcher.json', {})
        print(f"Stage25 {status.get('state', 'CPU_preparation')}/{p.get('phase', launcher.get('state', 'not_started'))} "
              f"parent3500+{p.get('updates', 0)}/100 pending={p.get('pending_queries', 0)}/64 "
              f"prefix={p.get('prefix_done', 0)}/1024 ETA={duration(p.get('overall_eta_seconds'))}", flush=True)
        for name, v in p.get('views', {}).items():
            print(f"  {name}: epoch{v['epoch']} {v['epoch_queries_done']}/{v['epoch_queries_total']} "
                  f"epochETA={duration(v['epoch_eta_seconds'])} "
                  f"evaluation={v.get('evaluation_queries_done', 0)}/{v.get('evaluation_queries_total', 0)} "
                  f"evalETA={duration(v.get('evaluation_eta_seconds'))}", flush=True)
        print('ETA includes fixed prefixes and conditional sample8 validation/confirmation/targets; resource pauses excluded.', flush=True)
        if not watch:
            return
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--output', required=True)
    parser.add_argument('--phase', choices=('prepare', 'diagnose', 'run', 'progress', 'report'), default='run')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    if args.phase == 'progress':
        return progress(output, args.watch)
    if args.phase == 'report':
        from gofa_f.idle_launcher import process_identity
        identity = read(output.with_suffix('.launch.json'), {})
        if (read(output/'activity.json', {}) or {}).get('worker') or identity.get('pid') and \
                process_identity(identity['pid']) == str(identity.get('start_ticks')):
            raise RuntimeError('Final CPU report requires supervisor/worker exit')
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise RuntimeError('CPU report requires disabled CUDA')
        print(json.dumps(report(output)), flush=True)
        return
    if not args.config:
        parser.error('--config is required for preparation or GPU work')
    cfg = configuration(args.config)
    from gofa_f.weather_state import prepare, validate
    validate(cfg)
    if args.phase == 'prepare':
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise RuntimeError('CPU preparation requires disabled CUDA')
        manifest = prepare(cfg, output)
        print(json.dumps(dict(prepared=True, train_queries=manifest['train_queries'])), flush=True)
    else:
        from gofa_f.weather_run import run
        run(cfg, output, args.resume, diagnostic_only=args.phase == 'diagnose')


if __name__ == '__main__':
    main()
