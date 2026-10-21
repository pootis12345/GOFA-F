"""Stage29 six CPU target models plus frozen ChatTime; no neural training entry."""
import argparse
from pathlib import Path
import time

from gofa_f.allnode_stage import read
from gofa_f.prefix_cache import atomic_json
from gofa_f.source_retraining import configuration
from gofa_f.supervised_ensemble_cpu import advance, develop, replay_test, report, seal
from gofa_f.supervised_ensemble_models import train_models
from gofa_f.supervised_ensemble_state import own_manifest, prepare, status


def dispatch(args):
    cfg, output = configuration(args.config), args.output
    output.mkdir(parents=True, exist_ok=True)
    if args.advance_cpu:
        return advance(cfg, output)
    if args.phase == 'prepare':
        return prepare(cfg, output)
    if args.phase in ('train-cpu', 'run-cpu'):
        train_models(cfg, output)
        if args.phase == 'train-cpu':
            return
    if args.phase in ('develop', 'run-cpu'):
        if not develop(cfg, output)['all_six_passed']:
            return report(output)
        if args.phase == 'develop':
            return
    if args.phase in ('seal', 'run-cpu'):
        seal(cfg, output)
        if read(output/'status.json')['state'] != 'complete':
            from gofa_f.supervised_ensemble_imports import import_confirmation
            import_confirmation(cfg, output, own_manifest(cfg, output))
        return report(output)
    if args.phase == 'replay-test':
        return replay_test(cfg, output)
    if args.phase == 'report':
        return report(output)
    if args.phase == 'progress':
        from gofa_f.supervised_ensemble_progress import show
        while True:
            show(output)
            if not args.watch:
                break
            time.sleep(10)
        return
    from gofa_f.supervised_ensemble_run import predict
    return predict(cfg, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('prepare', 'train-cpu', 'run-cpu', 'develop', 'seal',
        'predict-confirm', 'replay-test', 'report', 'progress'))
    parser.add_argument('--advance-cpu', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    try:
        dispatch(args)
    except (KeyboardInterrupt, MemoryError) as exc:
        if read(args.output/'status.json', {}).get('state') != 'complete':
            status(args.output, 'resource_paused' if isinstance(exc, MemoryError) else 'incomplete',
                'cpu_interrupted', reason=type(exc).__name__, completed_cpu_models_retained=True)
            report(args.output)
        raise
    except Exception as exc:
        if read(args.output/'status.json', {}).get('state') != 'complete':
            args.output.mkdir(parents=True, exist_ok=True)
            atomic_json(args.output/'failures.json', dict(phase=args.phase or 'predict-confirm',
                type=type(exc).__name__, error=str(exc)))
            status(args.output, 'failed', args.phase or 'predict-confirm', error=str(exc))
            report(args.output)
        raise


if __name__ == '__main__':
    main()
