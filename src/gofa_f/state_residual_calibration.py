"""Stage28 frozen-five/SDWPF residual calibration; no neural training entry."""
import argparse
from pathlib import Path
import time

from gofa_f.source_retraining import configuration
from gofa_f.state_residual_cpu import advance, confirmation, develop, replay_test, report, seal
from gofa_f.state_residual_state import import_confirmation, own_manifest, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('prepare', 'predict-development', 'develop', 'seal',
        'predict-confirm', 'confirm-gate', 'replay-test', 'progress', 'report'))
    parser.add_argument('--advance-cpu', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    cfg, output = configuration(args.config), args.output
    output.mkdir(parents=True, exist_ok=True)
    if args.advance_cpu:
        advance(cfg, output)
    elif args.phase == 'prepare':
        prepare(cfg, output)
    elif args.phase == 'develop':
        develop(cfg, output)
    elif args.phase == 'seal':
        seal(cfg, output)
        from gofa_f.allnode_stage import read
        if read(output/'status.json')['state'] != 'complete':
            import_confirmation(cfg, output, own_manifest(cfg, output))
    elif args.phase == 'confirm-gate':
        confirmation(cfg, output)
    elif args.phase == 'replay-test':
        replay_test(cfg, output)
    elif args.phase == 'report':
        report(output)
    elif args.phase == 'progress':
        from gofa_f.state_residual_progress import show
        while True:
            show(output)
            if not args.watch:
                break
            time.sleep(10)
    else:
        from gofa_f.allnode_stage import read
        if args.phase == 'predict-confirm' and not (output/'coefficient_lock.json').exists():
            raise ValueError('Confirmation requires sealed parameters')
        if args.phase == 'predict-development' and (output/'coefficient_lock.json').exists():
            raise ValueError('Cannot return to development after sealing')
        if read(output/'status.json', {}).get('state') in ('complete', 'failed'):
            raise ValueError('Terminal Stage28 is sealed')
        from gofa_f.state_residual_run import predict
        predict(cfg, output)


if __name__ == '__main__':
    main()
