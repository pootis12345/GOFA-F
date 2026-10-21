"""Stage26 target-validation calibration, without a neural training interface."""
import argparse
from pathlib import Path

from gofa_f.six_calibration_cpu import advance, confirm, fit_coefficients, replay_test, report
from gofa_f.six_calibration_state import prepare
from gofa_f.source_retraining import configuration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('prepare', 'predict-pilot', 'fit', 'pilot-gate',
        'predict-confirm', 'confirm-gate', 'replay-test', 'report', 'progress'), default=None)
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
    elif args.phase == 'fit':
        fit_coefficients(cfg, output)
    elif args.phase in ('pilot-gate', 'confirm-gate'):
        confirm(cfg, output, args.phase.split('-')[0])
    elif args.phase == 'replay-test':
        replay_test(cfg, output)
    elif args.phase == 'report':
        report(output)
    elif args.phase == 'progress':
        from gofa_f.six_target_progress import show
        import time
        while True:
            show(output)
            if not args.watch:
                break
            time.sleep(10)
    else:
        from gofa_f.allnode_stage import read
        from gofa_f.six_calibration_run import predict
        if args.phase == 'predict-confirm' and read(output/'pilot_gate.json') is None:
            raise ValueError('Independent confirmation requires sealed pilot success')
        if args.phase == 'predict-pilot' and read(output/'pilot_gate.json') is not None:
            raise ValueError('Cannot return to pilot after coefficient confirmation')
        predict(cfg, output)


if __name__ == '__main__':
    main()
