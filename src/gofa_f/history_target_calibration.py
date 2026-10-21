"""Stage27 causal historical-expert calibration; no neural training entry."""
import argparse
from pathlib import Path

from gofa_f.history_calibration_cpu import advance, confirmation, develop, replay_test, report, seal
from gofa_f.history_calibration_state import prepare
from gofa_f.source_retraining import configuration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('prepare', 'develop', 'seal', 'predict-confirm',
        'confirm-gate', 'replay-test', 'progress', 'report'))
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
    elif args.phase == 'confirm-gate':
        confirmation(cfg, output)
    elif args.phase == 'replay-test':
        replay_test(cfg, output)
    elif args.phase == 'report':
        report(output)
    elif args.phase == 'progress':
        from gofa_f.history_target_progress import show
        import time
        while True:
            show(output)
            if not args.watch:
                break
            time.sleep(10)
    else:
        from gofa_f.history_calibration_run import predict
        predict(cfg, output)


if __name__ == '__main__':
    main()
