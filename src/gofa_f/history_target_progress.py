"""Stage27 status-first progress; terminal skipped work has no pending ETA."""
import argparse
from pathlib import Path
import time

from gofa_f.allnode_stage import read
from gofa_f.generative_progress import duration
from gofa_f.six_target_progress import Progress as BaseProgress


class Progress(BaseProgress):
    def evaluation(self, *args, **kwargs):
        super().evaluation(*args, **kwargs)
        self.tick(total_model_datasets=12)


def show(output):
    status = read(output/'status.json', {})
    p = read(output/'progress.json', {})
    ledger = read(output/'activity.json', {})
    print(f"Stage27 {status.get('state')} phase={status.get('phase')} "
          f"GPU activity={duration(ledger.get('spent_seconds'))} "
          f"peak={ledger.get('peak_group_mib', 0)}MiB", flush=True)
    if status.get('state') in ('complete', 'failed', 'incomplete'):
        print(f"Stopped: {status.get('stop_reason', status.get('error'))}; "
              f"goal_achieved={status.get('goal_achieved')}; ETA=not applicable. "
              f"Confirmation={p.get('confirmation', 'not completed')}; test={p.get('test_replay', 'not completed')}", flush=True)
        return
    if p.get('state') == 'cpu_development':
        print(f"CPU {p.get('dataset')}/{p.get('model')} fits={p.get('fit_tasks_done')}/{p.get('fit_tasks_total')} "
              f"ETA={duration(p.get('overall_eta_seconds'))}; GPU confirmation is conditional", flush=True)
    else:
        print(f"{p.get('dataset')}/{p.get('model')} queries={p.get('evaluation_done', 0)}/64 "
              f"sample={p.get('evaluation_sample', 0)}/8 "
              f"overallETA={duration(p.get('overall_eta_seconds'))}; unknown waits excluded", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        show(args.output)
        if not args.watch:
            break
        time.sleep(10)


if __name__ == '__main__':
    main()
