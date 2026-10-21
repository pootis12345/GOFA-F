"""Stage29 CPU tree rounds, lazy LLM paths and status-first conditional ETA."""
import argparse
from pathlib import Path
import time

from gofa_f.allnode_stage import read
from gofa_f.generative_progress import duration
from gofa_f.prefix_cache import atomic_json
from gofa_f.six_target_progress import Progress as BaseProgress


class Progress(BaseProgress):
    def __init__(self, output, manifest):
        sealed = read(output/'coefficients.json')
        self.path_count = sum(1+int(p['full']['weights'][1] != 0.) for p in sealed['coefficients'].values())
        super().__init__(output, dict(manifest, total_numeric_work=sealed['required_numeric_work']))

    def tick(self, phase=None, force=False, **fields):
        if phase:
            self.state['state'] = phase
        self.state.update(fields, stage=29, updated_at=time.time(), total_model_datasets=self.path_count,
            eta_basis='conditional on six sequential confirmation gates; unknown pauses excluded')
        if force or time.monotonic()-self.last_write >= 5:
            atomic_json(self.output/'progress.json', self.state)
            print(f"Stage29 {self.state.get('state')} {self.state.get('dataset')}/{self.state.get('model')} "
                f"queries={self.state.get('evaluation_done', 0)}/{self.state.get('evaluation_total', 64)} "
                f"sample={self.state.get('evaluation_sample', 0)}/8 "
                f"overallETA={duration(self.state.get('overall_eta_seconds'))}", flush=True)
            self.last_write = time.monotonic()


def show(output):
    s, p = read(output/'status.json', {}), read(output/'progress.json', {})
    a, c = read(output/'activity.json', {}), read(output/'cpu_activity.json', {})
    print(f"Stage29 {s.get('state')} phase={s.get('phase')} CPU activity={duration(c.get('spent_seconds'))} "
        f"GPU activity={duration(a.get('spent_seconds'))} peak={a.get('peak_group_mib', 0)}MiB", flush=True)
    if s.get('state') in ('complete', 'failed', 'incomplete'):
        print(f"Stopped: {s.get('stop_reason', s.get('error', s.get('reason')))}; "
            f"goal_achieved={s.get('goal_achieved')}; ETA=not applicable. "
            f"Confirmation={p.get('confirmation', 'not completed')}; test={p.get('test_replay', 'not completed')}", flush=True)
        return
    if p.get('state') in ('cpu_features', 'cpu_training', 'cpu_preparation', 'cpu_development'):
        print(f"CPU {p.get('dataset')} models={p.get('cpu_models_done', 0)}/6 "
            f"queries={p.get('preprocessing_done', 0)}/{p.get('preprocessing_total', 4096)} "
            f"round={p.get('boosting_iteration', 0)}/150 overallETA={duration(p.get('overall_eta_seconds'))}", flush=True)
        print(f"Remaining GPU is conditional; initial estimate={duration(p.get('conditional_confirmation_estimate_seconds'))}; "
            'no training epochs or neural optimizer updates', flush=True)
    else:
        print(f"{p.get('dataset')}/{p.get('model')} queries={p.get('evaluation_done', 0)}/64 "
            f"sample={p.get('evaluation_sample', 0)}/8 overallETA={duration(p.get('overall_eta_seconds'))}; "
            'zero-weight graph paths skipped; unknown pauses excluded', flush=True)


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
