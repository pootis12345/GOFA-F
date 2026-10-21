"""Stage26 per-dataset/model/path progress; overall ETA is conditional on gates."""
import argparse
from pathlib import Path
import time

from gofa_f.allnode_stage import read
from gofa_f.generative_progress import duration
from gofa_f.prefix_cache import atomic_json


class Progress:
    def __init__(self, output, manifest):
        self.output = output
        self.state = read(output/'progress.json', {})
        self.total = manifest['total_numeric_work']
        self.completed = {p.name.removesuffix('-result.json'): read(p)
                          for p in output.glob('*-result.json')}
        self.last_write = 0.

    def tick(self, phase=None, force=False, **fields):
        if phase:
            self.state['state'] = phase
        self.state.update(fields, updated_at=time.time())
        if force or time.monotonic()-self.last_write >= 5:
            atomic_json(self.output/'progress.json', self.state)
            print(render(self.state), flush=True)
            self.last_write = time.monotonic()

    def evaluation(self, done, total, work_done, work_total, seconds, **fields):
        finished = {k: v for k, v in self.completed.items() if k != fields['evaluation_kind']}
        all_done = sum(v['numeric_slots_in_eight_paths'] for v in finished.values())+work_done
        all_seconds = sum(v['active_seconds'] for v in finished.values())+seconds
        rate = all_seconds/all_done if all_done else None
        return self.tick(evaluation_done=done, evaluation_total=total,
            evaluation_work_done=work_done, evaluation_work_total=work_total,
            evaluation_eta_seconds=seconds/work_done*(work_total-work_done) if work_done else None,
            overall_work_done=all_done, overall_work_total=self.total,
            overall_eta_seconds=rate*(self.total-all_done) if rate is not None else None,
            completed_model_datasets=len(finished), total_model_datasets=24,
            eta_basis='conditional on both gates; numeric-slot rate including graph encoding; unknown waits excluded',
            **fields)


def render(p):
    return (f"Stage26 {p.get('state')} dataset={p.get('dataset')} model={p.get('model')} "
        f"queries={p.get('evaluation_done', 0)}/{p.get('evaluation_total', '?')} "
        f"sample={p.get('evaluation_sample', '?')}/8 phaseETA={duration(p.get('evaluation_eta_seconds'))}\n"
        f"overall slots={p.get('overall_work_done', 0)}/{p.get('overall_work_total', '?')} "
        f"ETA={duration(p.get('overall_eta_seconds'))} (conditional on gates; unknown waits excluded)")


def show(output):
    state = read(output/'status.json', {})
    ledger = read(output/'activity.json', {})
    print(f"state={state.get('state')} phase={state.get('phase')} reason={state.get('stop_reason')} "
          f"GPU activity={duration(ledger.get('spent_seconds'))} peak={ledger.get('peak_group_mib')}MiB", flush=True)
    print(render(read(output/'progress.json', {})), flush=True)
    if state.get('state') == 'complete':
        print('Terminal run: no further GPU work; goal_achieved='+str(state.get('goal_achieved')), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        show(args.output)
        if not args.watch:
            return
        time.sleep(10)


if __name__ == '__main__':
    main()
