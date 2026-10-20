"""Six-dataset, twelve-model-phase progress and activity-based ETA."""
import argparse
from pathlib import Path
import time

from gofa_f.graph_benchmark import read
from gofa_f.generative_progress import duration
from gofa_f.prefix_cache import atomic_json


class Progress:
    def __init__(self, output, total_work, clock=time.monotonic):
        self.path = Path(output)/'progress.json'
        self.state = read(self.path) if self.path.exists() else {}
        self.total_work, self.clock = total_work, clock
        self.last_write, self.completed = 0., {}

    def tick(self, phase=None, force=False, **fields):
        if phase is not None:
            self.state['state'] = phase
        self.state.update(fields, updated_at=time.time(), total_model_datasets=12)
        if force or self.clock()-self.last_write >= 5:
            atomic_json(self.path, self.state)
            self.last_write = self.clock()
            print(render(self.state), flush=True)
        return self.state

    def evaluation(self, done, total, work_done, work_total, seconds, **fields):
        label = fields['evaluation_kind']
        finished = {k:v for k,v in self.completed.items() if k != label}
        overall = sum(v['numeric_slots_in_eight_paths'] for v in finished.values())+work_done
        active = sum(v['active_seconds'] for v in finished.values())+seconds
        return self.tick(evaluation_done=done, evaluation_total=total,
            evaluation_work_done=work_done, evaluation_work_total=work_total,
            evaluation_eta_seconds=seconds/work_done*(work_total-work_done) if work_done else None,
            overall_work_done=overall, overall_work_total=self.total_work,
            overall_fraction=overall/self.total_work,
            overall_eta_seconds=active/overall*(self.total_work-overall) if overall else None,
            completed_model_datasets=len(finished),
            eta_basis='eight-path numeric slots; approximate, excludes unknown waits and future pauses', **fields)


def render(p):
    return (f"phase={p.get('state')} dataset={p.get('dataset')} model={p.get('model')} "
        f"completed={p.get('completed_model_datasets', 0)}/12\n"
        f"queries {p.get('evaluation_done', 0)}/{p.get('evaluation_total', '?')} "
        f"sample={p.get('evaluation_sample', '?')}/8 phase ETA={duration(p.get('evaluation_eta_seconds'))}\n"
        f"overall numeric slots {p.get('overall_work_done', 0)}/{p.get('overall_work_total', '?')} "
        f"ETA={duration(p.get('overall_eta_seconds'))}; excludes unknown waits; no training epochs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=float, default=10.)
    args = parser.parse_args()
    while True:
        root = Path(args.output)
        launcher = read(root/'launcher.json') if (root/'launcher.json').exists() else {}
        progress = read(root/'progress.json') if (root/'progress.json').exists() else {}
        print('launcher='+str(launcher.get('state')), flush=True)
        print(render(progress), flush=True)
        if not args.watch:
            break
        time.sleep(max(1., args.interval))


if __name__ == '__main__':
    main()
