"""Stage22 phase and overall ETA, measured new work versus reused forecasts."""
import argparse
from pathlib import Path
import time

from gofa_f.graph_benchmark import atomic, read


class Progress:
    def __init__(self, output):
        self.output = Path(output)
        self.state = read(self.output/'progress.json') if (self.output/'progress.json').exists() else {}
        self.state.setdefault('evaluations', {})
        self.state.setdefault('maximum_new_numeric_work', 167936)

    def evaluation(self, done, total, work, total_work, seconds, **fields):
        key = fields['evaluation_kind']
        self.state['evaluations'][key] = dict(queries_done=done, queries_total=total,
            new_numeric_work_done=work, new_numeric_work_total=total_work, measured_seconds=seconds)
        self.state.update(phase=key, phase_done=done, phase_total=total,
            phase_eta_seconds=max(0., (total_work-work)*seconds/work) if work else None, **fields)
        self.save()

    def phase(self, name, **fields):
        self.state.update(phase=name, **fields)
        self.save()

    def save(self):
        work = sum(v['new_numeric_work_done'] for v in self.state['evaluations'].values())
        seconds = sum(v['measured_seconds'] for v in self.state['evaluations'].values())
        maximum = self.state['maximum_new_numeric_work']
        self.state.update(updated_at=time.time(), new_numeric_work_done=work,
            new_work_fraction_of_upper_bound=min(1., work/maximum),
            overall_eta_upper_bound_seconds=max(0., (maximum-work)*seconds/work) if work else None,
            eta_excludes_unknown_waiting=True, eta_is_upper_bound_until_actions_sealed=True)
        if self.state.get('phase') == 'complete':
            self.state.update(phase_eta_seconds=0., overall_eta_upper_bound_seconds=0.)
        atomic(self.output/'progress.json', self.state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output')
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        root = Path(args.output)
        p = read(root/'progress.json') if (root/'progress.json').exists() else {}
        status = read(root/'status.json') if (root/'status.json').exists() else {}
        launcher = read(root/'launcher.json') if (root/'launcher.json').exists() else {}
        def duration(v):
            return 'unknown' if v is None else f'{v/3600:.2f}h'
        print(f"launcher={launcher.get('state')} state={status.get('state')} phase={p.get('phase')}\n"
              f"phase={p.get('phase_done',0)}/{p.get('phase_total','?')} "
              f"phase ETA={duration(p.get('phase_eta_seconds'))}\n"
              f"new numeric work={p.get('new_numeric_work_done',0)}/"
              f"{p.get('maximum_new_numeric_work',167936)} (upper bound); "
              f"overall ETA <= {duration(p.get('overall_eta_upper_bound_seconds'))}; unknown waiting excluded\n"
              f"reused queries={p.get('reused_queries',0)}; no neural training/epochs", flush=True)
        if not args.watch:
            return
        time.sleep(10)


if __name__ == '__main__':
    main()
