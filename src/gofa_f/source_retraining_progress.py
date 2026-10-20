"""Stage23 query/epoch, per-view sample8 validation and total remaining-cost ETA."""
from pathlib import Path
import time

from gofa_f.prefix_cache import atomic_json
from gofa_f.source_retraining_data import VIEWS


class Progress:
    def __init__(self, output, state=None, clock=time.monotonic):
        self.output, self.clock = Path(output), clock
        self.state = state if state is not None else dict(train_seconds=0., updates=0, measured_updates=0,
            evaluations={}, pending_queries=0)
        self.phase, self.last, self.written = None, clock(), 0.

    def tick(self, phase=None, force=False, **fields):
        now = self.clock()
        if self.phase == 'train':
            self.state['train_seconds'] += max(0., now-self.last)
        self.last = now
        if phase is not None:
            self.phase = phase
        p = self.state
        p.update(fields)
        consumed = p['updates']*64+p.get('pending_queries', 0)
        rate = p['train_seconds']/p['measured_updates'] if p['measured_updates'] else None
        domain_cycle, rem = divmod(consumed, 3)
        counts = [domain_cycle//2+int(domain_cycle % 2 == 1)+int(rem > 0 and domain_cycle % 2 == 0),
                  domain_cycle//2+int(rem > 0 and domain_cycle % 2 == 1),
                  domain_cycle+int(rem > 1), domain_cycle]
        views = {name: dict(query_presentations=n, epoch=n//8192+1, epoch_queries_done=n % 8192,
                    epoch_queries_total=8192) for name, n in zip(VIEWS, counts)}
        for index, name in enumerate(VIEWS):
            n = views[name]['query_presentations']
            frequency = 6 if index < 2 else 3
            views[name]['epoch_eta_seconds'] = ((8192-n % 8192)*frequency*rate/64
                                                if rate is not None else None)
            views[name]['eta_basis'] = 'observed global training rate and fixed domain presentation frequency'
        if p.get('current_view') in views:
            views[p['current_view']]['evaluation_queries_done'] = p.get('evaluation_done', 0)
            views[p['current_view']]['evaluation_queries_total'] = p.get('evaluation_total', 0)
            views[p['current_view']]['evaluation_eta_seconds'] = p.get('evaluation_eta_seconds')
        # Eight baseline/confirmation model rounds plus at most four candidate
        # rounds and four64-query graph controls. Early stop/parent reuse reduce it.
        estimates = p.get('evaluation_seconds_per_slot', [])
        eval_rate = max(estimates) if estimates else None
        total_slots = 8*(128*12+256*24)
        expected = ['validation-native', 'validation-parent']+[
            f'validation-update-{u:06d}' for u in (250, 500, 750, 1000)]+[
            'confirmation-native', 'confirmation-parent', 'confirmation-selected']
        remaining = sum(total_slots for label in expected if label not in p['evaluations'])
        remaining += sum(8*(32*12+32*24) for label in
            ('diagnostic-parent-real', 'diagnostic-parent-self', 'diagnostic-selected-real', 'diagnostic-selected-self')
            if label not in p['evaluations'])
        if self.phase in expected and self.phase not in p['evaluations']:
            remaining -= p.get('evaluation_work_done', 0)
        train_eta = (1000-p['updates'])*rate if rate is not None else None
        eta = train_eta+remaining*eval_rate if train_eta is not None and eval_rate is not None else None
        if self.phase == 'complete':
            eta = train_eta = 0.
        p.update(phase=self.phase, parent_updates=3500, maximum_new_updates=1000, views=views,
            overall_training_fraction=consumed/64000, query_presentations=consumed,
            training_eta_seconds=train_eta, overall_eta_seconds=eta,
            eta_includes_remaining_sample8_source_validation_confirmation_and_controls=True,
            eta_excludes_unknown_idle_wait=True, eta_is_upper_update_cap_estimate=True,
            updated_at=time.time())
        if force or now-self.written >= 5:
            atomic_json(self.output/'progress.json', p)
            self.written = now
            print(f'Stage23 {self.phase} parent3500+{p["updates"]}/1000 pending={p.get("pending_queries", 0)}/64 '
                  f'eval={p.get("evaluation_done", 0)}/{p.get("evaluation_total", 0)} ETA={eta}', flush=True)
        return p

    def evaluation(self, done, total, work_done, work_total, seconds, **fields):
        return self.tick(evaluation_done=done, evaluation_total=total, evaluation_work_done=work_done,
            evaluation_work_total=work_total, evaluation_eta_seconds=(seconds/work_done*(work_total-work_done)
            if work_done else None), **fields)

    def completed(self, label, result):
        self.state['evaluations'][label] = dict(active_seconds=result['active_seconds'], queries=result['queries'])
        slots = result['numeric_slots_in_eight_paths']
        self.state.setdefault('evaluation_seconds_per_slot', []).append(result['active_seconds']/slots)
        self.tick(force=True)


def main():
    import argparse
    import json
    from datetime import timedelta
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output')
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    def read(name):
        path = output/name
        return json.loads(path.read_text()) if path.exists() else {}
    def duration(seconds):
        return 'unknown' if seconds is None else str(timedelta(seconds=round(seconds)))
    while True:
        p, launch, bootstrap = read('progress.json'), read('launcher.json'), read('bootstrap.json')
        print(f"Stage23 {p.get('phase', bootstrap.get('state', 'not_started'))} / "
              f"{launch.get('state', 'CPU_preparation')} parent3500+{p.get('updates', 0)}/1000 "
              f"pending={p.get('pending_queries', 0)}/64 overallETA={duration(p.get('overall_eta_seconds'))}", flush=True)
        for name, own in p.get('views', {}).items():
            print(f"  {name}: epoch{own['epoch']} {own['epoch_queries_done']}/8192 "
                  f"epochETA={duration(own['epoch_eta_seconds'])} "
                  f"evaluation={own.get('evaluation_queries_done', 0)}/{own.get('evaluation_queries_total', 0)} "
                  f"evalETA={duration(own.get('evaluation_eta_seconds'))}", flush=True)
        print('ETA includes remaining sample8 source evaluations; excludes unknown idle waiting; early stop may shorten it.', flush=True)
        if not args.watch:
            break
        time.sleep(10)


if __name__ == '__main__':
    main()
