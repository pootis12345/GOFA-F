"""Stage24 source epochs and conditional full-pipeline ETA; no time budget."""
from pathlib import Path
import time

from gofa_f.prefix_cache import atomic_json
from gofa_f.source_retraining_progress import Progress as SourceProgress


class Progress(SourceProgress):
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
        cycle, rem = divmod(consumed, 3)
        counts = [cycle//2+int(cycle % 2 == 1)+int(rem > 0 and cycle % 2 == 0),
                  cycle//2+int(rem > 0 and cycle % 2 == 1), cycle+int(rem > 1), cycle]
        rate = p['train_seconds']/p['measured_updates'] if p['measured_updates'] else None
        from gofa_f.source_retraining_data import VIEWS
        views = {}
        for i, (name, count) in enumerate(zip(VIEWS, counts)):
            views[name] = dict(query_presentations=count, epoch=count//8192+1,
                epoch_queries_done=count % 8192, epoch_queries_total=8192,
                epoch_eta_seconds=(8192-count % 8192)*(6 if i < 2 else 3)*rate/64 if rate else None)
        if p.get('current_view') in views:
            views[p['current_view']].update(evaluation_queries_done=p.get('evaluation_done', 0),
                evaluation_queries_total=p.get('evaluation_total', 0),
                evaluation_eta_seconds=p.get('evaluation_eta_seconds'))
        p.update(views=views, phase=self.phase, parent_updates=3500, query_presentations=consumed)
        rates = p.get('evaluation_seconds_per_slot', [])
        eval_rate = max(rates) if rates else None
        full = 8*(128*12+256*24)
        expected = {'quick-000050': full//4, 'validation-update-000100': full,
                    'validation-update-000250': full, 'confirmation-native': full//2,
                    'confirmation-parent': full//2, 'confirmation-selected': full//2}
        expected.update({f'target-{name}-new': 8*128*h for name, h in
            [('pems03', 12), ('pems_bay', 12), ('meteonet_nw2018', 24), ('engrad', 24),
             ('sdwpf', 24), ('kelmarsh2019', 24)]})
        remaining = sum(n for key, n in expected.items() if key not in p['evaluations'])
        if self.phase in expected and self.phase not in p['evaluations']:
            # Source evaluator reports view-local progress; this is deliberately
            # a conservative upper remaining estimate for an unfinished round.
            remaining -= min(expected[self.phase], p.get('evaluation_work_done', 0))
        train_eta = max(0, 250-p['updates'])*rate if rate is not None else None
        eta = train_eta+remaining*eval_rate if train_eta is not None and eval_rate is not None else None
        if self.phase == 'complete':
            eta = train_eta = 0.
        p.update(maximum_new_updates=250, overall_training_fraction=(p['updates']*64+p.get('pending_queries', 0))/16000,
            overall_eta_seconds=eta, training_eta_seconds=train_eta, updated_at=time.time(),
            eta_is_conditional_on_all_gates_passing=True, eta_excludes_unknown_idle_wait=True)
        if force or time.monotonic()-getattr(self, '_robust_written', 0) >= 5:
            atomic_json(self.output/'progress.json', p)
            self._robust_written = time.monotonic()
            print(f"Stage24 {self.phase} parent3500+{p['updates']}/250 "
                  f"pending={p.get('pending_queries', 0)}/64 eval={p.get('evaluation_done', 0)}/"
                  f"{p.get('evaluation_total', 0)} overallETA={eta}", flush=True)
        return p


def main():
    import argparse
    import json
    from datetime import timedelta
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output')
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    while True:
        def read(name):
            p = output/name
            return json.loads(p.read_text()) if p.exists() else {}
        p, status, launch = read('progress.json'), read('status.json'), read('launcher.json')
        eta = p.get('overall_eta_seconds')
        print(f"Stage24 {status.get('state', 'CPU_preparation')}/{p.get('phase', launch.get('state', 'not_started'))} "
              f"parent3500+{p.get('updates', 0)}/250 pending={p.get('pending_queries', 0)}/64 "
              f"overallETA={'unknown' if eta is None else str(timedelta(seconds=round(eta)))}", flush=True)
        for name, v in p.get('views', {}).items():
            print(f"  {name}: epoch{v['epoch']} {v['epoch_queries_done']}/{v['epoch_queries_total']} "
                  f"epochETA={v['epoch_eta_seconds']}", flush=True)
        print('ETA includes conditional source/confirmation/target sample8 work; excludes unknown idle waiting.', flush=True)
        if not args.watch:
            break
        time.sleep(10)


if __name__ == '__main__':
    main()
