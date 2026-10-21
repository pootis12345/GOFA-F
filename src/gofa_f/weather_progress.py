"""Stage25 per-view epochs and all-gates-passing ETA, including fixed prefixes."""
import time

from gofa_f.prefix_cache import atomic_json
from gofa_f.source_retraining_progress import Progress as Base
from gofa_f.weather_policy import COUNTS, PER_BATCH


class Progress(Base):
    def tick(self, phase=None, force=False, **fields):
        now = self.clock()
        if self.phase == 'train':
            self.state['train_seconds'] += max(0., now-self.last)
        self.last = now
        if phase is not None:
            self.phase = phase
        p = self.state
        p.update(fields)
        update, pending = p['updates'], p.get('pending_queries', 0)
        measured = p['train_seconds']/p['measured_updates'] if p['measured_updates'] else None
        rate = measured if p['measured_updates'] >= 10 else p.get('reference_update_seconds', measured)
        views = {}
        starts = dict(boston2017=0, pems04_flow=32, pems04_speed=40, penmanshiel2017=48)
        for name, n in COUNTS.items():
            frequency = PER_BATCH[name]
            used = update*frequency+max(0, min(frequency, pending-starts[name]))
            views[name] = dict(query_presentations=used, epoch=used//n+1,
                epoch_queries_done=used % n, epoch_queries_total=n,
                epoch_eta_seconds=(n-used % n)*rate/frequency if rate is not None else None)
        if p.get('current_view') in views:
            views[p['current_view']].update(evaluation_queries_done=p.get('evaluation_done', 0),
                evaluation_queries_total=p.get('evaluation_total', 0),
                evaluation_eta_seconds=p.get('evaluation_eta_seconds'))
        estimates = p.get('evaluation_seconds_per_slot', [])
        eval_rate = max(estimates) if estimates else None
        full = 8*(128*12+256*24)
        expected = {'quick-000025': full//4, 'validation-update-000050': full,
                    'validation-update-000100': full, 'confirmation-native': full//2,
                    'confirmation-parent': full//2, 'confirmation-selected': full//2}
        expected.update({f'target-{n}-new': 8*128*(12 if n in ('pems03', 'pems_bay') else 24)
                         for n in ('pems03', 'pems_bay', 'meteonet_nw2018', 'engrad', 'sdwpf', 'kelmarsh2019')})
        if p.get('locked_update'):
            expected.pop('validation-update-000100', None)
        remaining = sum(n for name, n in expected.items() if name not in p['evaluations'])
        if self.phase in expected and self.phase not in p['evaluations']:
            remaining -= min(expected[self.phase], p.get('evaluation_previous_slots', 0)+p.get('evaluation_work_done', 0))
        prefix_done = p.get('prefix_done', 0)
        prefix_eta = ((1024-prefix_done)*p.get('prefix_seconds', 0)/prefix_done if prefix_done else
                      21504*eval_rate if eval_rate else None)
        train_eta = (0 if p.get('locked_update') else max(0, 100-update)*rate) if rate is not None else None
        eta = train_eta+remaining*eval_rate+prefix_eta if None not in (train_eta, eval_rate, prefix_eta) else None
        if self.phase == 'complete':
            eta = train_eta = prefix_eta = 0.
        p.update(phase=self.phase, views=views, parent_updates=3500, maximum_new_updates=100,
            query_presentations=update*64+pending, overall_training_fraction=(update*64+pending)/6400,
            overall_eta_seconds=eta, training_eta_seconds=train_eta, prefix_eta_seconds=prefix_eta,
            eta_is_conditional_on_all_source_gates_passing=True, eta_excludes_unknown_resource_pauses=True,
            eta_basis='actual_after10_updates' if p['measured_updates'] >= 10 else 'prior_stage_reference',
            updated_at=time.time())
        if force or now-self.written >= 5:
            atomic_json(self.output/'progress.json', p)
            self.written = now
            print(f"Stage25 {self.phase} parent3500+{update}/100 pending={pending}/64 "
                  f"prefix={prefix_done}/1024 eval={p.get('evaluation_done', 0)}/{p.get('evaluation_total', 0)} ETA={eta}", flush=True)
        return p
