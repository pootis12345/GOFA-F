"""Progress to the authorized 1500-update review, including sampled evaluations."""
import math
from pathlib import Path
import time

from gofa_f.prefix_cache import atomic_json
from gofa_f.short_progress import render


class Progress:
    def __init__(self, output, state, clock=time.monotonic):
        self.path, self.state, self.clock = Path(output)/'progress.json',state,clock
        self.last,self.last_write,self.phase = clock(),0.,None

    def tick(self, phase=None, force=False, **fields):
        now = self.clock()
        if self.phase == 'train':
            self.state['train_seconds'] += max(0.,now-self.last)
        self.last = now
        if phase is not None:
            self.phase = phase
        p = self.state
        p.update(fields)
        consumed = p['updates']*64+p.get('pending_queries',0)
        epoch, done = divmod(consumed,32768)
        size = min(32768,1500*64-epoch*32768)
        measured = p['measured_updates']-p['throughput_anchor_updates']
        rate = ((p['train_seconds']-p['throughput_anchor_seconds'])/measured if measured>0 else None)
        durations = p.get('sampled_evaluation_durations',{})
        remaining_eval = (3-len(durations))*max(durations.values()) if durations else None
        if durations and self.phase not in durations and self.phase==p.get('evaluation_kind') \
                and p.get('evaluation_eta_seconds') is not None:
            remaining_eval=p['evaluation_eta_seconds']+max(0,2-len(durations))*max(durations.values())
        p.update(state=self.phase,max_updates=1500,original_update_cap=4000,epoch=epoch+1,
            maximum_epochs=math.ceil(1500*64/32768),epoch_queries_done=done,epoch_queries_total=size,
            overall_queries_done=consumed,overall_queries_total=1500*64,overall_fraction=consumed/(1500*64),
            updated_at=time.time(),epoch_eta_seconds=(size-done)/64*rate if rate is not None else None,
            training_eta_seconds=(1500-p['updates'])*rate if rate is not None else None,
            overall_eta_seconds=((1500-p['updates'])*rate+remaining_eval if rate is not None and remaining_eval is not None else None),
            eta_excludes_unknown_idle_wait=True,
            eta_basis='to_1500_review_including_remaining_sample8_evaluations; approximate, no target evaluation')
        if force or now-self.last_write>=5:
            atomic_json(self.path,p)
            self.last_write=now
            print(render({'progress':p}),flush=True)
        return p

    def evaluation(self,done,total,work_done,work_total,seconds,**fields):
        return self.tick(evaluation_done=done,evaluation_total=total,
            evaluation_work_done=work_done,evaluation_work_total=work_total,
            evaluation_eta_seconds=seconds/work_done*(work_total-work_done) if work_done else None,
            evaluation_eta_basis='numeric_slots_across_all_eight_independent_paths',**fields)
