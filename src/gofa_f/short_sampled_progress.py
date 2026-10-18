"""Progress to the authorized update cap, including remaining sample8 evaluations."""
import math
from pathlib import Path
import time

from gofa_f.prefix_cache import atomic_json
from gofa_f.short_progress import render


class Progress:
    def __init__(self, output, state, clock=time.monotonic, cap=1500):
        self.path, self.state, self.clock = Path(output)/'progress.json',state,clock
        self.cap=cap
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
        cap=self.cap
        epoch=min(consumed//32768,(cap*64-1)//32768)
        done=consumed-epoch*32768
        size = min(32768,cap*64-epoch*32768)
        measured = p['measured_updates']-p['throughput_anchor_updates']
        rate = ((p['train_seconds']-p['throughput_anchor_seconds'])/measured if measured>0 else None)
        durations = p.get('sampled_evaluation_durations',{})
        phases={'sampled-native','sampled-update-001000'}|{
            f'sampled-update-{u:06d}' for u in range(1500,cap+1,500)}
        remaining=len(phases-set(durations))
        remaining_eval = remaining*max(durations.values()) if durations else None
        if durations and self.phase not in durations and self.phase==p.get('evaluation_kind') \
                and p.get('evaluation_eta_seconds') is not None:
            remaining_eval=p['evaluation_eta_seconds']+max(0,remaining-1)*max(durations.values())
        p.update(state=self.phase,max_updates=cap,original_update_cap=4000,epoch=epoch+1,
            maximum_epochs=math.ceil(cap*64/32768),epoch_queries_done=done,epoch_queries_total=size,
            overall_queries_done=consumed,overall_queries_total=cap*64,overall_fraction=consumed/(cap*64),
            updated_at=time.time(),epoch_eta_seconds=(size-done)/64*rate if rate is not None else None,
            training_eta_seconds=(cap-p['updates'])*rate if rate is not None else None,
            overall_eta_seconds=((cap-p['updates'])*rate+remaining_eval if rate is not None and remaining_eval is not None else None),
            sampled_validation_rounds_done=len(set(durations)-{'sampled-native'}),
            sampled_validation_rounds_total=len(phases)-1,
            eta_excludes_unknown_idle_wait=True,
            eta_basis=f'to_{cap}_including_remaining_sample8_evaluations; may early stop, excludes targets')
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
