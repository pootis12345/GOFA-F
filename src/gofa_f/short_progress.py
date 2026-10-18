"""Stage19 progress uses the actual 500-update schedule and numeric-slot work."""
import argparse
import math
from pathlib import Path
import time

from gofa_f.generative_progress import duration, inspect
from gofa_f.prefix_cache import atomic_json


class Progress:
    def __init__(self, output, state=None, clock=time.monotonic):
        self.path = Path(output)/"progress.json"
        self.state = state or {"updates": 0, "max_updates": 4000, "effective_batch": 64,
            "epoch_queries": 32768, "train_seconds": 0., "validation_seconds": 0.,
            "validation_runs": 0, "measured_updates": 0, "validation_every": 500}
        self.clock, self.last, self.last_write, self.phase = clock, clock(), 0., None

    def tick(self, phase=None, force=False, **fields):
        now = self.clock()
        if self.phase in ("train", "source_validation"):
            key = "train_seconds" if self.phase == "train" else "validation_seconds"
            self.state[key] += max(0., now-self.last)
        self.last = now
        if phase is not None:
            self.phase = phase
        self.state.update(fields)
        p = self.state
        updates, cap, batch = p["updates"], p["max_updates"], p["effective_batch"]
        consumed, total = updates*batch+p.get("pending_queries", 0), cap*batch
        epoch = min(consumed//p["epoch_queries"], (total-1)//p["epoch_queries"])
        size = min(p["epoch_queries"], total-epoch*p["epoch_queries"])
        done = consumed-epoch*p["epoch_queries"]
        measured = p["measured_updates"]-p.get("throughput_anchor_updates", 0)
        seconds = p["train_seconds"]-p.get("throughput_anchor_seconds", 0.)
        rate = seconds/measured if measured > 0 else None
        validation_mean = p["validation_seconds"]/p["validation_runs"] if p["validation_runs"] else None
        remaining_rounds = cap//p["validation_every"]-p["validation_runs"]
        p.update(state=self.phase, epoch=epoch+1, maximum_epochs=math.ceil(total/p["epoch_queries"]),
            epoch_queries_done=done, epoch_queries_total=size, overall_queries_done=consumed,
            overall_queries_total=total, overall_fraction=consumed/total, updated_at=time.time(),
            epoch_eta_seconds=(size-done)/batch*rate if rate is not None else None,
            training_eta_seconds=(cap-updates)*rate if rate is not None else None,
            overall_eta_seconds=((cap-updates)*rate+remaining_rounds*validation_mean
                if rate is not None and validation_mean is not None else None),
            eta_excludes_unknown_idle_wait=True, eta_basis="to_4000_plus_source_validation; target ETA shown separately")
        if force or now-self.last_write >= 5:
            atomic_json(self.path, p)
            self.last_write = now
            print(render({"progress": p}), flush=True)
        return p

    def evaluation(self, done, total, work_done, work_total, seconds, **fields):
        return self.tick(force=True, evaluation_done=done, evaluation_total=total,
            evaluation_work_done=work_done, evaluation_work_total=work_total,
            evaluation_round=self.state["validation_runs"]+1,
            evaluation_maximum_rounds=self.state["max_updates"]//self.state["validation_every"],
            evaluation_eta_seconds=seconds/work_done*(work_total-work_done) if work_done else None,
            evaluation_eta_basis="numeric_slots", **fields)


def render(record):
    p = record.get("progress", {})
    phase = record.get("launcher", {}).get("state", record.get("status", {}).get("state", p.get("state", "preparing")))
    text = f"state={phase} phase={p.get('state')} active={duration(record.get('activity', {}).get('spent_seconds'))}"
    if "epoch" in p:
        text += (f"\nepoch {p['epoch']}/{p['maximum_epochs']} {p['epoch_queries_done']}/{p['epoch_queries_total']} "
            f"ETA={duration(p.get('epoch_eta_seconds'))}\nupdates {p['updates']}/{p['max_updates']} "
            f"ETA(train+source validation)={duration(p.get('overall_eta_seconds'))}; excludes unknown waits/targets")
    if "evaluation_total" in p:
        text += (f"\n{p.get('evaluation_kind')} {p['evaluation_done']}/{p['evaluation_total']} queries, "
            f"{p.get('evaluation_work_done')}/{p.get('evaluation_work_total')} numeric slots, "
            f"ETA={duration(p.get('evaluation_eta_seconds'))}")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=10.)
    args = parser.parse_args()
    while True:
        print(render(inspect(args.output)), flush=True)
        if not args.watch:
            return
        time.sleep(max(1., args.interval))


if __name__ == "__main__":
    main()
