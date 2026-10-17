"""Durable phase/epoch/overall progress; ETA excludes unknown GPU idle waits."""
import argparse
import json
import math
from pathlib import Path
import time

from gofa_f.prefix_cache import atomic_json


def duration(seconds):
    if seconds is None:
        return "estimating"
    seconds = max(0, int(seconds))
    return f"{seconds//3600:02d}:{seconds%3600//60:02d}:{seconds%60:02d}"


class Progress:
    def __init__(self, output, state=None, clock=time.monotonic):
        self.path = Path(output)/"progress.json"
        self.state = state or {"train_seconds": 0., "validation_seconds": 0., "validation_runs": 0,
                              "measured_updates": 0, "updates": 0, "max_updates": 4000,
                              "epoch_queries": 32768, "effective_batch": 64}
        self.clock = clock
        self.last = clock()
        self.phase = None
        self.last_write = 0.

    def tick(self, phase=None, force=False, **fields):
        now = self.clock()
        if self.phase == "train":
            self.state["train_seconds"] += max(0., now-self.last)
        elif self.phase == "source_validation":
            self.state["validation_seconds"] += max(0., now-self.last)
        self.last = now
        if phase is not None:
            self.phase = phase
        self.state.update(fields)
        updates = self.state["updates"]
        processed = updates*self.state["effective_batch"]+self.state.get("pending_queries", 0)
        length = self.state["epoch_queries"]
        total_queries = self.state["max_updates"]*self.state["effective_batch"]
        epoch_index = min(processed//length, (total_queries-1)//length)
        epoch_size = min(length, total_queries-epoch_index*length)
        epoch_done = processed-epoch_index*length
        self.state.update(state=self.phase, epoch=processed//length+1, epoch_queries_done=processed%length,
            epoch_queries_total=length, overall_queries_done=processed,
            overall_queries_total=self.state["max_updates"]*self.state["effective_batch"],
            overall_fraction=min(1., processed/(self.state["max_updates"]*self.state["effective_batch"])),
            eta_excludes_gpu_wait=True, eta_to_update_cap=True, updated_at=time.time())
        self.state.update(epoch=epoch_index+1, epoch_queries_done=epoch_done, epoch_queries_total=epoch_size,
                          maximum_epochs=math.ceil(total_queries/length))
        measured = self.state["measured_updates"]
        per_update = self.state["train_seconds"]/measured if measured else None
        self.state["epoch_eta_seconds"] = ((epoch_size-epoch_done)/self.state["effective_batch"]*per_update
                                            if per_update is not None else None)
        remaining = self.state["max_updates"]-updates
        validation_mean = (self.state["validation_seconds"]/self.state["validation_runs"]
                           if self.state["validation_runs"] else None)
        self.state["training_eta_seconds"] = remaining*per_update if per_update is not None else None
        self.state["overall_eta_seconds"] = (remaining*per_update+((self.state["max_updates"]//250)
            -(updates//250))*validation_mean if per_update is not None and validation_mean is not None else None)
        if self.phase == "complete":
            self.state.update(overall_eta_seconds=0., training_eta_seconds=0., epoch_eta_seconds=0.,
                              task_complete=True)
        if force or now-self.last_write >= 5:
            atomic_json(self.path, self.state)
            self.last_write = now
            activity = self.path.parent/"activity.json"
            record = {"progress": self.state}
            if activity.exists():
                record["activity"] = json.loads(activity.read_text())
            print(render(record), flush=True)
        return self.state

    def evaluation(self, done, total, work_done, work_total, seconds, **fields):
        return self.tick(force=True, evaluation_done=done, evaluation_total=total,
            evaluation_round=self.state["validation_runs"]+1,
            evaluation_maximum_rounds=self.state["max_updates"]//250,
            evaluation_overall_done=self.state["validation_runs"]*total+done,
            evaluation_overall_total=self.state["max_updates"]//250*total,
            evaluation_fraction=done/total if total else 0., evaluation_work_done=work_done,
            evaluation_work_total=work_total, evaluation_eta_seconds=(seconds/work_done*(work_total-work_done)
                if work_done else None), evaluation_eta_basis="generated_numeric_steps", **fields)


def inspect(output):
    root = Path(output)
    result = {}
    for name in ("launcher", "status", "progress", "activity", "final_selection"):
        path = root/(name+".json")
        if path.exists():
            result[name] = json.loads(path.read_text())
    return result


def render(record):
    p = record.get("progress", {})
    state = record.get("launcher", {}).get("state", record.get("status", {}).get("state", p.get("state", "preparing")))
    lines = [f"state={state} | phase={p.get('state', 'not_started')} | GPU active="
             f"{duration(record.get('activity', {}).get('spent_seconds'))}"]
    if p:
        lines.append(f"epoch {p['epoch']}/{p.get('maximum_epochs', 8)} | "
            f"{p['epoch_queries_done']}/{p['epoch_queries_total']} queries | "
            f"epoch ETA {duration(p.get('epoch_eta_seconds'))}")
        lines.append(f"overall {p['updates']}/{p['max_updates']} updates "
            f"({100*p['overall_fraction']:.2f}%) | ETA {duration(p.get('overall_eta_seconds'))} "
            "to update cap; excludes unknown idle waits and may end earlier by patience")
        if "evaluation_total" in p:
            lines.append(f"{p.get('evaluation_kind', 'evaluation')} {p['evaluation_done']}/{p['evaluation_total']} "
                f"queries | ETA {duration(p.get('evaluation_eta_seconds'))} | "
                f"current query {p.get('evaluation_current_step', 0)}/{p.get('evaluation_current_horizon', 0)}")
            if p.get("evaluation_kind") != "native":
                lines.append(f"validation round {p['evaluation_round']}/{p['evaluation_maximum_rounds']} | "
                    f"overall validation {p['evaluation_overall_done']}/{p['evaluation_overall_total']}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=10)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    while True:
        record = inspect(args.output)
        print(json.dumps(record, indent=2) if args.json else render(record), flush=True)
        if not args.watch:
            break
        time.sleep(max(1, args.interval))


if __name__ == "__main__":
    main()
