"""Stage18 epoch/total/validation ETA plus the active all-node query shape."""
import argparse
import time

from gofa_f.generative_progress import Progress, inspect, render as base_render

__all__ = ["Progress", "render"]


def render(record):
    text = base_render(record)
    p = record.get("progress", {})
    if "current_nodes" in p:
        text += (f"\nquery: nodes={p['current_nodes']}/9 history={p.get('current_history')} "
                 f"chunk={p.get('current_chunk')} tokens={p.get('current_tokens', 'pending')} "
                 f"subphase={p.get('training_subphase', 'pending')}")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=10.)
    args = parser.parse_args()
    if args.interval <= 0:
        raise ValueError("Positive refresh interval required")
    while True:
        print(render(inspect(args.output)), flush=True)
        if not args.watch:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
