"""Persisted wall-time budgets; active run time conservatively bounds GPU time."""
import time
from pathlib import Path

from gofa_f.prefix_cache import atomic_json


class BudgetExhausted(RuntimeError):
    pass


class Budget:
    def __init__(self, path, total, state=None):
        self.path = Path(path)
        self.total = total
        self.state = state or {"spent_seconds": 0.0, "scopes": {}}
        if state and state.get("active_scope"):
            # Ledger checkpoints occur every five seconds. Conservatively charge the unrecorded tail.
            self.state["spent_seconds"] += 5
            self.state["scopes"][state["active_scope"]] += 5
        self.scope = None
        self.limit = 0
        self.last = time.monotonic()
        self.persisted = 0

    def tick(self, force=False):
        now = time.monotonic()
        elapsed = now - self.last
        self.last = now
        if self.scope is not None:
            self.state["spent_seconds"] += elapsed
            self.state["scopes"][self.scope] += elapsed
        if force or now - self.persisted >= 5:
            atomic_json(self.path, self.state)
            self.persisted = now

    def activate(self, scope, limit):
        self.tick(force=True)
        self.scope, self.limit = scope, limit
        self.state["active_scope"] = scope
        self.state["scopes"].setdefault(scope, 0.0)
        self.tick(force=True)

    def remaining(self):
        self.tick()
        return min(self.total - self.state["spent_seconds"],
                   self.limit - self.state["scopes"].get(self.scope, 0.0))

    def check(self):
        if self.remaining() <= 0:
            raise BudgetExhausted(self.scope)

    def close(self):
        self.tick()
        self.scope = None
        self.state["active_scope"] = None
        self.tick(force=True)
