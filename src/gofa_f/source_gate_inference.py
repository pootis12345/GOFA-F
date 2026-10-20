"""Stage22 scoped all-node amplitude intervention; original tensor code untouched."""
from contextlib import contextmanager

from gofa_f.source_gate import AMPLITUDES


@contextmanager
def amplitude(model, alpha):
    if alpha not in AMPLITUDES or model._active is not None or model._kernel_active \
            or getattr(model, '_source_gate_scope', False) or getattr(model, '_stability_scope', False):
        raise ValueError('Invalid/nested source gate amplitude scope')
    handles = []
    model.cache_generation += 1
    model._source_gate_scope = True
    try:
        for module in model.adapter.exchanges:
            handles.append(module.register_forward_hook(lambda m, args, raw: raw*alpha))
        yield
    finally:
        for handle in handles:
            handle.remove()
        model.cache_generation += 1
        model._source_gate_scope = False


def label(alpha):
    if alpha not in AMPLITUDES:
        raise ValueError('Unregistered amplitude')
    return {0.: 'native', .25: 'alpha025', .5: 'alpha050', 1.: 'alpha100'}[alpha]
