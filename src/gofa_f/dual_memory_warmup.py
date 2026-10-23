"""Source-only temporal initialization for the existing dual-Memory expert.

No ChatTime loading, target access, new model architecture, or target selection.
The exported GRU/head fit is transferred before creating a joint optimizer.
"""
import argparse
from copy import deepcopy
import fcntl
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from torch import nn
import yaml

from gofa_f.dual_memory_data import GROUPS, SHAPES
from gofa_f.dual_memory_engines import model_digest
from gofa_f.dual_memory_fitting import fit, read
from gofa_f.dual_memory_policy import digest, DOMAINS
from gofa_f.dual_memory_store import SourceStore
from gofa_f.generative_data import sha
from gofa_f.graph_memory_data import flatten_queries, fresh_queries
from gofa_f.prefix_cache import atomic_json

FILES = ('dual_memory_warmup.py', 'dual_memory_model.py', 'dual_memory_data.py',
         'dual_memory_engines.py', 'dual_memory_fitting.py', 'dual_memory_policy.py',
         'dual_memory_store.py', 'graph_memory_data.py', 'generative_data.py')
PROTOCOL = 'stage60-source-warmup-v1'


class TemporalWarmup(nn.Module):
    """Exactly the production GRU and head, with zero Memory-read columns."""
    def __init__(self, slots=4, width=128):
        super().__init__()
        self.slots, self.width = slots, width
        self.temporal = nn.GRU(3, width, batch_first=True)
        self.head = nn.Sequential(nn.Linear(slots*width+width+4, width),
                                  nn.SiLU(), nn.Linear(width, 1))
        with torch.no_grad():
            self.head[0].weight[:, :slots*width].zero_()
            self.head[-1].weight.zero_(); self.head[-1].bias.zero_()

    def forecasts(self, items):
        """Bucket same shapes; future labels/masks never enter this accessor."""
        buckets = {}
        for index, item in enumerate(items):
            buckets.setdefault((item.history.shape[1], item.horizon), []).append(index)
        predictions = [None]*len(items)
        for (length, horizon), indices in buckets.items():
            counts = [len(items[i].node_ids) for i in indices]
            normalized = torch.cat([items[i].normalized.float() for i in indices])
            observed = torch.cat([items[i].observed.float() for i in indices])
            clock = torch.linspace(-1., 0., length).expand(len(normalized), length)
            encoded, _ = self.temporal(torch.stack((normalized, observed, clock), -1))
            temporal = encoded[:, -1]
            u = (torch.arange(1, horizon+1, dtype=torch.float32)/horizon)[:, None]
            times = torch.cat((u, u.square(), torch.full_like(u, horizon/24),
                               torch.full_like(u, length/48)), -1)
            features = torch.cat((torch.zeros(len(temporal), horizon, self.slots*self.width),
                temporal[:, None].expand(-1, horizon, -1),
                times[None].expand(len(temporal), -1, -1)), -1)
            residual = self.head(features).squeeze(-1).double()
            last = torch.cat([items[i].last for i in indices]).double()
            extent = torch.cat([items[i].extent for i in indices]).double()
            values = last[:, None]+extent[:, None]*residual
            for index, value in zip(indices, values.split(counts)):
                predictions[index] = value
        return predictions


def transfer(adapter, selected, binding):
    """Initialize a fresh production adapter, before any live graph/optimizer."""
    if selected['binding'] != binding:
        raise ValueError('Temporal initialization binding changed')
    if any(p.grad is not None for p in adapter.parameters()):
        raise ValueError('Transfer requires a fresh adapter without gradients')
    width, slots = adapter.temporal.hidden_size, adapter.slots
    model = TemporalWarmup(slots, width)
    model.load_state_dict(selected['model'], strict=True)
    if model_digest(model) != selected['model_sha256']:
        raise ValueError('Temporal initialization tensor SHA changed')
    if bool(model.head[0].weight[:, :slots*width].any()):
        raise ValueError('Warmup cannot contain a pre-fitted Memory contribution')
    adapter.temporal.load_state_dict(model.temporal.state_dict(), strict=True)
    adapter.head.load_state_dict(model.head.state_dict(), strict=True)


class TemporalEngine:
    def __init__(self, manifest, store, microbatch=16):
        self.model = TemporalWarmup()
        self.state_model = self.model
        self.manifest, self.store = manifest, store
        self.microbatch, self.cuda = microbatch, False

    def warmup(self, enabled):
        pass

    def loss(self, rows):
        from gofa_f.dual_memory_policy import query_loss
        inputs = [self.store.get(row, 'train') for row in rows]
        predictions = self.model.forecasts([x[0] for x in inputs])
        return sum(query_loss(p, y, mask, item.present,
            self.manifest['training_unit_normalizers'][row['source']])
            for row, p, (item, y, mask) in zip(rows, predictions, inputs))

    @torch.no_grad()
    def predict(self, row, split, role='fit'):
        item, y, mask = self.store.get(row, split, role)
        p = self.model.forecasts([item])[0]
        return p[0].numpy(), y[0].numpy(), mask[0].numpy(), item.identity

    def close(self):
        self.model = self.state_model = None


def cpu_only():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('Explicit CUDA-disabled CPU execution required')
    if os.environ.get('OMP_NUM_THREADS') != '2' or os.environ.get('MKL_NUM_THREADS') != '2':
        raise ValueError('Two OMP and MKL threads required')
    torch.set_num_threads(2)


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text().replace('${GOFA_WORKSPACE}', os.environ['GOFA_WORKSPACE']))
    if cfg['protocol'] != PROTOCOL or cfg['target_training_enabled'] or cfg['target_evaluation_enabled']:
        raise ValueError('Source-only temporal-initialization configuration required')
    if cfg['warmup'] != dict(max_updates=2000, effective_batch=16, microbatch=16,
        learning_rate=.001, warmup_updates=50, interface_only_updates=0,
        weight_decay=.0001, gradient_clip=1., validation_every=100,
        early_min_updates=1000, early_patience=8, significant_improvement=.001):
        raise ValueError('Fixed finite warmup recipe changed')
    if cfg['joint_memory_training'] != dict(depths=[6, 3], max_updates=3000,
        effective_batch=8, microbatch=1, learning_rate=.0001, warmup_updates=50,
        interface_only_updates=50, weight_decay=.0001, gradient_clip=1.,
        validation_every=100, early_min_updates=1200, early_patience=8,
        significant_improvement=.001):
        raise ValueError('Preregistered joint recipe changed')
    return cfg


def inherited_cost(parent, cfg):
    path = Path(cfg['inherited_costs_file'])
    if sha(path) != cfg['inherited_costs_sha256']:
        raise ValueError('Inherited cumulative cost evidence changed')
    value = read(path)
    source = read(Path(parent)/'activity.json')['spent_seconds']
    if not math.isclose(source, value['inherited_source_gpu_seconds'], rel_tol=0., abs_tol=1e-6):
        raise ValueError('Inherited source GPU ledger changed')
    total = value['cumulative_gpu_seconds']
    if not math.isclose(total, source+value['new_evaluation_gpu_seconds'], rel_tol=0., abs_tol=1e-6):
        raise ValueError('Inherited GPU cost reset or double counted')
    return total


def prepare(cfg, output):
    cpu_only()
    parent = Path(cfg['parent_output'])
    if sha(parent/'manifest.json') != cfg['parent_manifest_sha256']:
        raise ValueError('Parent source manifest changed')
    if read(parent/'status.json')['state'] != 'complete' or read(parent/'activity.json')['worker'] is not None:
        raise ValueError('Completed and exited parent required')
    old = read(parent/'manifest.json')
    for name, value in old['code'].items():
        if sha(Path(__file__).parent/name) != value:
            raise ValueError('Accepted architecture/input core changed: '+name)
    for name, value in old['formal_runtime_code'].items():
        if sha(Path(__file__).parent/name) != value:
            raise ValueError('Inherited exact fitting implementation changed: '+name)
    m = deepcopy(old)
    m['configuration']['training'] = cfg['warmup']
    m['configuration']['queries']['initial_training_per_view'] = 8192
    m['warmup_configuration'] = cfg
    m['warmup_runtime_code'] = {n: sha(Path(__file__).parent/n) for n in FILES}
    m['parent_source_manifest_sha256'] = cfg['parent_manifest_sha256']
    m['source_cache_index_sha256'] = sha(parent/'source_inputs/index.json')
    # Reserve NEW confirmation windows before fitting, excluding old source
    # development/confirmation/diagnostic reservations and every parent query.
    exclusions = list(flatten_queries(old['queries']))
    previous = old['configuration']
    prior = Path(previous['source_output'])/'manifest.json'
    if sha(prior) != previous['source_manifest_sha256']:
        raise ValueError('Prior source reservation lineage changed')
    prior = read(prior)
    for name in ('validation', 'confirmation', 'diagnostic'):
        exclusions.extend(flatten_queries(prior['pools'][name]))
    reserved = Path(previous['reservation_output'])/'manifest.json'
    if sha(reserved) != old['prior_reservations_sha256']:
        raise ValueError('Prior unopened source reservation changed')
    for value in read(reserved)['queries'].values():
        exclusions.extend(value['development']); exclusions.extend(value['reserved'])
    store = SourceStore(old, parent/'source_inputs')
    for name in GROUPS:
        data = store.source(name)
        for length, horizon in SHAPES:
            rows = fresh_queries(data, name, length, horizon, 'test', 32,
                old['partitions'][name]['confirmation'], exclusions, 60)
            m['queries'][name][f'{length}-{horizon}']['confirmation'] = rows
            exclusions.extend(rows)
    m['source_confirmation_labels_opened'] = False
    m['target_labels_opened'] = False
    m['inherited_gpu_seconds'] = inherited_cost(parent, cfg)
    m['inherited_gpu_costs_sha256'] = cfg['inherited_costs_sha256']
    output.mkdir(parents=True, exist_ok=True)
    if (output/'manifest.json').exists() and read(output/'manifest.json') != m:
        raise ValueError('Warmup manifest is immutable')
    atomic_json(output/'manifest.json', m)
    atomic_json(output/'status.json', dict(state='prepared', phase='cpu_temporal_source_initialization',
        goal_achieved=False, target_predictions=0, source_confirmation_labels_opened=False))
    return m


def run(cfg, output):
    cpu_only()
    handle = (output/'worker.lock').open('a'); fcntl.flock(handle, fcntl.LOCK_EX|fcntl.LOCK_NB)
    m = read(output/'manifest.json'); binding = sha(output/'manifest.json')
    if m['warmup_configuration'] != cfg or m['warmup_runtime_code'] != {n: sha(Path(__file__).parent/n) for n in FILES}:
        raise ValueError('Warmup runtime or configuration changed')
    proof = read(output/'cpu_preflight.json', {})
    if not proof.get('passed') or proof.get('manifest_sha256') != binding or proof.get('runtime_code') != m['warmup_runtime_code']:
        raise ValueError('Related CPU tests must bind this immutable runtime/manifest')
    parent = Path(cfg['parent_output'])
    if sha(parent/'source_inputs/index.json') != m['source_cache_index_sha256']:
        raise ValueError('Shared source cache changed')
    stopped = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    started = time.monotonic(); prior = read(output/'activity.json', {}).get('spent_cpu_residency_seconds', 0.)
    from gofa_f.idle_launcher import process_identity
    worker = dict(pid=os.getpid(), start_ticks=process_identity(os.getpid()))
    visible = {}
    def progress(**fields):
        if fields.get('fit') != visible.get('fit') or fields.get('phase') != visible.get('phase'):
            visible.clear()
        visible.update(fields)
        atomic_json(output/'status.json', dict(state='running', goal_achieved=False,
            target_predictions=0, source_confirmation_labels_opened=False, **visible))
        atomic_json(output/'activity.json', dict(worker=worker,
            spent_cpu_residency_seconds=prior+time.monotonic()-started,
            inherited_gpu_seconds=m['inherited_gpu_seconds'], new_gpu_seconds=0,
            cuda_initialized=False, threads=2))
    def check():
        if stopped[0]:
            raise InterruptedError('cooperative CPU warmup pause')
    selections = {}; engine = None
    try:
        # Avoid three more fits if the existing temporal branch cannot even
        # learn the included-source development task with cheap CPU updates.
        for held in (None, *DOMAINS):
            check(); random.seed(42); np.random.seed(42); torch.manual_seed(42)
            name = 'all' if held is None else 'hold-'+held
            store = SourceStore(m, parent/'source_inputs', held_group=held)
            engine = TemporalEngine(m, store)
            best = fit(engine, m, cfg, output/name, name, None,
                       digest([binding, 'temporal', name]), check, progress)
            selections[name] = read(output/name/'selection.json')
            if held is None and best['update'] == 0:
                atomic_json(output/'status.json', dict(state='failed_source_warmup',
                    reason='all_source_best_remains_persistence', goal_achieved=False,
                    target_predictions=0, joint_gpu_training_started=False))
                return
            engine.close(); engine = None
        atomic_json(output/'selection_seal.json', dict(binding=binding,
            selections={n:dict(selection_sha256=sha(output/n/'selection.json'),
                checkpoint_sha256=s['checkpoint_sha256'], model_sha256=s['model_sha256'],
                best_update=s['best_update'], excluded_group=s['excluded_group']) for n,s in selections.items()},
            target_labels_used=False, source_confirmation_labels_used=False,
            joint_memory_training_complete=False, goal_achieved=False))
        atomic_json(output/'status.json', dict(state='complete', phase='temporal_initializations_locked',
            goal_achieved=False, target_predictions=0, joint_memory_training_complete=False))
    except InterruptedError:
        atomic_json(output/'status.json', dict(state='paused', phase='cpu_temporal_source_initialization', goal_achieved=False))
        raise SystemExit(75)
    except BaseException as exc:
        atomic_json(output/f'failure-{time.time_ns()}.json', dict(error=repr(exc), binding=binding))
        atomic_json(output/'status.json', dict(state='failed', error=repr(exc), goal_achieved=False))
        raise
    finally:
        if engine is not None: engine.close()
        atomic_json(output/'activity.json', dict(worker=None,
            spent_cpu_residency_seconds=prior+time.monotonic()-started,
            inherited_gpu_seconds=m['inherited_gpu_seconds'], new_gpu_seconds=0,
            cuda_initialized=False, threads=2,
            peak_rss_kib=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--prepare', action='store_true'); args = parser.parse_args()
    cfg = load_config(args.config); output = Path(args.output)
    if args.prepare: prepare(cfg, output)
    else: run(cfg, output)


if __name__ == '__main__': main()
