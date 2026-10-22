"""Source-only, CPU graph utility screen. No fitting or target evaluation entry.

This is a deletion/rewiring diagnostic of a frozen, mixed-support source GNN,
not an independently trained graph ablation or proof of LLM graph transfer.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from gofa_f.allnode_graph import build_graph
from gofa_f.generative_data import Query, Series, sha
from gofa_f.prefix_cache import atomic_json
from gofa_f.source_dcrnn_data import sample, stack_samples
from gofa_f.source_dcrnn_model import DCRNN

VIEWS = ('pems04_flow', 'pems04_speed', 'boston2017', 'penmanshiel2017')
CONDITIONS = ('mixed', 'physical', 'self', 'rewired_mixed', 'rewired_physical')
PROTOCOL = 'stage35-source-graph-audit-v1'
FILES = ('source_graph_audit.py', 'source_dcrnn_model.py', 'source_dcrnn_data.py',
         'allnode_graph.py', 'generative_graph.py', 'generative_data.py')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def immutable(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError('Immutable audit evidence changed: '+path.name)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, value)


def configuration(path):
    text = Path(path).read_text(encoding='utf-8')
    if '${GOFA_WORKSPACE}' in text:
        workspace = os.environ.get('GOFA_WORKSPACE')
        if not workspace:
            raise ValueError('Set GOFA_WORKSPACE to the existing audited workspace')
        text = text.replace('${GOFA_WORKSPACE}', workspace.replace('\\', '/'))
    cfg = yaml.safe_load(text)
    if cfg['protocol'] != PROTOCOL or cfg['seed'] != 42 \
            or cfg['queries_per_view'] != 64 or cfg['bootstrap_draws'] != 2000 \
            or cfg['conditions'] != list(CONDITIONS) \
            or cfg['policy'] != dict(source_labels_only=True, target_reads=False,
                optimizer_updates=0, gpu_enabled=False, hourly_followup=False) \
            or cfg['gate'] != dict(minimum_improvement=.005, minimum_domains=2,
                maximum_other_domain_degradation=.02, minimum_supported_domains=1):
        raise ValueError('Fixed source-only screen changed')
    return cfg


def fresh_queries(data, name, variables, excluded, count=64):
    """ID/time stratification, last source20%, no error-based selection."""
    if name not in VIEWS or len(set(variables)) != len(variables):
        raise ValueError('Source view and unique variable identities required')
    length, horizon = (12, 12) if name.startswith('pems') else (48, 24)
    rng = np.random.default_rng(np.random.SeedSequence([42, 35, VIEWS.index(name)]))
    order = rng.permutation(variables).tolist()
    allocation = [order[i % len(order)] for i in range(count)]
    result = []
    for variable in order:
        number = allocation.count(variable)
        if not number:
            continue
        points = data.candidates(variable, length, horizon, 'test').copy()
        for old in excluded:
            if old['source'] == name and old['variable'] == variable:
                points = points[(points+horizon <= old['origin']) |
                                (points >= old['origin']+old['horizon'])]
        if len(points) < number:
            raise ValueError('Insufficient disjoint fresh source origins')
        chosen = []
        for pool in np.array_split(points, number):
            legal = np.array([p for p in pool if all(
                p+horizon <= o or p >= o+horizon for o in chosen)])
            if not len(legal):
                raise ValueError('Source time-stratum capacity exhausted; no substitution')
            origin = int(rng.choice(legal))
            chosen.append(origin)
            result.append(asdict(Query(name, int(variable), origin, length, horizon)))
    if len(result) != count or len({Query(**q).key for q in result}) != count:
        raise ValueError('Fresh source query coverage differs')
    return sorted(result, key=lambda q: (q['origin'], q['variable']))


def dual_support(adjacency, present):
    a = np.asarray(adjacency, np.float32).copy()
    p = np.asarray(present, bool)
    if a.shape != (len(p), len(p)) or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError('Invalid nonnegative receiver-row adjacency')
    a *= p[:, None] & p[None, :]
    def normalize(x):
        return np.divide(x, x.sum(-1, keepdims=True), out=np.zeros_like(x),
                         where=x.sum(-1, keepdims=True) > 0)
    return np.stack((normalize(a), normalize(a.T.copy())))


def rewire(support, present, query):
    """Fix the target and permute other present node identities, not histories.

    Preserves weighted degree multiset and target degree. Complete/degenerate
    graphs may remain identical; expose that fact rather than inventing edges.
    """
    key = hashlib.sha256(Query(**query).key.encode()).digest()
    seed = int.from_bytes(key[:8], 'little')
    nodes = np.flatnonzero(present)
    if 0 not in nodes:
        raise ValueError('Observed target required in the source screen')
    others = nodes[nodes != 0]
    permutation = np.arange(len(present))
    if len(others) > 1:
        rng = np.random.default_rng(np.random.SeedSequence([42, 35, seed]))
        order = rng.permutation(others)
        # A cyclic reassignment guarantees no fixed non-target node identity.
        permutation[order] = np.roll(order, 1)
    changed = support[:, permutation][:, :, permutation].copy()
    return changed, permutation


def audit_sample(data, query):
    """Identical input/roster for all five conditions; change supports only."""
    own = build_graph(data, Query(**query))
    packed = sample(data, query, labels=True)
    n = len(own.variable_ids)
    a = np.zeros((9, 9), np.float32)
    # prior is EXTERNAL topology plus self whenever audited external exists.
    if data.external is None:
        raise ValueError('Published/coordinate external graph required')
    a[:n, :n] = own.graph.prior.numpy()
    physical = dual_support(a, packed['present'])
    self_support = dual_support(np.eye(9, dtype=np.float32), packed['present'])
    rewired_physical, permutation = rewire(physical, packed['present'], query)
    rewired_mixed, _ = rewire(packed['supports'], packed['present'], query)
    supports = dict(mixed=packed['supports'].copy(), physical=physical,
        self=self_support, rewired_mixed=rewired_mixed, rewired_physical=rewired_physical)
    evidence = dict(nodes=list(own.variable_ids), permutation=permutation.tolist(),
        changed_physical=bool(not np.array_equal(physical, rewired_physical)),
        changed_mixed=bool(not np.array_equal(packed['supports'], rewired_mixed)),
        physical_cross_edges=int(np.count_nonzero(a-np.diag(np.diag(a)))),
        independent_node_scaling=True, target_fixed=True)
    return packed, supports, evidence


def code_hashes():
    return {name: sha(Path(__file__).parent/name) for name in FILES}


def source_data(manifest):
    root = Path(manifest['source_directory'])
    result = {}
    for name in VIEWS:
        path = root/(name+'.npz')
        if sha(path) != manifest['source_hashes'][name] \
                or sha(path.with_suffix('.json')) != manifest['metadata_hashes'][name]:
            raise ValueError('Source observation or graph metadata changed')
        result[name] = Series(path)
    return result


def prepare(cfg, output):
    source = Path(cfg['source_output'])
    prior = Path(cfg['gnn_output'])
    for path, value in ((source/'manifest.json', cfg['source_manifest_sha256']),
                        (prior/'manifest.json', cfg['gnn_manifest_sha256']),
                        (prior/'best.pt', cfg['gnn_checkpoint_sha256'])):
        if sha(path) != value:
            raise ValueError('Locked source-only input SHA changed')
    sm, gm = read(source/'manifest.json'), read(prior/'manifest.json')
    # Deliberately do not open old target results, coefficients, or target data.
    partial = dict(source_directory=str(Path(gm['source_configuration']['data']['directory'])/'sources'),
        source_hashes=gm['source_hashes'], metadata_hashes=gm['metadata_hashes'])
    data = source_data(partial)
    excluded = sm['pools']['confirmation']+sm['pools']['diagnostic']
    rows = {}
    for name in VIEWS:
        variables = sm['pools']['traffic_variables'] if name.startswith('pems') \
            else list(range(len(data[name].ids)))
        rows[name] = fresh_queries(data[name], name, variables, excluded)
    value = dict(configuration=cfg, code=code_hashes(), **partial, queries=rows,
        excluded_source_query_sha256=digest(excluded),
        training_updates=0, target_data_opened=False, checkpoint_selected_by_sources_only=True,
        new_graph_disjoint_from_training=False, independent_blind_source_test=False,
        inference_only_deletion_distribution_shift=True,
        old_source_labels_exposed_in_prior_research=True,
        interpretation='feasibility screen; matched trained and held-out graph controls still required')
    immutable(output/'manifest.json', value)
    atomic_json(output/'status.json', dict(state='prepared', stage=35, gpu_seconds=0.))
    return value


def verify(cfg, output):
    manifest = read(output/'manifest.json')
    if manifest['configuration'] != cfg or manifest['code'] != code_hashes() \
            or sha(Path(cfg['gnn_output'])/'best.pt') != cfg['gnn_checkpoint_sha256']:
        raise ValueError('Prepared audit binding changed')
    return manifest


@torch.no_grad()
def predict(model, rows, supports, horizon):
    a = stack_samples(rows)
    # Labels, teacher values and masks are intentionally not passed to the model.
    prediction = model(torch.from_numpy(a['x']), torch.from_numpy(np.stack(supports)),
        torch.from_numpy(a['present']), horizon)[:, :, 0].double().numpy()
    result = (prediction+.5)*a['scale'][:, None]+a['low'][:, None]
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite source forecast')
    return result


def variable_errors(records, condition):
    groups = {}
    for row in records:
        valid = np.asarray(row['observed'], bool)
        e = np.abs(np.asarray(row['predictions'][condition])-np.asarray(row['target']))[valid]
        if not len(e):
            raise ValueError('Empty evaluation target')
        groups.setdefault(str(row['query']['variable']), []).extend(e.tolist())
    return {k: float(np.mean(v)) for k, v in groups.items()}


def summarize(records, cfg):
    metrics, per_variable, structure = {}, {}, {}
    for name in VIEWS:
        own = records[name]
        per_variable[name] = {c: variable_errors(own, c) for c in CONDITIONS}
        metrics[name] = {c: float(np.mean(list(per_variable[name][c].values()))) for c in CONDITIONS}
        structure[name] = dict(queries=len(own), variables=len(per_variable[name]['physical']),
            physical_rewiring_changed_queries=sum(r['graph']['changed_physical'] for r in own),
            mixed_rewiring_changed_queries=sum(r['graph']['changed_mixed'] for r in own),
            no_cross_edge_queries=sum(r['graph']['physical_cross_edges'] == 0 for r in own))
    domains = {}
    for name, views in dict(traffic=VIEWS[:2], weather=VIEWS[2:3], wind=VIEWS[3:]).items():
        comparisons = {}
        common = sorted(set.intersection(*(set(per_variable[v]['physical']) for v in views)))
        if any(set(common) != set(per_variable[v]['physical']) for v in views):
            raise ValueError('Traffic views must use the same sensor coverage')
        rng = np.random.default_rng(np.random.SeedSequence([42, 35, list(('traffic','weather','wind')).index(name)]))
        draws = rng.integers(len(common), size=(cfg['bootstrap_draws'], len(common)))
        for control in ('self', 'rewired_physical'):
            ratios, simulated = [], []
            for view in views:
                a = np.array([per_variable[view]['physical'][v] for v in common])
                b = np.array([per_variable[view][control][v] for v in common])
                if b.mean() <= 0 or np.any(b[draws].mean(1) <= 0):
                    raise ValueError('Zero baseline cannot establish graph gain')
                ratios.append(a.mean()/b.mean())
                simulated.append(a[draws].mean(1)/b[draws].mean(1))
            ratio = float(np.exp(np.log(ratios).mean()))
            interval = np.quantile(np.exp(np.log(simulated).mean(0)), [.025, .975]).tolist()
            comparisons[control] = dict(physical_to_control_mae_ratio=ratio,
                paired_variable_cluster_ratio_95pct=interval, clusters=len(common),
                physical_point_improvement=1-ratio)
        domains[name] = comparisons
    policy = cfg['gate']
    point_domains = [d for d, c in domains.items() if all(
        x['physical_to_control_mae_ratio'] <= 1-policy['minimum_improvement'] for x in c.values())]
    supported = [d for d in point_domains if all(
        x['paired_variable_cluster_ratio_95pct'][1] < 1 for x in domains[d].values())]
    safe = all(x['physical_to_control_mae_ratio'] <= 1+policy['maximum_other_domain_degradation']
               for c in domains.values() for x in c.values())
    informative = all(structure[v]['physical_rewiring_changed_queries'] >= 32 for v in VIEWS)
    passed = len(point_domains) >= policy['minimum_domains'] \
        and len(supported) >= policy['minimum_supported_domains'] and safe and informative
    return dict(complete=True, source_view_variable_equal_mae=metrics, source_domains=domains,
        graph_structure_audit=structure, gate=dict(passed=passed, point_supported_domains=point_domains,
            interval_supported_domains=supported, no_large_domain_regression=safe,
            rewiring_informative=informative,
            next_action='matched_trained_controls_and_heldout_graph_pilot' if passed
                        else 'stop_before_expensive_chattime_training'),
        gpu_seconds=0., target_forecasts=0, optimizer_updates=0,
        graph_llm_transfer_proven=False, strict_heldout_graph_transfer_proven=False,
        limitations=['frozen model trained on mixed physical/correlation supports',
            'support deletion can shift the model input distribution',
            'all source graphs seen during old training; fresh time windows only',
            'single seed; few weather/wind variable clusters; prior source development exposure',
            'coordinate nearest-neighbor graphs are geographic proxies, not measured weather/wake causality'])


def run(cfg, output, charge):
    m = verify(cfg, output)
    proof = read(output/'cpu_preflight.json')
    if not proof['passed'] or proof['manifest_sha256'] != sha(output/'manifest.json'):
        raise ValueError('Bound full CPU regression proof required')
    if (output/'report.json').exists():
        return read(output/'report.json')
    data = source_data(m)
    model = DCRNN()
    ckpt = torch.load(Path(cfg['gnn_output'])/'best.pt', map_location='cpu', weights_only=False)
    if ckpt['binding'] != cfg['gnn_manifest_sha256']:
        raise ValueError('Source-selected GNN binding differs')
    model.load_state_dict(ckpt['model'])
    model.eval().requires_grad_(False)
    all_records = {}
    for name in VIEWS:
        rows, all_records[name] = m['queries'][name], []
        folder = output/'records'/name
        folder.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(rows), 16):
            chunk = rows[start:start+16]
            bindings = [digest(dict(manifest=sha(output/'manifest.json'), query=q)) for q in chunk]
            paths = [folder/f'{start+i:06d}.json' for i in range(len(chunk))]
            if all(p.exists() for p in paths):
                saved = [read(p) for p in paths]
                if any(r['binding'] != b or r['query'] != q for r, b, q in zip(saved, bindings, chunk)):
                    raise ValueError('Saved source record binding differs')
            else:
                packed = [audit_sample(data[name], q) for q in chunk]
                predictions = {c: predict(model, [r[0] for r in packed],
                    [r[1][c] for r in packed], chunk[0]['horizon']) for c in CONDITIONS}
                saved = []
                for i, (q, (a, _, graph)) in enumerate(zip(chunk, packed)):
                    row = dict(binding=bindings[i], query=q, graph=graph,
                        predictions={c: predictions[c][i].tolist() for c in CONDITIONS},
                        target=a['raw_target'].tolist(), observed=a['observed'][:, 0].tolist())
                    immutable(paths[i], row)
                    saved.append(row)
            all_records[name].extend(saved)
            charge()
            atomic_json(output/'progress.json', dict(state='running', phase='source_graph_screen',
                view=name, completed=start+len(chunk), per_view=len(rows),
                completed_records=sum(map(len, all_records.values())), total_records=256,
                gpu_seconds=0., optimizer_updates=0))
        print(json.dumps(dict(view=name, complete=True, queries=len(rows))), flush=True)
    if sha(Path(cfg['gnn_output'])/'best.pt') != cfg['gnn_checkpoint_sha256']:
        raise ValueError('Read-only source checkpoint unexpectedly changed')
    report = summarize(all_records, cfg)
    immutable(output/'report.json', report)
    atomic_json(output/'status.json', dict(state='screen_passed' if report['gate']['passed'] else 'stopped_at_screen',
        stage=35, phase='source_graph_screen', gate=report['gate'], gpu_seconds=0.,
        expensive_training_started=False, hourly_followup=False))
    atomic_json(output/'progress.json', dict(state='complete', phase='source_graph_screen', eta_seconds=None))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mode', required=True, choices=('prepare', 'run', 'progress'))
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('Explicitly disable CUDA for source-only CPU screen')
    torch.set_num_threads(2)
    cfg = configuration(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'progress':
        for name in ('status.json', 'progress.json', 'activity.json'):
            if (args.output/name).exists():
                print(json.dumps({name: read(args.output/name)}))
        return
    import fcntl
    with (args.output/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.mode == 'prepare':
            prepare(cfg, args.output)
            return
        path = args.output/'activity.json'
        activity = read(path) if path.exists() else dict(cpu_active_wall_seconds=0., gpu_seconds=0.)
        before = [time.monotonic()]
        def charge():
            now = time.monotonic()
            activity['cpu_active_wall_seconds'] += now-before[0]
            before[0] = now
            if torch.cuda.is_initialized():
                raise RuntimeError('CPU screen unexpectedly initialized CUDA')
            stat = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
            activity.update(worker=dict(pid=os.getpid(), start_ticks=stat[19]),
                gpu_seconds=0., cuda_initialized=False)
            atomic_json(path, activity)
        charge()
        try:
            report = run(cfg, args.output, charge)
            print(json.dumps(report['gate']), flush=True)
        except Exception as exc:
            atomic_json(args.output/'failure.json', dict(error=repr(exc), time=time.time()))
            atomic_json(args.output/'status.json', dict(state='failed', stage=35, error=repr(exc)))
            raise
        finally:
            charge()
            activity['worker'] = None
            atomic_json(path, activity)


if __name__ == '__main__':
    main()
