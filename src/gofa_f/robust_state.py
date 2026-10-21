"""Stage24 immutable source pools, prior evidence and guarded selection."""
from collections import defaultdict
from pathlib import Path
import shutil

import numpy as np

from gofa_f.allnode_stage import read
from gofa_f.generative_data import Series, sha
from gofa_f.graph_benchmark import ADAPTER_SHA, atomic
from gofa_f.short_data import origins, stratified
from gofa_f.short_sampled_state import exited
from gofa_f.source_gate_state import TARGETS
from gofa_f.source_retraining_data import VIEWS


def validate(cfg):
    if cfg['protocol'] != 'stage24-robust-source-adaptation-v1' or cfg['seed'] != 42 \
            or cfg['adapter_sha256'] != ADAPTER_SHA or cfg['targets'] != list(TARGETS) \
            or cfg['loss'] != dict(ce=1., cdf=1., parent_kl=.1, temperature=1.) \
            or cfg['training'] != dict(max_updates=250, effective_batch=64, microbatch=2,
                learning_rate=2e-5, warmup_updates=20, weight_decay=1e-4, gradient_clip=1,
                quick_validation_update=50, full_validation_updates=[100, 250]) \
            or cfg['sampling'] != dict(samples=8, top_k=100, temperature=1., top_p=1.) \
            or cfg['data']['target_training_enabled'] or cfg['data']['target_evaluation_enabled'] != 'gated_only' \
            or cfg['resources'] != dict(admission_policy='idle', admission_samples=3, resource_check_seconds=10,
                memory_cap_gib=40, torch_cap_gib=38, activity_limit_seconds=None, wall_deadline=None,
                yield_on_foreign=False, leave_free_gib=0):
        raise ValueError('Fixed Stage24 policy changed')


def gate100(native_score, parent_score, parent_reference):
    return all(x <= 1.02 for x in parent_score['domains'].values()) \
        and native_score['W'] <= parent_reference['W']*.995


def validation_boundary(update, pending):
    """A restored partial effective batch continues accumulation, never validation."""
    return update in (50, 100, 250) and pending == 0


def parent_replay_bindings(prior):
    """The parent source probes use the original four sources, not Stage23 views."""
    if not prior.get('probes') or not prior.get('source_raw_hashes') or not prior.get('source_configuration'):
        raise ValueError('Original source replay configuration/probes/raw hashes required')
    for probe in prior['probes']:
        if probe['query']['source'] not in prior['source_raw_hashes'] \
                or set(probe['references']) != {'native', 'gofa'}:
            raise ValueError('Original source replay hash/reference incomplete')
    return dict(source_raw_hashes=prior['source_raw_hashes'])


def select(history):
    eligible = [r for r in history if all(x <= 1.02 for x in r['parent_score']['domains'].values())]
    return min(eligible, key=lambda r: (r['native_score']['W'], r['native_score']['S'], r['update'])) \
        if eligible else None


def passed(native_score, parent_score):
    return native_score['S'] <= .99 and all(x <= 1.02 for x in native_score['domains'].values()) \
        and all(x <= 1.02 for x in parent_score['domains'].values())


def quick_pool(pool):
    result = []
    traffic = sorted({q['variable'] for q in pool if q['source'] == 'pems04_flow'})[:16]
    for index, name in enumerate(VIEWS):
        own = [q for q in pool if q['source'] == name]
        variables = traffic if index < 2 else sorted({q['variable'] for q in own})
        wanted = 16 if index < 2 else 32
        for j, variable in enumerate(variables):
            rows = sorted([q for q in own if q['variable'] == variable], key=lambda q: q['origin'])
            count = wanted//len(variables)+int(j < wanted % len(variables))
            result.extend(rows[int(i)] for i in np.linspace(0, len(rows)-1, count))
    if len(result) != 96:
        raise ValueError('Quick validation budget not met')
    return result


def recorded_windows(outputs, series):
    """Exclude recorded labels by IDs/time, never forecast error; no target arrays."""
    exclusions = {n: defaultdict(set) for n in VIEWS}
    receipts = {}
    def walk(value):
        if isinstance(value, dict):
            if {'source', 'variable', 'origin', 'horizon'} <= value.keys():
                name = value['source']
                names = ('pems04_flow', 'pems04_speed') if name == 'pems04' else (name,)
                for name in names:
                    if name not in VIEWS:
                        continue
                    v, start, horizon = int(value['variable']), int(value['origin']), int(value['horizon'])
                    if start+horizon > int(.8*len(series[name].values)):
                        exclusions[name][v].add((start, start+horizon))
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    for path in sorted(Path(outputs).glob('*/manifest.json')):
        receipts[str(path)] = sha(path)
        walk(read(path))
    return exclusions, receipts


def confirmation_pool(series, old_pool, exclusions):
    result = []
    wind_order = np.random.default_rng(np.random.SeedSequence([42, 24])).permutation(14).tolist()
    for index, name in enumerate(VIEWS):
        variables = old_pool['traffic_variables'] if index < 2 else list(range(16)) if index == 2 else wind_order
        for j, v in enumerate(variables):
            count = 1 if index < 2 else 4 if index == 2 else 4+int(j < 8)
            points = origins(series[name], int(v), VIEWS[name][1], VIEWS[name][2], 'test')
            for low, high in exclusions[name].get(int(v), []):
                points = points[(points+VIEWS[name][2] <= low) | (points >= high)]
            rng = np.random.default_rng(np.random.SeedSequence([42, 24, index, int(v)]))
            result.extend(dict(source=name, variable=int(v), origin=int(t), length=VIEWS[name][1],
                               horizon=VIEWS[name][2], chunk=0)
                          for t in stratified(points, count, VIEWS[name][2], rng, disjoint=True))
    if len(result) != 192:
        raise ValueError('Fresh confirmation coverage unavailable')
    return result


def reference_files(root, labels, queries):
    """Seal committed medians and every raw path needed for replay/reuse."""
    result = {}
    original = read(Path(root)/'manifest.json')['pools']['validation']
    for label in labels:
        aggregate = Path(root)/(label+'-result.json')
        item = read(aggregate)
        if not item['complete']:
            raise ValueError('Incomplete reference: '+label)
        result[str(aggregate)] = sha(aggregate)
        for name in VIEWS:
            own = [q for q in queries if q['source'] == name]
            folder = Path(root)/'evaluations'/(label+'-'+name)
            for q in own:
                # Index is view-local in the original sealed validation list.
                view = [x for x in original if x['source'] == name]
                i = view.index(q)
                median = folder/f'{i:06d}.json'
                row = read(median)
                if row['query'] != q or not row['format_complete'] or len(row['seeds']) != 8:
                    raise ValueError('Reference median/query incomplete')
                result[str(median)] = sha(median)
                for s in range(8):
                    path = folder/'trajectories'/f'{i:06d}-{s:02d}.json'
                    sample = read(path)
                    if sample['query'] != q or sample['seed'] != row['seeds'][s] or sample['sample'] != s:
                        raise ValueError('Reference trajectory/query/seed mismatch')
                    result[str(path)] = sha(path)
    return result


def prepare(cfg, output):
    validate(cfg)
    output, prior = Path(output), Path(cfg['stage23_output'])
    if (output/'manifest.json').exists():
        return verify(cfg, output)
    exited(prior)
    old = read(prior/'manifest.json')
    replay = parent_replay_bindings(old)
    original_sources = Path(old['source_configuration']['data']['directory'])/'sources'
    for probe in old['probes']:
        name = probe['query']['source']
        if sha(original_sources/(name+'.npz')) != replay['source_raw_hashes'][name]:
            raise ValueError('Original parent replay observations changed')
    if read(prior/'status.json')['state'] != 'complete' or sha(prior/'parent.pt') != ADAPTER_SHA:
        raise ValueError('Sealed Stage23/parent3500 required')
    for name in old['unchanged_tensor_files']:
        if sha(Path(__file__).parent/name) != old['code'][name]:
            raise ValueError('Accepted tensor core changed: '+name)
    series = {n: Series(Path(cfg['data']['directory'])/'sources'/(n+'.npz')) for n in VIEWS}
    for name, data in series.items():
        if sha(data.path) != old['source_hashes'][name] or sha(data.path.with_suffix('.json')) != old['metadata_hashes'][name]:
            raise ValueError('Audited source data changed')
    pools = {k: old['pools'][k] for k in ('train', 'validation', 'diagnostic', 'epsilons', 'traffic_variables')}
    pools['quick'] = quick_pool(pools['validation'])
    exclusions, receipts = recorded_windows(prior.parent, series)
    pools['confirmation'] = confirmation_pool(series, pools, exclusions)
    repair = cfg.get('repair_from')
    if repair:
        repair = Path(repair)
        failed = read(repair/'status.json')
        fault = read(repair/'failures.json')
        if (read(repair/'activity.json')['worker'] is not None or failed['state'] != 'failed'
                or failed['phase'] != 'cuda_mechanisms' or failed['new_updates'] != 0
                or failed['pending_queries'] != 0 or failed['target_evaluations'] != 0
                or fault != dict(type='KeyError', error="'source_raw_hashes'", phase='cuda_mechanisms')
                or (repair/'evaluations').exists() or (repair/'mechanisms.json').exists()):
            raise ValueError('Only the exact no-training/no-prediction parent-replay binding failure is repairable')
        previous = read(repair/'manifest.json')
        for key in ('train', 'validation', 'diagnostic', 'quick', 'epsilons', 'traffic_variables'):
            if pools[key] != previous['pools'][key]:
                raise ValueError('Repair changed source pools: '+key)
        # These windows were sealed but never evaluated in v1. Preserve their IDs,
        # rather than treating preparation as new label exposure or reselecting.
        pools['confirmation'] = previous['pools']['confirmation']
    target = Path(cfg['stage21_output'])
    exited(target)
    tm = read(target/'manifest.json')
    if tm['configuration']['adapter_sha256'] != ADAPTER_SHA or tm['source_selection']['updates'] != 3500:
        raise ValueError('Target suite was not bound to the parent3500')
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(protocol=cfg['protocol'], configuration=cfg, pools=pools, **replay,
        source_configuration=old['source_configuration'], weights_sha256=old['weights_sha256'],
        weight_stats=old['weight_stats'], alphabet=old['alphabet'], probes=old['probes'],
        source_hashes=old['source_hashes'], metadata_hashes=old['metadata_hashes'],
        parent_stage23_manifest_sha256=sha(prior/'manifest.json'), target_manifest_sha256=sha(target/'manifest.json'),
        confirmation_manifest_receipts=receipts, diagnostic_source=old['pools']['diagnostic'],
        references=reference_files(prior, ['validation-native', 'validation-parent'], pools['validation']),
        diagnostic_references=reference_files(prior, ['validation-update-000250', 'validation-update-000500'], pools['diagnostic']),
        candidate_hashes={n: sha(prior/n) for n in ('parent.pt', 'candidate-000250.pt', 'candidate-000500.pt')},
        code={p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        parent_updates=3500, new_optimizer=True, exact_stage23_resume=False,
        target_predictions_allowed='only_after_locked_source_confirmation')
    previous = read(output/'manifest.json')
    if previous and previous != manifest:
        raise ValueError('Prepared Stage24 binding changed')
    atomic(output/'manifest.json', manifest)
    if not (output/'parent.pt').exists():
        shutil.copyfile(prior/'parent.pt', output/'parent.pt')
    if not (output/'gpu_identity.json').exists():
        atomic(output/'gpu_identity.json', read(prior/'gpu_identity.json'))
    if not (output/'status.json').exists():
        atomic(output/'status.json', dict(state='prepared', new_updates=0, target_evaluations=0))
    return manifest


def verify(cfg, output):
    validate(cfg)
    m = read(Path(output)/'manifest.json')
    if m['configuration'] != cfg or m['code'] != {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')} \
            or sha(Path(output)/'parent.pt') != ADAPTER_SHA \
            or sha(Path(cfg['stage23_output'])/'manifest.json') != m['parent_stage23_manifest_sha256'] \
            or sha(Path(cfg['stage21_output'])/'manifest.json') != m['target_manifest_sha256']:
        raise ValueError('Immutable Stage24 code/provenance differs')
    for name in VIEWS:
        root = Path(cfg['data']['directory'])/'sources'
        if sha(root/(name+'.npz')) != m['source_hashes'][name] or sha(root/(name+'.json')) != m['metadata_hashes'][name]:
            raise ValueError('Source content changed')
    for records in (m['references'], m['diagnostic_references'], m['confirmation_manifest_receipts']):
        for path, digest in records.items():
            if sha(Path(path)) != digest:
                raise ValueError('Prior evidence changed: '+path)
    prior = Path(cfg['stage23_output'])
    if any(sha(prior/n) != digest for n, digest in m['candidate_hashes'].items()):
        raise ValueError('Diagnostic checkpoint weights changed')
    weights = Path(m['source_configuration']['weights']['directory'])
    if sha(weights/'snapshot.json') != m['weights_sha256'] or any(
            (weights/n).stat().st_size != v['size'] or (weights/n).stat().st_mtime_ns != v['mtime_ns']
            for n, v in m['weight_stats'].items()):
        raise ValueError('Official weights changed')
    return m
