"""Source-only inductive DCRNN: one fixed training run, one frozen target pass."""
import argparse
import copy
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from gofa_f.allnode_stage import read
from gofa_f.autonomous_iteration import DEADLINE
from gofa_f.generative_data import Series, sha
from gofa_f.idle_launcher import process_identity
from gofa_f.prefix_cache import atomic_json
from gofa_f.six_calibration_state import immutable
from gofa_f.source_dcrnn_data import sample, stack_samples
from gofa_f.source_dcrnn_model import DCRNN, masked_query_loss
from gofa_f.source_retraining import configuration
from gofa_f.target_calibration import cpu_only
from gofa_f.target_calibration_state import code_hashes, digest

VIEWS = ('pems04_flow', 'pems04_speed', 'boston2017', 'penmanshiel2017')
CYCLE = (VIEWS[0], VIEWS[2], VIEWS[3], VIEWS[1], VIEWS[2], VIEWS[3])


def state(output, mode, phase, **extra):
    atomic_json(output/'status.json', dict(state=mode, phase=phase, stage=34,
        target_training_enabled=False, strict_zero_shot_definition='no target parameter update',
        independent_blind_test=False, gpu_activity_seconds=0., updated_at=time.time(), **extra))


def save_torch(path, value):
    temporary = path.with_suffix('.part')
    torch.save(value, temporary)
    temporary.replace(path)


def prepare(cfg, output):
    cpu_only()
    if cfg['protocol'] != 'stage34-source-dcrnn-v1' or cfg['deadline_unix'] != DEADLINE \
            or cfg['training']['max_updates'] != 10000 or cfg['seed'] != 42 \
            or cfg['data']['target_training_enabled'] or cfg['data']['target_calibration_enabled']:
        raise ValueError('Fixed source-only comparison changed')
    source = Path(cfg['source_output'])
    target = Path(cfg['target_output'])
    gate = read(Path(cfg['supervised_result_output'])/'test_replay.json')
    if not gate or not gate['all_six_strictly_improved']:
        raise ValueError('Conditional comparison requires prior six-target success')
    if sha(source/'manifest.json') != cfg['source_manifest_sha256'] \
            or sha(target/'manifest.json') != cfg['target_manifest_sha256']:
        raise ValueError('Source/target query archive changed')
    sm, tm = read(source/'manifest.json'), read(target/'manifest.json')
    if list(sm['pools']['train']) != list(VIEWS) \
            or any(len(sm['pools']['train'][v]) != 8192 for v in VIEWS) \
            or len(sm['pools']['validation']) != 384 \
            or sum(map(len, tm['queries']['targets'].values())) != 768:
        raise ValueError('Fixed source pools/target query budget changed')
    baseline = read(source/'validation-native-result.json')
    if not baseline['complete'] or baseline['queries'] != 384 or len(baseline['errors']) != 384 \
            or not baseline['binding'].startswith(cfg['source_manifest_sha256']+':sample8:validation-native:'):
        raise ValueError('Complete bound source-native baseline required')
    m = dict(configuration=cfg, code=code_hashes(), source_configuration=sm['configuration'],
        source_hashes=sm['source_hashes'], metadata_hashes=sm['metadata_hashes'],
        pools=sm['pools'], target_queries=tm['queries']['targets'],
        target_data_directory=tm['data_directory'] if 'data_directory' in tm else tm['configuration']['data']['directory'],
        target_hashes=tm['raw_hashes'], target_metadata_hashes=tm['metadata_hashes'],
        baseline_errors=baseline['errors'], baseline_sha256=sha(source/'validation-native-result.json'),
        inherited_gpu_seconds=read(Path(cfg['supervised_result_output'])/'activity.json')['spent_seconds'],
        target_labels_used_for_training=False, goal_result_read_only_for_conditional_authorization=True,
        reference_revision='602afd9d767d3aa1c9b3eac51710d6aeee12c227')
    output.mkdir(parents=True, exist_ok=True)
    immutable(output/'manifest.json', m)
    if not (output/'status.json').exists():
        state(output, 'prepared', 'source_cache')
    return m


def verify(cfg, output):
    m = read(output/'manifest.json')
    if not m or m['configuration'] != cfg or m['code'] != code_hashes():
        raise ValueError('Comparison manifest binding changed')
    directory = Path(m['source_configuration']['data']['directory'])/'sources'
    for name, expected in m['source_hashes'].items():
        if sha(directory/(name+'.npz')) != expected or sha(directory/(name+'.json')) != m['metadata_hashes'][name]:
            raise ValueError('Audited source changed')
    return m


def caches(cfg, output, m, charge=lambda: None):
    directory = Path(m['source_configuration']['data']['directory'])/'sources'
    result = {}
    started = time.monotonic()
    completed, total = 0, 32768+384
    folder = output/'cache'
    folder.mkdir(exist_ok=True)
    for name in VIEWS:
        data = Series(directory/(name+'.npz'))
        for split in ('train', 'validation'):
            queries = m['pools']['train'][name] if split == 'train' else [q for q in m['pools']['validation'] if q['source'] == name]
            path = folder/(name+'-'+split+'.npz')
            binding = digest(dict(manifest=sha(output/'manifest.json'), queries=queries, split=split))
            proof = read(path.with_suffix('.json'))
            if path.exists():
                if not proof or proof['binding'] != binding or proof['sha256'] != sha(path):
                    raise ValueError('Source feature cache changed')
            else:
                rows = []
                for index, query in enumerate(queries):
                    if time.time() >= DEADLINE-180:
                        raise TimeoutError('No time for new source fitting')
                    rows.append(sample(data, query, labels=True, split=split))
                    charge()
                    completed += 1
                    if completed % 64 == 0:
                        elapsed = time.monotonic()-started
                        atomic_json(output/'progress.json', dict(phase='source_cache', prepared_queries=completed,
                            total_queries=total, dataset=name, split=split, eta_seconds=(total-completed)*elapsed/completed,
                            gpu_activity_seconds=0., target_forecasts=0))
                packed = stack_samples(rows)
                temp = path.with_suffix('.part')
                with temp.open('wb') as f:
                    np.savez_compressed(f, **packed)
                temp.replace(path)
                immutable(path.with_suffix('.json'), dict(binding=binding, sha256=sha(path), queries=len(queries),
                    source_only=True, split=split, all_sampled_nodes_supervised=True))
            with np.load(path, allow_pickle=False) as loaded:
                result[name, split] = {k: loaded[k] for k in loaded.files}
    immutable(output/'cache_lock.json', dict(manifest=sha(output/'manifest.json'),
        files={p.name: sha(p) for p in sorted(folder.glob('*.npz'))}, target_cache_built=False))
    return result


def batch(arrays, indices):
    return {k: torch.from_numpy(v[indices]) for k, v in arrays.items()}


@torch.no_grad()
def forecast(model, arrays, size=64):
    model.eval()
    predictions = []
    for start in range(0, len(arrays['x']), size):
        b = batch(arrays, slice(start, start+size))
        horizon = arrays['y'].shape[1] if 'y' in arrays else int(arrays['horizon'][0])
        output = model(b['x'], b['supports'], b['present'], horizon)
        values = (output[:, :, 0].double()+.5)*b['scale'][:, None]+b['low'][:, None]
        values = torch.where(b['empty_target'][:, None], torch.zeros_like(values), values)
        if not values.isfinite().all():
            raise ValueError('Nonfinite continuous GNN forecast')
        predictions.extend(values.double().numpy())
    return predictions


def selection_score(errors, baseline, queries, epsilons):
    grouped = {}
    for e, b, q in zip(errors, baseline, queries):
        grouped.setdefault((q['source'], q['variable']), []).append((float(e), float(b)))
    logs = {v: [] for v in VIEWS}
    for (view, variable), values in grouped.items():
        a = np.asarray(values, np.float64).mean(0)
        floor = epsilons[view][variable]
        logs[view].append(np.log(max(a[0], floor)/max(a[1], floor)))
    ratios = {v: float(np.exp(np.mean(logs[v]))) for v in VIEWS}
    traffic = np.sqrt(ratios[VIEWS[0]]*ratios[VIEWS[1]])
    score = float(np.exp(np.mean(np.log([traffic, ratios[VIEWS[2]], ratios[VIEWS[3]]]))))
    return dict(score=score, views=ratios, domains=dict(traffic=float(traffic), meteorology=ratios[VIEWS[2]], wind=ratios[VIEWS[3]]))


@torch.no_grad()
def validate(model, packed, m):
    errors = {}
    for name in VIEWS:
        a = packed[name, 'validation']
        predictions = forecast(model, a)
        true = a['raw_target']
        queries = [q for q in m['pools']['validation'] if q['source'] == name]
        for q, p, y, valid in zip(queries, predictions, true, a['observed'][:, :, 0]):
            errors[digest(q)] = float(np.abs(p-y)[valid].mean())
    ordered = [errors[digest(q)] for q in m['pools']['validation']]
    value = selection_score(ordered, m['baseline_errors'], m['pools']['validation'], m['pools']['epsilons'])
    value['query_errors'] = ordered
    return value


def train(cfg, output, m, packed, charge=lambda: None):
    model = DCRNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-3)
    rng = torch.Generator().manual_seed(42)
    order_rng = np.random.default_rng(42)
    progress = dict(updates=0, best_score=None, best_update=None, early_reference=None,
        bad_validations=0, last_validated_update=0, history=[], epochs={v: 0 for v in VIEWS},
        cursors={v: 0 for v in VIEWS}, orders={v: order_rng.permutation(8192).tolist() for v in VIEWS},
        cpu_seconds=0.)
    binding = sha(output/'manifest.json')
    if (output/'last.pt').exists():
        saved = torch.load(output/'last.pt', map_location='cpu', weights_only=False)
        if saved['binding'] != binding:
            raise ValueError('Training resume binding changed')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        progress = saved['progress']
        rng.set_state(saved['sampling_rng'])
        torch.set_rng_state(saved['torch_rng'])
        order_rng.bit_generator.state = saved['order_rng']
    stopped = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    anchor = time.monotonic()
    charged = progress['cpu_seconds']
    def save():
        progress['cpu_seconds'] = charged+time.monotonic()-anchor
        save_torch(output/'last.pt', dict(binding=binding, model=model.state_dict(),
            optimizer=optimizer.state_dict(), progress=copy.deepcopy(progress), sampling_rng=rng.get_state(),
            torch_rng=torch.get_rng_state(), order_rng=copy.deepcopy(order_rng.bit_generator.state)))
    save()
    reason = 'max_updates'
    while progress['updates'] < 10000:
        if stopped[0]:
            save()
            state(output, 'paused', 'source_train', updates=progress['updates'])
            return None
        if time.time() >= DEADLINE-180:
            reason = 'fixed_window_cutoff_not_convergence'
            break
        model.train()
        view = CYCLE[progress['updates'] % len(CYCLE)]
        cursor = progress['cursors'][view]
        indices = progress['orders'][view][cursor:cursor+64]
        progress['cursors'][view] += 64
        if progress['cursors'][view] == 8192:
            progress['epochs'][view] += 1
            progress['orders'][view] = order_rng.permutation(8192).tolist()
            progress['cursors'][view] = 0
        b = batch(packed[view, 'train'], indices)
        probability = 2000/(2000+np.exp(progress['updates']/2000))
        optimizer.zero_grad(set_to_none=True)
        prediction = model(b['x'], b['supports'], b['present'], b['y'].shape[1],
                           teacher=b['teacher'], probability=probability, rng=rng)
        loss = masked_query_loss(prediction, b['y'], b['observed'])
        if not loss.isfinite():
            raise ValueError('Nonfinite source loss; no target evaluation')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        if not torch.isfinite(norm):
            raise ValueError('Nonfinite source gradient')
        optimizer.step()
        charge()
        progress['updates'] += 1
        progress['last_loss'] = float(loss)
        update = progress['updates']
        if update % 250 == 0:
            result = validate(model, packed, m)
            result['update'] = update
            progress['history'].append(result)
            score = result['score']
            if progress['best_score'] is None or score < progress['best_score']:
                progress['best_score'], progress['best_update'] = score, update
                save_torch(output/'best.pt', dict(binding=binding, model=model.state_dict(), update=update, score=score))
            if progress['early_reference'] is None or score <= .999*progress['early_reference']:
                progress['early_reference'], progress['bad_validations'] = score, 0
            else:
                progress['bad_validations'] += 1
            progress['last_validated_update'] = update
            immutable(output/f'validation-{update:06}.json', result)
            print('Stage34 source validation', update, 'S', score, 'best', progress['best_score'], flush=True)
        save()
        if update % 10 == 0:
            speed = progress['cpu_seconds']/max(update, 1)
            atomic_json(output/'progress.json', dict(phase='source_train', updates=update, maximum_updates=10000,
                epochs=progress['epochs'], source_cursors=progress['cursors'], loss=progress['last_loss'],
                best_source_score=progress['best_score'], best_update=progress['best_update'],
                eta_seconds=(10000-update)*speed+768*speed/64,
                eta_note='includes approximate source validation/target cost; early stop/cutoff may shorten',
                gpu_activity_seconds=0., cpu_seconds=progress['cpu_seconds'], target_forecasts=0))
            state(output, 'running', 'source_train', updates=update, best_update=progress['best_update'])
        if update >= 2000 and progress['bad_validations'] >= 8:
            reason = 'source_early_stop'
            break
    save()
    if progress['best_update'] is None:
        state(output, 'incomplete', 'source_train', stop_reason='no_complete_source_validation')
        return None
    immutable(output/'source_selection.json', dict(binding=binding, checkpoint_sha256=sha(output/'best.pt'),
        updates=progress['updates'], selected_update=progress['best_update'], score=progress['best_score'],
        stop_reason=reason, target_labels_used=False, parameters=sum(p.numel() for p in model.parameters()),
        cpu_seconds=progress['cpu_seconds'], gpu_seconds=0., converged_claimed=False))
    return model


def numeric_summary(rows):
    from gofa_f.short_forecast import summarize
    entries = []
    for row in rows:
        n = row['query']['horizon']
        entries.append(dict(row, format_complete=False, inverse_valid=[True]*n, boundary=[False]*n,
            true_outside_range=[False]*n, empty_history=False, generated_token_count=0, forward_calls=0))
    result = summarize(entries)
    result.pop('formatting', None)
    result['forecast_kind'] = 'continuous DCRNN, no language generation coverage'
    return result


@torch.no_grad()
def target_evaluation(cfg, output, m, charge=lambda: None):
    lock = read(output/'source_selection.json')
    if not lock or lock['checkpoint_sha256'] != sha(output/'best.pt') or lock['target_labels_used']:
        raise ValueError('Unique source-selected checkpoint must be locked before target reads')
    model = DCRNN()
    model.load_state_dict(torch.load(output/'best.pt', map_location='cpu', weights_only=False)['model'])
    model.eval()
    final = {}
    directory = Path(m['target_data_directory'])/'targets'
    comparison = read(Path(cfg['supervised_result_output'])/'test_replay.json')
    for name, queries in m['target_queries'].items():
        data = Series(directory/(name+'.npz'))
        if sha(data.path) != m['target_hashes'][name] or sha(data.path.with_suffix('.json')) != m['target_metadata_hashes'][name]:
            raise ValueError('Frozen target data changed')
        folder = output/'targets'/name
        folder.mkdir(parents=True, exist_ok=True)
        records = []
        for index, q in enumerate(queries):
            charge()
            if time.time() >= DEADLINE:
                state(output, 'incomplete', 'target_evaluation', dataset=name, completed_queries=index)
                return
            path = folder/f'{index:06d}.json'
            old = read(path)
            binding = digest(dict(manifest=sha(output/'manifest.json'), checkpoint=lock['checkpoint_sha256'], query=q))
            if old:
                valid = np.asarray(old['observed'], bool)
                values = np.asarray(old['prediction'], np.float64)
                if old['binding'] != binding or old['query'] != q \
                        or values.shape != (q['horizon'],) or not np.isfinite(values).all() \
                        or float(np.abs(values-np.asarray(old['target']))[valid].mean()) != old['mae']:
                    raise ValueError('Target forecast binding changed')
                records.append(old)
                continue
            own = stack_samples([sample(data, q, labels=False)])
            own['horizon'] = np.array([q['horizon']], np.int64)
            prediction = forecast(model, own)[0]
            origin, horizon, variable = q['origin'], q['horizon'], q['variable']
            y = data.values[origin:origin+horizon, variable].astype(float)
            valid = data.mask[origin:origin+horizon, variable].astype(bool)&np.isfinite(y)
            row = dict(binding=binding, query=q, prediction=prediction.tolist(), target=y.tolist(),
                observed=valid.tolist(), mae=float(np.abs(prediction-y)[valid].mean()),
                source_checkpoint_sha256=lock['checkpoint_sha256'], target_parameter_updates=0)
            immutable(path, row)
            records.append(row)
            atomic_json(output/'progress.json', dict(phase='target_evaluation', dataset=name,
                completed_queries=index+1, total_queries=len(queries), gpu_activity_seconds=0., eta_seconds=None))
        metrics = numeric_summary(records)
        refs = comparison['datasets'][name]['metrics']
        paired = {}
        own_variables = metrics['variables']
        for reference in ('native', 'gofa', 'calibrated'):
            variables = list(own_variables)
            delta = np.array([own_variables[v]['mae']-refs[reference]['variables'][v]['mae'] for v in variables])
            rng = np.random.default_rng(4234)
            draws = delta[rng.integers(len(delta), size=(2000, len(delta)))].mean(1)
            paired[reference] = dict(gnn_minus_reference_mae=float(delta.mean()), clusters=len(delta),
                draws=2000, seed=4234, mae_difference_95pct=np.quantile(draws, [.025, .975]).tolist())
        final[name] = dict(dcrnn=metrics, references={k:refs[k]['macro'] for k in ('native','gofa','calibrated')},
            paired_variable_cluster_bootstrap=paired,
            supervised_hybrid_is_different_training_and_input_regime=True)
        immutable(output/(name+'-result.json'), final[name])
    immutable(output/'target_results.json', dict(complete=True, datasets=final,
        source_checkpoint_sha256=lock['checkpoint_sha256'], no_target_training_or_calibration=True,
        independent_blind_test=False, desired_ordering_enforced=False, gpu_activity_seconds=0.))
    state(output, 'complete', 'target_evaluation', source_selection=lock, target_datasets=len(final))
    atomic_json(output/'progress.json', dict(state='complete', phase='target_evaluation', eta_seconds=None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--progress', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cpu_only()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    cfg = configuration(args.config)
    if args.progress:
        print(read(args.output/'status.json'), read(args.output/'progress.json'))
        return
    if args.prepare:
        return prepare(cfg, args.output)
    m = verify(cfg, args.output)
    proof = read(args.output/'cpu_preflight.json')
    if not proof or not proof['passed'] or proof['manifest_sha256'] != sha(args.output/'manifest.json'):
        raise ValueError('Bound full CPU acceptance required')
    if read(args.output/'status.json', {}).get('state') == 'complete':
        raise ValueError('Terminal baseline remains sealed')
    activity = read(args.output/'activity.json', dict(cpu_active_wall_seconds=0., gpu_seconds=0.,
        inherited_gpu_seconds=m['inherited_gpu_seconds'], cumulative_gpu_seconds=m['inherited_gpu_seconds']))
    old = activity.get('worker')
    if old and process_identity(old['pid']) == str(old['start_ticks']):
        raise ValueError('Another verified CPU comparison worker is alive')
    if old:
        activity['cpu_active_wall_seconds'] += min(10., max(0., time.monotonic()-old['monotonic']))
    pid, tick = os.getpid(), process_identity(os.getpid())
    anchor = [time.monotonic()]
    def charge(force=False):
        if torch.cuda.is_initialized():
            raise RuntimeError('CPU-only comparison unexpectedly initialized CUDA')
        now = time.monotonic()
        if not force and now-anchor[0] < 10:
            return
        activity['cpu_active_wall_seconds'] += now-anchor[0]
        anchor[0] = now
        activity.update(worker=dict(pid=pid, start_ticks=tick, monotonic=now), gpu_seconds=0.,
                        cuda_initialized=torch.cuda.is_initialized())
        atomic_json(args.output/'activity.json', activity)
    charge(True)
    try:
        if not (args.output/'source_selection.json').exists():
            packed = caches(cfg, args.output, m, charge)
            if train(cfg, args.output, m, packed, charge) is None:
                return
            del packed
        target_evaluation(cfg, args.output, m, charge)
    finally:
        charge(True)
        activity['worker'] = None
        atomic_json(args.output/'activity.json', activity)


if __name__ == '__main__':
    main()
