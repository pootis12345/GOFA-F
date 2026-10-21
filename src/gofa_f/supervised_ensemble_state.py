"""Stage29 explicit target supervision and immutable train/develop/confirm isolation."""
import copy
from importlib.metadata import version
import os
from pathlib import Path
import shutil

from gofa_f.allnode_stage import read
from gofa_f.generative_data import Series, sha
from gofa_f.graph_benchmark_data import TARGETS
from gofa_f.history_calibration_state import development_records, verify_data
from gofa_f.prefix_cache import atomic_json
from gofa_f.short_sampled_state import exited
from gofa_f.short_sampling_diagnostic import SETTINGS
from gofa_f.six_calibration_state import ADAPTER, immutable
from gofa_f.state_residual_state import development_pair
from gofa_f.supervised_ensemble_data import CADENCE_SECONDS, feature_names, features, training_queries
from gofa_f.target_calibration_state import code_hashes, digest

MODEL_SETTINGS = dict(loss='absolute_error', max_iter=150, learning_rate=.05, max_leaf_nodes=15,
    max_depth=4, min_samples_leaf=40, l2_regularization=1., max_bins=63, random_state=42,
    early_stopping=False, categorical_features=None)
DATA_FLAGS = dict(target_training_enabled=True, neural_target_training_enabled=False,
    target_evaluation_enabled=True, target_label_calibration_enabled=True, strict_zero_shot=False)


def validate_config(cfg):
    resources = dict(admission_policy='memory_cap_only', admission_samples=1, resource_check_seconds=10,
        memory_cap_gib=40, torch_cap_gib=38, activity_limit_seconds=None, wall_deadline=None,
        yield_on_foreign=False, leave_free_gib=0)
    if cfg['protocol'] != 'stage29-supervised-target-ensemble-v1' or cfg['seed'] != 42 \
            or cfg['adapter_sha256'] != ADAPTER or cfg['targets'] != list(TARGETS) \
            or cfg['sampling'] != SETTINGS or cfg['resources'] != resources or cfg['data'] != DATA_FLAGS \
            or cfg['training'] != dict(max_updates=0, cpu_target_training=True, models=6, queries_per_dataset=4096) \
            or cfg['cpu_model'] != MODEL_SETTINGS or cfg['sklearn_version'] != '1.5.2' \
            or cfg['development_minimum_improvement'] != .005 or cfg['skip_zero_graph_weight'] is not True:
        raise ValueError('Fixed Stage29 target-supervised protocol changed')


def status(output, state, phase, **fields):
    import time
    atomic_json(output/'status.json', dict(state=state, phase=phase, optimizer_updates=0,
        neural_training_enabled=False, cpu_target_training_enabled=True, strict_zero_shot=False,
        updated_at=time.time(), **fields))


def original_development(cfg):
    parent = Path(cfg['stage28_output'])
    prior = read(parent/'manifest.json')
    if sha(parent/'manifest.json') != cfg['stage28_manifest_sha256'] \
            or sha(parent/'cpu_gate.json') != cfg['stage28_cpu_gate_sha256'] \
            or sha(parent/'development.json') != cfg['stage28_development_sha256']:
        raise ValueError('Stage28 evidence changed')
    old, first = development_records(cfg)
    wind, second = development_pair(cfg, parent, prior)
    old['sdwpf'] = wind
    return old, digest(dict(stage26=first, sdwpf_stage28=second))


def prepare(cfg, output):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('OMP_NUM_THREADS') != '2' \
            or os.environ.get('MKL_NUM_THREADS') != '2':
        raise RuntimeError('Preparation requires CUDA disabled and OMP/MKL2')
    validate_config(cfg)
    if version('scikit-learn') != cfg['sklearn_version']:
        raise ValueError('Pinned installed scikit-learn version changed; do not install replacements')
    root = Path(cfg['stage28_output'])
    for name in ('stage21_output', 'stage26_output', 'stage27_output', 'stage28_output'):
        parent = Path(cfg[name])
        if parent.resolve() == output.resolve():
            raise ValueError('Separate Stage29 output required')
        exited(parent)
    if read(root/'status.json')['stop_reason'] != 'failed_sdwpf_development_gate' \
            or (root/'confirm_gate.json').exists() or (root/'test_replay.json').exists():
        raise ValueError('Stopped Stage28 without confirmation/test required')
    prior = read(root/'manifest.json')
    for key in ('stage26_output', 'stage26_manifest_sha256', 'stage26_report_sha256',
                'stage26_pilot_gate_sha256', 'stage27_output', 'stage27_manifest_sha256',
                'stage21_output', 'stage21_manifest_sha256', 'stage21_report_sha256'):
        if cfg[key] != prior['configuration'][key]:
            raise ValueError('Inherited archive changed: '+key)
    pairs, inputs = original_development(cfg)
    code = code_hashes()
    if any(code.get(n) != h for n, h in prior['unchanged_tensor_files'].items()):
        raise ValueError('Accepted frozen LLM numerical core changed')
    m = {k: copy.deepcopy(prior[k]) for k in ('source_configuration', 'weights_sha256', 'weight_stats',
        'alphabet', 'source_selection', 'source_raw_hashes', 'probes', 'unchanged_tensor_files',
        'data_directory', 'raw_hashes', 'metadata_hashes', 'queries', 'coverage', 'test_queries', 'test_hashes')}
    verify_data(m)
    queries, coverage = {}, {}
    output.mkdir(parents=True, exist_ok=True)
    for name, (_, _, length, horizon) in TARGETS.items():
        data = Series(Path(m['data_directory'])/'targets'/(name+'.npz'))
        def notify(done, total):
            atomic_json(output/'progress.json', dict(state='cpu_preparation', dataset=name,
                preparation_variables_done=done, preparation_variables_total=total, overall_eta_seconds=None))
        queries[name], coverage[name] = training_queries(data, name, length, horizon, notify=notify)
        # Real shape/cadence checks at both ends of each variable's fixed pool.
        for v in {q['variable'] for q in queries[name]}:
            own = [q for q in queries[name] if q['variable'] == v]
            for q in (own[0], own[-1]):
                features(data, q)
    inherited = read(root/'cpu_preflight.json')
    if not inherited['passed'] or inherited['manifest_sha256'] != cfg['stage28_manifest_sha256']:
        raise ValueError('Inherited bound frozen-core CPU proof missing')
    rates = {}
    for mode in ('native', 'gofa'):
        result = read(root/f'development-sdwpf-{mode}-result.json')
        if not result['complete'] or result['queries'] != 64:
            raise ValueError('Measured inference-rate reference is incomplete')
        rates[mode] = result['active_seconds']/result['numeric_slots_in_eight_paths']
    initial_estimate = 8*sum(q['horizon'] for rows in m['queries'].values() for q in rows[64:])*sum(rates.values())
    m.update(configuration=cfg, code=code, training_queries=queries, training_coverage=coverage,
        development_queries={n: [r['query'] for r in pair['native']] for n, pair in pairs.items()},
        development_inputs_sha256=inputs, feature_names={n: feature_names(v[2]) for n, v in TARGETS.items()},
        cadence_seconds=CADENCE_SECONDS, inherited_cpu_proof_sha256=sha(root/'cpu_preflight.json'),
        target_supervised=True, neural_training_allowed=False, optimizer_updates=0, cpu_model_count=6,
        graph_alpha=1., strict_zero_shot=False, confirmation_previously_partially_generated=True,
        prior_inference_seconds_per_numeric_slot=rates, initial_confirmation_estimate_seconds=initial_estimate,
        total_numeric_work=16*sum(q['horizon'] for rows in m['queries'].values() for q in rows[64:]))
    immutable(output/'manifest.json', m)
    if not (output/'adapter.pt').exists():
        shutil.copyfile(root/'adapter.pt', output/'adapter.pt')
    if sha(output/'adapter.pt') != ADAPTER:
        raise ValueError('Frozen adapter changed')
    immutable(output/'gpu_identity.json', read(root/'gpu_identity.json'))
    if not (output/'status.json').exists():
        status(output, 'prepared', 'cpu_preparation')
    if not (output/'activity.json').exists():
        atomic_json(output/'activity.json', dict(spent_seconds=0., worker=None, waiting_seconds=0.,
            resource_pause_count=0, peak_group_mib=0))
    return m


def own_manifest(cfg, output):
    validate_config(cfg)
    m = read(output/'manifest.json')
    if not m or m['configuration'] != cfg or m['code'] != code_hashes():
        raise ValueError('Stage29 code/config/manifest binding changed')
    return m


def verify_inference(cfg, output):
    m = own_manifest(cfg, output)
    verify_data(m)
    root = Path(m['source_configuration']['weights']['directory'])
    if sha(output/'adapter.pt') != ADAPTER or sha(root/'snapshot.json') != m['weights_sha256'] \
            or any((root/n).stat().st_size != v['size'] or (root/n).stat().st_mtime_ns != v['mtime_ns']
                   for n, v in m['weight_stats'].items()):
        raise ValueError('Frozen LLM weights changed')
    return m


def inference_binding(m, binding, name, mode):
    weight = m['weights_sha256'] if mode == 'native' else ADAPTER
    return f'{binding}:sample8:confirm-{name}-{mode}:{weight}:'+m['alphabet']['alphabet_sha256']
