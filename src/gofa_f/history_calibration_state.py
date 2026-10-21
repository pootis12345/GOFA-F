"""Stage27 inherits immutable Stage26 identities, never resamples confirmation."""
import os
from pathlib import Path
import shutil

from gofa_f.allnode_stage import read
from gofa_f.generative_data import sha
from gofa_f.graph_benchmark_data import TARGETS
from gofa_f.prefix_cache import atomic_json
from gofa_f.short_sampled_state import exited
from gofa_f.short_sampling_diagnostic import SETTINGS
from gofa_f.six_calibration_cpu import records
from gofa_f.six_calibration_state import ADAPTER, immutable
from gofa_f.target_calibration_state import code_hashes, digest


def validate_config(cfg):
    resource = dict(admission_policy='memory_cap_only', admission_samples=1, resource_check_seconds=10,
        memory_cap_gib=40, torch_cap_gib=38, activity_limit_seconds=None, wall_deadline=None,
        yield_on_foreign=False, leave_free_gib=0)
    if cfg['protocol'] != 'stage27-history-target-calibration-v1' or cfg['seed'] != 42 \
            or cfg['adapter_sha256'] != ADAPTER or cfg['targets'] != list(TARGETS) \
            or cfg['sampling'] != SETTINGS or cfg['resources'] != resource \
            or cfg['training'] != {'max_updates': 0} \
            or cfg['taus'] != [.01, .1, 1.] or cfg['bias_limit'] != .25 \
            or cfg['folds'] != [[32, 48], [48, 64]] \
            or cfg['data'] != dict(target_training_enabled=False, target_evaluation_enabled=True,
                target_label_calibration_enabled=True, strict_zero_shot=False):
        raise ValueError('Fixed Stage27 protocol changed')


def development_records(cfg):
    root = Path(cfg['stage26_output'])
    if sha(root/'manifest.json') != cfg['stage26_manifest_sha256'] \
            or sha(root/'report.json') != cfg['stage26_report_sha256'] \
            or sha(root/'pilot_gate.json') != cfg['stage26_pilot_gate_sha256']:
        raise ValueError('Stage26 development provenance changed')
    prior = read(root/'manifest.json')
    first, ha = records(root, prior, 'fit')
    last, hb = records(root, prior, 'pilot')
    return {n: {mode: first[n][mode]+last[n][mode] for mode in ('native', 'gofa')} for n in TARGETS}, \
        digest(dict(fit=ha, old_pilot_now_development=hb))


def prepare(cfg, output):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU preparation requires CUDA disabled')
    validate_config(cfg)
    root = Path(cfg['stage26_output'])
    exited(root)
    if root.resolve() == output.resolve() or read(root/'status.json')['stop_reason'] != 'failed_pilot_gate':
        raise ValueError('Separate output and stopped Stage26 required')
    prior = read(root/'manifest.json')
    if read(root/'confirm_gate.json') is not None or read(root/'test_replay.json') is not None \
            or list((root/'evaluations').glob('confirm-*')):
        raise ValueError('Confirmation must still be unseen and unexecuted')
    if cfg['stage21_output'] != prior['configuration']['stage21_output'] \
            or cfg['stage21_manifest_sha256'] != prior['configuration']['stage21_manifest_sha256'] \
            or cfg['stage21_report_sha256'] != prior['configuration']['stage21_report_sha256']:
        raise ValueError('Stage21 frozen test archive differs')
    _, inputs = development_records(cfg)
    code = code_hashes()
    if any(code.get(n) != h for n, h in prior['unchanged_tensor_files'].items()):
        raise ValueError('Accepted numerical core changed')
    proof = read(root/'cpu_preflight.json')
    if not proof['passed'] or proof['manifest_sha256'] != cfg['stage26_manifest_sha256']:
        raise ValueError('Inherited CPU proof must be bound')
    inherited = ('source_configuration', 'weights_sha256', 'weight_stats', 'alphabet', 'source_selection',
        'source_raw_hashes', 'probes', 'unchanged_tensor_files', 'data_directory', 'raw_hashes',
        'metadata_hashes', 'queries', 'coverage', 'test_queries', 'test_hashes')
    manifest = {k: prior[k] for k in inherited}
    manifest.update(configuration=cfg, code=code, development_inputs_sha256=inputs,
        inherited_cpu_proof_sha256=sha(root/'cpu_preflight.json'),
        development_queries=384, independent_confirmation_queries=384,
        confirmation_previously_evaluated=False,
        total_numeric_work=16*sum(q['horizon'] for rows in prior['queries'].values() for q in rows[64:]),
        optimizer_updates=0, graph_alpha=1., strict_zero_shot=False, neural_training_allowed=False)
    verify_data(manifest)
    output.mkdir(parents=True, exist_ok=True)
    immutable(output/'manifest.json', manifest)
    if not (output/'adapter.pt').exists():
        shutil.copyfile(root/'adapter.pt', output/'adapter.pt')
    if sha(output/'adapter.pt') != ADAPTER:
        raise ValueError('Frozen adapter changed')
    immutable(output/'gpu_identity.json', read(root/'gpu_identity.json'))
    if not (output/'status.json').exists():
        atomic_json(output/'status.json', dict(state='prepared', phase='prepared', optimizer_updates=0))
    return manifest


def verify_data(manifest):
    root = Path(manifest['data_directory'])
    for name in TARGETS:
        if sha(root/'targets'/(name+'.npz')) != manifest['raw_hashes'][name] \
                or sha(root/'targets'/(name+'.json')) != manifest['metadata_hashes'][name]:
            raise ValueError('Raw target data/metadata changed')


def own_manifest(cfg, output):
    validate_config(cfg)
    value = read(output/'manifest.json')
    if value is None or value['configuration'] != cfg or value['code'] != code_hashes():
        raise ValueError('Stage27 code/config/manifest changed')
    return value


def verify_inference(cfg, output):
    manifest = own_manifest(cfg, output)
    verify_data(manifest)
    root = Path(manifest['source_configuration']['weights']['directory'])
    if sha(output/'adapter.pt') != ADAPTER or sha(root/'snapshot.json') != manifest['weights_sha256'] \
            or any((root/n).stat().st_size != v['size'] or (root/n).stat().st_mtime_ns != v['mtime_ns']
                   for n, v in manifest['weight_stats'].items()):
        raise ValueError('Frozen model changed')
    return manifest
