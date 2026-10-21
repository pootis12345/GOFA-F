"""Immutable Stage25 source-only provenance and never-evaluated confirmation handoff."""
from pathlib import Path
import shutil

from gofa_f.allnode_stage import read
from gofa_f.generative_data import Series, sha
from gofa_f.graph_benchmark import ADAPTER_SHA, atomic
from gofa_f.robust_state import parent_replay_bindings, reference_files
from gofa_f.short_sampled_state import exited
from gofa_f.source_gate_state import TARGETS
from gofa_f.source_retraining_data import VIEWS
from gofa_f.weather_policy import subset


def validate(cfg):
    if cfg['protocol'] != 'stage25-weather-prefix-adaptation-v1' or cfg['seed'] != 42 \
            or cfg['adapter_sha256'] != ADAPTER_SHA or cfg['targets'] != list(TARGETS) \
            or cfg['training'] != dict(max_updates=100, effective_batch=64, microbatch=2,
                learning_rate=1e-5, warmup_updates=10, weight_decay=1e-4, gradient_clip=1,
                quick_validation_update=25, full_validation_updates=[50, 100]) \
            or cfg['sampling'] != dict(samples=8, top_k=100, temperature=1., top_p=1.) \
            or cfg['resources'] != dict(admission_policy='memory_cap_only', admission_samples=1,
                resource_check_seconds=10, memory_cap_gib=40, torch_cap_gib=38,
                activity_limit_seconds=None, wall_deadline=None, yield_on_foreign=False, leave_free_gib=0) \
            or cfg['data']['target_training_enabled'] or cfg['data']['target_evaluation_enabled'] != 'gated_only':
        raise ValueError('Fixed Stage25 policy changed')


def prepare(cfg, output):
    validate(cfg)
    output = Path(output)
    if (output/'manifest.json').exists():
        return verify(cfg, output)
    old_root, prior = Path(cfg['stage24_output']), Path(cfg['stage23_output'])
    for root in (old_root, prior, Path(cfg['stage21_output'])):
        exited(root)
    old, source = read(old_root/'manifest.json'), read(prior/'manifest.json')
    if read(old_root/'status.json')['stop_reason'] != 'failed_update100_source_gate' \
            or read(old_root/'status.json')['new_updates'] != 100 \
            or not read(old_root/'diagnostics/complete.json')['complete'] \
            or not read(old_root/'mechanisms.json')['passed'] \
            or sha(old_root/'candidate-000100.pt') != read(old_root/'update100_gate.json')['sha256'] \
            or sha(old_root/'parent.pt') != ADAPTER_SHA:
        raise ValueError('Completed Stage24 diagnostic-only candidate and parent required')
    if (old_root/'source_confirmation.json').exists() or (old_root/'final_selection.json').exists() \
            or any(old_root.glob('confirmation*-result.json')) \
            or any((old_root/'evaluations').glob('confirmation*')):
        raise ValueError('Sealed confirmation windows must never have been evaluated or selected')
    if len(old['pools']['confirmation']) != 192:
        raise ValueError('Exactly192 sealed confirmation queries required')
    for name in source['unchanged_tensor_files']:
        if sha(Path(__file__).parent/name) != source['code'][name]:
            raise ValueError('Accepted shared model/inference changed: '+name)
    sources = {n: Series(Path(cfg['data']['directory'])/'sources'/(n+'.npz')) for n in VIEWS}
    for name, data in sources.items():
        if sha(data.path) != old['source_hashes'][name] or sha(data.path.with_suffix('.json')) != old['metadata_hashes'][name]:
            raise ValueError('Audited source data changed')
    train = subset(old['pools']['train'])
    for name, queries in train.items():
        if any(q['origin']+q['horizon'] > int(.6*len(sources[name].values)) for q in queries):
            raise ValueError('Training labels cross the first60% boundary')
    diagnostic = []
    for name, count in zip(VIEWS, [4, 4, 16, 8]):
        diagnostic.extend([q for q in old['pools']['diagnostic'] if q['source'] == name][:count])
    if len(diagnostic) != 32:
        raise ValueError('Fixed diagnostic coverage unavailable')
    references = reference_files(prior, ['validation-native', 'validation-parent'], old['pools']['validation'])
    for i, q in enumerate(diagnostic):
        original_index = old['pools']['diagnostic'].index(q)
        p = old_root/'diagnostics'/f'parent-{original_index:03d}.json'
        if read(p)['query'] != q or read(p)['binding'] != sha(old_root/'manifest.json'):
            raise ValueError('Reusable parent diagnostic binding differs')
        references[str(p)] = sha(p)
        view = [x for x in old['pools']['validation'] if x['source'] == q['source']]
        index = view.index(q)
        for root, label in [(prior, 'validation-parent'), (old_root, 'validation-update-000100')]:
            path = root/'evaluations'/(label+'-'+q['source'])/'trajectories'/f'{index:06d}-00.json'
            if read(path)['query'] != q or read(path)['sample'] != 0:
                raise ValueError('Diagnostic saved prefix/query differs')
            references[str(path)] = sha(path)
    manifest = dict(protocol=cfg['protocol'], configuration=cfg,
        pools=dict(train=train, validation=old['pools']['validation'], quick=old['pools']['quick'],
                   confirmation=old['pools']['confirmation'], diagnostic=diagnostic,
                   epsilons=old['pools']['epsilons']),
        source_configuration=old['source_configuration'], weights_sha256=old['weights_sha256'],
        weight_stats=old['weight_stats'], alphabet=old['alphabet'], probes=old['probes'],
        source_hashes=old['source_hashes'], metadata_hashes=old['metadata_hashes'],
        **parent_replay_bindings(old), references=references,
        parent_stage23_manifest_sha256=sha(prior/'manifest.json'),
        parent_stage24_manifest_sha256=sha(old_root/'manifest.json'),
        target_manifest_sha256=sha(Path(cfg['stage21_output'])/'manifest.json'),
        diagnostic_checkpoint_sha256=sha(old_root/'candidate-000100.pt'),
        code={p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        parent_updates=3500, new_optimizer=True, exact_stage24_resume=False,
        prefix_mode='fixed_parent_greedy_no_random_sampling', train_queries=1024,
        primary_metric='observed_points_within_variable_then_equal_variables',
        confirmation_previously_evaluated=False, hourly_followup_enabled=False)
    output.mkdir(parents=True, exist_ok=True)
    atomic(output/'manifest.json', manifest)
    shutil.copyfile(old_root/'parent.pt', output/'parent.pt')
    atomic(output/'gpu_identity.json', read(old_root/'gpu_identity.json'))
    atomic(output/'status.json', dict(state='prepared', parent_updates=3500, new_updates=0, target_evaluations=0))
    return manifest


def verify(cfg, output):
    validate(cfg)
    output = Path(output)
    m = read(output/'manifest.json')
    if m['configuration'] != cfg or m['code'] != {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')} \
            or sha(output/'parent.pt') != ADAPTER_SHA:
        raise ValueError('Stage25 code/configuration/parent binding differs')
    for key, name in [('stage23_output', 'parent_stage23_manifest_sha256'),
                      ('stage24_output', 'parent_stage24_manifest_sha256'),
                      ('stage21_output', 'target_manifest_sha256')]:
        if sha(Path(cfg[key])/'manifest.json') != m[name]:
            raise ValueError('Parent manifest changed: '+key)
    if sha(Path(cfg['stage24_output'])/'candidate-000100.pt') != m['diagnostic_checkpoint_sha256']:
        raise ValueError('Diagnostic-only candidate changed')
    for name in VIEWS:
        root = Path(cfg['data']['directory'])/'sources'
        if sha(root/(name+'.npz')) != m['source_hashes'][name] or sha(root/(name+'.json')) != m['metadata_hashes'][name]:
            raise ValueError('Source observations changed')
    for path, digest in m['references'].items():
        if sha(Path(path)) != digest:
            raise ValueError('Reusable evidence changed: '+path)
    weights = Path(m['source_configuration']['weights']['directory'])
    if sha(weights/'snapshot.json') != m['weights_sha256'] or any(
            (weights/n).stat().st_size != v['size'] or (weights/n).stat().st_mtime_ns != v['mtime_ns']
            for n, v in m['weight_stats'].items()):
        raise ValueError('Official backbone changed')
    return m
