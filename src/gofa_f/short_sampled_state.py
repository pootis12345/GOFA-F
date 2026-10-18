"""Exact trained-state import with a separate sampled selection/patience ledger."""
import copy
from contextlib import contextmanager
from pathlib import Path
import shutil

import torch

from gofa_f.allnode_stage import cpu_copy, read, rng_state, restore_rng
from gofa_f.generative_data import SOURCES, sha
from gofa_f.generative_stage import verify_weights
from gofa_f.idle_launcher import process_identity
from gofa_f.prefix_cache import atomic_json
from gofa_f.short_generative_stage import config_check
from gofa_f.short_migration import INFERENCE_FILES, assert_equal
from gofa_f.short_sampling_diagnostic import SETTINGS

TENSOR_FILES = INFERENCE_FILES+('short_training.py','allnode_stage.py')


def exited(root, source=False):
    for suffix in ('.launch.json','.bootstrap.launch.json'):
        identity = read(Path(str(root)+suffix),{})
        if identity.get('pid') and process_identity(identity['pid'])==str(identity['start_ticks']):
            raise ValueError('Prior controller is still alive: '+str(root))
    activity, launcher = read(root/'activity.json'),read(root/'launcher.json')
    expected = 'stopped_by_user' if source else 'complete'
    if activity.get('worker') is not None or launcher['state']!=expected:
        raise ValueError('Prior run has not safely exited: '+str(root))
    return activity


def fresh_ledger():
    return dict(best_score=None,best_update=None,early_reference=None,bad_validations=0,
                last_validated_update=0,validated_updates=[],protocol='sample8_median_only')


def atomic_torch(path,value):
    temporary = path.with_suffix('.tmp')
    torch.save(value,temporary)
    temporary.replace(path)


def migrated_payload(original,binding):
    state=copy.deepcopy(original)
    state['binding']=binding
    state['sampling_ledger']=fresh_ledger()
    state['legacy_greedy_progress']=copy.deepcopy(state['progress'])
    state['progress'].update(throughput_anchor_updates=state['progress']['measured_updates'],
        throughput_anchor_seconds=state['progress']['train_seconds'],pending_queries=state['state']['pending'],
        sampled_evaluation_durations={})
    return state


def prepare(cfg,output):
    if cfg['protocol']!='stage19-sampled-continuation-v1' or cfg['sampling']!=SETTINGS \
            or cfg['review_at_update']!=1500 or any(cfg['data'].values()):
        raise ValueError('Only the authorized sampled source continuation to 1500 is allowed')
    source = Path(cfg['source_output'])
    activity = exited(source,True)
    base = read(source/'manifest.json')
    config_check(base['configuration'])
    if base['code_revision']!=cfg['source_revision'] or sha(source/'last.pt')!=cfg['source_last_sha256'] \
            or sha(source/'best.pt')!=cfg['reference_1000_sha256'] \
            or sha(source/'candidates/update-001000.pt')!=cfg['reference_1000_sha256']:
        raise ValueError('Locked paused/reference checkpoint differs')
    if cfg['resources']!=base['configuration']['resources'] or cfg['training']!={'max_updates':4000}:
        raise ValueError('Resource or original update cap changed')
    code = {p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    if any(code[n]!=base['code'][n] for n in TENSOR_FILES):
        raise ValueError('Frozen, graph, loss or training tensor implementation changed')
    proof = read(source/'cpu_preflight.json')
    gpu = read(source/'mechanisms.json')
    if not proof['passed'] or proof['manifest_sha256']!=sha(source/'manifest.json') \
            or not proof['real_weights']['passed'] or not proof['continued_training_optimizations']['passed'] \
            or not gpu['passed'] or not gpu['training_optimizations']['passed']:
        raise ValueError('Unchanged tensor model lacks original full CPU/CUDA acceptance')
    official_hash = verify_weights(base['configuration'])
    if official_hash!=base['weights_sha256']:
        raise ValueError('Official backbone hash differs')
    data_root = Path(base['configuration']['data']['directory'])
    raw = {s:sha(data_root/'sources'/(s+'.npz')) for s in SOURCES}
    if any(raw[s]!=base['raw_hashes'][s] for s in SOURCES):
        raise ValueError('Source observations changed')
    pool = read(source/'pools/queries.json')
    if sha(source/'pools/queries.json')!=base['query_sha256'] or len(pool['validation'])!=2264 \
            or sha(source/'pools/metric_epsilons.json')!=base['epsilon_sha256']:
        raise ValueError('Full fixed source pool/epsilon binding changed')
    costs = []
    names={'stage19-source-diagnostic-v1':'diagnostic_manifest.json',
        'stage19-native-case-probe-v1':'case_manifest.json',
        'stage19-sampling-diagnostic-v1':'sampling_manifest.json',
        'stage19-stability-diagnostic-v1':'stability_manifest.json'}
    if {Path(p).name for p in cfg['diagnostic_cost_sources']}!=set(names) or len(cfg['diagnostic_cost_sources'])!=4:
        raise ValueError('Exactly four distinct completed diagnostic ledgers required')
    for name in cfg['diagnostic_cost_sources']:
        root = Path(name)
        ledger = exited(root)
        prior=read(root/names[root.name])
        if Path(prior['configuration']['source_output']).resolve()!=source.resolve() \
                or prior['configuration']['paused_last_sha256']!=cfg['source_last_sha256']:
            raise ValueError('Diagnostic cost has a different training lineage')
        costs.append(dict(output=str(root),activity=ledger,sha256=sha(root/'activity.json'),
                          manifest_sha256=sha(root/names[root.name])))
    for record in costs:
        for key in ('spent_seconds','waiting_seconds','resource_pause_count'):
            activity[key]=activity.get(key,0)+record['activity'].get(key,0)
        activity['peak_group_mib']=max(activity.get('peak_group_mib',0),record['activity'].get('peak_group_mib',0))
    state = torch.load(source/'last.pt',map_location='cpu',weights_only=False)
    if state['binding']!=sha(source/'manifest.json') or (state['state']['updates'],state['state']['pending'],
            state['state']['pending_tokens'])!=(1177,30,390) or not state['optimizer']['state'] \
            or not any(v is not None for v in state['gradients'].values()):
        raise ValueError('Unexpected paused optimizer/gradient/cursor state')
    weight_root=Path(base['configuration']['weights']['directory'])
    weight_stats={name:dict(size=(weight_root/name).stat().st_size,mtime_ns=(weight_root/name).stat().st_mtime_ns)
                  for name in read(weight_root/'snapshot.json')['files']}
    manifest = dict(protocol=cfg['protocol'],configuration=cfg,source_configuration=base['configuration'],
        code=code,source_manifest_sha256=sha(source/'manifest.json'),source_pool_sha256=base['query_sha256'],
        source_last_sha256=cfg['source_last_sha256'],reference_1000_sha256=cfg['reference_1000_sha256'],
        raw_hashes=raw,epsilon_sha256=base['epsilon_sha256'],weights_sha256=official_hash,
        cpu_proof_sha256=sha(source/'cpu_preflight.json'),gpu_proof_sha256=sha(source/'mechanisms.json'),
        unchanged_tensor_files={n:code[n] for n in TENSOR_FILES},diagnostic_costs=costs,
        weight_stats=weight_stats,dependencies=base['dependencies'],
        source_training_updates=1177,review_at_update=1500,target_evaluation_enabled=False)
    if read(output/'manifest.json',manifest)!=manifest:
        raise ValueError('Immutable sampled continuation manifest differs')
    atomic_json(output/'manifest.json',manifest)
    if (output/'migration.json').exists():
        return manifest
    if (output/'last.pt').exists():
        raise ValueError('Refuse to overwrite an unaudited destination checkpoint')
    (output/'pools').mkdir(exist_ok=True)
    atomic_json(output/'pools/queries.json',{k:pool[k] for k in ('train','validation')})
    shutil.copyfile(source/'pools/metric_epsilons.json',output/'pools/metric_epsilons.json')
    original = copy.deepcopy(state)
    state=migrated_payload(state,sha(output/'manifest.json'))
    atomic_torch(output/'last.pt',state)
    restored=torch.load(output/'last.pt',map_location='cpu',weights_only=False)
    for key in ('adapter','optimizer','gradients','rng','parameter_version','state'):
        assert_equal(original[key],restored[key])
    assert_equal(state,restored)
    shutil.copyfile(source/'candidates/update-001000.pt',output/'reference-001000.pt')
    atomic_json(output/'activity.json',activity)
    atomic_json(output/'gpu_identity.json',read(source/'gpu_identity.json'))
    atomic_json(output/'migration.json',dict(state_restore='exact',source=str(source),
        source_checkpoint_sha256=cfg['source_last_sha256'],target_initial_sha256=sha(output/'last.pt'),
        preserved=['adapter','optimizer','gradients','rng','parameter_version','state'],
        updates=1177,pending=30,pending_tokens=390,greedy_selection_preserved_in_training_state=True,
        sampling_ledger='new independent ledger; 1000 sampled anchor then 1500',
        cost_merge_once=costs,merged_activity=activity,old_processes_exited=True))
    return manifest


def verify(cfg,output):
    manifest=read(output/'manifest.json')
    source=Path(cfg['source_output'])
    exited(source,True)
    if manifest['configuration']!=cfg or sha(source/'last.pt')!=cfg['source_last_sha256'] \
            or sha(source/'best.pt')!=cfg['reference_1000_sha256'] \
            or sha(output/'reference-001000.pt')!=cfg['reference_1000_sha256'] \
            or manifest['code']!={p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}:
        raise ValueError('Source, code or sampled phase binding changed')
    pool=read(source/'pools/queries.json')
    if read(output/'pools/queries.json')!={k:pool[k] for k in ('train','validation')} \
            or sha(output/'pools/metric_epsilons.json')!=manifest['epsilon_sha256']:
        raise ValueError('Current source-only query/epsilon pool changed')
    root=Path(manifest['source_configuration']['data']['directory'])
    if any(sha(root/'sources'/(s+'.npz'))!=v for s,v in manifest['raw_hashes'].items()):
        raise ValueError('Current source observations changed')
    weights=Path(manifest['source_configuration']['weights']['directory'])
    if sha(weights/'snapshot.json')!=manifest['weights_sha256'] or any(
            (weights/n).stat().st_size!=s['size'] or (weights/n).stat().st_mtime_ns!=s['mtime_ns']
            for n,s in manifest['weight_stats'].items()):
        raise ValueError('Official weights changed since CPU content hash verification')
    return manifest


def save(path,model,optimizer,state,progress,ledger,binding):
    progress.tick(force=True)
    payload=dict(binding=binding,adapter=model.trainable_state(),optimizer=optimizer.state_dict(),
        gradients={n:cpu_copy(p.grad) for n,p in model.adapter.named_parameters()},state=copy.deepcopy(state),
        progress=copy.deepcopy(progress.state),rng=rng_state(),parameter_version=model.parameter_version,
        sampling_ledger=copy.deepcopy(ledger))
    atomic_torch(path,payload)


@contextmanager
def temporary_adapter(model,state=None):
    original,version,rng=model.trainable_state(),model.parameter_version,rng_state()
    gradients={n:cpu_copy(p.grad) for n,p in model.adapter.named_parameters()}
    try:
        if state is not None:
            model.load_trainable(state)
        yield
    finally:
        model.load_trainable(original)
        model.parameter_version=version
        restore_rng(rng)
        for n,p in model.adapter.named_parameters():
            assert_equal(cpu_copy(p.grad),gradients[n])
