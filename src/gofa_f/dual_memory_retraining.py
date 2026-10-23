"""Joint source-only Memory fitting from sealed temporal initializations.

Reuses unchanged Native source paths and independently trained official UFG.
No target path, target score, or confirmation label is accepted by this runner.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import shutil
import signal
import time

import numpy as np
import torch
import yaml

from gofa_f.dual_memory_data import GROUPS, SHAPES
from gofa_f.dual_memory_engines import MemoryEngine, model_digest
from gofa_f.dual_memory_fitting import fit, evaluate, read
from gofa_f.dual_memory_policy import DOMAINS, digest, rng_state, restore_rng
from gofa_f.dual_memory_store import SourceStore, HistoryCache
from gofa_f.dual_memory_training import development_rows, ratios
from gofa_f.dual_memory_warmup import transfer, TemporalWarmup
from gofa_f.generative_data import sha, Query
from gofa_f.prefix_cache import atomic_json

PROTOCOL = 'stage60-dual-memory-source-v1'
FILES = ('dual_memory_retraining.py', 'dual_memory_warmup.py', 'dual_memory_training.py',
         'dual_memory_engines.py', 'dual_memory_fitting.py', 'dual_memory_policy.py',
         'dual_memory_store.py', 'dual_memory_fusion.py', 'short_launcher.py')


def configuration(path):
    cfg = yaml.safe_load(Path(path).read_text().replace('${GOFA_WORKSPACE}', os.environ['GOFA_WORKSPACE']))
    if cfg['protocol'] != PROTOCOL or cfg['training']['max_updates'] != 3000 or any(cfg['data'].values()):
        raise ValueError('Fixed source-only joint training required')
    if cfg['resources'] != dict(admission_policy='memory_cap_only', admission_samples=1,
        resource_check_seconds=10, memory_cap_gib=40, torch_cap_gib=38,
        activity_limit_seconds=None, wall_deadline=None, yield_on_foreign=False, leave_free_gib=0):
        raise ValueError('Resource policy changed')
    return cfg


def initialization_specs(warmup, manifest):
    warmup = Path(warmup)
    if read(warmup/'status.json')['state'] != 'complete' or read(warmup/'activity.json')['worker'] is not None:
        raise ValueError('All CPU source fits must complete and exit before joint preparation')
    from gofa_f.idle_launcher import process_identity
    launch=read(Path(str(warmup)+'.launch.json'),{})
    if launch and process_identity(launch['pid'])==launch['start_ticks']:
        raise ValueError('Completed CPU worker has not exited yet')
    seal = read(warmup/'selection_seal.json')
    if seal['binding'] != sha(warmup/'manifest.json') or seal['target_labels_used'] or seal['source_confirmation_labels_used']:
        raise ValueError('CPU initialization seal changed or used forbidden labels')
    if set(seal['selections']) != {'all', *('hold-'+g for g in DOMAINS)}:
        raise ValueError('All four source initialization fits required')
    result = {}
    for name, entry in seal['selections'].items():
        folder = warmup/name
        selection = read(folder/'selection.json')
        expected = None if name == 'all' else name[5:]
        binding = digest([sha(warmup/'manifest.json'), 'temporal', name])
        if selection['binding'] != binding or selection['excluded_group'] != expected or selection['target_labels_used'] or selection['held_source_used_for_checkpoint_selection']:
            raise ValueError('A held source entered the wrong initialization')
        if sha(folder/'selection.json') != entry['selection_sha256'] or sha(folder/'best.pt') != entry['checkpoint_sha256']:
            raise ValueError('A sealed initialization file changed')
        value = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
        model = TemporalWarmup(); model.load_state_dict(value['model'], strict=True)
        if value['binding'] != binding or model_digest(model) != value['model_sha256'] or value['model_sha256'] != entry['model_sha256']:
            raise ValueError('Initialization tensor layout/hash changed')
        result[name] = dict(path=str(folder/'best.pt'), binding=binding,
            checkpoint_sha256=entry['checkpoint_sha256'], model_sha256=entry['model_sha256'],
            excluded_group=expected, best_update=entry['best_update'])
    return result


def initialization(manifest, name):
    entry = manifest['temporal_initializations'][name]
    if sha(Path(entry['path'])) != entry['checkpoint_sha256']:
        raise ValueError('Sealed temporal model changed')
    value = torch.load(entry['path'], map_location='cpu', weights_only=False)
    if value['binding'] != entry['binding'] or value['model_sha256'] != entry['model_sha256']:
        raise ValueError('Temporal checkpoint binding changed')
    return value, entry['binding']


def initialize_engine(engine, manifest, name):
    # Called only on a fresh engine, before fit creates its optimizer. Exact
    # resumed state, if any, is restored subsequently by the existing fitter.
    value, binding = initialization(manifest, name)
    transfer(engine.state_model, value, binding)


def source_baseline(parent, parent_manifest, destination):
    """Verify each old Native median against all8 unchanged saved sample paths."""
    from gofa_f.short_sampling_diagnostic import sample_seed, SETTINGS
    parent = Path(parent); destination = Path(destination)
    baseline = []; proofs = []
    store = SourceStore(parent_manifest, parent/'source_inputs')
    for row in development_rows(parent_manifest):
        key = Query(**row).key; name = digest(key)
        path = parent/'native/development'/(name+'.json'); value = read(path)
        expected = dict(binding=sha(parent/'manifest.json'), query=row,
                        weights=parent_manifest['weights_sha256'], sampling=SETTINGS)
        item, y, mask = store.get(row, 'development')
        if value['identity'] != expected or value['input_sha256'] != item.identity or value['query'] != row:
            raise ValueError('Native reuse input/query/kernel binding changed')
        if not np.array_equal(value['target'], y[0].numpy()) or not np.array_equal(value['observed'], mask[0].numpy()):
            raise ValueError('Native source labels or valid slots changed')
        paths = []; hashes = []
        for sample in range(8):
            trajectory = parent/'native/development/trajectories'/f'{name}-{sample}.json'
            saved = read(trajectory)
            if saved['identity'] != expected or saved['seed'] != sample_seed(row, sample):
                raise ValueError('Native trajectory seed or binding changed')
            p = np.asarray(saved['result']['prediction'], np.float64)
            if p.shape != (row['horizon'],) or not np.isfinite(p).all():
                raise ValueError('Incomplete inherited Native sample')
            paths.append(p); hashes.append(sha(trajectory))
        if not np.array_equal(np.median(paths, axis=0), value['prediction']):
            raise ValueError('Inherited Native median differs from its8 samples')
        baseline.append(value)
        proofs.append(dict(query=row, record_sha256=sha(path), trajectory_sha256=hashes,
            input_sha256=item.identity, new_generation_seconds=0))
    destination.mkdir(parents=True, exist_ok=True)
    atomic_json(destination/'records.json', dict(records=baseline,
        parent_manifest_sha256=sha(parent/'manifest.json'), target_labels_read=False))
    atomic_json(destination/'reuse_proof.json', dict(records=proofs, queries=len(proofs),
        new_predictions=0, new_generation_seconds=0, original_costs_preserved=True))


def prepare(cfg, output):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('Joint preparation is CPU-only')
    warmup = Path(cfg['warmup_output']); parent = Path(cfg['parent_source_output'])
    if sha(warmup/'manifest.json') != cfg['warmup_manifest_sha256'] or sha(parent/'manifest.json') != cfg['parent_source_manifest_sha256']:
        raise ValueError('Source lineage changed')
    base = read(parent/'manifest.json'); m = deepcopy(read(warmup/'manifest.json'))
    if m['source_cache_index_sha256'] != sha(parent/'source_inputs/index.json'):
        raise ValueError('Inherited common source cache changed')
    for name in GROUPS:
        for shape in ('12-12', '48-24'):
            for split in ('train', 'development'):
                if m['queries'][name][shape][split] != base['queries'][name][shape][split]:
                    raise ValueError('Native/UFG reuse requires identical source queries')
    for name, value in base['code'].items():
        if sha(Path(__file__).parent/name) != value:
            raise ValueError('Accepted input/tensor core changed')
    from gofa_f.dual_memory_eval_data import CORE
    accepted=Path(cfg['accepted_snapshot'])/'src/gofa_f'
    core={name:sha(Path(__file__).parent/name) for name in CORE}
    if core!={name:sha(accepted/name) for name in CORE}:
        raise ValueError('Native sampling or accepted tensor core changed')
    m['source_accepted_core_sha256']=core
    m['temporal_initializations'] = initialization_specs(warmup, m)
    settings = deepcopy(m['warmup_configuration']['joint_memory_training']); settings.pop('depths')
    m['configuration']['training'] = settings
    m['formal_configuration'] = cfg
    m['formal_runtime_code'] = {n:sha(Path(__file__).parent/n) for n in FILES}
    m['warmup_selection_seal_sha256'] = sha(warmup/'selection_seal.json')
    m['formal_source_input_cache'] = str(parent/'source_inputs')
    m['source_initialization_is_new_experiment_not_optimizer_resume'] = True
    m['source_confirmation_labels_opened'] = False
    output.mkdir(parents=True, exist_ok=True)
    if (output/'manifest.json').exists() and read(output/'manifest.json') != m:
        raise ValueError('Joint source manifest is immutable')
    atomic_json(output/'manifest.json', m)
    source_baseline(parent, base, output/'native_reuse')
    for shape in ('12-12', '48-24'):
        old = parent/f'ufg-{shape}'; selected = read(old/'selection.json')
        if selected['target_labels_used'] or sha(old/'best.pt') != selected['checkpoint_sha256']:
            raise ValueError('Official source UFG lock changed')
        target = output/f'ufg-{shape}'; target.mkdir(exist_ok=True)
        names=['best.pt','selection.json','curve.jsonl']+[p.name for p in sorted(old.glob('validation-*.json'))]
        for name in names:
            path = target/name
            if path.exists():
                if sha(path) != sha(old/name):raise ValueError('Inherited UFG artifact changed')
            else:shutil.copyfile(old/name, path)
    atomic_json(output/'ufg_reuse.json', dict(parent_manifest_sha256=sha(parent/'manifest.json'),
        same_source_inputs=True, same_all_node_label_slots=True, official_width=128,
        new_training_updates=0, new_cpu_training_seconds=0,
        inherited_source_cpu_activity=read(parent/'ufg_activity.json')))
    atomic_json(output/'ufg_status.json', dict(state='complete', phase='unchanged_official_source_bests_reused'))
    atomic_json(output/'ufg_activity.json', dict(read(parent/'ufg_activity.json'), worker=None, reused=True))
    if not (output/'activity.json').exists():
        atomic_json(output/'activity.json', dict(spent_seconds=m['inherited_gpu_seconds'],
            inherited_prior_gpu_seconds=m['inherited_gpu_seconds'], worker=None,
            waiting_seconds=0., resource_pause_count=0, peak_group_mib=23676))
    atomic_json(output/'status.json', dict(state='prepared', phase='joint_source_training',
        goal_achieved=False, target_predictions=0, source_confirmation_opened=False))


def execution_proof(native, tokenizer, codec, manifest, store, cache, output, check, progress):
    """Only new transferred initialization is checked; reuse old31node proof."""
    from gofa_f.dual_memory_evaluation import source_replays
    binding = sha(output/'manifest.json'); existing = read(output/'mechanisms.json')
    if existing:
        if existing['binding'] != binding or not existing['passed']:
            raise ValueError('Joint execution proof binding changed')
        return
    saved_rng = rng_state(True); cases = []; replay = None
    try:
        for depth in (6,3):
            random.seed(42); np.random.seed(42); torch.manual_seed(42)
            engine = MemoryEngine(native, tokenizer, codec, manifest, store, cache, depth)
            try:
                initialize_engine(engine, manifest, 'all')
                init, _ = initialization(manifest, 'all'); cpu = TemporalWarmup()
                cpu.load_state_dict(init['model']); cpu.eval()
                for source, shape in (('pems04_flow', '12-12'), ('boston2017', '48-24')):
                    check(); row = manifest['queries'][source][shape]['train'][0]
                    cpu_item, _, _ = store.get(row, 'train')
                    item, tokens, metadata, prefix, _, _ = engine.input(row, 'train')
                    with torch.no_grad():
                        expected = cpu.forecasts([cpu_item])[0]
                        actual, _ = engine.model.enhanced(tokens,item,metadata,prefix=prefix,optimized=True)
                    torch.testing.assert_close(actual.cpu(),expected,rtol=.002,atol=.002)
                    initial = engine.model.trainable_state()
                    # Discarded QA only: open Memory readout and graph gates to
                    # test the trained-head path, then restore every parameter.
                    with torch.no_grad():
                        columns=engine.state_model.slots*engine.state_model.temporal.hidden_size
                        engine.state_model.head[0].weight[:,:columns].normal_(std=.002)
                        for exchange in engine.state_model.exchanges:
                            exchange.gate.fill_(.1); exchange.ff_gate.fill_(.1)
                    engine.model.zero_grad(set_to_none=True)
                    full, fm = engine.model.enhanced(tokens,item,metadata,prefix=prefix,optimized=False)
                    (full/item.extent[:,None]).square().mean().backward()
                    gradients = {n:p.grad.detach().clone() for n,p in engine.state_model.named_parameters()}
                    engine.model.zero_grad(set_to_none=True)
                    cached, cm = engine.model.enhanced(tokens,item,metadata,prefix=prefix,optimized=True,checkpoint_tail=True)
                    torch.testing.assert_close(cached,full,rtol=.002,atol=.002)
                    torch.testing.assert_close(cm,fm,rtol=.002,atol=.002)
                    (cached/item.extent[:,None]).square().mean().backward()
                    for name,p in engine.state_model.named_parameters():
                        if p.grad is None or not p.grad.isfinite().all():raise ValueError('Transferred Memory gradient missing/nonfinite')
                        torch.testing.assert_close(p.grad,gradients[name],rtol=.002,atol=.002)
                    cases.append(dict(depth=depth,source=source,shape=shape,real_nodes=len(item.node_ids),
                        transfer_max_difference=float((actual.cpu()-expected).abs().max()),
                        full_cached_max_difference=float((cached-full).abs().max()),new_optimizer_updates=0))
                    engine.model.zero_grad(set_to_none=True);engine.model.load_trainable(initial)
                    progress(phase='transferred_initialization_acceptance',depth=depth,completed=len(cases),total=4)
                    del actual,expected,full,fm,cached,cm,gradients,prefix
                if replay is None:replay = source_replays(engine.model,codec,manifest,check)
            finally:engine.close()
    finally:restore_rng(saved_rng)
    atomic_json(output/'mechanisms.json',dict(binding=binding,passed=True,cases=cases,
        source_replays=replay,cpu_tolerance=1e-6,cuda_tolerance=.002,
        unchanged_maximum31node_proof=manifest['accepted_mechanisms_sha256'],
        discarded_optimizer_updates=0,target_predictions=0))


def run(cfg, output):
    from transformers import LlamaForCausalLM,LlamaTokenizer
    from gofa_f.short_codec import ShortCodec
    from gofa_f.allnode_stage import supervisor_alive,Paused
    from gofa_f.dual_memory_fusion import fit_gate
    if os.environ.get('GOFA_STAGE19_ADMITTED') != '1':raise ValueError('Cap-supervised worker required')
    m=read(output/'manifest.json');binding=sha(output/'manifest.json')
    if cfg!=m['formal_configuration'] or m['formal_runtime_code']!={n:sha(Path(__file__).parent/n) for n in FILES}:
        raise ValueError('Joint runtime/configuration changed')
    proof=read(output/'formal_cpu_preflight.json',{})
    if not proof.get('passed') or proof.get('manifest_sha256')!=binding:
        raise ValueError('Bound related CPU proof required')
    stopped=[False]
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    def check():
        if stopped[0] or not supervisor_alive(os.environ['GOFA_STAGE19_SUPERVISOR_PID'],os.environ['GOFA_STAGE19_SUPERVISOR_TICKS']):
            raise Paused('cooperative_resource_pause')
    visible={}
    def progress(**fields):
        if fields.get('fit')!=visible.get('fit') or fields.get('phase')!=visible.get('phase'):visible.clear()
        visible.update(fields)
        value=dict(state='running',goal_achieved=False,target_predictions=0,**visible)
        name=visible.get('fit')
        if name in m['temporal_initializations']:
            value.update(initialization_best_update=m['temporal_initializations'][name]['best_update'],
                fit_scope='fresh_all_source_fit' if name=='all' else 'source_group_crossfit',
                best_update_is_local_to_this_fit=True)
        atomic_json(output/'status.json',value)
    engine=None
    try:
        torch.cuda.set_per_process_memory_fraction(38*2**30/torch.cuda.get_device_properties(0).total_memory)
        root=Path(m['source_configuration']['weights']['directory'])
        if sha(root/'snapshot.json')!=m['weights_sha256']:raise ValueError('Frozen ChatTime changed')
        progress(phase='loading_frozen_chattime')
        native=LlamaForCausalLM.from_pretrained(root,torch_dtype=torch.bfloat16,
            attn_implementation='sdpa',local_files_only=True).to('cuda:0')
        versions={n:p._version for n,p in native.named_parameters()}
        tokenizer=LlamaTokenizer.from_pretrained(root,local_files_only=True);codec=ShortCodec(tokenizer)
        cache=HistoryCache(output/'history_cache',int(cfg['prefix_cache']['ram_gib']*2**30),int(cfg['prefix_cache']['disk_gib']*2**30))
        store=SourceStore(m,m['formal_source_input_cache'])
        execution_proof(native,tokenizer,codec,m,store,cache,output,check,progress)
        baseline=read(output/'native_reuse/records.json')['records'];crossfit={}
        for depth in (6,3):
            combined=[]
            for held in DOMAINS:
                check();random.seed(42);np.random.seed(42);torch.manual_seed(42)
                name='hold-'+held;folder=output/f'depth{depth}'/name
                own=SourceStore(m,m['formal_source_input_cache'],held_group=held)
                engine=MemoryEngine(native,tokenizer,codec,m,own,cache,depth);initialize_engine(engine,m,name)
                best=fit(engine,m,cfg,folder,name,depth,digest([binding,depth,name]),check,progress)
                own.permit_out_of_fit(folder/'selection.json')
                rows=development_rows(m,[v for v in GROUPS if GROUPS[v]==held])
                records=evaluate(engine,rows,'development',folder/'out_of_fit',binding,best['model_sha256'],check,progress,role='out_of_fit')
                for r in records:r['query_key']=Query(**r['query']).key
                combined.extend(records);engine.close();engine=None
            crossfit[depth]=combined
            atomic_json(output/f'crossfit-depth{depth}.json',dict(binding=binding,records=combined,
                traffic_channels_held_out_together=True,target_labels_used=False))
        scores={d:ratios(rows,baseline,m['training_unit_normalizers']) for d,rows in crossfit.items()}
        chosen=min((6,3),key=lambda d:(scores[d]['worst'],scores[d]['geometric'],d))
        selection=dict(binding=binding,depth=chosen,source_out_of_fit_scores=scores,selection_uses_target_labels=False)
        old=read(output/'depth_selection.json')
        if old is not None and old!=selection:raise ValueError('Locked global depth changed')
        atomic_json(output/'depth_selection.json',selection)
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        engine=MemoryEngine(native,tokenizer,codec,m,store,cache,chosen);initialize_engine(engine,m,'all')
        best=fit(engine,m,cfg,output/'final-source-model','all',chosen,digest([binding,chosen,'all']),check,progress)
        gate=fit_gate(crossfit[chosen],baseline,store,cfg['fusion'])
        gate.update(binding=binding,depth=chosen,source_expert_sha256=best['model_sha256'])
        old=read(output/'shared_gate.json')
        if old is not None and old!=gate:raise ValueError('Locked source gate changed')
        atomic_json(output/'shared_gate.json',gate)
        atomic_json(output/'source_model_lock.json',dict(binding=binding,depth=chosen,
            expert_checkpoint_sha256=sha(output/'final-source-model/best.pt'),expert_model_sha256=best['model_sha256'],
            shared_gate_sha256=sha(output/'shared_gate.json'),source_confirmation_labels_used=False,target_labels_used=False,
            initialization_seal_sha256=m['warmup_selection_seal_sha256']))
        if versions!={n:p._version for n,p in native.named_parameters()} or any(p.grad is not None for p in native.parameters()):
            raise ValueError('Frozen ChatTime weights/gradients changed')
        atomic_json(output/'status.json',dict(state='complete',phase='source_model_and_shared_fusion_locked_confirmation_pending',
            goal_achieved=False,target_predictions=0,source_confirmation_opened=False,selection=selection,
            final_best_is_pretrained_temporal_initialization=best['update']==0))
    except Paused:
        atomic_json(output/'status.json',dict(state='resource_paused',phase='joint_source_training',goal_achieved=False));raise SystemExit(75)
    except BaseException as exc:
        atomic_json(output/f'failure-{time.time_ns()}.json',dict(error=repr(exc),binding=binding))
        atomic_json(output/'status.json',dict(state='failed',phase='joint_source_training',error=repr(exc),goal_achieved=False));raise
    finally:
        if engine is not None:engine.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--prepare',action='store_true');args=parser.parse_args();cfg=configuration(args.config);output=Path(args.output)
    if args.prepare:prepare(cfg,output)
    else:run(cfg,output)


if __name__=='__main__':main()
