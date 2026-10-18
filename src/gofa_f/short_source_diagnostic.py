"""Read-only source diagnostics while production training remains user-paused.

No optimizer, backward, target access or checkpoint writes. Self-only is an
inference intervention, never a trained control or selection candidate.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
import yaml

from gofa_f.allnode_graph import build_graph, time_weights
from gofa_f.allnode_stage import read, supervisor_alive
from gofa_f.generative_data import Query, SOURCES, Series, sha
from gofa_f.idle_launcher import process_identity
from gofa_f.prefix_cache import atomic_json
from gofa_f.short_data import TEMPLATES
from gofa_f.short_forecast import forecast_batch
from gofa_f.short_generative_stage import load_model
from gofa_f.short_migration import INFERENCE_FILES


def select_queries(queries):
    """32 ID-stratified queries; 2 previously disclosed symptom cases separate."""
    selected = []
    for source, nodes in SOURCES.items():
        for variable in (nodes//3, 2*nodes//3):
            for length, horizon in TEMPLATES:
                matches = [(i, q) for i, q in enumerate(queries) if q['source'] == source
                    and q['variable'] == variable and (q['length'], q['horizon']) == (length, horizon)]
                if len(matches) != 1:
                    raise ValueError('Exactly one fixed source query per ID/template required')
                selected.append(dict(index=matches[0][0], query=matches[0][1], group='id_stratified'))
    for variable in (16, 92):
        matches = [(i, q) for i, q in enumerate(queries) if q['source'] == 'nn5'
            and q['variable'] == variable and (q['length'], q['horizon']) == (48, 24)]
        if len(matches) != 1:
            raise ValueError('Disclosed symptom query missing')
        selected.append(dict(index=matches[0][0], query=matches[0][1], group='known_symptom'))
    return selected


def self_only(graph):
    original = graph.graph
    present = original.observed.any(0)
    weights = torch.diag(present.to(original.prior.dtype))
    return replace(graph, graph=replace(original, prior=weights, base=weights.clone(),
        identity=hashlib.sha256((original.identity+':self_only_diagnostic').encode()).hexdigest()))


def verify_source(cfg):
    if cfg['diagnostic_updates'] != 0 or cfg['resume_training_after_diagnostic'] \
            or cfg['data']['target_evaluation_enabled']:
        raise ValueError('Only read-only source diagnostics are permitted')
    source = Path(cfg['source_output'])
    if read(source/'launcher.json')['state'] != 'stopped_by_user' or read(source/'activity.json')['worker']:
        raise ValueError('Production training must remain user-paused')
    launch = read(source.with_suffix('.launch.json'))
    if process_identity(launch['pid']) == launch['start_ticks']:
        raise ValueError('Production supervisor is still alive')
    manifest = read(source/'manifest.json')
    if manifest['code_revision'] != cfg['source_revision']:
        raise ValueError('Unexpected production runtime')
    paths = {'last.pt':cfg['paused_last_sha256'], **{
        f'candidates/update-{int(k):06d}.pt':v for k, v in cfg['checkpoints'].items()}}
    for path, expected in paths.items():
        if sha(source/path) != expected:
            raise ValueError('Paused source checkpoint changed: '+path)
    for name in INFERENCE_FILES+('short_training.py',):
        if sha(Path(__file__).parent/name) != manifest['code'][name]:
            raise ValueError('Diagnostic changed a model/inference execution file: '+name)
    return source, manifest, paths


def prepare(cfg, output):
    source, manifest, paths = verify_source(cfg)
    pools = read(source/'pools/queries.json')
    selected = select_queries(pools['validation'])
    raw = {}
    for name in SOURCES:
        path = Path(manifest['configuration']['data']['directory'])/'sources'/(name+'.npz')
        raw[name] = sha(path)
        if raw[name] != manifest['raw_hashes'][name]:
            raise ValueError('Source data changed')
    inputs = {}
    for label in ('native', 'update-000500', 'update-001000'):
        complete = source/'evaluations'/label/'complete.json'
        if read(complete)['complete'] is not True:
            raise ValueError('Complete source evaluation required')
        inputs[label] = sha(complete)
    saved = torch.load(source/'last.pt', map_location='cpu', weights_only=False)
    if saved['state']['updates'] != 1177 or saved['state']['pending'] != 30 \
            or not saved['gradients'] or not saved['optimizer']['state']:
        raise ValueError('Paused optimizer/partial-gradient state differs')
    record = {'configuration':cfg, 'source_manifest_sha256':sha(source/'manifest.json'),
        'checkpoints':paths, 'source_data_sha256':raw, 'complete_evaluations':inputs, 'queries':selected,
        'code':{p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        'paused_state':{'updates':saved['state']['updates'], 'pending':saved['state']['pending'],
            'pending_tokens':saved['state']['pending_tokens'],
            'gradient_tensors':sum(v is not None for v in saved['gradients'].values()),
            'optimizer_parameters':len(saved['optimizer']['state']),
            'parameter_version':saved['parameter_version']},
        'no_target_predictions':True, 'no_training_updates':True, 'no_automatic_training_resume':True,
        'cache_comparison_queries':['nn5:92:48:24', 'pems04:102:48:24'],
        'tolerance':{'rtol':.002, 'atol':.002},
        'interpretation':'teacher-conditioned predictions use true earlier targets; self-only is not a trained control'}
    old = read(output/'diagnostic_manifest.json')
    if old is not None and old != record:
        raise ValueError('Diagnostic manifest changed')
    atomic_json(output/'diagnostic_manifest.json', record)
    return record


def distribution(scores, codec, record=None):
    scores = scores.float()
    probabilities = scores.softmax(-1)
    ids = codec.alphabet(scores.device)[scores.argmax(-1)]
    result = {'argmax_ids':ids.tolist(), 'upper_probability':probabilities[:, -1].tolist(),
        'lower_probability':probabilities[:, 0].tolist(),
        'max_probability':probabilities.max(-1).values.tolist(),
        'top2_margin':(scores.topk(2,dim=-1).values[:,0]-scores.topk(2,dim=-1).values[:,1]).tolist()}
    if record is not None:
        prediction, _, boundary = codec.decode_ids(ids.tolist(), *record['scaling'], record['last'])
        result.update(prediction=prediction.tolist(), boundary=boundary.tolist())
    return result


class Trace:
    def __init__(self, model, codec, cached):
        self.model, self.codec, self.cached = model, codec, cached
        self.native = model.native
        self.scores = []

    def __call__(self, tokens, **kwargs):
        output = self.model(tokens, **kwargs)
        if kwargs.get('cache') is not None or not self.cached:
            scores = output.logits[:, -1]
        else:
            ends = tokens.prompt_ends+len(self.codec.first_separator)
            scores = output.logits[torch.arange(len(ends), device=ends.device), ends]
        self.scores.append(scores.float().index_select(-1, self.codec.alphabet(scores.device)).cpu())
        return output


def exchange_measurements(module, args, delta):
    hidden, graph = args
    with torch.no_grad(), torch.autocast(hidden.device.type, enabled=False):
        z = module.down(module.norm(hidden.float())).transpose(0, 1)
        gate = module.gate(z).sigmoid().squeeze(-1)
        base = time_weights(graph.base, graph.observed)
        support = base > 0
        logits = module.query(z) @ module.key(z).transpose(-1,-2)/4
        logits += base.clamp_min(torch.finfo(torch.float32).tiny).log() \
            + module.edge_score(graph.edges.float()).squeeze(-1)[None]
        empty = ~support.any(-1, keepdim=True)
        logits = logits.masked_fill(~support, -torch.inf)
        attention = torch.where(empty, torch.zeros_like(logits), logits).softmax(-1)*support.float()
        valid = graph.observed[:, 0]
        h, d = hidden[0, valid].float(), delta[0, valid].float()
        g = gate[valid, 0]
        return {'target_delta_to_hidden_norm':float(d.norm()/h.norm().clamp_min(1e-30)),
            'target_delta_rms':float(d.square().mean().sqrt()),
            'target_gate_mean':float(g.mean()), 'target_gate_min':float(g.min()),
            'target_gate_max':float(g.max()), 'target_gate_fraction_above_095':float((g>.95).float().mean()),
            'target_attention_self_mean':float(attention[valid,0,0].mean()),
            'target_base_self_mean':float(base[valid,0,0].mean()),
            'nodes':len(hidden)}


def compare_scores(a, b, tolerance=.002):
    actual, expected = torch.cat(a), torch.cat(b)
    error = (actual-expected).abs()
    return {'max_abs':float(error.max()),
        'violations':int((error > tolerance+tolerance*expected.abs()).sum()),
        'argmax_equal':bool(torch.equal(actual.argmax(-1),expected.argmax(-1)))}


@torch.no_grad()
def inspect_query(model, codec, series, query, mode, check, compare_cache=False):
    q = Query(**query)
    data = series[q.source]
    history = np.where(data.mask[q.origin-q.length:q.origin,q.variable],
                       data.values[q.origin-q.length:q.origin,q.variable],np.nan)
    target = np.where(data.mask[q.origin:q.origin+q.horizon,q.variable],
                      data.values[q.origin:q.origin+q.horizon,q.variable],np.nan)
    record = codec.record(history,target)
    contexts, telemetry = None, []
    handles = []
    if mode != 'native':
        graph = build_graph(data,q)
        if mode == 'self_only':
            graph = self_only(graph)
        for layer_index,module in enumerate(model.adapter.exchanges):
            def hook(module,args,delta,index=layer_index):
                telemetry.append(dict(layer=index+1,**exchange_measurements(module,args,delta)))
            handles.append(module.register_forward_hook(hook))
        try:
            contexts = model.encode_origins([graph],codec,check=check)
        finally:
            for handle in handles:
                handle.remove()
    check()
    tokens = codec.batch([record]).to(next(model.native.parameters()).device)
    output = model(tokens,contexts=contexts)
    positions = torch.tensor(record['answer_positions'],device=tokens.input_ids.device)-1
    scores = output.logits[0].index_select(0,positions).float().index_select(-1,codec.alphabet(positions.device))
    teacher = distribution(scores,codec,record)
    labels = tokens.labels[0,positions+1]
    valid = labels != -100
    targets = codec.lookup(model.native.config.vocab_size,positions.device)[labels[valid]]
    teacher['conditional_ce'] = float(-scores.log_softmax(-1)[valid].gather(1,targets[:,None]).mean())
    teacher['observed_slots'] = valid.tolist()
    observed = np.isfinite(target)
    teacher['conditional_mae'] = float(np.abs(np.asarray(teacher['prediction'])-target)[observed].mean())
    del output,scores,tokens
    check()
    trace = Trace(model,codec,True)
    free = forecast_batch(trace,codec,[history],q.horizon,contexts,check=check)
    free_scores = torch.cat(trace.scores)
    free_ce = float(-free_scores.log_softmax(-1)[valid.cpu()].gather(1,targets.cpu()[:,None]).mean())
    result = {'query':query,'mode':mode,'teacher':teacher,'free':{
        **distribution(free_scores,codec,record), 'conditional_ce_to_true':free_ce,
        'numeric_ids':free['numeric_ids'][0], 'generated_ids':free['generated_ids'][0],
        'mae':float(np.abs(free['predictions'][0]-target)[observed].mean())},'exchanges':telemetry}
    if compare_cache:
        check()
        uncached = Trace(model,codec,False)
        other = forecast_batch(uncached,codec,[history],q.horizon,contexts,check=check,cached=False)
        comparison = compare_scores(trace.scores,uncached.scores)
        comparison['raw_tokens_equal'] = free['generated_ids'] == other['generated_ids']
        result['cache_comparison'] = comparison
        if comparison['violations'] or not comparison['argmax_equal'] or not comparison['raw_tokens_equal']:
            raise ValueError('Actual trained checkpoint full/incremental equivalence failed: '+str(comparison))
    return result


def run(cfg,output):
    if os.environ.get('GOFA_STAGE19_ADMITTED') != '1':
        raise RuntimeError('Source diagnostic requires fully-idle CPU supervisor admission')
    proof = read(output/'cpu_preflight.json',{})
    manifest = read(output/'diagnostic_manifest.json')
    if not proof.get('passed') or proof['manifest_sha256'] != sha(output/'diagnostic_manifest.json') \
            or manifest['configuration'] != cfg or manifest['code'] != \
                {p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}:
        raise ValueError('Code-bound CPU verification required')
    source, source_manifest, _ = verify_source(cfg)
    stopped = [False]
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    def check():
        if stopped[0] or not supervisor_alive(os.environ['GOFA_STAGE19_SUPERVISOR_PID'],
                                              os.environ['GOFA_STAGE19_SUPERVISOR_TICKS']):
            raise InterruptedError('Cooperative diagnostic pause')
    def status(state,**extra):
        atomic_json(output/'status.json',dict(state=state,updated_at=time.time(),**extra))
    torch.cuda.set_per_process_memory_fraction(38*2**30/torch.cuda.get_device_properties(0).total_memory)
    base = source_manifest['configuration']
    model,codec = load_model(base,torch.device('cuda:0'))
    model.adapter.requires_grad_(False)
    series = {name:Series(Path(base['data']['directory'])/'sources'/(name+'.npz')) for name in SOURCES}
    total = len(manifest['queries'])*5
    completed = 0
    try:
        for checkpoint in ('native','500','1000'):
            check()
            if checkpoint != 'native':
                model.load_trainable(torch.load(source/'candidates'/f'update-{int(checkpoint):06d}.pt',
                                               map_location='cpu',weights_only=False))
            for row in manifest['queries']:
                for mode in (('native',) if checkpoint == 'native' else ('full','self_only')):
                    check()
                    path = output/'queries'/f"{checkpoint}-{mode}-{row['index']:06d}.json"
                    if path.exists():
                        completed += 1
                        continue
                    status('running',phase='source_diagnostic',checkpoint=checkpoint,mode=mode,done=completed,total=total)
                    q = row['query']
                    compare = mode == 'full' and (q['source'],q['variable'],q['length'],q['horizon']) \
                        in (('nn5',92,48,24),('pems04',102,48,24))
                    report = inspect_query(model,codec,series,q,mode,check,compare)
                    report['sampling_group'] = row['group']
                    if mode != 'self_only':
                        label = 'native' if checkpoint == 'native' else f'update-{int(checkpoint):06d}'
                        existing = read(source/'evaluations'/label/f"{row['index']:06d}.json")
                        if existing['numeric_ids'] != report['free']['numeric_ids'] \
                                or existing['generated_ids'] != report['free']['generated_ids']:
                            raise ValueError('Diagnostic changed existing source prediction raw tokens')
                        report['existing_prediction_tokens_equal'] = True
                    atomic_json(path,report)
                    completed += 1
        verify_source(cfg)
        if any(p.grad is not None for p in model.parameters()):
            raise RuntimeError('Diagnostic created gradients')
        atomic_json(output/'result.json',{'complete':True,'queries':len(manifest['queries']),
            'records':completed,'optimizer_updates':0,'source_unchanged':True,
            'allocated_bytes':torch.cuda.max_memory_allocated(),'reserved_bytes':torch.cuda.max_memory_reserved(),
            'interpretation':'source-only fixed subset; teacher forcing is not forecasting; self-only is not a trained control'})
        status('complete',phase='source_diagnostic',done=completed,total=total)
    except InterruptedError as exc:
        status('resource_paused',reason=str(exc),done=completed,total=total)
    except Exception as exc:
        atomic_json(output/'failures.json',{'type':type(exc).__name__,'error':str(exc),'done':completed})
        status('failed',error=str(exc),done=completed,total=total)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--prepare',action='store_true')
    args = parser.parse_args()
    cfg,output = yaml.safe_load(Path(args.config).read_text()),Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    (output/'queries').mkdir(exist_ok=True)
    if args.prepare:
        print(json.dumps(prepare(cfg,output),ensure_ascii=False),flush=True)
    else:
        run(cfg,output)


if __name__ == '__main__':
    main()
