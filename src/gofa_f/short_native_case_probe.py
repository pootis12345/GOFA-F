"""Official FP16 ChatTime eight-sample NN5 case probe, with narrow controls.

The official predict method is executed unchanged. Observational pipeline capture
retains every sample, block scaler and median. No graph, optimizer or training.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np
import torch
import yaml

from gofa_f.allnode_codec import OFFICIAL_PROMPT as STAGE19_PROMPT
from gofa_f.allnode_stage import read, supervisor_alive
from gofa_f.generative_data import Series, sha
from gofa_f.prefix_cache import atomic_json
from gofa_f.short_source_diagnostic import verify_source

OFFICIAL_BLOBS = {'model/model.py':'af932802db789a7293b2a4bf35eec1877b8db379',
    'utils/prompt.py':'89f65f4609266f0dde0e2bce9ba00ece9850f61f',
    'utils/tools.py':'ec41d9b88296a197d12fb7d811f4f8e6afe7d79e'}


def load_official(root):
    root = Path(root).resolve()
    hashes = {}
    for name, expected in OFFICIAL_BLOBS.items():
        content = (root/name).read_bytes()
        blob = hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        if blob != expected:
            raise ValueError('Official fixed-commit file differs: '+name)
        hashes[name] = sha(root/name)
    sys.path.insert(0,str(root))
    module = importlib.import_module('model.model')
    if Path(module.__file__).resolve() != root/'model/model.py':
        raise ValueError('Official module resolved outside verified source')
    return module,hashes


def numbers(values):
    return [float(x) if np.isfinite(x) else None for x in np.asarray(values).reshape(-1)]


def repetition(values):
    values = np.asarray(values)
    finite = np.isfinite(values)
    longest = current = 0
    previous = None
    for value in values:
        current = current+1 if np.isfinite(value) and previous is not None and value == previous else int(np.isfinite(value))
        longest = max(longest,current)
        previous = value if np.isfinite(value) else None
    return {'finite_slots':int(finite.sum()),'requested_slots':len(values),
        'unique_finite_values':len(np.unique(values[finite])),
        'constant_complete':bool(len(values)>1 and finite.all() and len(np.unique(values))==1),
        'longest_exact_constant_run':longest}


def capture_predict(native,module,history,protocol,check=lambda:None,on_block=lambda *_:None):
    if protocol not in ('official_sample8','official_greedy1','stage19_prompt_greedy1'):
        raise ValueError('Unknown fixed case protocol')
    original_pipeline,original_prompt = module.pipeline,module.getPrompt
    blocks = []
    native.num_samples = 8 if protocol == 'official_sample8' else 1
    def factory(*args,**kwargs):
        if protocol != 'official_sample8':
            kwargs['do_sample'],kwargs['num_return_sequences'] = False,1
            for key in ('top_k','top_p','temperature'):
                kwargs.pop(key,None)
        pipe = original_pipeline(*args,**kwargs)
        settings = {k:v for k,v in kwargs.items() if k not in ('model','tokenizer')}
        def call(prompt):
            check()
            on_block(len(blocks)+1)
            samples = pipe(prompt)
            check()
            count = min(native.max_pred_len,native.pred_len-sum(b['horizon'] for b in blocks))
            predictions,saved = [],[]
            for sample in samples:
                response = sample['generated_text'].split('### Response:\n')[1]
                dispersed = native.serializer.inverse_serialize(response)
                raw = native.discretizer.inverse_discretize(dispersed)
                padded = np.concatenate((raw,np.full(max(0,count-len(raw)),np.nan)))[:count]
                predictions.append(padded)
                saved.append({'serialized_response':response,'parsed_values':numbers(raw),
                    'used_prediction':numbers(padded),'repetition':repetition(padded)})
            median = np.nanmedian(predictions,axis=0)
            scaler = native.discretizer.scaler
            blocks.append({'horizon':count,'generation':settings,'prompt':prompt,
                'scaler':{k:numbers(getattr(scaler,k)) for k in ('scale_','min_','data_min_','data_max_')},
                'samples':saved,'median':numbers(median),'median_repetition':repetition(median)})
            return samples
        return call
    if protocol == 'stage19_prompt_greedy1':
        def legacy_prompt(flag,instruction=None,input=None,response=None,context=None):
            if flag != 'prediction' or context is not None:
                raise ValueError('Only fixed source prediction prompt control is permitted')
            return STAGE19_PROMPT.format(input)+(response or '')
        module.getPrompt = legacy_prompt
    module.pipeline = factory
    try:
        prediction = native.predict(history.copy())
    finally:
        module.pipeline,module.getPrompt = original_pipeline,original_prompt
    expected = np.concatenate([np.array([np.nan if v is None else v for v in b['median']]) for b in blocks])
    if not np.array_equal(prediction,expected,equal_nan=True):
        raise ValueError('Observational capture changed official median or rolling result')
    return {'protocol':protocol,'prediction':numbers(prediction),'repetition':repetition(prediction),
            'blocks':blocks,'capture_median_matches_official':True}


def prepare(cfg,output):
    source,manifest,_ = verify_source(cfg)
    if cfg['official_revision'] != '8c2d4c209302d2b2bd6cc3c154586842341e3247' \
            or cfg['seed'] != 42 or cfg['protocols'] != ['official_sample8','official_greedy1','stage19_prompt_greedy1']:
        raise ValueError('Fixed official case probe changed')
    module,hashes = load_official(cfg['official_directory'])
    data = Series(Path(manifest['configuration']['data']['directory'])/'sources/nn5.npz')
    if sha(data.path)!=manifest['raw_hashes']['nn5']:
        raise ValueError('Fixed source data changed')
    q = cfg['query']
    existing = read(source/'pools/queries.json')['validation']
    selected = [i for i,x in enumerate(existing) if x==q]
    if len(selected)!=1 or q != dict(source='nn5',variable=92,origin=568,length=48,horizon=24,chunk=0):
        raise ValueError('Fixed source symptom case changed')
    history = np.where(data.mask[520:568,92],data.values[520:568,92],np.nan)
    if not np.isfinite(history).all():
        raise ValueError('This fixed case must retain its fully observed history')
    words = module.Serializer().serialize(module.Discretizer().discretize(history))
    official = module.getPrompt(flag='prediction',input=words)
    legacy = STAGE19_PROMPT.format(words)
    record = {'configuration':cfg,'official_files':hashes,'official_git_blobs':OFFICIAL_BLOBS,
        'source_manifest_sha256':sha(source/'manifest.json'),'source_data_sha256':sha(data.path),
        'paused_last_sha256':sha(source/'last.pt'),'query_index':selected[0],
        'history':numbers(history),'target':numbers(data.values[568:592,92]),
        'observed':data.mask[568:592,92].tolist(),
        'stage19_native_reference':read(source/'evaluations/native'/f'{selected[0]:06d}.json'),
        'prompt_comparison':{'official':official,'stage19_style':legacy,'identical':official==legacy},
        'code':{p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        'no_training_updates':True,'no_graph_or_adapter':True,'no_automatic_training_resume':True,
        'interpretation':'official primary uses FP16, 8 samples per block, 16+8 rolling; auxiliary controls isolate sampling/prompt, not model selection'}
    atomic_json(output/'case_manifest.json',record)
    return record


def run(cfg,output):
    if os.environ.get('GOFA_STAGE19_ADMITTED') != '1':
        raise RuntimeError('Native case probe requires fully-idle supervisor admission')
    verify_source(cfg)
    manifest = read(output/'case_manifest.json')
    proof = read(output/'cpu_preflight.json',{})
    if not proof.get('passed') or proof['manifest_sha256']!=sha(output/'case_manifest.json') \
            or manifest['configuration']!=cfg or manifest['code']!={p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}:
        raise ValueError('Code-bound CPU proof required')
    stopped = [False]
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    def check():
        if stopped[0] or not supervisor_alive(os.environ['GOFA_STAGE19_SUPERVISOR_PID'],os.environ['GOFA_STAGE19_SUPERVISOR_TICKS']):
            raise InterruptedError('Cooperative native case stop')
    def status(state,**fields):
        atomic_json(output/'status.json',dict(state=state,updated_at=time.time(),**fields))
    torch.cuda.set_per_process_memory_fraction(38*2**30/torch.cuda.get_device_properties(0).total_memory)
    module,_ = load_official(cfg['official_directory'])
    base = read(Path(cfg['source_output'])/'manifest.json')['configuration']
    native = module.ChatTime(base['weights']['directory'],hist_len=48,pred_len=24)
    native.model.requires_grad_(False).eval()
    if native.model.device.type!='cuda' or native.model.dtype!=torch.float16:
        raise ValueError('Official FP16 model must fit on admitted GPU without offload')
    results = []
    try:
        for protocol in cfg['protocols']:
            check()
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            result = capture_predict(native,module,np.asarray(manifest['history'],np.float32),protocol,check,
                lambda block:status('running',protocol=protocol,block=block))
            prediction = np.array([np.nan if x is None else x for x in result['prediction']])
            target,observed = np.array(manifest['target']),np.array(manifest['observed'],bool)
            valid = observed & np.isfinite(prediction)
            result['observed_prediction_coverage'] = float(valid.sum()/observed.sum())
            result['mae_complete_case'] = float(np.abs(prediction-target)[observed].mean()) if valid.sum()==observed.sum() else None
            result['mae_valid_only'] = float(np.abs(prediction-target)[valid].mean()) if valid.any() else None
            atomic_json(output/(protocol+'.json'),result)
            results.append(result)
        verify_source(cfg)
        atomic_json(output/'result.json',{'complete':True,'results':results,'seed':42,'dtype':'float16',
            'optimizer_updates':0,'source_unchanged':True,'allocated_bytes':torch.cuda.max_memory_allocated(),
            'reserved_bytes':torch.cuda.max_memory_reserved()})
        status('complete',phase='native_case_probe')
    except InterruptedError as exc:
        status('resource_paused',reason=str(exc))
    except Exception as exc:
        atomic_json(output/'failures.json',{'type':type(exc).__name__,'error':str(exc)})
        status('failed',error=str(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--prepare',action='store_true')
    args = parser.parse_args()
    cfg,output = yaml.safe_load(Path(args.config).read_text()),Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    if args.prepare:
        print(json.dumps(prepare(cfg,output)))
    else:
        run(cfg,output)


if __name__=='__main__':
    main()
