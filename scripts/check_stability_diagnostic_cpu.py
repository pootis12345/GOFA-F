"""CPU proof and input audit; never starts a CUDA worker."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from transformers import LlamaTokenizer
import yaml

from gofa_f.allnode_graph import AllNodeAdapter, build_graph
from gofa_f.allnode_stage import read
from gofa_f.generative_data import Query, Series, sha
from gofa_f.prefix_cache import atomic_json
from gofa_f.short_codec import ShortCodec
from gofa_f.short_sampling_diagnostic import choose, generate
from gofa_f.short_source_diagnostic import compare_scores
from gofa_f.short_generative_stage import load_model
from gofa_f.short_stability_diagnostic import REPLAYS, amplitude, analyze_existing, prepare


@torch.no_grad()
def real_cpu_cases(base, cfg, manifest):
    model, codec = load_model(base, torch.device('cpu'), torch.float32)
    model.load_trainable(torch.load(Path(cfg['source_output'])/'candidates/update-001000.pt',
                                   map_location='cpu', weights_only=False))
    model.adapter.requires_grad_(False)
    results = []
    for row in manifest['queries']:
        q = Query(**row['query'])
        if (q.source, q.variable, q.length, q.horizon) not in REPLAYS:
            continue
        data = Series(Path(base['data']['directory'])/'sources'/(q.source+'.npz'))
        history = np.where(data.mask[q.origin-q.length:q.origin, q.variable],
                           data.values[q.origin-q.length:q.origin, q.variable], np.nan)
        with amplitude(model, .25):
            contexts = model.encode_origins([build_graph(data, q)], codec)
            cached_scores = []
            seed = manifest['shared_seeds'][str(row['index'])][0]
            cached = generate(model, codec, history, q.horizon, contexts, seed=seed, trace=cached_scores)
            # One complete causal forward on the actually sampled prefix is enough:
            # matching every conditional draw proves the full autoregressive trace
            # by induction, without 24 repeated CPU prefills. No true future input.
            record = codec.record(history)
            tokens = codec.batch([record]).append(torch.tensor(cached['generated_ids'])[None])
            tokens.input_ids = tokens.full_ids
            full_output = model(tokens, contexts=contexts)
            positions = torch.arange(q.horizon)*(1+len(codec.separator)) \
                + record['prompt_end']+len(codec.first_separator)
            full_scores = list(full_output.logits[0].index_select(0, positions).float()
                .index_select(-1, codec.alphabet('cpu')).split(1))
            generator = torch.Generator(device='cpu').manual_seed(seed)
            replay = list(codec.first_separator)
            for slot, scores in enumerate(full_scores):
                replay.append(int(codec.alphabet('cpu')[choose(scores, generator)].item()))
                if slot+1 < q.horizon:
                    replay.extend(codec.separator)
            comparison = compare_scores(cached_scores, full_scores, tolerance=1e-6)
            comparison['raw_tokens_equal'] = cached['generated_ids'] == replay
            comparison['full_path'] = 'single_complete_causal_forward_actual_sample_prefix'
            if comparison['violations'] or not comparison['argmax_equal'] or not comparison['raw_tokens_equal']:
                raise ValueError('FP32 quarter-amplitude full/KV mismatch: '+str(comparison))
            results.append(dict(query=row['query'], seed=seed, **comparison))
            print('Real 7B CPU case passed: '+q.key, flush=True)
    if len(results) != 2 or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError('Both real CPU cases must pass without gradients')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or any(os.environ.get(k) != '2'
            for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS')):
        raise RuntimeError('Disable CUDA and set OMP/MKL each to two threads')
    repo, output = Path(__file__).resolve().parents[1], Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for name in ('teachers', 'samples', 'queries'):
        (output/name).mkdir(exist_ok=True)
    subprocess.run(['/home/dcg/dcg_data/GOFA-F/.venv/bin/ruff', 'check',
        'src/gofa_f/short_stability_diagnostic.py', 'src/gofa_f/short_stability_metrics.py',
        'src/gofa_f/short_launcher.py', 'tests/test_short_stability_diagnostic.py', str(Path(__file__))],
        cwd=repo, check=True)
    subprocess.run([sys.executable, '-m', 'pytest', '-q', '--junitxml', str(output/'cpu_tests.xml')],
                   cwd=repo, check=True)
    cfg = yaml.safe_load(Path(args.config).read_text())
    manifest = prepare(cfg, output)
    analyze_existing(cfg, output)
    source = Path(cfg['source_output'])
    base = read(source/'manifest.json')['configuration']
    codec = ShortCodec(LlamaTokenizer.from_pretrained(base['weights']['directory'], local_files_only=True))
    alphabet = codec.audit_alphabet()
    if not (np.diff([codec.id_centers[i] for i in codec.finite_ids]) > 0).all():
        raise ValueError('Numeric probabilities must be accumulated in increasing value order')
    adapter = torch.load(source/'candidates/update-001000.pt', map_location='cpu', weights_only=False)
    AllNodeAdapter(4096).load_state_dict(adapter, strict=True)
    if not all(torch.isfinite(v).all() for v in adapter.values()) or torch.cuda.is_initialized():
        raise RuntimeError('Nonfinite adapter or unexpected CUDA initialization')
    real_cases = real_cpu_cases(base, cfg, manifest)
    atomic_json(output/'cpu_preflight.json', dict(passed=True,
        manifest_sha256=sha(output/'stability_manifest.json'), cpu_tests_sha256=sha(output/'cpu_tests.xml'),
        existing_analysis_sha256=sha(output/'existing_analysis.json'), alphabet=alphabet,
        adapter_parameters=sum(v.numel() for v in adapter.values()), paused_state=manifest['paused_state'],
        original_inference_files_unchanged=True, official_weight_content_hashes_verified=True,
        real_7b_fp32_cases=real_cases, cuda_initialized=False,
        scope='tiny Llama plus real 7B quarter-amplitude full/KV sampled cases; BF16 checks after idle admission'))
    print('Stability CPU proof passed; no training or GPU launched', flush=True)


if __name__ == '__main__':
    main()
