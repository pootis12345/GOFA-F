"""Full CPU regression plus exact trained migration and unchanged tensor proofs."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import torch
from transformers import LlamaTokenizer
import yaml

from gofa_f.allnode_graph import AllNodeAdapter
from gofa_f.allnode_stage import read
from gofa_f.generative_data import sha
from gofa_f.prefix_cache import atomic_json
from gofa_f.short_codec import ShortCodec
from gofa_f.short_sampled_state import TENSOR_FILES,prepare


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(os.environ.get(k)!='2' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS')):
        raise RuntimeError('Disable CUDA, OMP/MKL each two threads')
    repo=Path(__file__).resolve().parents[1]
    files=['src/gofa_f/short_sampled_continuation.py','src/gofa_f/short_sampled_state.py',
        'src/gofa_f/short_sampled_forecast.py','src/gofa_f/short_sampled_progress.py',
        'src/gofa_f/short_launcher.py','tests/test_short_sampled_continuation.py',str(Path(__file__))]
    subprocess.run(['/home/dcg/dcg_data/GOFA-F/.venv/bin/ruff','check',*files],cwd=repo,check=True)
    cfg,output=yaml.safe_load(Path(args.config).read_text()),Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,'-m','pytest','-q','--junitxml',str(output/'cpu_tests.xml')],cwd=repo,check=True)
    manifest=prepare(cfg,output)
    checkpoint=torch.load(output/'last.pt',map_location='cpu',weights_only=False)
    AllNodeAdapter(4096).load_state_dict(checkpoint['adapter'],strict=True)
    values=list(checkpoint['adapter'].values())+[v for v in checkpoint['gradients'].values() if v is not None]
    values += [v for p in checkpoint['optimizer']['state'].values() for v in p.values() if isinstance(v,torch.Tensor)]
    if not all(torch.isfinite(v).all() for v in values):
        raise ValueError('Paused weights, gradients or optimizer state are nonfinite')
    base=manifest['source_configuration']
    codec=ShortCodec(LlamaTokenizer.from_pretrained(base['weights']['directory'],local_files_only=True))
    alphabet=codec.audit_alphabet()
    sample_root=Path(cfg['diagnostic_cost_sources'][2])
    sample_manifest=read(sample_root/'sampling_manifest.json')
    for name in TENSOR_FILES+('short_sampling_diagnostic.py',):
        if sample_manifest['code'][name]!=manifest['code'][name]:
            raise ValueError('Accepted sample8 numerical path changed: '+name)
    prior=read(sample_root/'result.json')
    comparisons=[s['cache_comparison'] for r in prior['records'] for s in r['samples'] if 'cache_comparison' in s]
    if len(comparisons)!=4 or any(c['violations'] or not c['raw_tokens_equal'] for c in comparisons):
        raise ValueError('Original-amplitude full/CUDA sampling proof incomplete')
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU preflight unexpectedly created CUDA context')
    cases=ET.parse(output/'cpu_tests.xml').getroot().findall('.//testcase')
    proof=dict(passed=True,manifest_sha256=sha(output/'manifest.json'),cases=len(cases),
        skipped=sum(c.find('skipped') is not None for c in cases),alphabet=alphabet,cuda_initialized=False,
        migration=read(output/'migration.json'),unchanged_tensor_proofs=dict(
            original_cpu_proof_sha256=manifest['cpu_proof_sha256'],original_gpu_proof_sha256=manifest['gpu_proof_sha256'],
            unchanged_files=manifest['unchanged_tensor_files'],real_trained_sample8_comparisons=comparisons,
            sampled_manifest_sha256=sha(sample_root/'sampling_manifest.json'),
            reason='new evaluation orchestration/ledger only; all tensor, training, sampler code byte-identical'))
    atomic_json(output/'cpu_preflight.json',proof)
    print('Sampled continuation CPU passed; no CUDA context or training launched',flush=True)


if __name__=='__main__':
    main()
