"""CPU-only proof for observational Stage19 source diagnostics."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from transformers import LlamaTokenizer
import yaml

from gofa_f.allnode_graph import AllNodeAdapter
from gofa_f.generative_data import sha
from gofa_f.prefix_cache import atomic_json
from gofa_f.short_codec import ShortCodec
from gofa_f.short_source_diagnostic import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or any(os.environ.get(k) != '2'
            for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS')):
        raise RuntimeError('Disable CUDA and use two OMP/MKL threads for CPU checks')
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    (output/'queries').mkdir(exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    ruff = Path('/home/dcg/dcg_data/GOFA-F/.venv/bin/ruff')
    subprocess.run([str(ruff),'check','src/gofa_f/short_source_diagnostic.py','src/gofa_f/short_launcher.py',
                    'tests/test_short_source_diagnostic.py',str(Path(__file__))],
                   cwd=repo,check=True)
    subprocess.run([sys.executable,'-m','pytest','-q','--junitxml',str(output/'cpu_tests.xml')],
                   cwd=repo,check=True)
    cfg = yaml.safe_load(Path(args.config).read_text())
    manifest = prepare(cfg,output)
    source = Path(cfg['source_output'])
    base = json.loads((source/'manifest.json').read_text())['configuration']
    codec = ShortCodec(LlamaTokenizer.from_pretrained(base['weights']['directory'],local_files_only=True))
    alphabet = codec.audit_alphabet()
    counts = {}
    adapter = AllNodeAdapter(4096)
    for step in cfg['checkpoints']:
        state = torch.load(source/'candidates'/f'update-{int(step):06d}.pt',map_location='cpu',weights_only=False)
        adapter.load_state_dict(state,strict=True)
        if not all(torch.isfinite(tensor).all() for tensor in state.values()):
            raise RuntimeError('Nonfinite checkpoint adapter parameter')
        counts[step] = {'parameters':sum(v.numel() for v in state.values()),
            'parameter_norms':{k:float(v.float().norm()) for k,v in state.items()}}
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU preflight unexpectedly initialized CUDA')
    atomic_json(output/'cpu_preflight.json',{'passed':True,
        'manifest_sha256':sha(output/'diagnostic_manifest.json'), 'cpu_tests_sha256':sha(output/'cpu_tests.xml'),
        'alphabet':alphabet,'checkpoints':counts,'paused_state':manifest['paused_state'],
        'unchanged_inference_files_checked':True,'cuda_initialized':False,
        'scope':'CPU tests use tiny real Llama plus official tokenizer; real 7B parameters unchanged, not a new real 7B FP32 forward proof'})
    print('CPU diagnostic preflight passed; no GPU launched',flush=True)


if __name__ == '__main__':
    main()
