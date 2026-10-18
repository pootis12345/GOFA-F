"""Disabled-CUDA proof for the paired eight-sample source pilot."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

import torch
from transformers import LlamaTokenizer
import yaml

from gofa_f.allnode_graph import AllNodeAdapter
from gofa_f.allnode_stage import read
from gofa_f.generative_data import sha
from gofa_f.prefix_cache import atomic_json
from gofa_f.short_codec import ShortCodec
from gofa_f.short_sampling_diagnostic import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or any(os.environ.get(k) != '2'
            for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS')):
        raise RuntimeError('Disable CUDA; OMP/MKL each two threads')
    repo, output = Path(__file__).resolve().parents[1], Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'queries').mkdir(exist_ok=True)
    subprocess.run(['/home/dcg/dcg_data/GOFA-F/.venv/bin/ruff', 'check',
        'src/gofa_f/short_sampling_diagnostic.py', 'src/gofa_f/short_launcher.py',
        'tests/test_short_sampling_diagnostic.py', str(Path(__file__))], cwd=repo, check=True)
    subprocess.run([sys.executable, '-m', 'pytest', '-q', '--junitxml', str(output/'cpu_tests.xml')],
                   cwd=repo, check=True)
    cfg = yaml.safe_load(Path(args.config).read_text())
    manifest = prepare(cfg, output)
    source = Path(cfg['source_output'])
    base = read(source/'manifest.json')['configuration']
    codec = ShortCodec(LlamaTokenizer.from_pretrained(base['weights']['directory'], local_files_only=True))
    alphabet = codec.audit_alphabet()
    state = torch.load(source/'candidates/update-001000.pt', map_location='cpu', weights_only=False)
    AllNodeAdapter(4096).load_state_dict(state, strict=True)
    if not all(torch.isfinite(v).all() for v in state.values()) or torch.cuda.is_initialized():
        raise RuntimeError('Nonfinite adapter or unexpected CUDA context')
    atomic_json(output/'cpu_preflight.json', dict(passed=True,
        manifest_sha256=sha(output/'sampling_manifest.json'), cpu_tests_sha256=sha(output/'cpu_tests.xml'),
        alphabet=alphabet, checkpoint_parameters=sum(v.numel() for v in state.values()),
        paused_state=manifest['paused_state'], unchanged_production_inference=True, cuda_initialized=False,
        scope='tiny real Llama plus official tokenizer; no new 7B FP32 forward; actual BF16 sampling checked after admission'))
    print('Sampling CPU proof passed; training remains paused', flush=True)


if __name__ == '__main__':
    main()
