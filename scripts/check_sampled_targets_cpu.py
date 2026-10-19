"""CPU-only regression, frozen target identities and numerical proof inheritance."""
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
from gofa_f.generative_data import sha
from gofa_f.prefix_cache import atomic_json
from gofa_f.short_codec import ShortCodec
from gofa_f.short_target_state import prepare


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(os.environ.get(k)!='2' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS')):
        raise ValueError('Disable CUDA, OMP/MKL each two threads')
    repo=Path(__file__).resolve().parents[1]
    cfg,output=yaml.safe_load(Path(args.config).read_text()),Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    files=['src/gofa_f/short_sampled_targets.py','src/gofa_f/short_target_state.py',
        'src/gofa_f/short_target_progress.py','src/gofa_f/short_launcher.py',
        'tests/test_short_sampled_targets.py',str(Path(__file__))]
    subprocess.run(['/home/dcg/dcg_data/GOFA-F/.venv/bin/ruff','check',*files],cwd=repo,check=True)
    subprocess.run([sys.executable,'-m','pytest','-q','--junitxml',str(output/'cpu_tests.xml')],cwd=repo,check=True)
    manifest=prepare(cfg,output)
    adapter=torch.load(output/'adapter.pt',map_location='cpu',weights_only=False)
    AllNodeAdapter(4096).load_state_dict(adapter,strict=True)
    if not all(torch.isfinite(v).all() for v in adapter.values()):
        raise ValueError('Nonfinite locked 3500 adapter')
    codec=ShortCodec(LlamaTokenizer.from_pretrained(manifest['source_configuration']['weights']['directory'],local_files_only=True))
    alphabet=codec.audit_alphabet()
    if alphabet['alphabet_sha256']!=manifest['alphabet']['alphabet_sha256'] or torch.cuda.is_initialized():
        raise ValueError('Official finite alphabet changed or CUDA context created')
    cases=ET.parse(output/'cpu_tests.xml').getroot().findall('.//testcase')
    atomic_json(output/'cpu_preflight.json',dict(passed=True,manifest_sha256=sha(output/'manifest.json'),
        cases=len(cases),skipped=sum(c.find('skipped') is not None for c in cases),alphabet=alphabet,
        cuda_initialized=False,coverage=manifest['queries']['coverage'],selected_update=3500,
        adapter_sha256=sha(output/'adapter.pt'),optimizer_updates=0,
        inherited_source_proof_sha256=manifest['source_proof_sha256'],
        unchanged_numerical_files=manifest['unchanged_tensor_files'],
        actual_gpu_source_replay_required=True))
    print('CPU target preparation passed; no GPU/target prediction or training',flush=True)


if __name__=='__main__':
    main()
