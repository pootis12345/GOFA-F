"""Validate orchestration before arming a future immutable update1500 handoff."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import torch
import yaml

from gofa_f.allnode_stage import read
from gofa_f.generative_data import sha
from gofa_f.prefix_cache import atomic_json
from gofa_f.short_sampled_handoff import audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(os.environ.get(k)!='2' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS')):
        raise ValueError('CPU validation requires disabled CUDA and two OMP/MKL threads')
    repo=Path(__file__).resolve().parents[1]
    cfg,output=yaml.safe_load(Path(args.config).read_text()),Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    files=['src/gofa_f/short_sampled_handoff.py','src/gofa_f/short_sampled_continuation.py',
        'src/gofa_f/short_sampled_progress.py','src/gofa_f/short_sampled_state.py','src/gofa_f/short_launcher.py',
        'tests/test_short_sampled_handoff.py',str(Path(__file__))]
    subprocess.run(['/home/dcg/dcg_data/GOFA-F/.venv/bin/ruff','check',*files],cwd=repo,check=True)
    subprocess.run([sys.executable,'-m','pytest','-q','--junitxml',str(output/'cpu_tests.xml')],cwd=repo,check=True)
    manifest=audit(cfg,output)
    inherited=read(Path(cfg['source_output'])/'cpu_preflight.json')
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU handoff validation created a CUDA context')
    cases=ET.parse(output/'cpu_tests.xml').getroot().findall('.//testcase')
    atomic_json(output/'cpu_preflight.json',dict(passed=True,manifest_sha256=sha(output/'manifest.json'),
        cases=len(cases),skipped=sum(c.find('skipped') is not None for c in cases),cuda_initialized=False,
        alphabet=inherited['alphabet'],unchanged_tensor_files=manifest['unchanged_tensor_files'],
        inherited_cpu_proof_sha256=manifest['inherited_cpu_proof_sha256'],
        proof_scope='orchestration, post-review state/shard migration and unchanged accepted numerical paths',
        real_post1500_state_audit='deferred until source completes and all controllers/workers exit',
        numerical_tolerances_unchanged=True,target_evaluation_enabled=False))
    print('CPU handoff validation passed; source run untouched, no CUDA context',flush=True)


if __name__=='__main__':
    main()
