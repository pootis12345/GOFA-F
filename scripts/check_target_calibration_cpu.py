"""Stage20 CPU regression and actual frozen data/query/weight preparation."""
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
from gofa_f.target_calibration_state import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or any(
            os.environ.get(k) != '2' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS')):
        raise ValueError('CUDA must be disabled and OMP/MKL each use two threads')
    repo = Path(__file__).resolve().parents[1]
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    cfg = yaml.safe_load(Path(args.config).read_text())
    files = [str(p) for p in (repo/'src/gofa_f').glob('target_calibration*.py')]
    files += ['src/gofa_f/short_launcher.py','tests/test_target_calibration.py',str(Path(__file__))]
    subprocess.run(['/home/dcg/dcg_data/GOFA-F/.venv/bin/ruff','check',*files],cwd=repo,check=True)
    subprocess.run([sys.executable,'-m','pytest','-q','--junitxml',str(output/'cpu_tests.xml')],cwd=repo,check=True)
    manifest = prepare(cfg,output)
    adapter = torch.load(output/'adapter.pt',map_location='cpu',weights_only=False)
    AllNodeAdapter(4096).load_state_dict(adapter,strict=True)
    if not all(torch.isfinite(v).all() for v in adapter.values()):
        raise ValueError('Nonfinite inherited adapter')
    codec = ShortCodec(LlamaTokenizer.from_pretrained(
        manifest['source_configuration']['weights']['directory'],local_files_only=True))
    alphabet = codec.audit_alphabet()
    if alphabet['alphabet_sha256'] != manifest['alphabet']['alphabet_sha256'] or torch.cuda.is_initialized():
        raise ValueError('Alphabet changed or CPU preparation created a CUDA context')
    cases = ET.parse(output/'cpu_tests.xml').getroot().findall('.//testcase')
    atomic_json(output/'cpu_preflight.json',dict(passed=True,manifest_sha256=sha(output/'manifest.json'),
        cases=len(cases),skipped=sum(c.find('skipped') is not None for c in cases),
        cuda_initialized=False,alphabet=alphabet,coverage=manifest['coverage'],
        total_numeric_work=manifest['total_numeric_work'],optimizer_updates=0,
        strict_zero_shot=False,validation_queries=1152,validation_trajectories=18432,
        inherited_numerical_files=manifest['inherited_numerical_files'],
        actual_gpu_source_replay_required=True,test_fusion_gpu_predictions=0))
    print('Stage20 complete CPU preparation passed; no GPU predictions or neural training',flush=True)


if __name__ == '__main__':
    main()
