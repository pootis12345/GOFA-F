"""CPU source/official-code and observational capture verification before GPU."""
import argparse
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from transformers import LlamaTokenizer
import yaml

from gofa_f.allnode_stage import read
from gofa_f.generative_data import sha
from gofa_f.prefix_cache import atomic_json
from gofa_f.short_native_case_probe import load_official,prepare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(os.environ.get(k)!='2' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS')):
        raise RuntimeError('CPU checks require disabled CUDA and OMP/MKL2')
    repo,output = Path(__file__).resolve().parents[1],Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    subprocess.run(['/home/dcg/dcg_data/GOFA-F/.venv/bin/ruff','check','src/gofa_f/short_native_case_probe.py',
        'src/gofa_f/short_launcher.py','tests/test_short_native_case_probe.py',str(Path(__file__))],cwd=repo,check=True)
    subprocess.run([sys.executable,'-m','pytest','-q','--junitxml',str(output/'cpu_tests.xml')],cwd=repo,check=True)
    cfg = yaml.safe_load(Path(args.config).read_text())
    manifest = prepare(cfg,output)
    module,_ = load_official(cfg['official_directory'])
    defaults = {k:v.default for k,v in inspect.signature(module.ChatTime).parameters.items() if v.default is not inspect.Parameter.empty}
    assert defaults['num_samples']==8 and defaults['max_pred_len']==16
    source_cfg = read(Path(cfg['source_output'])/'manifest.json')['configuration']
    tokenizer = LlamaTokenizer.from_pretrained(source_cfg['weights']['directory'],local_files_only=True)
    prompts = manifest['prompt_comparison']
    lengths = {k:len(tokenizer(prompts[k])['input_ids']) for k in ('official','stage19_style')}
    if torch.cuda.is_initialized():
        raise RuntimeError('Unexpected CUDA context in CPU proof')
    atomic_json(output/'cpu_preflight.json',{'passed':True,'manifest_sha256':sha(output/'case_manifest.json'),
        'cpu_tests_sha256':sha(output/'cpu_tests.xml'),'official_defaults':defaults,'prompt_token_lengths':lengths,
        'official_blobs_verified':True,'no_training_resume':True,'cuda_initialized':False,
        'scope':'official parsing/rolling tested with mocked generation; real FP16 inference only after idle admission'})
    print(json.dumps({'cpu_passed':True,'defaults':defaults,'prompt_token_lengths':lengths}))


if __name__=='__main__':
    main()
