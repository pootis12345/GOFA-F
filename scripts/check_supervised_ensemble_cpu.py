"""CPU preparation/regression proof; frozen accepted numerical core is reused by SHA."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('OMP_NUM_THREADS') != '2' \
            or os.environ.get('MKL_NUM_THREADS') != '2':
        raise RuntimeError('CPU acceptance requires CUDA disabled and OMP/MKL two threads')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from gofa_f.allnode_stage import read
    from gofa_f.generative_data import sha
    from gofa_f.prefix_cache import atomic_json
    from gofa_f.supervised_ensemble_state import prepare
    from gofa_f.source_retraining import configuration
    import torch
    cfg, output = configuration(args.config), args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest = prepare(cfg, output)
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', '--disable-warnings',
        '--junitxml='+str(output/'cpu_tests.xml')], check=False)
    suites = ET.parse(output/'cpu_tests.xml').getroot()
    cases = list(suites.iter('testcase'))
    skipped = sum(c.find('skipped') is not None for c in cases)
    failures = sum(c.find('failure') is not None or c.find('error') is not None for c in cases)
    inherited = read(Path(cfg['stage28_output'])/'cpu_preflight.json')
    value = dict(passed=result.returncode == 0 and failures == 0 and not torch.cuda.is_initialized(),
        manifest_sha256=sha(output/'manifest.json'), cases=len(cases), skipped=skipped,
        failures=failures, cuda_disabled=True, cuda_initialized=torch.cuda.is_initialized(),
        inference_core_proof_reused=True, inherited_cpu_proof_sha256=manifest['inherited_cpu_proof_sha256'],
        numerical_core_sha256=manifest['unchanged_tensor_files'],
        scope='New float64 LP, isolation, query, restart and orchestration regression; frozen numerical core unchanged',
        inherited_acceptance_passed=inherited['passed'],
        runtime_gpu_acceptance_required='four exact source replays and new full/incremental dataset cases')
    atomic_json(output/'cpu_preflight.json', value)
    if not value['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
