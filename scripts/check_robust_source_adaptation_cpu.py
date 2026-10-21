"""Full CPU regression plus new actual-FP32 combined-loss/teacher acceptance."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or any(os.environ.get(k) != '2'
            for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS')):
        raise RuntimeError('CPU proof needs disabled CUDA and two OMP/MKL threads')
    import torch
    from gofa_f.allnode_stage import read
    from gofa_f.generative_data import Series, sha
    from gofa_f.graph_benchmark import atomic
    from gofa_f.robust_checks import mechanisms
    from gofa_f.robust_state import prepare, verify
    from gofa_f.short_generative_stage import load_model
    from gofa_f.source_retraining import configuration
    from gofa_f.source_retraining_data import VIEWS
    cfg, output = configuration(args.config), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    repo = Path(__file__).resolve().parents[1]
    ruff = str(Path(os.environ['GOFA_WORKSPACE'])/'.venv/bin/ruff')
    files = list(map(str, (repo/'src/gofa_f').glob('robust*.py')))+[
        'src/gofa_f/short_launcher.py', 'tests/test_robust_source_adaptation.py',
        'scripts/bootstrap_robust_source_adaptation.py', str(Path(__file__))]
    subprocess.run([ruff, 'check', *files], cwd=repo, check=True)
    xml = output/'cpu_tests.xml'
    subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                    '--junitxml', str(xml)], cwd=repo, check=True)
    m = prepare(cfg, output)
    cases = ET.parse(xml).getroot().findall('.//testcase')
    proof = dict(passed=False, manifest_sha256=sha(output/'manifest.json'), cases=len(cases),
        skipped=sum(c.find('skipped') is not None for c in cases), cuda_disabled=True,
        target_predictions=0, real_weight_dtype='float32')
    atomic(output/'cpu_preflight.json', proof)
    try:
        # Only unchanged model/inference proof is reused; the new loss, teacher,
        # shared prefixes and gradients below are checked on real weights anew.
        prior = Path(cfg['stage23_output'])
        previous = read(prior/'cpu_preflight.json')
        original = read(prior/'manifest.json')
        if not previous['passed'] or not previous['real_weights']['passed'] \
                or previous['manifest_sha256'] != sha(prior/'manifest.json'):
            raise ValueError('Prior unchanged-inference proof is not bound and complete')
        for name in original['unchanged_tensor_files']:
            if sha(repo/'src/gofa_f'/name) != original['code'][name]:
                raise ValueError('Inference proof cannot be reused for changed tensor core')
        proof['unchanged_inference_provenance'] = dict(source_proof_sha256=sha(prior/'cpu_preflight.json'),
            tensor_files_identical=True, reused_only_unchanged_inference=True,
            changed_loss_requires_new_real_weight_acceptance=True)
        model, codec = load_model(m['source_configuration'], torch.device('cpu'), torch.float32)
        model.load_trainable(torch.load(output/'parent.pt', map_location='cpu', weights_only=False))
        model.parameter_version = 3500
        proof['alphabet'] = codec.audit_alphabet()
        sources = {n: Series(Path(cfg['data']['directory'])/'sources'/(n+'.npz')) for n in VIEWS}
        proof['real_weights'] = mechanisms(model, codec, sources, m['pools']['validation'], 1e-6, output)
        verify(cfg, output)
        if torch.cuda.is_initialized():
            raise ValueError('CPU verification initialized CUDA')
        proof['passed'] = True
    except Exception as exc:
        proof.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        atomic(output/'cpu_preflight.json', proof)
    print('Stage24 full CPU acceptance passed', flush=True)


if __name__ == '__main__':
    main()
