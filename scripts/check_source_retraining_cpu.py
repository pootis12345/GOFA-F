"""Stage23 full CPU regression plus real FP32 new-source nonzero path acceptance."""
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
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or any(
            os.environ.get(k) != '2' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS')):
        raise RuntimeError('CPU proof requires disabled CUDA and two OMP/MKL threads')
    import torch
    from gofa_f.generative_data import Series, sha
    from gofa_f.graph_benchmark import atomic
    from gofa_f.short_generative_stage import load_model
    from gofa_f.source_retraining import configuration
    from gofa_f.source_retraining_checks import mechanisms
    from gofa_f.source_retraining_data import VIEWS
    from gofa_f.source_retraining_state import prepare, verify
    cfg, output = configuration(args.config), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    repository = Path(__file__).resolve().parents[1]
    files = list(map(str, (repository/'src/gofa_f').glob('source_retraining*.py')))+[
        'src/gofa_f/short_launcher.py', 'tests/test_source_retraining.py', str(Path(__file__)),
        'scripts/fetch_stage23_sources.py']
    ruff = os.environ.get('GOFA_RUFF', str(Path(os.environ['GOFA_WORKSPACE'])/'.venv/bin/ruff'))
    subprocess.run([ruff, 'check', *files], cwd=repository, check=True)
    xml = output/'cpu_tests.xml'
    subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                    '--junitxml', str(xml)], cwd=repository, check=True)
    manifest = prepare(cfg, output)
    cases = ET.parse(xml).getroot().findall('.//testcase')
    proof = dict(passed=False, manifest_sha256=sha(output/'manifest.json'), cases=len(cases),
        skipped=sum(c.find('skipped') is not None for c in cases), cuda_disabled=True,
        target_predictions=0, real_weight_dtype='float32')
    atomic(output/'cpu_preflight.json', proof)
    try:
        model, codec = load_model(manifest['source_configuration'], torch.device('cpu'), torch.float32)
        model.load_trainable(torch.load(output/'parent.pt', map_location='cpu', weights_only=False))
        model.parameter_version = 3500
        proof['alphabet'] = codec.audit_alphabet()
        sources = {n: Series(Path(cfg['data']['directory'])/'sources'/(n+'.npz')) for n in VIEWS}
        proof['real_weights'] = mechanisms(model, codec, sources, manifest['pools']['validation'],
                                           1e-6, output)
        verify(cfg, output)
        if torch.cuda.is_initialized():
            raise ValueError('CPU proof created a CUDA context')
        proof['passed'] = True
    except Exception as exc:
        proof.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        atomic(output/'cpu_preflight.json', proof)
    print('Stage23 full CPU acceptance passed', flush=True)


if __name__ == '__main__':
    main()
