"""Stage23 full CPU regression plus real FP32 new-source nonzero path acceptance."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


def reuse_real_weights(old_output, old_snapshot, manifest, output):
    """Reuse numerical proof only for the exact optional-JSON-read repair.

    New CPU regression still runs. Every scientific binding and all numerical
    Python files must match; only the CLI optional-record read import may differ.
    """
    from gofa_f.generative_data import sha
    from gofa_f.graph_benchmark import read
    old_output, old_snapshot = Path(old_output), Path(old_snapshot)
    previous = read(old_output/'cpu_preflight.json')
    original = read(old_output/'manifest.json')
    if not previous or not previous.get('passed') or not previous.get('cuda_disabled') \
            or previous.get('real_weight_dtype') != 'float32' \
            or not previous.get('real_weights', {}).get('passed') \
            or previous['manifest_sha256'] != sha(old_output/'manifest.json'):
        raise ValueError('Complete original bound real FP32 proof required')
    cases = ET.parse(old_output/'cpu_tests.xml').getroot().findall('.//testcase')
    if len(cases) != previous['cases'] or any(
            c.find('failure') is not None or c.find('error') is not None for c in cases):
        raise ValueError('Original CPU regression evidence is incomplete')
    if {k: v for k, v in original.items() if k != 'code'} != \
            {k: v for k, v in manifest.items() if k != 'code'}:
        raise ValueError('Scientific configuration/data/query/weights bindings changed')
    differences = {n for n in set(original['code']) | set(manifest['code'])
                   if original['code'].get(n) != manifest['code'].get(n)}
    if differences != {'source_retraining.py'}:
        raise ValueError('Numerical proof reuse permits only the exact CLI read repair')
    old_entry = old_snapshot/'src/gofa_f/source_retraining.py'
    new_entry = Path(__file__).resolve().parents[1]/'src/gofa_f/source_retraining.py'
    if sha(old_entry) != original['code']['source_retraining.py']:
        raise ValueError('Original CLI source does not match the accepted manifest')
    expected = old_entry.read_text().replace(
        '    from gofa_f.graph_benchmark import atomic, read, sha',
        '    from gofa_f.allnode_stage import read\n    from gofa_f.graph_benchmark import atomic, sha').replace(
        '        from gofa_f.graph_benchmark import read', '        from gofa_f.allnode_stage import read')
    if new_entry.read_text() != expected:
        raise ValueError('CLI change exceeds the audited optional-JSON-read fix')
    for name, digest in original['code'].items():
        if sha(old_snapshot/'src/gofa_f'/name) != digest:
            raise ValueError('Original accepted snapshot changed: '+name)
    if sha(old_output/'parent.pt') != manifest['configuration']['adapter_sha256']:
        raise ValueError('Original parent adapter changed')
    if (old_output/'last.pt').exists() or (old_output/'activity.json').exists() \
            or old_output.with_suffix('.launch.json').exists():
        raise ValueError('This repair proof reuse is limited to the never-launched bootstrap failure')
    return dict(alphabet=previous['alphabet'], real_weights=previous['real_weights'],
        reuse_provenance=dict(original_proof_sha256=sha(old_output/'cpu_preflight.json'),
            original_manifest_sha256=previous['manifest_sha256'],
            original_tests_sha256=sha(old_output/'cpu_tests.xml'),
            changed_non_numerical_files=sorted(differences),
            all_scientific_bindings_identical=True, all_numerical_files_identical=True,
            new_full_cpu_regression=True, prior_gpu_started=False,
            reuse_verifier_sha256=sha(Path(__file__))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--reuse-real-proof-from')
    parser.add_argument('--original-snapshot')
    args = parser.parse_args()
    if bool(args.reuse_real_proof_from) != bool(args.original_snapshot):
        parser.error('Proof reuse requires both the original output and snapshot')
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
        'scripts/fetch_stage23_sources.py', 'scripts/bootstrap_source_retraining.py']
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
        if args.reuse_real_proof_from:
            proof.update(reuse_real_weights(args.reuse_real_proof_from, args.original_snapshot,
                                            manifest, output))
        else:
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
