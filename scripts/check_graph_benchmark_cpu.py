"""Stage21 CPU-only real-data, frozen adapter and regression acceptance."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import torch
from transformers import LlamaTokenizer
import yaml

from gofa_f.allnode_graph import AllNodeAdapter, build_graph
from gofa_f.generative_data import Query, Series, sha
from gofa_f.graph_benchmark_state import prepare, verify
from gofa_f.prefix_cache import atomic_json
from gofa_f.short_codec import ShortCodec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or any(
            os.environ.get(k) != '2' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS')):
        raise ValueError('CPU acceptance requires disabled CUDA and OMP/MKL two threads')
    repo, output = Path(__file__).resolve().parents[1], Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config).read_text())
    files = [str(p) for p in (repo/'src/gofa_f').glob('graph_benchmark*.py')]
    files += ['src/gofa_f/short_launcher.py', 'tests/test_graph_benchmark_v2.py', str(Path(__file__))]
    subprocess.run(['/home/dcg/dcg_data/GOFA-F/.venv/bin/ruff', 'check', *files], cwd=repo, check=True)
    subprocess.run([sys.executable, '-m', 'pytest', '-q', '--junitxml', str(output/'cpu_tests.xml')],
                   cwd=repo, check=True)
    manifest = prepare(cfg, output, convert=False)
    adapter = torch.load(output/'adapter.pt', map_location='cpu', weights_only=False)
    AllNodeAdapter(4096).load_state_dict(adapter, strict=True)
    if not all(torch.isfinite(v).all() for v in adapter.values()):
        raise ValueError('Nonfinite inherited adapter')
    codec = ShortCodec(LlamaTokenizer.from_pretrained(
        manifest['source_configuration']['weights']['directory'], local_files_only=True))
    alphabet = codec.audit_alphabet()
    if alphabet['alphabet_sha256'] != manifest['alphabet']['alphabet_sha256']:
        raise ValueError('Inherited10000-token numeric alphabet differs')
    graphs = {}
    for name, rows in manifest['queries']['targets'].items():
        data = Series(Path(cfg['data_directory'])/'targets'/(name+'.npz'))
        graph = build_graph(data, Query(**rows[0]))
        if len(graph.variable_ids) > 9 or not torch.isfinite(graph.graph.prior).all() \
                or not torch.isfinite(graph.graph.base).all():
            raise ValueError('Actual target subgraph failed shape/finite checks')
        graphs[name] = dict(nodes=len(graph.variable_ids), variable_ids=graph.variable_ids,
            identity=graph.identity, original_nodes=len(data.ids),
            external_edges=int((data.external > 0).sum()), missing_fraction=data.meta['missing_fraction'])
    verify(cfg, output)
    if torch.cuda.is_initialized():
        raise ValueError('CPU preparation initialized CUDA')
    cases = ET.parse(output/'cpu_tests.xml').getroot().findall('.//testcase')
    atomic_json(output/'cpu_preflight.json', dict(passed=True, manifest_sha256=sha(output/'manifest.json'),
        cases=len(cases), skipped=sum(c.find('skipped') is not None for c in cases),
        cuda_initialized=False, alphabet=alphabet, selected_update=3500,
        adapter_sha256=cfg['adapter_sha256'], queries=768, trajectories=12288,
        total_numeric_work=manifest['total_numeric_work'], coverage=manifest['queries']['coverage'],
        actual_graphs=graphs, optimizer_updates=0, actual_gpu_source_replay_required=True))
    print('Stage21 complete CPU preparation passed; GPU replay remains required', flush=True)


if __name__ == '__main__':
    main()
