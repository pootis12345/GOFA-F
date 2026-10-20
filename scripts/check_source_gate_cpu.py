"""Stage22 CPU regressions, actual pools, alphabet and frozen-state acceptance."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import torch
from transformers import LlamaTokenizer
import yaml

from gofa_f.allnode_graph import AllNodeAdapter, build_graph
from gofa_f.generative_data import Query, Series
from gofa_f.graph_benchmark import atomic, sha
from gofa_f.short_codec import ShortCodec
from gofa_f.source_gate import features
from gofa_f.source_gate_state import prepare, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or any(
            os.environ.get(k) != '2' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS')):
        raise RuntimeError('Disable CUDA and use two OMP/MKL threads for CPU acceptance')
    repo, output = Path(__file__).resolve().parents[1], Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    files = [str(p) for p in (repo/'src/gofa_f').glob('source_gate*.py')]
    files += ['src/gofa_f/source_reliability_gate.py', 'src/gofa_f/short_launcher.py',
              'tests/test_source_gate.py', str(Path(__file__))]
    subprocess.run(['/home/dcg/dcg_data/GOFA-F/.venv/bin/ruff', 'check', *files], cwd=repo, check=True)
    subprocess.run([sys.executable, '-m', 'pytest', '-q', '--junitxml', str(output/'cpu_tests.xml')],
                   cwd=repo, check=True)
    cfg = yaml.safe_load(Path(args.config).read_text())
    manifest = prepare(cfg, output)
    state = torch.load(output/'adapter.pt', map_location='cpu', weights_only=False)
    AllNodeAdapter(4096).load_state_dict(state, strict=True)
    if not all(torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Nonfinite frozen adapter')
    codec = ShortCodec(LlamaTokenizer.from_pretrained(
        manifest['source_configuration']['weights']['directory'], local_files_only=True))
    if codec.alphabet_sha != manifest['alphabet']['alphabet_sha256']:
        raise ValueError('Frozen finite alphabet changed')
    summaries = {}
    for source in manifest['source_raw_hashes']:
        data = Series(Path(manifest['source_configuration']['data']['directory'])/'sources'/(source+'.npz'))
        own = []
        for split in ('fit', 'confirmation'):
            for row in manifest['pools'][split]:
                q = Query(**row['query'])
                if q.source != source:
                    continue
                graph = build_graph(data, q)
                mask = data.mask[q.origin-q.length:q.origin, q.variable]
                history = np.where(mask, data.values[q.origin-q.length:q.origin, q.variable], np.nan)
                observed = features(history, mask, graph, q.length, q.horizon)
                own.append(dict(split=split, query=row['query'], features=observed,
                    graph_identity=graph.identity, nodes=len(graph.variable_ids)))
                if len(graph.variable_ids) > 9:
                    raise ValueError('Invalid actual source graph size')
        summaries[source] = own
    verify(cfg, output)
    if torch.cuda.is_initialized():
        raise ValueError('CPU proof created a CUDA context')
    cases = ET.parse(output/'cpu_tests.xml').getroot().findall('.//testcase')
    atomic(output/'cpu_preflight.json', dict(passed=True, manifest_sha256=sha(output/'manifest.json'),
        cases=len(cases), skipped=sum(c.find('skipped') is not None for c in cases), cuda_initialized=False,
        actual_source_features=summaries, source_fit_queries=128, source_confirmation_queries=128,
        source_holdout_audit=manifest['source_holdout_audit'], adapter_sha256=cfg['adapter_sha256'],
        new_neural_parameters=0, neural_optimizer_updates=0, target_predictions=0,
        actual_bf16_full_incremental_acceptance_required=True))


if __name__ == '__main__':
    main()
