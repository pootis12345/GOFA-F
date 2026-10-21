"""Locked-only Stage24 six-target forecasts; no target fitting or checkpoint choice."""
from pathlib import Path

import numpy as np
import torch

from gofa_f.allnode_stage import read
from gofa_f.generative_data import Series, sha
from gofa_f.graph_benchmark import atomic
from gofa_f.graph_benchmark_run import target_path_acceptance
from gofa_f.short_sampled_forecast import evaluate
from gofa_f.short_sampling_diagnostic import SETTINGS
from gofa_f.source_gate_state import TARGETS


def bootstrap_interval(native, current, queries):
    left, right = native['metrics']['variables'], current['metrics']['variables']
    if set(left) != set(right) or not left or len(queries) != len(native['errors']) \
            or len(queries) != len(current['errors']):
        raise ValueError('Complete paired variable metrics required for bootstrap')
    # Match the actual point-within-variable macro MAE, including missingness.
    values = np.asarray([[left[v]['mae'], right[v]['mae']] for v in sorted(left)], np.float64)
    rng = np.random.default_rng(4224)
    samples = []
    for _ in range(2000):
        batch = values[rng.integers(0, len(values), len(values))].mean(0)
        samples.append(batch[1]-batch[0])
    return dict(variable_cluster_bootstrap_draws=2000, seed=4224,
                paired_macro_mae_difference_95pct=np.quantile(samples, [.025, .975]).tolist(),
                conditional_on_fixed_origins_and_training_seed=True)


@torch.no_grad()
def run(model, codec, manifest, output, progress, check, gate_name='update100_gate.json'):
    output = Path(output)
    locked, confirmation = read(output/'final_selection.json'), read(output/'source_confirmation.json')
    if not locked or not confirmation or not confirmation['complete'] or not confirmation['passed'] \
            or locked['checkpoint_sha256'] != confirmation['checkpoint_sha256']:
        raise ValueError('Target labels/predictions inaccessible before locked source confirmation passes')
    binding = sha(output/'manifest.json')
    if gate_name not in ('update100_gate.json', 'source_gate.json'):
        raise ValueError('Unknown source barrier')
    gate = read(output/gate_name)
    if locked.get('binding') != binding or confirmation.get('binding') != binding \
            or not locked.get('source_gate_passed') or not gate or not gate['passed'] or gate['binding'] != binding:
        raise ValueError('Source gate/selection/confirmation binding mismatch')
    selected = Path(locked['path'])
    if sha(selected) != locked['checkpoint_sha256']:
        raise ValueError('Locked model changed')
    prior = Path(manifest['configuration']['stage21_output'])
    old = read(prior/'manifest.json')
    root = Path(old['data_directory'])/'targets'
    model.load_trainable(torch.load(selected, map_location='cpu', weights_only=False))
    result = {}
    for name in TARGETS:
        check()
        # Reuse only the two original bound complete reference systems. No
        # original target record is loaded until after the source gate above.
        pair = {key: read(prior/f'target-{name}-{key}-result.json') for key in ('native', 'gofa')}
        queries = old['queries']['targets'][name]
        for key, value in pair.items():
            digest = old['weights_sha256'] if key == 'native' else old['configuration']['adapter_sha256']
            expected = f'{sha(prior/"manifest.json")}:sample8:target-{name}-{key}:{digest}:'+codec.alphabet_sha
            if not value['complete'] or value['queries'] != 128 or value['sampling'] != SETTINGS \
                    or value['binding'] != expected:
                raise ValueError('Original target reference binding differs')
            for i, q in enumerate(queries):
                path = prior/'evaluations'/f'target-{name}-{key}'/f'{i:06d}.json'
                median = read(path)
                if median['query'] != q or median['binding'] != expected:
                    raise ValueError('Original target median changed')
                for s in range(8):
                    sample = read(path.parent/'trajectories'/f'{i:06d}-{s:02d}.json')
                    if sample['query'] != q or sample['seed'] != median['seeds'][s] or sample['binding'] != expected:
                        raise ValueError('Original target sample changed')
        if sha(root/(name+'.npz')) != old['raw_hashes'][name] \
                or sha(root/(name+'.json')) != old['metadata_hashes'][name]:
            raise ValueError('Target observation/metadata changed')
        data = {name: Series(root/(name+'.npz'))}
        label = f'target-{name}-new'
        progress.tick(label, force=True)
        target_path_acceptance(model, codec, data[name], queries[0], False, output, label,
                               sha(output/'manifest.json'), check)
        current = evaluate(model, codec, data, queries, output, label,
            sha(output/'manifest.json')+':'+locked['checkpoint_sha256']+':'+codec.alphabet_sha,
            progress, check)
        atomic(output/(label+'-result.json'), current)
        progress.completed(label, current)
        base, new = pair['native']['metrics']['macro']['mae'], current['metrics']['macro']['mae']
        result[name] = dict(native_metrics=pair['native']['metrics'], parent_metrics=pair['gofa']['metrics'],
            new_metrics=current['metrics'], passed=new < base-max(1e-12, 1e-9*base),
            improved_queries=sum(b < a for a, b in zip(pair['native']['errors'], current['errors'])),
            equal_queries=sum(b == a for a, b in zip(pair['native']['errors'], current['errors'])),
            worse_queries=sum(b > a for a, b in zip(pair['native']['errors'], current['errors'])),
            uncertainty=bootstrap_interval(pair['native'], current, queries))
        atomic(output/'target_results.json', dict(complete=len(result) == 6,
            checkpoint_sha256=locked['checkpoint_sha256'], datasets=result,
            all_six_improved=len(result) == 6 and all(v['passed'] for v in result.values()),
            target_label_fitting=False, development_exposure_disclosed=True))
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError('No gradients may exist during target evaluation')
    return result
