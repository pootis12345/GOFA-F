"""Stage22 entry: source-only CPU fit, conditional holdout and six-target evaluation."""
import argparse
import os
from pathlib import Path

import numpy as np
import yaml

from gofa_f.graph_benchmark import atomic, read, sha
from gofa_f.source_gate import AMPLITUDES, fit_and_cross_validate


def fit_gate(cfg, output):
    """Only source fitting records/epsilons may be read by this entry."""
    from gofa_f.source_gate_state import verify
    manifest = verify(cfg, output)
    binding = sha(output/'manifest.json')
    records = read(output/'source_fit.json')
    if records['binding'] != binding or not records['complete'] or len(records['records']) != 128:
        raise ValueError('Complete bound source action forecasts required')
    if records['action_order'] != list(AMPLITUDES) or [r['query'] for r in records['records']] != [
            r['query'] for r in manifest['pools']['fit']]:
        raise ValueError('Only the sealed source fitting queries/actions may fit the shared gate')
    for row in records['records']:
        errors, extent = np.asarray(row['errors'], np.float64), row['origin_range']
        if errors.shape != (4,) or not np.isfinite(errors).all() or (errors < 0).any() \
                or not np.isfinite(extent) or extent <= 0 or not np.array_equal(
                    np.asarray(row['risks']), (errors[1:]-errors[0])/extent):
            raise ValueError('Risk labels differ from source free-forecast errors/origin scale')
    eps = read(Path(cfg['source_output'])/'pools/metric_epsilons.json')
    cv, gate = fit_and_cross_validate(records['records'], eps)
    cv['binding'] = binding
    gate.update(binding=binding, adapter_sha256=cfg['adapter_sha256'],
        fitting_records_sha256=sha(output/'source_fit.json'), target_labels_used=False)
    for name, item in (('source_cross_validation.json', cv), ('gate.json', gate)):
        if (output/name).exists() and read(output/name) != item:
            raise ValueError('Frozen source gate/validation changed')
        atomic(output/name, item)
    return cv


def report(output):
    manifest = read(output/'manifest.json')
    def optional(name):
        return read(output/name) if (output/name).exists() else None
    status = optional('status.json') or {}
    cv, confirmation, targets = (optional(n) for n in
        ('source_cross_validation.json', 'source_confirmation.json', 'target_results.json'))
    gate = optional('gate.json')
    value = dict(complete=status.get('state') == 'complete', stop_reason=status.get('stop_reason'),
        source_cross_validation=cv, source_confirmation=confirmation, targets=targets,
        source_gate_sha256=sha(output/'gate.json') if gate else None,
        adapter_sha256=manifest['configuration']['adapter_sha256'], selected_update=3500,
        neural_optimizer_updates=0, target_label_fitting=False,
        goal_achieved=bool(targets and targets.get('all_six_strictly_improved')),
        activity=optional('activity.json'), memory=optional('cuda_memory.json'),
        limitations=['small source pilot; single source/checkpoint seed and one sample seed set',
            'base adapter saw all four source domains; leave-source-out applies only to gate fitting',
            'previous six-target outcomes seen: development reevaluation, not an independent blind test',
            'no target labels fitted; unknown dataset improvements are not guaranteed',
            'no graph ablation; BF16 constrained sample8 differs from official ChatTime protocol',
            'ChatTime pretraining exposure cannot be independently verified'])
    atomic(output/'report.json', value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare', action='store_true')
    modes.add_argument('--fit-gate', action='store_true')
    modes.add_argument('--predict', action='store_true')
    modes.add_argument('--report-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    output = Path(args.output)
    if args.prepare:
        from gofa_f.source_gate_state import prepare
        result = prepare(cfg, output)
        print('Prepared fixed source pools; target labels excluded; '+str(len(result['pools']['fit']))+' fitting queries')
    elif args.fit_gate:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise RuntimeError('Standalone CPU fitting must disable CUDA')
        print(fit_gate(cfg, output)['validation'])
    elif args.predict:
        from gofa_f.source_gate_run import run
        run(cfg, output)
    else:
        value = report(output)
        print({k: value[k] for k in ('complete', 'stop_reason', 'goal_achieved', 'source_gate_sha256')})


if __name__ == '__main__':
    main()
