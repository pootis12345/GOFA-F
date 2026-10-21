"""Stage27 CPU proof then one resource supervisor; no GPU allocation in this controller."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    from gofa_f.allnode_stage import read
    from gofa_f.graph_benchmark import atomic
    from gofa_f.idle_launcher import process_identity
    from gofa_f.generative_data import sha
    from gofa_f.history_target_calibration import configuration
    from gofa_f.history_calibration_state import verify_inference as verify
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    lock = output.with_suffix('.bootstrap.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    snapshot = Path(__file__).resolve().parents[1]
    identity = read(output.with_suffix('.launch.json'), {})
    worker = (read(output/'activity.json', {}) or {}).get('worker')
    if worker or identity.get('pid') and process_identity(identity['pid']) == str(identity.get('start_ticks')):
        raise RuntimeError('Existing Stage27 supervisor/worker; refusing duplicate deployment')
    if read(output/'status.json', {}).get('state') == 'complete':
        raise RuntimeError('Completed Stage27 is sealed; no automatic restart')
    # A sealed calibrator can resume only independent confirmation/test replay.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
        PYTHONPATH=str(snapshot/'src'), PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    def status(state, **fields):
        atomic(output/'bootstrap.json', dict(state=state, pid=os.getpid(),
            start_ticks=process_identity(os.getpid()), updated_at=time.time(), **fields))
    proof = read(output/'cpu_preflight.json')
    if proof is not None:
        verify(configuration(args.config), output)
        if not proof['passed'] or proof['manifest_sha256'] != sha(output/'manifest.json') \
                or not proof.get('inference_core_proof_reused') or not proof['cuda_disabled']:
            raise ValueError('Existing CPU proof is not a complete bound proof')
        status('complete_bound_cpu_proof_reused', cpu_proof_sha256=sha(output/'cpu_preflight.json'))
    else:
        status('cpu_preflight_running')
        result = subprocess.run([sys.executable, '-u', str(snapshot/'scripts/check_history_calibration_cpu.py'),
            '--config', args.config, '--output', str(output)], cwd=snapshot, env=env)
        if result.returncode:
            status('failed', returncode=result.returncode)
            return result.returncode
    # CPU development and the all-six gate run before even querying GPU admission.
    from gofa_f.history_calibration_cpu import develop, seal, report
    cfg = configuration(args.config)
    gate = develop(cfg, output)
    if not gate['all_six_strictly_improved']:
        status('complete_cpu_gate_failed_no_gpu')
        report(output)
        return 0
    seal(cfg, output)
    identity = read(output.with_suffix('.launch.json'), {})
    if identity.get('pid') and process_identity(identity['pid']) == str(identity.get('start_ticks')):
        raise RuntimeError('Another launcher appeared during CPU calibration')
    log = output.with_suffix('.log').open('ab', buffering=0)
    command = [sys.executable, '-u', '-m', 'gofa_f.short_launcher', '--output', str(output),
               '--config', args.config, '--snapshot', str(snapshot), '--worker-mode', 'history_target_calibration']
    process = subprocess.Popen(command, cwd=snapshot, env=env, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True, stdin=subprocess.DEVNULL)
    tick = process_identity(process.pid)
    if not tick:
        raise RuntimeError('Supervisor exited before deployment identity was recorded')
    atomic(output.with_suffix('.launch.json'), dict(pid=process.pid, start_ticks=tick,
        command=command, deployed_at=time.time(), snapshot=str(snapshot), worker_mode='history_target_calibration'))
    status('cpu_passed_resource_supervisor_deployed', launcher_pid=process.pid, launcher_start_ticks=tick)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
