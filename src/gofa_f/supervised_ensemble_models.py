"""CPU-only fixed HGB fitting, atomic artifacts, deterministic prediction caches."""
from contextlib import redirect_stdout
import os
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from gofa_f.allnode_stage import read
from gofa_f.generative_data import Series, sha
from gofa_f.graph_benchmark_data import TARGETS
from gofa_f.history_calibration_state import verify_data
from gofa_f.prefix_cache import atomic_json
from gofa_f.six_calibration_cpu import no_worker
from gofa_f.six_calibration_state import immutable
from gofa_f.supervised_ensemble_data import FitLog, features, training_arrays
from gofa_f.supervised_ensemble_state import MODEL_SETTINGS, own_manifest, status
from gofa_f.target_calibration import cpu_only
from gofa_f.target_calibration_state import digest


class CPUCharge:
    def __init__(self, output):
        self.path = output/'cpu_activity.json'
        self.value = read(self.path, dict(spent_seconds=0.))
        now = time.monotonic()
        old = self.value.get('active')
        if old:
            tail = min(5., max(0., now-old['last_monotonic']))
            self.value['spent_seconds'] += tail
            self.value['conservative_unclean_tail_seconds'] = self.value.get('conservative_unclean_tail_seconds', 0.)+tail
        self.last = self.saved = now
        self.value['active'] = dict(pid=os.getpid(), last_monotonic=now)
        atomic_json(self.path, self.value)

    def tick(self, force=False):
        now = time.monotonic()
        self.value['spent_seconds'] += max(0., now-self.last)
        self.last = now
        self.value['active']['last_monotonic'] = now
        if force or now-self.saved >= 5:
            atomic_json(self.path, self.value)
            self.saved = now

    def close(self):
        self.tick(True)
        self.value['active'] = None
        atomic_json(self.path, self.value)


def proof_required(output):
    proof = read(output/'cpu_preflight.json', {})
    if not proof.get('passed') or not proof.get('cuda_disabled') or proof['manifest_sha256'] != sha(output/'manifest.json'):
        raise ValueError('Bound complete CPU acceptance required')


def model_binding(manifest, name):
    return digest(dict(code=manifest['code'], configuration=manifest['configuration'], source=name,
        queries=manifest['training_queries'][name], raw=manifest['raw_hashes'][name],
        metadata=manifest['metadata_hashes'][name], features=manifest['feature_names'][name]))


def load_model(output, manifest, name):
    metadata = read(output/'models'/(name+'.json'))
    path = output/'models'/(name+'.joblib')
    if not metadata or metadata['binding'] != model_binding(manifest, name) \
            or metadata['model_sha256'] != sha(path) or metadata['sklearn_version'] != '1.5.2':
        raise ValueError('CPU model/input binding changed: '+name)
    model = joblib.load(path)
    if not isinstance(model, HistGradientBoostingRegressor) or model.n_iter_ != 150 \
            or model.n_features_in_ != len(manifest['feature_names'][name]) \
            or any(model.get_params()[k] != v for k, v in MODEL_SETTINGS.items()):
        raise ValueError('CPU model type/rounds/hyperparameters changed')
    return model, metadata


def train_models(cfg, output):
    cpu_only()
    no_worker(output)
    if os.environ.get('OMP_NUM_THREADS') != '2' or os.environ.get('MKL_NUM_THREADS') != '2':
        raise RuntimeError('CPU training requires OMP/MKL2')
    import fcntl
    stream = (output/'cpu_training.lock').open('a')
    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = own_manifest(cfg, output)
    verify_data(manifest)
    proof_required(output)
    if (output/'coefficient_lock.json').exists():
        raise ValueError('Cannot train after ensemble sealing')
    if read(output/'status.json', {}).get('state') in ('complete', 'failed'):
        raise ValueError('Terminal Stage29 cannot automatically retrain')
    (output/'models').mkdir(exist_ok=True)
    cost = CPUCharge(output)
    completed, durations, last_write = 0, [], [0.]
    all_models = {}
    try:
        for name in TARGETS:
            if (output/'models'/(name+'.json')).exists():
                _, metadata = load_model(output, manifest, name)
                durations.append(metadata['cpu_wall_seconds'])
                completed += 1
                all_models[name] = metadata['model_sha256']
                continue
            start = time.monotonic()
            status(output, 'running', 'cpu_training', dataset=name, cpu_models_done=completed)
            def notify(done, total, stage='cpu_features', force=False):
                cost.tick()
                now = time.monotonic()
                if force or done == total or now-last_write[0] >= 5:
                    atomic_json(output/'progress.json', dict(state=stage, dataset=name, cpu_models_done=completed,
                        cpu_models_total=6, preprocessing_done=done if stage == 'cpu_features' else 4096,
                        preprocessing_total=total if stage == 'cpu_features' else 4096,
                        boosting_iteration=done if stage == 'cpu_training' else 0, boosting_total=150,
                        cpu_phase_eta_seconds=(now-start)/done*(total-done) if done else None,
                        overall_eta_seconds=max(0., float(np.mean(durations))*(6-completed)-(now-start))
                            +manifest['initial_confirmation_estimate_seconds'] if durations else None,
                        eta_basis='remaining CPU models plus conditional GPU confirmation; unknown pauses excluded',
                        conditional_confirmation_estimate_seconds=manifest['initial_confirmation_estimate_seconds'], updated_at=time.time()))
                    last_write[0] = now
            data = Series(Path(manifest['data_directory'])/'targets'/(name+'.npz'))
            x, y, weights, stats = training_arrays(data, manifest['training_queries'][name], notify)
            model = HistGradientBoostingRegressor(**MODEL_SETTINGS, verbose=1)
            with redirect_stdout(FitLog(sys.stdout, lambda i, total, _now: notify(i, total, 'cpu_training'))):
                model.fit(x, y, sample_weight=weights)
            if model.n_iter_ != 150:
                raise ValueError('CPU model stopped before150 fixed rounds')
            path = output/'models'/(name+'.joblib')
            temporary = path.with_suffix('.joblib.part')
            joblib.dump(model, temporary)
            reloaded = joblib.load(temporary)
            probe = x[:min(128, len(x))]
            before, after = model.predict(probe), reloaded.predict(probe)
            if not np.allclose(before, after, rtol=1e-6, atol=1e-6):
                raise ValueError('CPU serialized model replay differs')
            temporary.replace(path)
            metadata = dict(binding=model_binding(manifest, name), model_sha256=sha(path),
                manifest_sha256=sha(output/'manifest.json'), sklearn_version='1.5.2', settings=MODEL_SETTINGS,
                source=name, supervised_target_labels=True, neural_training=False, stats=stats,
                cpu_wall_seconds=time.monotonic()-start, saved_prediction_max_abs=float(np.abs(before-after).max()),
                training_queries=4096, rounds=150)
            immutable(output/'models'/(name+'.json'), metadata)
            completed += 1
            all_models[name] = metadata['model_sha256']
            durations.append(metadata['cpu_wall_seconds'])
            notify(150, 150, 'cpu_training', True)
        value = dict(binding=sha(output/'manifest.json'), models=all_models, complete=True, cpu_models=6,
            target_supervised=True, neural_optimizer_updates=0)
        immutable(output/'models_complete.json', value)
        status(output, 'cpu_models_complete', 'cpu_training', cpu_models_done=6)
        return value
    finally:
        cost.close()


def require_models(output, manifest):
    marker = read(output/'models_complete.json')
    if not marker or not marker['complete'] or marker['binding'] != sha(output/'manifest.json') \
            or set(marker['models']) != set(TARGETS):
        raise ValueError('All six fixed CPU models required')
    for name, expected in marker['models'].items():
        path = output/'models'/(name+'.joblib')
        metadata = read(output/'models'/(name+'.json'))
        if sha(path) != expected or metadata['model_sha256'] != expected \
                or metadata['binding'] != model_binding(manifest, name):
            raise ValueError('Sealed CPU model changed')
    return marker


def cpu_predictions(output, manifest, name, queries, phase):
    require_models(output, manifest)
    if phase not in ('development', 'confirm', 'test'):
        raise ValueError('Explicit CPU prediction subset required')
    root = Path(manifest['data_directory'])/'targets'
    if sha(root/(name+'.npz')) != manifest['raw_hashes'][name] \
            or sha(root/(name+'.json')) != manifest['metadata_hashes'][name]:
        raise ValueError('CPU prediction source data changed')
    if phase != 'development':
        lock, post = read(output/'coefficient_lock.json'), read(output/'postfit_gate.json')
        if not lock or not post or not post['passed'] or lock['coefficients_sha256'] != sha(output/'coefficients.json'):
            raise ValueError('CPU confirmation/test requires sealed nonidentity coefficients')
        if phase == 'test':
            gate, admission = read(output/'confirm_gate.json'), read(output/'test_admission.json')
            if not gate or not admission or not gate['all_six_strictly_improved'] \
                    or gate['coefficients_sha256'] != lock['coefficients_sha256'] \
                    or admission['confirm_gate_sha256'] != sha(output/'confirm_gate.json'):
                raise ValueError('CPU test predictions require all six confirmations and test admission')
    marker = read(output/'models_complete.json')
    binding = digest(dict(manifest=sha(output/'manifest.json'), model=marker['models'][name],
        queries=queries, phase=phase, schema=manifest['feature_names'][name]))
    path = output/'cpu_predictions'/f'{phase}-{name}.json'
    cached = read(path)
    if cached is not None:
        if cached['binding'] != binding or cached['queries'] != queries:
            raise ValueError('CPU prediction cache rebound')
        result = [np.asarray(v, np.float64) for v in cached['predictions']]
        if len(result) != len(queries) or any(v.shape != (q['horizon'],) or not np.isfinite(v).all() for v, q in zip(result, queries)):
            raise ValueError('Invalid cached CPU predictions')
        return result
    if phase == 'development' and (output/'coefficient_lock.json').exists():
        raise ValueError('Cannot add development predictions after coefficient sealing')
    data = Series(Path(manifest['data_directory'])/'targets'/(name+'.npz'))
    model, _ = load_model(output, manifest, name)
    result = []
    for q in queries:
        x, last = features(data, q)
        own = last+model.predict(x)
        if own.shape != (q['horizon'],) or not np.isfinite(own).all():
            raise ValueError('Nonfinite CPU prediction; never fill using future data')
        result.append(own)
    path.parent.mkdir(exist_ok=True)
    immutable(path, dict(binding=binding, queries=queries, predictions=[v.tolist() for v in result],
        model_sha256=marker['models'][name], supervised_target_model=True, generated_text=False))
    return result
