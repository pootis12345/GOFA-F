import copy
import json

import pytest

from gofa_f.scada_wind_iteration import confirmation_fingerprint, sealed_confirmation
from gofa_f.scada_wind_recovery import numeric_replay, source_exited
from gofa_f.generative_data import sha


def record():
    return dict(binding='B', coefficients_sha256='C', inputs_sha256='D', refitted=False,
        comparison=dict(strictly_improved=True,
            metrics=dict(native=dict(macro=dict(mae=10.)), calibrated=dict(macro=dict(mae=9.)))))


def test_reuse_preserves_exact_artifact_and_checks_original_strict_gate():
    original = record()
    assert sealed_confirmation(original, 'B', 'C', 'D') is original
    assert sealed_confirmation(copy.deepcopy(original), 'B', 'C', 'D') == original
    for key, value in [('binding', 'bad'), ('coefficients_sha256', 'bad'), ('inputs_sha256', 'bad')]:
        altered = copy.deepcopy(original)
        altered[key] = value
        with pytest.raises(ValueError, match='binding changed'):
            sealed_confirmation(altered, 'B', 'C', 'D')
    altered = copy.deepcopy(original)
    altered['comparison']['strictly_improved'] = False
    with pytest.raises(ValueError, match='strict rule'):
        sealed_confirmation(altered, 'B', 'C', 'D')


def test_numeric_replay_does_not_relax_state_or_cpu_tolerance():
    a = dict(error=1., count=64, passed=True)
    b = dict(error=1.+1e-14, count=64, passed=True)
    assert numeric_replay(a, b) < 1e-12
    with pytest.raises(ValueError, match='tolerance'):
        numeric_replay(a, dict(a, error=1.01))
    with pytest.raises(ValueError, match='state/gate'):
        numeric_replay(a, dict(a, passed=False))
    with pytest.raises(ValueError, match='state/gate'):
        numeric_replay(a, dict(a, count=63))


def test_confirm_dataset_validates_inputs_without_recomputing_forests(tmp_path, monkeypatch):
    import gofa_f.scada_wind_iteration as module
    m = dict(queries={'pems03': [{}]*128}, raw_hashes={'pems03': 'data'},
             metadata_hashes={'pems03': 'metadata'}, unchanged_tensor_files={})
    p = dict(binding='B', coefficients={'pems03': dict(full=dict(weights=dict(gofa=0.)))})
    lock = dict(coefficients_sha256='C')
    (tmp_path/'manifest.json').write_text('{}')
    folder = tmp_path/'evaluations/confirm-pems03-native'
    folder.mkdir(parents=True)
    (folder/'complete.json').write_text('{"complete":true}')
    hashes = {str(folder/'complete.json'): sha(folder/'complete.json')}
    fingerprint = confirmation_fingerprint(tmp_path, m, p, lock, 'pems03', hashes)
    original = dict(record(), inputs_sha256=fingerprint)
    (tmp_path/'confirmation').mkdir()
    (tmp_path/'confirmation/pems03.json').write_text(json.dumps(original))
    monkeypatch.setattr(module, 'load_coefficients', lambda *_: (m, p, lock))
    monkeypatch.setattr(module, 'cpu_only', lambda: None)
    monkeypatch.setattr(module, 'no_worker', lambda *_: None)
    monkeypatch.setattr(module, 'read_shard', lambda *_: {})
    monkeypatch.setattr(module, 'inference_binding', lambda *_: 'sample-binding')
    monkeypatch.setattr(module, 'compare', lambda *_: (_ for _ in ()).throw(AssertionError('must not recompute')))
    assert module.confirm_dataset({}, tmp_path, 'pems03') == original
    (folder/'complete.json').write_text('{"complete":true,"changed":true}')
    with pytest.raises(ValueError, match='binding changed'):
        module.confirm_dataset({}, tmp_path, 'pems03')


def test_fingerprint_includes_scientific_parameters_and_data(tmp_path):
    (tmp_path/'manifest.json').write_text('{}')
    m = dict(raw_hashes={'wind': 'data'}, metadata_hashes={'wind': 'meta'}, unchanged_tensor_files={'core': 'sha'})
    p = dict(coefficients={'wind': dict(weight=.25, model_sha256='model')})
    lock = dict(coefficients_sha256='C')
    a = confirmation_fingerprint(tmp_path, m, p, lock, 'wind', {})
    changed = copy.deepcopy(p)
    changed['coefficients']['wind']['weight'] = .5
    assert confirmation_fingerprint(tmp_path, m, changed, lock, 'wind', {}) != a


def test_recovery_refuses_nonfinalized_or_unknown_source(tmp_path):
    (tmp_path/'activity.json').write_text('{"worker":null}')
    (tmp_path/'launcher.json').write_text('{"state":"complete"}')
    (tmp_path/'status.json').write_text('{"state":"complete"}')
    with pytest.raises(ValueError, match='Only exited'):
        source_exited(tmp_path)
