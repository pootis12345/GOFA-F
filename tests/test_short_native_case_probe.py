from pathlib import Path
import numpy as np
import pytest

from gofa_f.short_native_case_probe import capture_predict, load_official, numbers, repetition, run


@pytest.fixture
def official():
    root = Path('/home/dcg/dcg_data/GOFA-F/external/ChatTime')
    if not root.exists():
        pytest.skip('Downloaded official source unavailable')
    module,_ = load_official(root)
    return module


def native_fixture(module):
    native = module.ChatTime.__new__(module.ChatTime)
    native.hist_len,native.pred_len,native.max_pred_len,native.num_samples = 48,24,16,8
    native.top_k,native.top_p,native.temperature,native.eos_token_id = 100,1.,1.,2
    native.model,native.tokenizer = object(),object()
    native.discretizer,native.serializer = module.Discretizer(),module.Serializer()
    return native


def fake_pipeline(calls):
    def factory(**kwargs):
        calls.append({k:v for k,v in kwargs.items() if k not in ('model','tokenizer')})
        size = (kwargs['max_new_tokens']-8)//2
        def call(prompt):
            return [{'generated_text':prompt+' '.join(f'###{value:.4f}###' for value in
                np.linspace(-.2,.2,size)+sample*.001)} for sample in range(kwargs['num_return_sequences'])]
        return call
    return factory


def test_official_file_identity_refuses_unverified_source(tmp_path):
    (tmp_path/'model').mkdir()
    (tmp_path/'model/model.py').write_text('raise RuntimeError("must not execute")')
    with pytest.raises(ValueError,match='fixed-commit'):
        load_official(tmp_path)


def test_repetition_counts_do_not_treat_partial_or_nan_as_constant():
    assert repetition([2.,2.,2.])['constant_complete']
    assert repetition([2.,2.,np.nan,2.])['longest_exact_constant_run']==2
    assert not repetition([2.,np.nan])['constant_complete']
    assert repetition([1.,2.,3.])['unique_finite_values']==3
    assert numbers([1.,np.nan,np.inf])==[1.,None,None]


def test_official_sample8_per_block_and_capture_preserves_median(official,monkeypatch):
    calls = []
    monkeypatch.setattr(official,'pipeline',fake_pipeline(calls))
    native = native_fixture(official)
    history = np.arange(48,dtype=np.float32)
    result = capture_predict(native,official,history,'official_sample8')
    assert result['capture_median_matches_official']
    assert [b['horizon'] for b in result['blocks']]==[16,8]
    assert [len(b['samples']) for b in result['blocks']]==[8,8]
    assert [c['max_new_tokens'] for c in calls]==[40,24]
    assert all(c['do_sample'] and c['num_return_sequences']==8 and c['top_k']==100 for c in calls)
    assert len(result['prediction'])==24
    assert result['blocks'][0]['prompt']!=result['blocks'][1]['prompt']
    assert native.hist_len==48 and np.array_equal(history,np.arange(48))


@pytest.mark.parametrize('protocol',['official_greedy1','stage19_prompt_greedy1'])
def test_greedy_controls_change_only_sample_arguments_and_optional_prompt(official,monkeypatch,protocol):
    calls = []
    original_prompt = official.getPrompt
    monkeypatch.setattr(official,'pipeline',fake_pipeline(calls))
    result = capture_predict(native_fixture(official),official,np.arange(48,dtype=np.float32),protocol)
    assert all(not c['do_sample'] and c['num_return_sequences']==1 for c in calls)
    assert all('temperature' not in c and 'top_k' not in c for c in calls)
    assert official.getPrompt is original_prompt
    first_prompt = result['blocks'][0]['prompt']
    assert ('\n\n### Instruction:' in first_prompt)==(protocol=='official_greedy1')
    assert ('\n\n### Response:' in first_prompt)==(protocol=='official_greedy1')


def test_capture_restores_official_functions_if_generation_fails(official,monkeypatch):
    def broken(**kwargs):
        raise RuntimeError('simulated inference failure')
    monkeypatch.setattr(official,'pipeline',broken)
    original = official.getPrompt
    with pytest.raises(RuntimeError,match='simulated'):
        capture_predict(native_fixture(official),official,np.arange(48,dtype=np.float32),'stage19_prompt_greedy1')
    assert official.pipeline is broken and official.getPrompt is original


def test_probe_cannot_run_without_idle_admission(monkeypatch,tmp_path):
    monkeypatch.delenv('GOFA_STAGE19_ADMITTED',raising=False)
    with pytest.raises(RuntimeError,match='fully-idle'):
        run({},tmp_path)
