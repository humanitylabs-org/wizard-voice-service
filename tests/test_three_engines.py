import io
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient
from batch_audio import pause_spans
from config import load_provider_config, ProviderConfigurationError
from local_engine_api import create_engine_app
from isolated_batch import request_local
from providers import ProviderError, route_transcription


def test_pause30_full_tail_and_no_gaps():
    samples = np.ones(361*16000+137, dtype=np.float32)
    samples[27*16000:28*16000] = 0
    spans = pause_spans(samples)
    assert spans[0][0] == 0 and spans[-1][1] == len(samples)
    assert all(b == c for (_, b), (c, _) in zip(spans, spans[1:]))
    assert all(0 < b-a <= 30*16000 for a, b in spans)
    assert 27*16000 < spans[0][1] < 28*16000


def test_short_partial_frame():
    assert pause_spans(np.ones(7)) == [(0,7)]
    assert pause_spans(np.zeros(0)) == []


def test_silence_ties_choose_latest():
    spans = pause_spans(np.zeros(61*16000))
    assert spans[0] == (0, 30*16000-320)


def fixture_app():
    class Runtime:
        receipt = {'status':'ready','model':'fixture','provider':'fixture','instance':'fixture'}
        def transcribe(self, data):
            return dict(self.receipt, text='' if data==b'silent' else 'known words')
    return create_engine_app(Runtime, 'fixture')


def test_openai_shape_readiness_wrong_model_and_silence():
    with TestClient(fixture_app()) as client:
        assert client.get('/v1/models').json()['data'][0]['task'] == 'automatic-speech-recognition'
        assert client.get('/health').json()['status'] == 'ready'
        for model, status in [('fixture',200),('other',400)]:
            response = client.post('/v1/audio/transcriptions', data={'model':model}, files={'file':('audio.wav',b'silent')})
            assert response.status_code == status
        assert client.post('/v1/audio/transcriptions', data={'model':'fixture'}, files={'file':('x',b'silent')}).json()['text'] == ''
        assert client.post('/v1/audio/transcriptions', data={'model':'fixture','prompt':'rewrite'}, files={'file':('x',b'a')}).status_code == 400


def test_local_profiles_explicit_and_no_credential(tmp_path, monkeypatch):
    root = yaml.safe_load((Path(__file__).parents[1]/'examples/three-engines.yaml').read_text())
    for name in ['parakeet-local','voxtral-local']:
        root['stt']['active_profile'] = name
        p=tmp_path/'config.yaml';p.write_text(yaml.safe_dump(root))
        monkeypatch.setenv('WIZARD_VOICE_CONFIG',str(p))
        with patch('providers.load_api_key', side_effect=AssertionError('cloud key loaded')), patch('providers.transcribe') as invoke:
            route_transcription(tmp_path/'audio.wav')
            assert invoke.call_args.kwargs['api_key'] == ''
        cfg=load_provider_config(p)
        assert cfg.batch_policy=='pause30' and cfg.privacy=='local-host'
        for url in ['https://example.com','http://127.0.0.1','http://user:pass@127.0.0.1:9','http://127.0.0.1:9/path']:
            bad=yaml.safe_load(yaml.safe_dump(root));bad['stt']['profiles'][name]['base_url']=url
            p.write_text(yaml.safe_dump(bad))
            with pytest.raises(ProviderConfigurationError):load_provider_config(p)


def test_local_down_no_cloud_fallback(tmp_path):
    from config import ProviderConfig
    p=tmp_path/'a.wav';p.write_bytes(b'a')
    cfg=ProviderConfig(name='local-engine',model='parakeet-tdt-0.6b-v3',endpoint='http://127.0.0.1:9/v1/audio/transcriptions',api_key_env='',attested=False)
    with patch('providers._open_transcription',side_effect=ConnectionRefusedError),patch('providers.load_api_key',side_effect=AssertionError):
        with pytest.raises(ProviderError,match='no fallback'):request_local(p,cfg)


def test_local_wrong_identity_fails(tmp_path):
    from config import ProviderConfig
    cfg=ProviderConfig(name='local-engine',model='parakeet-tdt-0.6b-v3',endpoint='http://127.0.0.1:9/v1/audio/transcriptions',api_key_env='',attested=False)
    p=tmp_path/'a.wav';p.write_bytes(b'a')
    response=SimpleNamespace(read=lambda _:b'{"text":"words","model":"wrong","provider":"parakeet","instance":"a"}')
    from contextlib import nullcontext
    with patch('providers._open_transcription',return_value=nullcontext(response)):
        with pytest.raises(ProviderError,match='no fallback'): request_local(p,cfg)


def test_scoring_deletions_endings_and_silence():
    import importlib.util
    spec=importlib.util.spec_from_file_location('benchmark',Path(__file__).parents[1]/'benchmarks/three_engines.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    s=module.score('one two three four','one two')
    assert s['deletions']==2 and s['omitted_trailing_words']==2 and s['wer']==0.5
    assert not s['ending_last12_exact']
    s=module.score('','invented sentence')
    assert s['hallucinated_words_on_empty_reference']==2 and s['wer'] is None
    s=module.score("Don't delete the blue folder.", "don't delete the blue folder")
    assert s['wer']==0 and s['ending_last12_exact']
