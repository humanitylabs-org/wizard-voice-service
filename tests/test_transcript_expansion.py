import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import providers
from config import ProviderConfig

CONFIG = ProviderConfig(name='tinfoil', model='voxtral-small-24b', endpoint='http://127.0.0.1:3301/v1/audio/transcriptions', api_key_env='TEST_KEY', attested=True)


def response(payload):
    result = MagicMock()
    result.__enter__.return_value = result
    result.status = 200
    result.read.return_value = json.dumps(payload).encode()
    return result


def setup(monkeypatch, tmp_path, recovered='Please summarize that book.'):
    audio = tmp_path / 'voice.mp3'
    audio.write_bytes(b'unit-test-audio')
    monkeypatch.setattr(providers, '_audio_duration', lambda path: 14.8, raising=False)
    reply = {'choices': [{'finish_reason': 'stop', 'message': {'content': recovered}}]}
    calls = MagicMock(side_effect=[response({'text': 'invented answer ' * 150}), response(reply)])
    monkeypatch.setattr(providers.urllib.request, 'urlopen', calls)
    return audio, calls


def test_answer_expansion_recovers_same_model_and_attested_origin(monkeypatch, tmp_path):
    audio, calls = setup(monkeypatch, tmp_path)
    assert providers._transcribe_chunk(audio, CONFIG, 'synthetic') == 'Please summarize that book.'
    assert calls.call_count == 2
    request = calls.call_args_list[1].args[0]
    assert request.full_url == 'http://127.0.0.1:3301/v1/chat/completions'
    body = json.loads(request.data)
    assert body['model'] == CONFIG.model
    assert body['temperature'] == 0
    content = body['messages'][0]['content']
    assert content[0]['type'] == 'audio_url'
    assert 'Do not answer' in content[1]['text']
    assert 'invented answer' not in content[1]['text']


@pytest.mark.parametrize('recovered', ['', None, 'more invented text ' * 150, "I'm sorry, but I'm unable to transcribe the audio."])
def test_invalid_recovery_fails_closed(monkeypatch, tmp_path, recovered):
    audio, calls = setup(monkeypatch, tmp_path, recovered)
    with pytest.raises(providers.ProviderError):
        providers._transcribe_chunk(audio, CONFIG, 'synthetic')
    assert calls.call_count == 2


def test_short_genuine_text_is_untouched(monkeypatch, tmp_path):
    audio = tmp_path / 'voice.mp3'
    audio.write_bytes(b'unit-test-audio')
    text = "Here's a brief summary of the book I read."
    calls = MagicMock(return_value=response({'text': text}))
    monkeypatch.setattr(providers.urllib.request, 'urlopen', calls)
    assert providers._transcribe_chunk(audio, CONFIG, 'synthetic') == text
    assert calls.call_count == 1


def test_long_valid_recording_is_untouched(monkeypatch, tmp_path):
    audio, calls = setup(monkeypatch, tmp_path)
    monkeypatch.setattr(providers, '_audio_duration', lambda path: 120)
    assert providers._transcribe_chunk(audio, CONFIG, 'synthetic') == ('invented answer ' * 150).strip()
    assert calls.call_count == 1


def test_truncated_recovery_fails_closed(monkeypatch, tmp_path):
    audio, calls = setup(monkeypatch, tmp_path)
    calls.side_effect = [response({'text': 'invented answer ' * 150}), response({'choices': [{'finish_reason': 'length', 'message': {'content': 'Please summarize'}}]})]
    with pytest.raises(providers.ProviderError):
        providers._transcribe_chunk(audio, CONFIG, 'synthetic')
