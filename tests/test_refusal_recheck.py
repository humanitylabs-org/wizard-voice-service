"""Refusal-classified chunks get one verbatim recheck on the same attested model.

Live failure: genuine dictation that mentions assistant-meta phrases
("transcribing audio", "access the link") was rejected on every retry,
so the phone showed "could not transcribe" forever.
"""
import json
from unittest.mock import MagicMock

import pytest

import providers
from config import ProviderConfig

CONFIG = ProviderConfig(name='tinfoil', model='voxtral-small-24b',
                        endpoint='http://127.0.0.1:3301/v1/audio/transcriptions',
                        api_key_env='TEST_KEY', attested=True)


def _resp(payload):
    result = MagicMock()
    result.__enter__.return_value = result
    result.status = 200
    result.read.return_value = json.dumps(payload).encode()
    return result


def _chat(text):
    return _resp({'choices': [{'finish_reason': 'stop', 'message': {'content': text}}]})


def _setup(monkeypatch, tmp_path, first, recheck):
    audio = tmp_path / 'voice.mp3'
    audio.write_bytes(b'unit-test-audio')
    monkeypatch.setattr(providers, '_audio_duration', lambda path: 9.0)
    responses = [_resp({'text': first})]
    if recheck is not None:
        responses.append(recheck if isinstance(recheck, Exception) else _resp({'text': recheck}))
    calls = MagicMock(side_effect=responses)
    monkeypatch.setattr(providers.urllib.request, 'urlopen', calls)
    return audio, calls


SPOKEN = "Our synthetic example mentions transcribing audio while describing a recording workflow"


def test_genuine_speech_with_assistant_phrase_is_kept_when_recheck_agrees(monkeypatch, tmp_path):
    assert providers.is_assistant_refusal(SPOKEN)
    audio, calls = _setup(monkeypatch, tmp_path, SPOKEN, SPOKEN + '.')
    text = providers._transcribe_chunk(audio, CONFIG, 'synthetic')
    assert text == SPOKEN
    assert isinstance(text, providers.VerifiedSpeech)
    assert calls.call_count == 2
    request = calls.call_args_list[1].args[0]
    assert request.full_url == CONFIG.endpoint
    assert b'name="model"\r\n\r\nvoxtral-small-24b' in request.data
    assert b'Verbatim dictation.' in request.data
    assert SPOKEN.encode() not in request.data


def test_assistant_answer_replaced_by_clean_verbatim_recheck(monkeypatch, tmp_path):
    answer = "Sure, I'd be happy to help with that! Here are some steps to get started."  # 15 words
    audio, _ = _setup(monkeypatch, tmp_path, answer, 'Can you help me get started?')
    assert providers._transcribe_chunk(audio, CONFIG, 'synthetic') == 'Can you help me get started?'


OPENERS = [
    "I'm sorry, but I'm unable to provide a transcription of this audio.",
    "Sorry, I'm not sure what you're asking. Could you please provide more context?",
]


@pytest.mark.parametrize('first', OPENERS)
def test_refusal_opener_is_replaced_by_clean_second_reading(monkeypatch, tmp_path, first):
    # Live failure: a short question got an apology on the first reading and
    # was rejected on every Retry with no second attempt.
    audio, calls = _setup(monkeypatch, tmp_path, first, 'Can we compare the two example diagrams?')
    assert providers._transcribe_chunk(audio, CONFIG, 'synthetic') == 'Can we compare the two example diagrams?'
    assert calls.call_count == 2


@pytest.mark.parametrize('first', OPENERS)
def test_refusal_openers_never_accepted_even_if_recheck_repeats(monkeypatch, tmp_path, first):
    audio, _ = _setup(monkeypatch, tmp_path, first, first)
    with pytest.raises(providers.ProviderError, match='assistant_refusal'):
        providers._transcribe_chunk(audio, CONFIG, 'synthetic')


def test_disagreeing_recheck_fails_closed_without_echo(monkeypatch, tmp_path):
    first = "Sure, I'd be happy to help with that! Feel free to ask anything."
    audio, _ = _setup(monkeypatch, tmp_path, first, "I'm here to help! Feel free to ask about transcribing audio.")
    with pytest.raises(providers.ProviderError) as info:
        providers._transcribe_chunk(audio, CONFIG, 'synthetic')
    assert str(info.value) == 'tinfoil returned assistant_refusal'


@pytest.mark.parametrize('recheck', [OSError('down'), TimeoutError('slow')])
def test_recheck_transport_failure_keeps_refusal_contract(monkeypatch, tmp_path, recheck):
    audio, _ = _setup(monkeypatch, tmp_path, SPOKEN, recheck)
    with pytest.raises(providers.ProviderError, match='assistant_refusal'):
        providers._transcribe_chunk(audio, CONFIG, 'synthetic')


def test_unattested_route_does_not_recheck(monkeypatch, tmp_path):
    other = ProviderConfig(name='openai-compatible', model='whisper-1',
                           endpoint='https://example.invalid/v1/audio/transcriptions',
                           api_key_env='TEST_KEY', attested=False)
    audio, calls = _setup(monkeypatch, tmp_path, SPOKEN, None)
    with pytest.raises(providers.ProviderError, match='assistant_refusal'):
        providers._transcribe_chunk(audio, other, 'synthetic')
    assert calls.call_count == 1


def test_whole_job_accepts_verified_chunk(monkeypatch, tmp_path):
    monkeypatch.setattr(providers, 'detect_no_speech', lambda path: False)
    def normalize(source, directory):
        path = directory / 'chunk.mp3'
        path.write_bytes(b'x')
        return [path]
    monkeypatch.setattr(providers, 'normalize_audio_chunks', normalize)
    _setup(monkeypatch, tmp_path, SPOKEN, SPOKEN)
    source = tmp_path / 'in.wav'
    source.write_bytes(b'x')
    result = providers.transcribe(source, CONFIG, api_key='synthetic')
    assert result.text.startswith('Our synthetic example')


@pytest.mark.parametrize('second', [
    "I'm ready to transcribe the audio. Please play the audio.",
    "Sure, I'd be happy to help with transcribing audio from your phone today.",
])
def test_assistant_voice_recheck_never_accepted(monkeypatch, tmp_path, second):
    audio, _ = _setup(monkeypatch, tmp_path, SPOKEN, second)
    with pytest.raises(providers.ProviderError, match='assistant_refusal'):
        providers._transcribe_chunk(audio, CONFIG, 'synthetic')


def test_agreeing_but_implausibly_long_answer_rejected(monkeypatch, tmp_path):
    long_answer = "You can access the link and browse the internet for the knowledge cutoff. " * 4
    audio, _ = _setup(monkeypatch, tmp_path, long_answer, long_answer)
    monkeypatch.setattr(providers, '_audio_duration', lambda path: 3.0)
    with pytest.raises(providers.ProviderError, match='assistant_refusal'):
        providers._transcribe_chunk(audio, CONFIG, 'synthetic')


def test_short_stalled_segment_retries_once_with_neutral_prompt(monkeypatch, tmp_path):
    audio = tmp_path / 'voice.mp3'
    audio.write_bytes(b'unit-test-audio')
    monkeypatch.setattr(providers, '_audio_duration', lambda path: 5.0)
    monkeypatch.setattr(providers, 'PROVIDER_CHUNK_SECONDS', 30)
    calls = MagicMock(side_effect=[TimeoutError('stalled'), _resp({'text': 'Is the sample ready?'})])
    monkeypatch.setattr(providers.urllib.request, 'urlopen', calls)
    assert providers._transcribe_segment(audio, CONFIG, 'synthetic') == 'Is the sample ready?'
    assert calls.call_count == 2
    assert b'Verbatim dictation.' in calls.call_args_list[1].args[0].data


def test_short_segment_that_stalls_twice_fails(monkeypatch, tmp_path):
    audio = tmp_path / 'voice.mp3'
    audio.write_bytes(b'unit-test-audio')
    monkeypatch.setattr(providers, '_audio_duration', lambda path: 5.0)
    monkeypatch.setattr(providers, 'PROVIDER_CHUNK_SECONDS', 30)
    monkeypatch.setattr(providers.urllib.request, 'urlopen', MagicMock(side_effect=[TimeoutError('a'), TimeoutError('b')]))
    with pytest.raises(providers.ProviderError, match='TimeoutError'):
        providers._transcribe_segment(audio, CONFIG, 'synthetic')
