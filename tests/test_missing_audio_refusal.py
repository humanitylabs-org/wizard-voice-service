"""Observed same-model recovery boilerplate must not pass as dictation."""
from unittest.mock import Mock

import pytest
import providers
from stt_refusal import is_assistant_refusal

REFUSAL = "I'm sorry, but I can't find any audio to transcribe. If you have any text or audio that you'd like me to transcribe, please provide it, and I'll be happy to help."


@pytest.mark.parametrize('text', [REFUSAL, "I cannot find the audio to transcribe.", "I can’t find audio to transcribe."])
def test_missing_audio_recovery_is_a_refusal(text):
    assert is_assistant_refusal(text)
    assert providers.is_refusal_opener(text)


@pytest.mark.parametrize('text', ["I'm sorry, but I can't find the audio cable.", "I cannot find any audio recordings from yesterday."])
def test_normal_missing_item_speech_is_not_refusal(text):
    assert not is_assistant_refusal(text)


def test_expansion_recovery_cannot_return_missing_audio_boilerplate(monkeypatch,tmp_path):
    config=providers.ProviderConfig('tinfoil','voxtral-small-24b',
        'http://127.0.0.1:3301/v1/audio/transcriptions','TINFOIL_API_KEY',True)
    monkeypatch.setattr(providers,'_verbatim_recovery',Mock(return_value=REFUSAL))
    with pytest.raises(providers.ProviderError,match='expansion recovery failed'):
        providers._recover_expanded_answer(tmp_path/'original.mp3',config,'test',14.8,30)
