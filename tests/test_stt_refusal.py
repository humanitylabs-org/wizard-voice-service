"""Synthetic policy cases only; no production transcripts or recordings."""
import pytest

from stt_refusal import is_assistant_refusal

# Eight invented suffixes exercise the common prefix shared by all eight diagnosed
# production variants. We deliberately do not retrieve the private originals.
REFUSALS = ["I'm sorry, but I'm unable to provide" + suffix for suffix in (
    " a response to that question.",
    " a transcript for the audio.",
    " assistance with this request.",
    " an answer.",
    " the requested information.",
    " a transcription.",
    " any further help.",
    ".",
)]


@pytest.mark.parametrize("text", REFUSALS)
def test_all_suffix_variants(text):
    assert is_assistant_refusal(text)


@pytest.mark.parametrize("text", [
    "  I’m sorry, but I’m unable to provide a transcript.\n",
    "I am sorry but I am unable to provide assistance.",
    "I'M SORRY, BUT I'M UNABLE TO PROVIDE.",
    "\nI'm\tsorry,\n but I'm unable  to provide something.",
])
def test_normalization(text):
    assert is_assistant_refusal(text)


@pytest.mark.parametrize("text", [
    "", None, {}, "Synthetic meeting text.",
    "I'm sorry we are late.",
    "I'm sorry, but I'm able to provide the file.",
    "I'm sorry, but I'm unable to attend.",
    "I'm sorry, but I'm unable to provider.",
    "They said I'm sorry, but I'm unable to provide a response.",
    '"I\'m sorry, but I\'m unable to provide" was the test string.',
])
def test_near_misses_remain_transcript(text):
    assert not is_assistant_refusal(text)
