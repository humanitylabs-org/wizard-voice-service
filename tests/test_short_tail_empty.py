"""Regression: a recording slightly past the 30 s segment boundary.

Live failure (9 Oct 2026): a 30.66 s phone note split into 30.02 s + 0.72 s.
The tail was near-silent, Voxtral returned empty text, and every retry failed
with "chunk 2/2 failed: tinfoil returned no transcript" (Wizard showed 502).
"""
from pathlib import Path

import pytest

import providers

CONFIG = providers.ProviderConfig(
    name="tinfoil", model="voxtral-small-24b",
    endpoint="http://127.0.0.1:3301/v1/audio/transcriptions",
    api_key_env="TINFOIL_API_KEY", attested=True, language="en", timeout=30,
)


def _setup(monkeypatch, tmp_path, count, tail_seconds):
    chunks = []
    for index in range(count):
        path = tmp_path / f"voice-{index:03d}.mp3"
        path.write_bytes(b"x")
        chunks.append(path)
    monkeypatch.setattr(providers, "normalize_audio_chunks", lambda *a, **k: chunks)
    monkeypatch.setattr(providers, "detect_no_speech", lambda chunk: False)
    monkeypatch.setattr(providers, "_audio_duration",
                        lambda audio: tail_seconds if audio == chunks[-1] else 30.0)

    def fake(chunk, config, api_key, prompt=""):
        if chunk == chunks[-1]:
            raise providers.ProviderError("tinfoil returned no transcript")
        return f"part{int(chunk.stem[-3:])}"

    monkeypatch.setattr(providers, "_transcribe_chunk", fake)
    return chunks


def test_empty_short_tail_is_silence(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, 2, 0.72)
    result = providers.transcribe(Path("in.webm"), CONFIG, api_key="k")
    assert result.text == "part0"


def test_empty_long_final_chunk_still_fails(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, 2, 12.0)
    with pytest.raises(providers.ProviderError, match=r"chunk 2/2 failed"):
        providers.transcribe(Path("in.webm"), CONFIG, api_key="k")


def test_single_short_empty_note_still_fails(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, 1, 0.72)
    with pytest.raises(providers.ProviderError, match=r"returned no transcript"):
        providers.transcribe(Path("in.webm"), CONFIG, api_key="k")
