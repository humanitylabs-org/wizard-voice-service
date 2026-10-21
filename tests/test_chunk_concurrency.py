import threading
import time
from pathlib import Path

import pytest

import providers

CONFIG = providers.ProviderConfig(
    name="tinfoil", model="voxtral-small-24b",
    endpoint="http://127.0.0.1:3301/v1/audio/transcriptions",
    api_key_env="TINFOIL_API_KEY", attested=True, language="en", timeout=30,
)


def _chunks(tmp_path: Path, count: int) -> list[Path]:
    paths = []
    for index in range(count):
        path = tmp_path / f"voice-{index:03d}.mp3"
        path.write_bytes(b"x")
        paths.append(path)
    return paths


def test_long_recording_chunks_run_concurrently_and_keep_order(monkeypatch, tmp_path):
    chunks = _chunks(tmp_path, 8)
    monkeypatch.setattr(providers, "normalize_audio_chunks", lambda *a, **k: chunks)
    monkeypatch.setattr(providers, "detect_no_speech", lambda chunk: False)
    monkeypatch.setattr(providers, "CHUNK_CONCURRENCY", 4)
    active = 0
    peak = 0
    lock = threading.Lock()

    def fake(chunk, config, api_key, prompt=""):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        # Reverse completion order to prove the joined text keeps segment order.
        time.sleep(0.05 * (8 - int(chunk.stem[-3:])))
        with lock:
            active -= 1
        i = int(chunk.stem[-3:])
        return f"part{i} shared context part{i+1} shared context"

    monkeypatch.setattr(providers, "_transcribe_chunk", fake)
    started = time.monotonic()
    result = providers.transcribe(Path("in.wav"), CONFIG, api_key="k")
    elapsed = time.monotonic() - started
    assert result.text == " ".join(f"part{i} shared context" for i in range(9))
    assert peak == 4
    assert elapsed < 0.05 * sum(range(1, 9))


def test_first_failing_chunk_by_position_is_reported(monkeypatch, tmp_path):
    chunks = _chunks(tmp_path, 5)
    monkeypatch.setattr(providers, "normalize_audio_chunks", lambda *a, **k: chunks)
    monkeypatch.setattr(providers, "detect_no_speech", lambda chunk: False)
    monkeypatch.setattr(providers, "CHUNK_CONCURRENCY", 4)

    def fake(chunk, config, api_key, prompt=""):
        index = int(chunk.stem[-3:])
        if index in {1, 3}:
            time.sleep(0.05 if index == 1 else 0)
            raise providers.ProviderError("tinfoil returned no transcript")
        return "ok"

    monkeypatch.setattr(providers, "_transcribe_chunk", fake)
    with pytest.raises(providers.ProviderError, match=r"chunk 2/5 failed"):
        providers.transcribe(Path("in.wav"), CONFIG, api_key="k")


def test_stalled_segment_is_resplit_on_same_model(monkeypatch, tmp_path):
    chunks = _chunks(tmp_path, 3)
    monkeypatch.setattr(providers, "normalize_audio_chunks", lambda *a, **k: chunks)
    monkeypatch.setattr(providers, "detect_no_speech", lambda chunk: False)
    monkeypatch.setattr(providers, "PROVIDER_CHUNK_SECONDS", 30)
    monkeypatch.setattr(providers, "_audio_duration", lambda audio: 30.0)

    def split(audio):
        parts = [audio.with_name(audio.stem + f"-split{i}.mp3") for i in range(2)]
        for part in parts:
            part.write_bytes(b"x")
        return parts

    monkeypatch.setattr(providers, "_split_segment", split)
    seen = []

    def fake(chunk, config, api_key, prompt=""):
        seen.append((chunk.stem, config.timeout, config.model))
        if chunk.stem == "voice-001":
            raise providers.ProviderError("tinfoil route unavailable (TimeoutError)")
        return {
            "voice-000": "Start at the old poster",
            "voice-001-split0": "at the old poster then follow the river",
            "voice-001-split1": "then follow the river until the bridge",
            "voice-002": "until the bridge and stop",
        }[chunk.stem]

    monkeypatch.setattr(providers, "_transcribe_chunk", fake)
    result = providers.transcribe(Path("in.wav"), CONFIG, api_key="k")
    assert result.text == "Start at the old poster then follow the river until the bridge and stop"
    assert all(model == "voxtral-small-24b" for _, _, model in seen)
    assert all(timeout <= providers.SEGMENT_ATTEMPT_TIMEOUT for _, timeout, _ in seen)


def test_non_stall_errors_are_not_masked_by_resplit(monkeypatch, tmp_path):
    chunks = _chunks(tmp_path, 1)
    monkeypatch.setattr(providers, "normalize_audio_chunks", lambda *a, **k: chunks)
    monkeypatch.setattr(providers, "detect_no_speech", lambda chunk: False)
    monkeypatch.setattr(providers, "PROVIDER_CHUNK_SECONDS", 30)
    monkeypatch.setattr(providers, "_split_segment", lambda audio: pytest.fail("must not split"))
    monkeypatch.setattr(providers, "_transcribe_chunk", lambda *a, **k: (_ for _ in ()).throw(
        providers.ProviderError("tinfoil returned HTTP 500")))
    with pytest.raises(providers.ProviderError, match="HTTP 500"):
        providers.transcribe(Path("in.wav"), CONFIG, api_key="k")
