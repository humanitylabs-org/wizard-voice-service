import io
import threading
import wave
from types import SimpleNamespace
import numpy as np
import pytest
from fastapi import HTTPException
from parakeet_adapter import Runtime


def wave_bytes(seconds):
    data = np.zeros(int(seconds*16000), dtype=np.int16)
    b = io.BytesIO()
    with wave.open(b, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(data.tobytes())
    return b.getvalue()


def runtime():
    r = Runtime.__new__(Runtime)
    r.lock = threading.Lock()
    r.receipt = {'instance': 'fixture-only'}
    lengths = []
    def recognize(x, *, sample_rate):
        assert sample_rate == 16000
        lengths.append(len(x))
        return 'spoken words'
    r.model = SimpleNamespace(recognize=recognize)
    return r, lengths


def test_adapter_contiguous_coverage():
    r, lengths = runtime()
    result = r.transcribe(wave_bytes(61))
    assert sum(lengths) == 61*16000
    assert max(lengths) <= 30*16000
    assert result['text'] == 'spoken words spoken words spoken words'
    assert not r.lock.locked()


def test_adapter_oversize_never_silently_truncates():
    r, lengths = runtime()
    with pytest.raises(HTTPException) as exc:
        r.transcribe(wave_bytes(901))
    assert exc.value.status_code == 413 and lengths == []
    assert not r.lock.locked()


def test_busy_adapter_rejects_without_decoding():
    r, lengths = runtime()
    r.lock.acquire()
    with pytest.raises(HTTPException) as exc:
        r.transcribe(b'not audio')
    assert exc.value.status_code == 429 and lengths == []


def test_invalid_audio_releases_lock():
    r, lengths = runtime()
    with pytest.raises(HTTPException) as exc:
        r.transcribe(b'not audio')
    assert exc.value.status_code == 422 and lengths == []
    assert not r.lock.locked()
