import io
import json
import wave
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import providers
import server
from config import ProviderConfig
from test_stt_refusal import REFUSALS


def synthetic_wav():
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0" * 3200)
    return out.getvalue()


def mock_provider(monkeypatch, texts):
    # These tests isolate response validation, not acoustic detection.
    monkeypatch.setattr(providers, "detect_no_speech", lambda source: False)
    config = ProviderConfig(name="tinfoil", model="voxtral-small-24b",
        endpoint="http://127.0.0.1:3301/v1/audio/transcriptions",
        api_key_env="SYNTHETIC_KEY", attested=True)
    monkeypatch.setattr(providers, "load_provider_config", lambda: config)
    monkeypatch.setattr(providers, "load_api_key", lambda name: "synthetic")
    def normalize(source, directory):
        paths = [directory / f"chunk-{index}.wav" for index in range(len(texts))]
        for path in paths:
            path.write_bytes(synthetic_wav())
        return paths
    monkeypatch.setattr(providers, "normalize_audio_chunks", normalize)
    responses = []
    for text in texts:
        response = MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.read.return_value = json.dumps({"text": text}).encode()
        responses.append(response)
    monkeypatch.setattr(providers.urllib.request, "urlopen", MagicMock(side_effect=responses))


@pytest.mark.parametrize("text", REFUSALS)
def test_real_broker_route_rejects_refusal_without_echoing_text(monkeypatch, text):
    mock_provider(monkeypatch, [text])
    response = TestClient(server.app).post("/v1/audio/transcriptions",
        files={"file": ("synthetic.wav", synthetic_wav(), "audio/wav")})
    assert response.status_code == 502
    assert response.json() == {"detail": "tinfoil returned assistant_refusal"}
    assert text not in response.text


def test_refusal_in_later_provider_segment_rejects_entire_job(monkeypatch):
    mock_provider(monkeypatch, ["Synthetic valid first segment.", REFUSALS[0]])
    response = TestClient(server.app).post("/v1/audio/transcriptions",
        files={"file": ("synthetic.wav", synthetic_wav(), "audio/wav")})
    assert response.status_code == 502
    assert "assistant_refusal" in response.json()["detail"]
    assert "text" not in response.json()


@pytest.mark.parametrize("payload", [{}, {"text": None}, {"text": 123}, [], {"error": "synthetic"}])
def test_malformed_provider_response_is_failure_never_no_speech(monkeypatch, payload):
    mock_provider(monkeypatch, [""])
    response_mock = MagicMock()
    response_mock.__enter__.return_value = response_mock
    response_mock.status = 200
    response_mock.read.return_value = json.dumps(payload).encode()
    monkeypatch.setattr(providers.urllib.request, "urlopen", lambda *args, **kwargs: response_mock)
    response = TestClient(server.app).post("/v1/audio/transcriptions",
        files={"file": ("synthetic.wav", synthetic_wav(), "audio/wav")})
    assert response.status_code == 502
    assert response.json() == {"detail": "tinfoil returned invalid transcription response"}


def test_explicit_empty_provider_text_preserves_no_speech_contract(monkeypatch):
    mock_provider(monkeypatch, [""])
    response = TestClient(server.app).post("/v1/audio/transcriptions",
        files={"file": ("synthetic.wav", synthetic_wav(), "audio/wav")})
    assert response.status_code == 502
    assert response.json() == {"detail": "tinfoil returned no transcript"}


def test_real_broker_route_accepts_valid_text_and_keeps_receipt(monkeypatch):
    mock_provider(monkeypatch, ["Synthetic valid transcription."])
    response = TestClient(server.app).post("/v1/audio/transcriptions",
        files={"file": ("synthetic.wav", synthetic_wav(), "audio/wav")})
    assert response.status_code == 200
    assert response.json()["provider"] == "tinfoil-attested"
    assert response.json()["model"] == "voxtral-small-24b"
    assert response.json()["text"] == "Synthetic valid transcription."
