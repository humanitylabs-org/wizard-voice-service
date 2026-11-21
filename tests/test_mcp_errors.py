"""wizard-voice-service#13: a failed MCP transcribe names the real cause."""
import http.client
import json
from pathlib import Path
import shutil
import subprocess

from fastapi import HTTPException
import pytest

import providers
from voice_mcp.front import explain, rpc

ROOT = Path(__file__).parents[1]


@pytest.fixture
def media(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text((ROOT / "packaging/default-config.yaml").read_text())
    monkeypatch.setenv("WIZARD_VOICE_CONFIG", str(config))
    root = tmp_path / "media"
    (root / "Inbox/2026-11-21").mkdir(parents=True)
    monkeypatch.setenv("WIZARD_VOICE_MEDIA_DIR", str(root))
    return root


def call(path, mode="verbatim"):
    result = rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                  "params": {"name": "transcribe", "arguments": {"path": path, "mode": mode}}})["result"]
    return result["isError"], json.loads(result["content"][0]["text"])


def opus_webm(path, seconds=2):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is required")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"sine=frequency=220:duration={seconds}",
                    "-ac", "1", "-ar", "48000", "-c:a", "libopus", str(path)], check=True)


def test_engine_unreachable_is_named_not_bare_http_exception(media, monkeypatch):
    note = media / "Inbox/2026-11-21/wizard-voice-2026-11-21T11-10-05-908Z.webm"
    opus_webm(note)
    monkeypatch.setattr(providers, "detect_no_speech", lambda _p: False)

    def unreachable(*_a, **_k):
        raise providers.ProviderError("openai-compatible route unavailable (URLError)")
    monkeypatch.setattr(providers, "_request_text", unreachable)
    is_error, body = call("Inbox/2026-11-21/" + note.name)
    assert is_error
    assert "HTTPException" not in body["error"]
    assert "Transcriber app" in body["error"] and "reached" in body["error"]


def test_opus_webm_voice_note_transcribes(media, monkeypatch):
    note = media / "Inbox/2026-11-21/wizard-voice-2026-11-21T10-31-54-847Z.webm"
    opus_webm(note)
    monkeypatch.setattr(providers, "detect_no_speech", lambda _p: False)
    monkeypatch.setattr(providers, "_request_text", lambda *_a, **_k: "Remind me to buy milk.")
    is_error, body = call("Inbox/2026-11-21/" + note.name)
    assert not is_error and body["text"] == "Remind me to buy milk."


def test_undecodable_file_and_missing_file_are_named(media):
    (media / "Inbox/broken.webm").write_bytes(b"not audio at all")
    is_error, body = call("Inbox/broken.webm")
    assert is_error and "could not be decoded" in body["error"]
    is_error, body = call("Inbox/missing.webm")
    assert is_error and "no file at that path" in body["error"]


def test_dropped_connection_is_an_unreachable_engine(media, monkeypatch, tmp_path):
    from dataclasses import replace
    from config import load_provider_config
    audio = tmp_path / "a.mp3"; audio.write_bytes(b"x")
    config = replace(load_provider_config(), endpoint="http://127.0.0.1:9/v1/audio/transcriptions")

    def dropped(*_a, **_k):
        raise http.client.RemoteDisconnected("Remote end closed connection without response")
    monkeypatch.setattr(providers, "_open_transcription", dropped)
    monkeypatch.setattr(providers.urllib.request, "urlopen", dropped)
    with pytest.raises(providers.ProviderError, match="route unavailable"):
        providers._request_text(audio, config, "")


@pytest.mark.parametrize("exc, expected", [
    (HTTPException(422, "No speech detected"), "No speech was detected"),
    (HTTPException(413, "audio too large"), "larger than the upload limit"),
    (HTTPException(502, "openai-compatible returned HTTP 500: /tmp/secret path"), "HTTP 500"),
    (HTTPException(502, "audio normalization failed: /tmp/wizard-stt-x.webm: Invalid data"), "could not be decoded"),
    (HTTPException(502, "transcription deadline exceeded"), "took too long"),
    (PermissionError("/media/x"), "permission denied"),
])
def test_explain_is_plain_and_never_echoes_paths(exc, expected):
    message = explain(exc)
    assert expected in message and "/tmp" not in message and "/media" not in message
