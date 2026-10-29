import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys

import pytest
import yaml
from fastapi.testclient import TestClient

from config import ProviderConfig, ProviderConfigurationError, load_provider_config
from postprocess import preview_transcript
from voice_cli import main, mutate
from voice_rules import read_rules


def correction(id="user.sample", pattern="Acem", replacement="Acme"):
    return {"id": id, "kind": "correction", "scope": "user", "provenance": "synthetic fixture",
            "enabled": True, "priority": 100, "modes": ["dictation", "meeting"],
            "operation": "regex", "pattern": pattern, "replacement": replacement}


def test_cli_roundtrip(tmp_path, capsys):
    target = tmp_path / "rules.yaml"
    source = tmp_path / "rule.yaml"
    source.write_text(yaml.safe_dump(correction()))
    assert main(["--file", str(target), "add", str(source)]) == 0
    capsys.readouterr()
    assert main(["--file", str(target), "list"]) == 0
    assert len(json.loads(capsys.readouterr().out)["rules"]) == 2
    assert main(["--file", str(target), "preview", "--text", "Acem"]) == 0
    assert json.loads(capsys.readouterr().out)["corrected_text"] == "Acme"
    assert main(["--file", str(target), "disable", "user.sample"]) == 0
    capsys.readouterr()
    assert main(["--file", str(target), "preview", "--text", "Acem"]) == 0
    assert json.loads(capsys.readouterr().out)["styled_text"] == "Acem"
    assert main(["--file", str(target), "enable", "user.sample"]) == 0
    capsys.readouterr()
    before = target.read_bytes()
    assert main(["--file", str(target), "add", str(source)]) == 2
    assert target.read_bytes() == before
    assert target.stat().st_mode & 0o777 == 0o600
    assert main(["add", str(source)]) == 2
    assert main(["--file", str(target), "disable", "missing"]) == 2


def test_cli_overlapping_process_writes_preserved(tmp_path):
    target = tmp_path / "rules.yaml"
    def add(index):
        source = tmp_path / f"rule{index}.yaml"
        source.write_text(yaml.safe_dump(correction(id=f"user.r{index}")))
        return subprocess.run([sys.executable, "-m", "voice_cli", "--file", str(target), "add", str(source)],
                              capture_output=True).returncode
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(add, range(8))) == [0] * 8
    assert len(read_rules(target)) == 9


def test_legacy_layer_kept_and_rule_file_hot_read(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("stt:\n  provider: tinfoil\n  postprocess:\n    replacements:\n      - {pattern: first, replacement: second}\n  providers:\n    tinfoil:\n      model: voxtral-small-24b\nvoice:\n  rules_file: rules.yaml\n")
    owned = tmp_path / "owned.yaml"
    owned.write_text("replacements:\n  - {pattern: second, replacement: third}\n")
    monkeypatch.setenv("WIZARD_STT_CORRECTIONS", str(owned))
    rules = tmp_path / "rules.yaml"
    mutate(rules, "add", rule=correction(pattern="third", replacement="fourth"))
    monkeypatch.setenv("WIZARD_VOICE_CONFIG", str(cfg))
    monkeypatch.setenv("WIZARD_STT_CONFIG", str(tmp_path / "not-selected"))
    assert preview_transcript("first", load_provider_config().postprocess)["styled_text"] == "fourth"
    mutate(rules, "disable", rule_id="user.sample")
    assert preview_transcript("first", load_provider_config().postprocess)["styled_text"] == "third"
    rules.write_text("broken: true")
    with pytest.raises(ProviderConfigurationError):
        load_provider_config()


@pytest.mark.parametrize("mode", ["meeting", "verbatim"])
def test_http_mode_reaches_provider(mode, monkeypatch):
    import server
    import providers
    calls = []
    monkeypatch.setattr(providers, "load_provider_config", lambda: ProviderConfig(
        name="tinfoil", model="voxtral-small-24b", endpoint="http://127.0.0.1:3301/v1/audio/transcriptions",
        api_key_env="TEST_KEY", attested=True))
    monkeypatch.setattr(providers, "load_api_key", lambda _: "synthetic")
    def transcribe(path, config, **kwargs):
        calls.append(config.postprocess["mode"])
        return providers.TranscriptionResult("Synthetic result.", "tinfoil-attested", config.model, 1)
    monkeypatch.setattr(providers, "transcribe", transcribe)
    client = TestClient(server.app)
    response = client.post("/inference", files={"file": ("test.wav", b"synthetic")}, data={"mode": mode})
    assert response.status_code == 200
    assert calls == [mode]


def test_default_job_worker_passes_mode(tmp_path, monkeypatch):
    import jobs
    calls = []
    def route(path, language, **kwargs):
        calls.append(kwargs)
        return jobs.TranscriptionResult("sample", "tinfoil-attested", "voxtral-small-24b", 1)
    # Explicit injection allows observing the same route identity used at runtime.
    monkeypatch.setattr(jobs, "route_transcription", route)
    store = jobs.VoiceJobStore(tmp_path / "state")
    source = tmp_path / "audio.wav"
    source.write_bytes(b"synthetic")
    admission = store.accept(source, source_app="test", device_id="test", idempotency_key="one", mode="meeting")
    worker = jobs.VoiceJobWorker(store, transcriber=route)
    assert worker.process_one()
    assert calls == [{"mode": "meeting"}]
    assert store.get(admission.job_id)["state"] == "completed"
    with pytest.raises(jobs.VoiceJobError):
        store.accept(source, source_app="test", device_id="test", idempotency_key="two", mode="unknown")


def test_failed_audio_retention_opt_in(tmp_path, monkeypatch):
    import server
    source = tmp_path / "audio.wav"
    source.write_bytes(b"synthetic")
    monkeypatch.delenv("WIZARD_VOICE_RETAIN_FAILED_AUDIO", raising=False)
    server._retain_failure(source, ".wav", "synthetic failure")
    assert not server.FAILED_AUDIO_DIR.exists()
    monkeypatch.setenv("WIZARD_VOICE_RETAIN_FAILED_AUDIO", "1")
    server._retain_failure(source, ".wav", "synthetic failure")
    files = list(server.FAILED_AUDIO_DIR.glob("*.wav"))
    assert len(files) == 1 and files[0].stat().st_mode & 0o777 == 0o600


def test_private_buffer_is_opt_in_and_never_breaks(tmp_path, monkeypatch):
    import json
    import server
    from types import SimpleNamespace
    source = tmp_path / "audio.m4a"
    source.write_bytes(b"synthetic")
    result = SimpleNamespace(text="that is so funny haha", provider="tinfoil-attested", model="voxtral", latency_ms=1)
    target = tmp_path / "buffer"
    monkeypatch.delenv("WIZARD_VOICE_PRIVATE_BUFFER_DIR", raising=False)
    server._buffer_private(source, ".m4a", "dictation", result)
    assert not target.exists()
    monkeypatch.setenv("WIZARD_VOICE_PRIVATE_BUFFER_DIR", str(target))
    server._buffer_private(source, ".m4a", "meeting", result)  # meeting mode is not buffered by default
    assert not target.exists()
    server._buffer_private(source, ".m4a", "dictation", result)
    sidecars = list(target.glob("*.json"))
    assert len(sidecars) == 1 and not list(target.glob(".*"))
    meta = json.loads(sidecars[0].read_text())
    assert meta["text"] == result.text and meta["mode"] == "dictation"
    audio = target / meta["audio"]
    assert audio.read_bytes() == b"synthetic" and audio.stat().st_mode & 0o777 == 0o600
    monkeypatch.setenv("WIZARD_VOICE_PRIVATE_BUFFER_DIR", str(source))  # a file, not a folder: must not raise
    server._buffer_private(source, ".m4a", "dictation", result)
