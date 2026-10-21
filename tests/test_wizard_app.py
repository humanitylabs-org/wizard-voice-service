import copy
from dataclasses import replace
import json
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from fastapi.testclient import TestClient
import pytest
import yaml

from config import load_provider_config, ProviderConfigurationError, is_local_transcriber
from profile_cli import select
import providers
import server
from voice_mcp.front import rpc

ROOT = Path(__file__).parents[1]


@pytest.fixture
def app_config(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text((ROOT / "packaging/default-config.yaml").read_text())
    monkeypatch.setenv("WIZARD_VOICE_CONFIG", str(path))
    monkeypatch.delenv("WIZARD_VOICE_TINFOIL_CONNECTION", raising=False)
    monkeypatch.delenv("WIZARD_VOICE_TINFOIL_KEY", raising=False)
    monkeypatch.delenv("WIZARD_VOICE_RULES", raising=False)
    return path


def test_packaged_engine_is_local_credential_free(app_config):
    c = load_provider_config()
    assert c.profile == "local-transcriber" and is_local_transcriber(c)
    assert c.model == "Systran/faster-whisper-small" and not c.attested
    assert c.postprocess["replacements"] == []
    assert not is_local_transcriber(replace(c, endpoint="http://other:8000/v1/audio/transcriptions"))


@pytest.mark.parametrize("url", ["http://other:8000", "http://transcriber:8001", "http://transcriber:8000/other",
    "http://user:pass@transcriber:8000", "http://transcriber:8000?x=1", "https://example.com"])
def test_engine_exception_cannot_escape_box(app_config, url):
    root = yaml.safe_load(app_config.read_text())
    root["stt"]["profiles"]["local-transcriber"]["base_url"] = url
    app_config.write_text(yaml.safe_dump(root))
    with pytest.raises(ProviderConfigurationError, match="local_engine"):
        load_provider_config()


def test_http_exception_requires_explicit_engine_marker(app_config):
    root = yaml.safe_load(app_config.read_text())
    del root["stt"]["profiles"]["local-transcriber"]["local_engine"]
    app_config.write_text(yaml.safe_dump(root))
    with pytest.raises(ProviderConfigurationError, match="HTTPS or loopback"):
        load_provider_config()


def test_engine_route_never_reads_cloud_key(app_config, monkeypatch, tmp_path):
    monkeypatch.setattr(providers, "load_api_key", lambda *a: pytest.fail("read cloud key"))
    def transcribe(source, config, *, api_key, prompt):
        assert api_key == "" and config.postprocess["mode"] == "meeting"
        return "local"
    monkeypatch.setattr(providers, "transcribe", transcribe)
    assert providers.route_transcription(tmp_path / "sample", mode="meeting") == "local"


def test_engine_request_has_no_bearer_and_safe_opener(app_config, monkeypatch, tmp_path):
    c = load_provider_config()
    path = tmp_path / "synthetic.mp3"; path.write_bytes(b"unit-fixture")
    class Response:
        status = 200
        def read(self): return b'{"text":"The blue notebook is on the table."}'
        def __enter__(self): return self
        def __exit__(self, *a): pass
    def opened(request, timeout):
        assert request.get_header("Authorization") is None
        assert timeout == 900 and request.full_url == c.endpoint
        return Response()
    monkeypatch.setattr(providers, "_open_transcription", opened)
    assert providers._request_text(path, c, "") == "The blue notebook is on the table."


def test_local_opener_refuses_redirect_even_with_proxy_env(monkeypatch):
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            calls.append(self.path)
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(302); self.send_header("Location", "/never"); self.end_headers()
        def log_message(self, *a): pass
    http = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True); thread.start()
    try:
        monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
        monkeypatch.setenv("no_proxy", "")
        request = providers.urllib.request.Request(f"http://127.0.0.1:{http.server_port}/audio", data=b"unit-fixture")
        with pytest.raises(providers.ProviderError, match="redirect refused"):
            providers._open_transcription(request, timeout=2)
        assert calls == ["/audio"]
    finally:
        http.shutdown(); http.server_close(); thread.join()


def test_tinfoil_disabled_refusal_is_atomic(app_config):
    before = app_config.read_bytes()
    with pytest.raises(ProviderConfigurationError, match="disabled: no router connection config"):
        select(app_config, "tinfoil-voxtral", expected="local-transcriber")
    assert app_config.read_bytes() == before


def connection_config(tmp_path, monkeypatch):
    binding = {"schema_version": 1, "provider": "tinfoil", "model": "voxtral-small-24b",
        "endpoint": "http://127.0.0.1:3301/v1/audio/transcriptions", "attestation_required": True,
        "api_key_env": "WIZARD_VOICE_TINFOIL_KEY"}
    path = tmp_path / "connection.json"; path.write_text(json.dumps(binding))
    monkeypatch.setenv("WIZARD_VOICE_TINFOIL_CONNECTION", str(path))
    return path, binding


def test_tinfoil_needs_runtime_credential_not_legacy_env(app_config, tmp_path, monkeypatch):
    connection_config(tmp_path, monkeypatch)
    monkeypatch.setenv("TINFOIL_API_KEY", "unit-not-a-credential")
    with pytest.raises(ProviderConfigurationError, match="credential is unavailable"):
        select(app_config, "tinfoil-voxtral", expected="local-transcriber")


@pytest.mark.parametrize("field,value", [("endpoint", "https://example.com/v1/audio/transcriptions"),
    ("endpoint", "http://proxy:3301/v1/audio/transcriptions"), ("attestation_required", False),
    ("model", "other"), ("api_key_env", "TINFOIL_API_KEY"), ("provider", "other"), ("schema_version", 2)])
def test_tinfoil_rejects_wrong_connection(app_config, tmp_path, monkeypatch, field, value):
    path, binding = connection_config(tmp_path, monkeypatch)
    binding[field] = value; path.write_text(json.dumps(binding))
    monkeypatch.setenv("WIZARD_VOICE_TINFOIL_KEY", "unit-not-a-credential")
    with pytest.raises(ProviderConfigurationError, match="approved loopback"):
        select(app_config, "tinfoil-voxtral", expected="local-transcriber")


def test_tinfoil_explicit_activation_and_revocation(app_config, tmp_path, monkeypatch):
    path, _ = connection_config(tmp_path, monkeypatch)
    monkeypatch.setenv("WIZARD_VOICE_TINFOIL_KEY", "unit-not-a-credential")
    select(app_config, "tinfoil-voxtral", expected="local-transcriber")
    cfg = load_provider_config()
    assert cfg.attested and cfg.privacy == "attested-cloud"
    assert cfg.api_key_env == "WIZARD_VOICE_TINFOIL_KEY"
    path.unlink()
    with pytest.raises(ProviderConfigurationError, match="unavailable or invalid"):
        load_provider_config()


def test_manifest_and_compose_packaging():
    manifest = yaml.safe_load((ROOT / "wizard.md").read_text().split("---")[1])
    assert manifest["provides"] == ["voice"]
    assert manifest["needs"] == ["speech-to-text-engine"]
    assert "view" not in manifest
    cfg = json.loads((ROOT / "runtipi/config.json").read_text())
    assert cfg["no_gui"] and not cfg["exposable"]
    compose = yaml.safe_load((ROOT / "runtipi/docker-compose.yml").read_text())
    service = compose["services"]["wizard-voice-service"]
    assert service["x-runtipi"]["add_to_main_network"] and service["read_only"]
    assert service["pull_policy"] == "never" and service["user"] == "1000:1000"
    assert service["environment"]["WIZARD_VOICE_RETAIN_FAILED_AUDIO"] == "0"
    assert service["volumes"][1].endswith(":/media:ro")


def test_mcp_list_status_and_initialize(app_config):
    client = TestClient(server.app)
    def call(method, params=None):
        r = client.post("/mcp", json={"jsonrpc":"2.0", "id":1, "method":method, "params":params or {}})
        assert r.status_code == 200
        return r.json()["result"]
    assert call("initialize")["serverInfo"]["name"] == "wizard-voice-service"
    assert {t["name"] for t in call("tools/list")["tools"]} == {"transcribe", "voice_status"}
    status = json.loads(call("tools/call", {"name":"voice_status"})["content"][0]["text"])
    assert status["personal_rules"] == status["enabled_style_rules"] == status["legacy_replacements"] == 0
    assert status["privacy"] == "local-box"
    assert client.post("/mcp", json={"jsonrpc":"2.0", "method":"notifications/initialized"}).status_code == 202
    assert client.get("/mcp").status_code == 405


def test_mcp_token_and_malformed_payloads(app_config, monkeypatch):
    client = TestClient(server.app)
    assert client.post("/mcp", json={}, headers={"Origin":"https://untrusted.example"}).status_code == 403
    monkeypatch.setenv("WIZARD_VOICE_MCP_TOKEN", "unit-token")
    assert client.post("/mcp", json={}).status_code == 401
    headers={"Authorization":"Bearer unit-token"}
    assert client.post("/mcp", content="bad-json", headers=headers).status_code == 400
    assert client.post("/mcp", json=[], headers=headers).status_code == 400
    assert client.post("/mcp", content=b"a"*((1<<20)+1), headers=headers).status_code == 413
    assert rpc({"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"transcribe","arguments":{}}})["error"]["code"] == -32602


@pytest.mark.parametrize("path", ["../outside.wav", "/etc/passwd", "escape.wav"])
def test_mcp_media_boundary(tmp_path, monkeypatch, path):
    media = tmp_path / "media"; media.mkdir()
    outside = tmp_path / "outside.wav"; outside.write_bytes(b"fixture")
    (media / "escape.wav").symlink_to(outside)
    monkeypatch.setenv("WIZARD_VOICE_MEDIA_DIR", str(media))
    from voice_mcp.tools import transcribe
    with pytest.raises(ValueError): transcribe(path)


def test_mcp_mode_and_upload_path(app_config, tmp_path, monkeypatch):
    media = tmp_path / "media"; media.mkdir(); (media / "synthetic.wav").write_bytes(b"unit-fixture")
    monkeypatch.setenv("WIZARD_VOICE_MEDIA_DIR", str(media))
    def upload(file, language, **kw):
        assert file.file.read() == b"unit-fixture" and language == "en" and kw["mode"] == "meeting"
        return {"text":"The blue notebook is on the table."}
    monkeypatch.setattr(server, "_transcribe_upload", upload)
    result = rpc({"jsonrpc":"2.0", "id":1, "method":"tools/call", "params":{"name":"transcribe", "arguments":{"path":"synthetic.wav", "mode":"meeting"}}})
    assert not result["result"]["isError"]
    assert json.loads(result["result"]["content"][0]["text"])["text"].startswith("The blue")
