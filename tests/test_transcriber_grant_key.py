"""Voice -> Transcriber carries Voice's own grant key (wizard-os issue 386, card 5).

With the browser boundary on, the Transcriber front serves its engine API only to granted callers (its grants file
`enforce` flag). Voice reads its key from its own grants file (`outgoing`, App Manager layer 2) and sends it as
X-Wizard-Grant on the packaged local-engine route only. The keys here are generated at runtime.
"""
import json
import secrets
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
import yaml

import providers
from config import load_provider_config

ROOT = Path(__file__).parents[1]


@pytest.fixture
def app_config(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text((ROOT / "packaging/default-config.yaml").read_text())
    monkeypatch.setenv("WIZARD_VOICE_CONFIG", str(path))
    for name in ("WIZARD_VOICE_TINFOIL_CONNECTION", "WIZARD_VOICE_TINFOIL_KEY", "WIZARD_VOICE_RULES", "WIZARD_GRANTS_FILE"):
        monkeypatch.delenv(name, raising=False)
    return path


def grants(tmp_path, outgoing):
    p = tmp_path / "grants.json"
    p.write_text(json.dumps({"outgoing": outgoing, "incoming": [], "enforce": False}))
    return p


def test_wizard_md_asks_for_the_transcriber_grant():
    head = (ROOT / "wizard.md").read_text().split("---")[1]
    assert "transcriber" in yaml.safe_load(head)["uses"]


def test_key_comes_from_the_apps_own_grants_file(tmp_path, monkeypatch):
    key = secrets.token_hex(32)
    assert providers.engine_grant_key(str(grants(tmp_path, {"transcriber": key}))) == key
    assert providers.engine_grant_key(str(grants(tmp_path, {"slot:speech-to-text-engine": key}))) == key
    assert providers.engine_grant_key(str(grants(tmp_path, {"files": key}))) == ""
    assert providers.engine_grant_key(str(grants(tmp_path, {"transcriber": "short"}))) == ""
    assert providers.engine_grant_key(str(tmp_path / "missing.json")) == ""
    (tmp_path / "bad.json").write_text("{not json")
    assert providers.engine_grant_key(str(tmp_path / "bad.json")) == ""
    monkeypatch.delenv("WIZARD_GRANTS_FILE", raising=False)
    assert providers.engine_grant_key() == ""


class Engine(BaseHTTPRequestHandler):
    seen = []

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        Engine.seen.append({"grant": self.headers.get("X-Wizard-Grant"), "auth": self.headers.get("Authorization")})
        # Like the Transcriber front with enforce on: no granted key, no engine.
        ok = self.headers.get("X-Wizard-Grant") == Engine.want
        body = json.dumps({"text": "The blue notebook is on the table."} if ok else {"error": "needs X-Wizard-Grant"}).encode()
        self.send_response(200 if ok else 403)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture
def engine():
    Engine.seen, Engine.want = [], secrets.token_hex(32)
    http = HTTPServer(("127.0.0.1", 0), Engine)
    threading.Thread(target=http.serve_forever, daemon=True).start()
    yield http
    http.shutdown()


def test_local_engine_request_sends_the_grant_key_and_no_bearer(app_config, tmp_path, monkeypatch, engine):
    c = load_provider_config()
    audio = tmp_path / "synthetic.mp3"
    audio.write_bytes(b"unit-fixture")
    monkeypatch.setenv("WIZARD_GRANTS_FILE", str(grants(tmp_path, {"transcriber": Engine.want})))
    real_open = providers._open_transcription

    def to_engine(request, *, timeout):
        assert request.full_url == c.endpoint   # the packaged route, unchanged
        request.full_url = f"http://127.0.0.1:{engine.server_port}/v1/audio/transcriptions"
        return real_open(request, timeout=timeout)
    monkeypatch.setattr(providers, "_open_transcription", to_engine)
    assert providers._request_text(audio, c, "") == "The blue notebook is on the table."
    assert Engine.seen == [{"grant": Engine.want, "auth": None}]
    # Without a grant the enforcing engine refuses (what the boundary means for an ungranted Voice).
    monkeypatch.delenv("WIZARD_GRANTS_FILE")
    with pytest.raises(providers.ProviderError, match="HTTP 403"):
        providers._request_text(audio, c, "")


def test_cloud_routes_never_get_the_grant_key(app_config, tmp_path, monkeypatch):
    c = replace(load_provider_config(), endpoint="https://stt.example/v1/audio/transcriptions", name="openai-compatible")
    audio = tmp_path / "synthetic.mp3"
    audio.write_bytes(b"unit-fixture")
    monkeypatch.setenv("WIZARD_GRANTS_FILE", str(grants(tmp_path, {"transcriber": secrets.token_hex(32)})))
    seen = []

    class Response:
        status = 200
        def read(self): return b'{"text":"ok"}'
        def __enter__(self): return self
        def __exit__(self, *a): pass
    monkeypatch.setattr(providers.urllib.request, "urlopen", lambda request, timeout: seen.append(request) or Response())
    assert providers._request_text(audio, c, "cloud-key") == "ok"
    assert seen[0].get_header("X-wizard-grant") is None
