"""Box-peer gate on /mcp (wizard-os issue 386, card 5). Default-off; synthetic keys generated per run."""
import hashlib
import json
import secrets
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

import server

ROOT = Path(__file__).parents[1]
LIST = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text((ROOT / "packaging/default-config.yaml").read_text())
    monkeypatch.setenv("WIZARD_VOICE_CONFIG", str(path))
    for k in ("WIZARD_BROWSER_BOUNDARY", "WIZARD_GATEWAY_KEY", "WIZARD_GRANTS_FILE", "WIZARD_VOICE_MCP_TOKEN",
              "WIZARD_VOICE_MCP_ORIGINS"):
        monkeypatch.delenv(k, raising=False)
    # TestClient talks as host "testserver", which the shipped allowlist does not carry; the compose list adds the
    # app's own names. Exercise the gate, not the Host allowlist.
    return TestClient(server.app)


def on(monkeypatch, key=None, rows=None, tmp_path=None):
    monkeypatch.setenv("WIZARD_BROWSER_BOUNDARY", "1")
    if key:
        monkeypatch.setenv("WIZARD_GATEWAY_KEY", key)
    if rows is not None:
        f = tmp_path / "grants.json"
        f.write_text(json.dumps({"outgoing": {}, "incoming": rows, "enforce": True}))
        monkeypatch.setenv("WIZARD_GRANTS_FILE", str(f))


def test_boundary_off_keeps_todays_open_mcp(client, monkeypatch):
    monkeypatch.setenv("WIZARD_GATEWAY_KEY", secrets.token_hex(32))
    assert client.post("/mcp", json=LIST).status_code == 200


def test_mcp_needs_the_gateway_key(client, monkeypatch):
    key = secrets.token_hex(32)
    on(monkeypatch, key)
    assert client.post("/mcp", json=LIST).status_code == 403
    assert client.post("/mcp", json=LIST, headers={"X-Wizard-Grant": secrets.token_hex(32)}).status_code == 403
    r = client.post("/mcp", json=LIST, headers={"X-Wizard-Grant": key})
    assert r.status_code == 200 and {t["name"] for t in r.json()["result"]["tools"]} == {"transcribe", "voice_status"}


@pytest.mark.parametrize("marker", [{"Origin": "http://127.0.0.1"}, {"Origin": "null"}, {"Cookie": "a=b"},
                                    {"Referer": "http://x/"}, {"Sec-Fetch-Site": "same-origin"},
                                    {"Tailscale-User-Login": "owner@example.com"}, {"X-Forwarded-For": "100.64.0.9"}])
def test_markers_refused_even_with_the_key(client, monkeypatch, marker):
    key = secrets.token_hex(32)
    on(monkeypatch, key)
    monkeypatch.setenv("WIZARD_VOICE_MCP_ORIGINS", "http://127.0.0.1")  # an allowlisted origin is still a browser
    assert client.post("/mcp", json=LIST, headers={"X-Wizard-Grant": key, **marker}).status_code == 403


def test_fails_closed_when_nothing_is_configured(client, monkeypatch):
    on(monkeypatch)
    assert client.post("/mcp", json=LIST, headers={"X-Wizard-Grant": secrets.token_hex(32)}).status_code == 503


def test_spec_host_agent_service_row_only(client, monkeypatch, tmp_path):
    gw, app = secrets.token_hex(32), secrets.token_hex(32)
    on(monkeypatch, rows=[{"from": "host:agent-service", "sha256": hashlib.sha256(gw.encode()).hexdigest()},
                          {"from": "wisdom", "sha256": hashlib.sha256(app.encode()).hexdigest()}], tmp_path=tmp_path)
    assert client.post("/mcp", json=LIST, headers={"X-Wizard-Grant": gw}).status_code == 200
    assert client.post("/mcp", json=LIST, headers={"X-Wizard-Grant": app}).status_code == 403


def test_operator_bearer_token_still_works(client, monkeypatch):
    on(monkeypatch, secrets.token_hex(32))
    monkeypatch.setenv("WIZARD_VOICE_MCP_TOKEN", "unit-token")
    assert client.post("/mcp", json=LIST, headers={"Authorization": "Bearer unit-token"}).status_code == 200


def test_compose_wires_the_gate_default_off():
    env = yaml.safe_load((ROOT / "runtipi/docker-compose.yml").read_text())["services"]["wizard-voice-service"]["environment"]
    assert env["WIZARD_BROWSER_BOUNDARY"] == "${WIZARD_BROWSER_BOUNDARY:-0}"
    assert env["WIZARD_GATEWAY_KEY"] == "${WIZARD_GATEWAY_KEY:-}"
