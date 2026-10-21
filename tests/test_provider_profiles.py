import copy
import json
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import yaml
from config import load_provider_config, load_voice_control_config, ProviderConfigurationError
from profile_cli import select, main
from parakeet_adapter import request_transcription
from providers import ProviderError

@pytest.fixture
def cfg(tmp_path):
    p = tmp_path / 'config.yaml'
    p.write_text((Path(__file__).parents[1] / 'examples/profiles.yaml').read_text())
    return p


def test_selection_roundtrip_and_no_other_mutation(cfg):
    before = yaml.safe_load(cfg.read_text())
    for current, following in [('tinfoil', 'current-vps-parakeet'),
                               ('current-vps-parakeet', 'mac-mini-parakeet'),
                               ('mac-mini-parakeet', 'tinfoil')]:
        receipt = select(cfg, following, expected=current)
        c = load_provider_config(cfg)
        assert c.profile == following and receipt['fallback'] is None
        assert c.attested == (following == 'tinfoil')
        assert load_voice_control_config(cfg).realtime_stt.enabled is False
        after = yaml.safe_load(cfg.read_text())
        after['stt']['active_profile'] = 'tinfoil'
        assert after == before
    assert cfg.stat().st_mode & 0o777 == 0o600


def test_legacy_realtime_is_not_switched_by_batch_profile(cfg):
    root = yaml.safe_load(cfg.read_text())
    del root['voice']
    cfg.write_text(yaml.safe_dump(root))
    before = load_voice_control_config(cfg).realtime_stt
    assert before.provider == 'tinfoil' and before.enabled
    select(cfg, 'current-vps-parakeet', expected='tinfoil')
    assert load_voice_control_config(cfg).realtime_stt == before


def test_generic_profile_status_retains_selected_identity(cfg):
    root = yaml.safe_load(cfg.read_text())
    root['stt']['profiles']['explicit-cloud'] = {'provider': 'openai-compatible',
        'model': 'owner-model', 'base_url': 'https://api.example.com/v1', 'secret_env': 'OWNER_KEY'}
    cfg.write_text(yaml.safe_dump(root))
    select(cfg, 'explicit-cloud', expected='tinfoil')
    assert load_provider_config(cfg).profile == 'explicit-cloud'


def test_invalid_profile_never_changes_file(cfg):
    before = cfg.read_bytes()
    with pytest.raises(ProviderConfigurationError):
        select(cfg, 'cloud-fallback', expected='tinfoil')
    with pytest.raises(ProviderConfigurationError):
        select(cfg, 'current-vps-parakeet', expected='stale')
    assert cfg.read_bytes() == before

@pytest.mark.parametrize('url', ['https://api.example.com', 'http://192.0.2.1:8888',
    'http://user:secret@localhost:8888', 'http://localhost:8888/other', 'http://localhost:8888?secret=x',
    'http://localhost:99999', 'http://localhost:bad'])
def test_private_route_boundary(cfg, url):
    root = yaml.safe_load(cfg.read_text())
    root['stt']['profiles']['current-vps-parakeet']['base_url'] = url
    cfg.write_text(yaml.safe_dump(root))
    before = cfg.read_bytes()
    with pytest.raises(ProviderConfigurationError):
        select(cfg, 'current-vps-parakeet', expected='tinfoil')
    assert cfg.read_bytes() == before


def test_no_credential_and_no_redirect_or_wrong_provider(cfg, tmp_path):
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            calls.append(dict(self.headers))
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(302 if len(calls) == 1 else 200)
            self.send_header('Location', '/must-not-follow')
            self.end_headers()
            self.wfile.write(json.dumps({'text': 'secret transcript', 'provider': 'other', 'model': 'other'}).encode())
        def log_message(self, *args):
            pass
    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        root = yaml.safe_load(cfg.read_text())
        root['stt']['profiles']['current-vps-parakeet']['base_url'] = f'http://127.0.0.1:{server.server_port}'
        cfg.write_text(yaml.safe_dump(root))
        select(cfg, 'current-vps-parakeet', expected='tinfoil')
        audio = tmp_path / 'audio'; audio.write_bytes(b'fixture')
        c = load_provider_config(cfg)
        for _ in range(2):
            with pytest.raises(ProviderError, match='no fallback') as exc:
                request_transcription(audio, c)
            assert 'secret' not in str(exc.value)
        assert len(calls) == 2
        assert all('Authorization' not in h for h in calls)
    finally:
        server.shutdown(); server.server_close(); thread.join()


def test_local_profile_does_not_read_cloud_credentials(cfg, monkeypatch, tmp_path):
    import providers
    select(cfg, 'current-vps-parakeet', expected='tinfoil')
    monkeypatch.setenv('WIZARD_VOICE_CONFIG', str(cfg))
    monkeypatch.setattr(providers, 'load_api_key', lambda *a: pytest.fail('read cloud key'))
    def transcribe(source, config, *, api_key, prompt):
        assert api_key == '' and config.privacy == 'local-host'
        assert config.postprocess['mode'] == 'meeting'
        return 'ok'
    monkeypatch.setattr(providers, 'transcribe', transcribe)
    assert providers.route_transcription(tmp_path/'audio', 'en', mode='meeting') == 'ok'


def test_missing_adapter_fails_closed(cfg, tmp_path):
    select(cfg, 'current-vps-parakeet', expected='tinfoil')
    from dataclasses import replace
    c = replace(load_provider_config(cfg), endpoint='http://127.0.0.1:1/v1/audio/transcriptions', timeout=1)
    p = tmp_path/'audio'; p.write_bytes(b'fixture')
    with pytest.raises(ProviderError, match='no fallback'):
        request_transcription(p, c)
