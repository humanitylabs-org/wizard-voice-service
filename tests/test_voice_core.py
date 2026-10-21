from __future__ import annotations

import os
import hashlib
import io
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml
from fastapi.testclient import TestClient
from urllib.error import HTTPError

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ProviderConfigurationError, load_voice_control_config  # noqa: E402
from control import VoiceControlError, no_op_patch, safe_voice_config  # noqa: E402
from jobs import VoiceJobConflict, VoiceJobStore, VoiceJobWorker  # noqa: E402
from providers import TranscriptionResult  # noqa: E402
import client  # noqa: E402
import server  # noqa: E402


def fixture(*, explicit_lanes: bool = True) -> dict:
    data = {
        "stt": {
            "enabled": True,
            "provider": "tinfoil",
            "language": "en",
            "providers": {
                "tinfoil": {
                    "model": "voxtral-small-24b",
                    "realtime_model": "voxtral-mini-4b-realtime",
                    "proxy_url": "http://127.0.0.1:3301",
                    "secret_env": "TINFOIL_API_KEY",
                },
                "openai-compatible": {
                    "model": "gpt-4o-mini-transcribe",
                    "base_url": "https://api.openai.com/v1",
                    "secret_env": "VOICE_TOOLS_OPENAI_KEY",
                },
            },
        },
        "voice": {"record_key": "ctrl+b"},
    }
    if explicit_lanes:
        data["voice"]["schema_version"] = 1
        data["voice"]["lanes"] = {
            "realtime_stt": {
                "enabled": True,
                "provider": "tinfoil",
                "model": "voxtral-mini-4b-realtime",
                "transport": "websocket",
            },
            "conversation": {
                "enabled": True,
                "provider": "tinfoil",
                "transport": "websocket",
                "models": {
                    "transcription": "voxtral-mini-4b-realtime",
                    "reasoning": "deepseek-v4-flash",
                    "speech": "qwen3-tts",
                    "voice": "ryan",
                },
            },
        }
    return data


class VoiceLaneTests(unittest.TestCase):
    def write_config(self, directory: str, payload: dict) -> Path:
        path = Path(directory) / "config.yaml"
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        return path

    def test_three_lanes_are_independent_and_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"TINFOIL_API_KEY": "secret"}):
            path = self.write_config(directory, fixture())
            loaded = load_voice_control_config(path)
            self.assertEqual(loaded.batch_stt.model, "voxtral-small-24b")
            self.assertEqual(loaded.realtime_stt.model, "voxtral-mini-4b-realtime")
            self.assertEqual(loaded.conversation.models["speech"], "qwen3-tts")
            self.assertNotEqual(loaded.batch_stt.model, loaded.realtime_stt.model)
            safe = safe_voice_config(path)
            self.assertEqual(safe["lanes"]["batch_stt"]["credential"], "configured")
            self.assertNotIn("secret", str(safe))
            self.assertNotIn("127.0.0.1:3301", str(safe))

    def test_legacy_realtime_model_is_visible_but_unmanaged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_config(directory, fixture(explicit_lanes=False))
            loaded = load_voice_control_config(path)
            self.assertFalse(loaded.realtime_stt.managed)
            self.assertEqual(loaded.realtime_stt.model, "voxtral-mini-4b-realtime")
            self.assertFalse(loaded.conversation.managed)
            self.assertFalse(loaded.conversation.enabled)

    def test_disabled_batch_does_not_hide_independent_voice_lanes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = fixture()
            payload["stt"]["enabled"] = False
            path = self.write_config(directory, payload)
            loaded = load_voice_control_config(path)
            self.assertFalse(loaded.batch_stt.enabled)
            self.assertTrue(loaded.realtime_stt.enabled)
            self.assertTrue(loaded.conversation.enabled)
            self.assertFalse(safe_voice_config(path)["lanes"]["batch_stt"]["enabled"])

    def test_realtime_model_must_match_the_selected_provider_capability(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = fixture()
            payload["voice"]["lanes"]["realtime_stt"]["model"] = "unsupported-model"
            path = self.write_config(directory, payload)
            with self.assertRaisesRegex(ProviderConfigurationError, "not approved"):
                load_voice_control_config(path)

    def test_hash_is_canonical_and_ignores_unrelated_edits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = fixture()
            path = self.write_config(directory, payload)
            first = load_voice_control_config(path).config_hash
            payload["unrelated"] = {"x": 1}
            path = self.write_config(directory, payload)
            self.assertEqual(first, load_voice_control_config(path).config_hash)

    def test_unapproved_lane_provider_and_incomplete_conversation_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = fixture()
            payload["voice"]["lanes"]["realtime_stt"]["provider"] = "unknown"
            path = self.write_config(directory, payload)
            with self.assertRaisesRegex(ProviderConfigurationError, "not approved"):
                load_voice_control_config(path)

            payload = fixture()
            del payload["voice"]["lanes"]["conversation"]["models"]["speech"]
            path = self.write_config(directory, payload)
            with self.assertRaisesRegex(ProviderConfigurationError, "conversation requires"):
                load_voice_control_config(path)

    def test_bridge_patch_accepts_only_exact_noop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_config(directory, fixture())
            safe = safe_voice_config(path)
            result = no_op_patch(safe, expected_hash=safe["config_hash"], path=path)
            self.assertEqual(result["action"], "noop")
            changed = {**safe, "lanes": {**safe["lanes"], "realtime_stt": {"provider": "attacker"}}}
            with self.assertRaisesRegex(VoiceControlError, "writes are not enabled"):
                no_op_patch(changed, expected_hash=safe["config_hash"], path=path)
            with self.assertRaisesRegex(VoiceControlError, "changed"):
                no_op_patch(safe, expected_hash="0" * 64, path=path)


class VoiceJobTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.audio = self.root / "sample.m4a"
        self.audio.write_bytes(b"audio-source")
        self.store = VoiceJobStore(self.root / "voice")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def accept(self, **overrides):
        values = {
            "source_app": "movement-play",
            "device_id": "browser-1",
            "idempotency_key": "turn-1",
            "destination": "client.transcript",
            "dispatch": "client",
        }
        values.update(overrides)
        return self.store.accept(self.audio, **values)

    def test_admission_is_durable_idempotent_and_content_bound(self) -> None:
        first = self.accept()
        second = self.accept()
        self.assertEqual(first.job_id, second.job_id)
        self.assertTrue(second.duplicate)
        self.assertEqual(self.store.get(first.job_id)["state"], "accepted")
        other = self.root / "other.m4a"
        other.write_bytes(b"different")
        with self.assertRaises(VoiceJobConflict):
            self.store.accept(
                other,
                source_app="movement-play",
                device_id="browser-1",
                idempotency_key="turn-1",
                destination="client.transcript",
            )

    def test_worker_completes_one_job_and_records_receipt(self) -> None:
        admission = self.accept()
        calls = []

        def transcribe(path: Path, language: str) -> TranscriptionResult:
            calls.append((path.read_bytes(), language))
            return TranscriptionResult("move gently", "tinfoil-attested", "voxtral-small-24b", 25.0)

        worker = VoiceJobWorker(self.store, transcriber=transcribe)
        self.assertTrue(worker.process_one())
        self.assertFalse(worker.process_one())
        job = self.store.get(admission.job_id)
        self.assertEqual(job["state"], "completed")
        self.assertEqual(job["transcript"], "move gently")
        self.assertEqual(job["provider"], "tinfoil-attested")
        self.assertEqual(len(calls), 1)

    def test_processing_job_is_recovered_after_process_restart(self) -> None:
        admission = self.accept()
        claimed = self.store.claim_for_processing()
        self.assertEqual(claimed["job_id"], admission.job_id)
        reopened = VoiceJobStore(self.root / "voice")
        self.assertEqual(reopened.get(admission.job_id)["state"], "accepted")

    def test_failed_job_retries_then_terminates(self) -> None:
        admission = self.accept()
        self.store.claim_for_processing()
        self.store.fail(admission.job_id, "provider down", attempts=1, max_attempts=2)
        with sqlite3.connect(self.store.db_path) as conn:
            conn.execute("UPDATE voice_jobs SET next_attempt_at=0 WHERE job_id=?", (admission.job_id,))
        retry = self.store.claim_for_processing()
        self.store.fail(admission.job_id, "provider down", attempts=int(retry["attempts"]), max_attempts=2)
        self.assertEqual(self.store.get(admission.job_id)["state"], "failed_terminal")

    def test_unexpected_transcriber_exception_becomes_visible_job_failure(self) -> None:
        admission = self.accept()

        def explode(_path: Path, _language: str) -> TranscriptionResult:
            raise ValueError("malformed provider receipt")

        worker = VoiceJobWorker(self.store, transcriber=explode)
        self.assertTrue(worker.process_one())
        job = self.store.get(admission.job_id)
        self.assertEqual(job["state"], "failed_retryable")
        self.assertIn("malformed provider receipt", str(job["error"]))


class VoiceApiTests(unittest.TestCase):
    def tearDown(self) -> None:
        server._job_store = None
        server._job_worker = None

    def test_capabilities_are_lane_aware(self) -> None:
        with patch.object(
            server,
            "safe_capabilities",
            return_value={"schema_version": 1, "lanes": {"batch_stt": {"provider": "tinfoil"}}},
        ):
            response = TestClient(server.app).get("/v1/voice/capabilities")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["lanes"]["batch_stt"]["provider"], "tinfoil")

    def test_untrusted_host_is_rejected_before_job_or_transcript_access(self) -> None:
        response = TestClient(server.app).get("/v1/voice/jobs/opaque-id", headers={"host": "attacker.invalid"})
        self.assertEqual(response.status_code, 400)

    def test_voice_job_endpoint_returns_durable_admission_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            server._job_store = VoiceJobStore(Path(directory))
            worker = MagicMock()
            server._job_worker = worker
            client = TestClient(server.app)
            response = client.post(
                "/v1/voice/jobs",
                files={"file": ("voice.m4a", b"audio-bytes", "audio/mp4")},
                data={
                    "source_app": "movement-play",
                    "device_id": "web",
                    "idempotency_key": "request-1",
                    "destination": "client.transcript",
                    "dispatch": "client",
                },
            )
            self.assertEqual(response.status_code, 202)
            receipt = response.json()
            self.assertEqual(receipt["state"], "accepted")
            self.assertEqual(receipt["audio_sha256"], hashlib.sha256(b"audio-bytes").hexdigest())
            status = client.get(receipt["status_url"])
            self.assertEqual(status.status_code, 200)
            self.assertEqual(status.json()["job_id"], receipt["job_id"])
            worker.wake.assert_called_once()

    def test_job_client_preserves_http_status_for_compatibility_decisions(self) -> None:
        error = HTTPError(
            "http://127.0.0.1:8864/v1/voice/jobs/missing",
            404,
            "Not Found",
            {},
            io.BytesIO(b'{"detail":"Not Found"}'),
        )
        with patch("client.urllib.request.urlopen", side_effect=error):
            with self.assertRaisesRegex(client.ClientError, "HTTP 404"):
                client.get_voice_job("/v1/voice/jobs/missing")

    def test_admin_requires_token_and_only_permits_noop_patch(self) -> None:
        current = {"schema_version": 1, "config_hash": "abc", "lanes": {}}
        with patch.dict(os.environ, {"WIZARD_VOICE_ADMIN_TOKEN": "admin-test-token"}, clear=False), patch.object(
            server,
            "no_op_patch",
            return_value={"action": "noop", **current},
        ) as noop:
            client = TestClient(server.app)
            rejected = client.patch(
                "/admin/v1/voice/config",
                headers={"X-Wizard-Voice-Admin": "wrong", "If-Match": "abc"},
                json=current,
            )
            accepted = client.patch(
                "/admin/v1/voice/config",
                headers={"X-Wizard-Voice-Admin": "admin-test-token", "If-Match": "abc"},
                json=current,
            )
        self.assertEqual(rejected.status_code, 401)
        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(accepted.json()["action"], "noop")
        noop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
