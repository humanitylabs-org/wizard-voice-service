from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


class ConfigTests(unittest.TestCase):
    def test_tinfoil_is_selected_from_the_single_hermes_key(self) -> None:
        from config import load_provider_config

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.yaml"
            config.write_text(
                """stt:
  enabled: true
  provider: tinfoil
  providers:
    tinfoil:
      model: voxtral-small-24b
      proxy_url: http://127.0.0.1:3301
      api_key_env: TINFOIL_API_KEY
""",
                encoding="utf-8",
            )
            selected = load_provider_config(config)
        self.assertEqual(selected.name, "tinfoil")
        self.assertEqual(selected.model, "voxtral-small-24b")
        self.assertEqual(selected.endpoint, "http://127.0.0.1:3301/v1/audio/transcriptions")
        self.assertTrue(selected.attested)
        self.assertEqual(selected.postprocess, {})

    def test_postprocess_config_is_carried_by_voice_core(self) -> None:
        from config import load_provider_config

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.yaml"
            config.write_text(
                """stt:
  enabled: true
  provider: tinfoil
  postprocess:
    normalize_spoken_times: true
    replacements:
      - pattern: '\\bAcem\\b'
        replacement: Acme
  providers:
    tinfoil:
      model: voxtral-small-24b
      proxy_url: http://127.0.0.1:3301
""",
                encoding="utf-8",
            )
            selected = load_provider_config(config)
        self.assertTrue(selected.postprocess["normalize_spoken_times"])
        self.assertEqual(selected.postprocess["replacements"][0]["replacement"], "Acme")

    def test_openai_compatible_is_switchable_with_the_same_key(self) -> None:
        from config import load_provider_config

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.yaml"
            config.write_text(
                """stt:
  enabled: true
  provider: openai-compatible
  providers:
    openai-compatible:
      model: gpt-4o-mini-transcribe
      base_url: https://api.openai.com/v1
      api_key_env: VOICE_TOOLS_OPENAI_KEY
""",
                encoding="utf-8",
            )
            selected = load_provider_config(config)
        self.assertEqual(selected.name, "openai-compatible")
        self.assertEqual(selected.endpoint, "https://api.openai.com/v1/audio/transcriptions")
        self.assertFalse(selected.attested)

    def test_unapproved_provider_fails_closed(self) -> None:
        from config import ProviderConfigurationError, load_provider_config

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.yaml"
            config.write_text("stt:\n  provider: local\n", encoding="utf-8")
            with self.assertRaises(ProviderConfigurationError):
                load_provider_config(config)


class ProviderTests(unittest.TestCase):
    POSTPROCESS = {
        "normalize_spoken_times": True,
        "replacements": [
            {"pattern": r"\bAcem\b", "replacement": "Acme"},
            {"pattern": r"\bAlice\b(?=(?:['\u2019]s)\b)", "replacement": "Alicia"},
            {
                "pattern": (
                    r"\bA[\s.-]*C[\s.-]*M[\s.-]*E\s+APP\b"
                ),
                "replacement": "Example.app",
            },
        ],
    }

    def test_postprocess_is_owned_by_voice_core(self) -> None:
        from postprocess import apply_transcription_postprocess

        self.assertEqual(
            apply_transcription_postprocess(
                "Ask Acem about Alice's A C M E APP notes at 09:00 p.m.",
                self.POSTPROCESS,
            ),
            "Ask Acme about Alicia's Example.app notes at 9pm",
        )

    def test_invalid_postprocess_rule_does_not_discard_valid_transcript(self) -> None:
        from postprocess import apply_transcription_postprocess

        self.assertEqual(
            apply_transcription_postprocess(
                "Acem",
                {
                    "replacements": [
                        {"pattern": "[", "replacement": "broken"},
                        {"pattern": r"\bAcem\b", "replacement": "Acme"},
                    ]
                },
            ),
            "Acme",
        )

    def test_tinfoil_contract_rejects_non_loopback_proxy(self) -> None:
        from config import ProviderConfig
        from providers import ProviderError, transcribe

        config = ProviderConfig(
            name="tinfoil",
            model="voxtral-small-24b",
            endpoint="https://example.com/v1/audio/transcriptions",
            api_key_env="TINFOIL_API_KEY",
            attested=True,
        )
        with tempfile.NamedTemporaryFile(suffix=".m4a") as audio:
            with self.assertRaises(ProviderError):
                transcribe(Path(audio.name), config, api_key="secret")

    @patch("providers.detect_no_speech", new=lambda source: False)
    @patch("providers.normalize_audio_chunks")
    @patch("providers.urllib.request.urlopen")
    def test_provider_returns_receipt_and_common_postprocessing(self, urlopen: MagicMock, normalize: MagicMock) -> None:
        from config import ProviderConfig
        from providers import transcribe

        def normalized_chunks(source: Path, directory: Path) -> list[Path]:
            output = directory / "voice-000.mp3"
            output.write_bytes(b"normalized")
            return [output]

        normalize.side_effect = normalized_chunks
        response = MagicMock()
        response.status = 200
        response.__enter__.return_value = response
        response.__exit__.return_value = False
        response.read.return_value = json.dumps(
            {"text": "Ask Acem about the sample project at 09:00 p.m."}
        ).encode()
        urlopen.return_value = response
        config = ProviderConfig(
            name="tinfoil",
            model="voxtral-small-24b",
            endpoint="http://127.0.0.1:3301/v1/audio/transcriptions",
            api_key_env="TINFOIL_API_KEY",
            attested=True,
            postprocess=self.POSTPROCESS,
        )
        with tempfile.NamedTemporaryFile(suffix=".m4a") as audio:
            result = transcribe(Path(audio.name), config, api_key="secret")
        self.assertEqual(result.text, "Ask Acme about the sample project at 9pm")
        self.assertEqual(result.provider, "tinfoil-attested")
        self.assertEqual(result.model, "voxtral-small-24b")

    @patch("providers.detect_no_speech", new=lambda source: False)
    @patch("providers.normalize_audio_chunks")
    @patch("providers.urllib.request.urlopen")
    def test_provider_transcribes_and_joins_multiple_chunks(self, urlopen: MagicMock, normalize: MagicMock) -> None:
        from config import ProviderConfig
        from providers import transcribe

        def normalized_chunks(source: Path, directory: Path) -> list[Path]:
            outputs = [directory / "voice-000.mp3", directory / "voice-001.mp3"]
            for index, output in enumerate(outputs):
                output.write_bytes(b"normalized-chunk-%d" % index)
            return outputs

        normalize.side_effect = normalized_chunks
        texts = {b"normalized-chunk-0": "First segment ends here.", b"normalized-chunk-1": "segment ends here. Second segment starts here."}

        def respond(request, *args, **kwargs):
            # Chunks upload concurrently: answer by the chunk actually sent, not by thread arrival order.
            body = request.data if isinstance(request.data, bytes) else b""
            text = next(t for marker, t in texts.items() if marker in body)
            response = MagicMock()
            response.status = 200
            response.__enter__.return_value = response
            response.__exit__.return_value = False
            response.read.return_value = json.dumps({"text": text}).encode()
            return response

        urlopen.side_effect = respond
        config = ProviderConfig(
            name="tinfoil",
            model="voxtral-small-24b",
            endpoint="http://127.0.0.1:3301/v1/audio/transcriptions",
            api_key_env="TINFOIL_API_KEY",
            attested=True,
        )
        with tempfile.NamedTemporaryFile(suffix=".m4a") as audio:
            result = transcribe(Path(audio.name), config, api_key="secret")

        self.assertEqual(result.text, "First segment ends here. Second segment starts here.")
        self.assertEqual(urlopen.call_count, 2)


class ServerTests(unittest.TestCase):
    @patch("server.route_transcription")
    def test_endpoint_returns_provider_and_model_receipt(self, route: MagicMock) -> None:
        from providers import TranscriptionResult
        import server

        route.return_value = TranscriptionResult(
            text="Which computer was the sample project?",
            provider="tinfoil-attested",
            model="voxtral-small-24b",
            latency_ms=123.4,
        )
        response = TestClient(server.app).post(
            "/v1/audio/transcriptions",
            files={"file": ("voice.m4a", b"audio-bytes", "audio/mp4")},
            data={"language": "en"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "text": "Which computer was the sample project?",
                "provider": "tinfoil-attested",
                "model": "voxtral-small-24b",
                "latency_ms": 123.4,
            },
        )
        route.assert_called_once()
        self.assertEqual(route.call_args.args[1:], ("en", ""))

    @patch("server.route_transcription")
    def test_endpoint_passes_request_scoped_context_prompt(self, route: MagicMock) -> None:
        from providers import TranscriptionResult
        import server

        route.return_value = TranscriptionResult(
            text="Synthetic transcription.", provider="tinfoil-attested",
            model="voxtral-small-24b", latency_ms=1.0,
        )
        response = TestClient(server.app).post(
            "/v1/audio/transcriptions",
            files={"file": ("voice.wav", b"audio-bytes", "audio/wav")},
            data={"language": "en", "prompt": "previous spoken context"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(route.call_args.args[1:], ("en", "previous spoken context"))

    @patch("server.route_transcription")
    def test_provider_failure_is_visible_without_fallback(self, route: MagicMock) -> None:
        from providers import ProviderError
        import server

        route.side_effect = ProviderError("attestation route unavailable")
        response = TestClient(server.app).post(
            "/v1/audio/transcriptions",
            files={"file": ("voice.m4a", b"audio-bytes", "audio/mp4")},
        )
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["detail"], "attestation route unavailable")


class ReconcileTests(unittest.TestCase):
    def test_tinfoil_restarts_realtime_service(self) -> None:
        import reconcile

        run = MagicMock()
        receipt = reconcile.reconcile("tinfoil", run=run)
        run.assert_called_once_with(
            ["/usr/bin/systemctl", "--user", "restart", "mizel-realtime.service"],
            check=True,
            timeout=60,
        )
        self.assertEqual(receipt["action"], "restart")

    def test_openai_compatible_stops_realtime_service(self) -> None:
        import reconcile

        run = MagicMock()
        receipt = reconcile.reconcile("openai-compatible", run=run)
        run.assert_called_once_with(
            ["/usr/bin/systemctl", "--user", "stop", "mizel-realtime.service"],
            check=True,
            timeout=60,
        )
        self.assertEqual(receipt["action"], "stop")


    def test_invalid_or_disabled_provider_stops_realtime_service(self) -> None:
        import reconcile

        run = MagicMock()
        receipt = reconcile.reconcile("invalid", run=run)
        run.assert_called_once_with(
            ["/usr/bin/systemctl", "--user", "stop", "mizel-realtime.service"],
            check=True,
            timeout=60,
        )
        self.assertEqual(receipt["action"], "stop")

    def test_unchanged_fingerprint_does_not_restart_live_sessions(self) -> None:
        import reconcile

        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state.json"
            state.write_text('{"fingerprint":"tinfoil:voxtral-mini-4b-realtime"}\n')
            run = MagicMock()
            receipt = reconcile.reconcile(
                "tinfoil",
                "voxtral-mini-4b-realtime",
                state_path=state,
                run=run,
            )
        run.assert_not_called()
        self.assertEqual(receipt["action"], "noop")


    def test_missing_enabled_key_defaults_to_enabled(self) -> None:
        import reconcile

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.yaml"
            config.write_text(
                "stt:\n  provider: tinfoil\n  providers:\n    tinfoil:\n      model: voxtral-small-24b\n      realtime_model: voxtral-mini-4b-realtime\n",
                encoding="utf-8",
            )
            with patch.object(reconcile, "DEFAULT_CONFIG_PATH", config):
                provider, model = reconcile.read_selection()
        self.assertEqual(provider, "tinfoil")
        self.assertEqual(model, "voxtral-mini-4b-realtime")

    def test_explicit_realtime_lane_is_independent_from_batch_provider(self) -> None:
        import reconcile

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.yaml"
            config.write_text(
                "stt:\n  enabled: true\n  provider: openai-compatible\n  providers:\n"
                "    openai-compatible:\n      model: gpt-4o-mini-transcribe\n      base_url: https://api.openai.com/v1\n"
                "    tinfoil:\n      model: voxtral-small-24b\n      realtime_model: voxtral-mini-4b-realtime\n"
                "voice:\n  lanes:\n    realtime_stt:\n      provider: tinfoil\n"
                "      model: voxtral-mini-4b-realtime\n",
                encoding="utf-8",
            )
            provider, model = reconcile.read_selection(config)
        self.assertEqual(provider, "tinfoil")
        self.assertEqual(model, "voxtral-mini-4b-realtime")

    def test_conversation_lane_has_its_own_model_fingerprint(self) -> None:
        import reconcile

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.yaml"
            config.write_text(
                "stt:\n  enabled: true\n  provider: tinfoil\n  providers:\n"
                "    tinfoil:\n      model: voxtral-small-24b\n      realtime_model: voxtral-mini-4b-realtime\n"
                "voice:\n  lanes:\n    conversation:\n      provider: tinfoil\n      models:\n"
                "        transcription: voxtral-mini-4b-realtime\n"
                "        reasoning: deepseek-v4-flash\n        speech: qwen3-tts\n        voice: ryan\n",
                encoding="utf-8",
            )
            provider, fingerprint = reconcile.read_conversation_selection(config)
        self.assertEqual(provider, "tinfoil")
        self.assertEqual(
            fingerprint,
            "voxtral-mini-4b-realtime:deepseek-v4-flash:qwen3-tts:ryan",
        )


    def test_realtime_condition_cli_uses_validated_explicit_lane(self) -> None:
        import reconcile

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.yaml"
            config.write_text(
                "stt:\n  enabled: true\n  provider: tinfoil\n  providers:\n"
                "    tinfoil:\n      model: voxtral-small-24b\n      realtime_model: voxtral-mini-4b-realtime\n"
                "voice:\n  lanes:\n    realtime_stt:\n      enabled: true\n      provider: tinfoil\n"
                "      model: voxtral-mini-4b-realtime\n",
                encoding="utf-8",
            )
            with patch.object(reconcile, "DEFAULT_CONFIG_PATH", config):
                self.assertEqual(reconcile.main(["--condition-lane", "realtime_stt"]), 0)

    def test_realtime_condition_cli_rejects_unsupported_model(self) -> None:
        import reconcile

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.yaml"
            config.write_text(
                "stt:\n  enabled: true\n  provider: tinfoil\n  providers:\n"
                "    tinfoil:\n      model: voxtral-small-24b\n      realtime_model: voxtral-mini-4b-realtime\n"
                "voice:\n  lanes:\n    realtime_stt:\n      enabled: true\n      provider: tinfoil\n"
                "      model: unsupported-realtime\n",
                encoding="utf-8",
            )
            with patch.object(reconcile, "DEFAULT_CONFIG_PATH", config):
                self.assertEqual(reconcile.main(["--condition-lane", "realtime_stt"]), 1)


class ClientTests(unittest.TestCase):
    @patch("client.urllib.request.urlopen")
    def test_client_writes_only_transcript_for_hermes_command_provider(self, urlopen: MagicMock) -> None:
        import client

        response = MagicMock()
        response.status = 200
        response.__enter__.return_value = response
        response.__exit__.return_value = False
        response.read.return_value = json.dumps(
            {
                "text": "Use the sample project.",
                "provider": "tinfoil-attested",
                "model": "voxtral-small-24b",
                "latency_ms": 101,
            }
        ).encode()
        urlopen.return_value = response
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "voice.m4a"
            output = Path(directory) / "transcript.txt"
            audio.write_bytes(b"audio")
            receipt = client.transcribe_file(audio)
            client.write_transcript(receipt, output)
            self.assertEqual(output.read_text(encoding="utf-8"), "Use the sample project.")
            self.assertEqual(receipt["provider"], "tinfoil-attested")


if __name__ == "__main__":
    unittest.main()
