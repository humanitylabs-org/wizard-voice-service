from __future__ import annotations

import hmac
import os
from dataclasses import asdict
from pathlib import Path

from dotenv import dotenv_values

from config import (
    DEFAULT_CONFIG_PATH,
    ProviderConfigurationError,
    VoiceControlConfig,
    VoiceLaneConfig,
    load_provider_config,
    load_voice_control_config,
)

HERMES_ENV = Path(os.environ.get("WIZARD_VOICE_ENV") or os.environ.get("HERMES_ENV") or str(Path.home() / ".hermes/.env"))


class VoiceControlError(RuntimeError):
    pass


def _credential_present(name: str, env_path: Path = HERMES_ENV) -> bool:
    if not name:
        return False
    if os.environ.get(name, "").strip():
        return True
    if env_path.is_file():
        return bool(str(dotenv_values(env_path).get(name) or "").strip())
    return False


def _safe_lane(lane: VoiceLaneConfig) -> dict[str, object]:
    return {
        "name": lane.name,
        "enabled": lane.enabled,
        "managed": lane.managed,
        "provider": lane.provider or None,
        "model": lane.model or None,
        "transport": lane.transport or None,
        "models": dict(lane.models),
    }


def safe_voice_config(path: Path | str | None = None) -> dict[str, object]:
    control = load_voice_control_config(path)
    batch_provider = load_provider_config(path, require_enabled=False)
    return {
        "schema_version": control.schema_version,
        "config_hash": control.config_hash,
        "lanes": {
            "batch_stt": {
                **_safe_lane(control.batch_stt),
                "attested": batch_provider.attested,
                "privacy": batch_provider.privacy if batch_provider.name == "parakeet" else ("attested-confidential" if batch_provider.attested else "provider-policy"),
                "credential": "not-required" if batch_provider.name == "parakeet" else ("configured" if _credential_present(batch_provider.api_key_env) else "missing"),
                "fallback": None,
            },
            "realtime_stt": _safe_lane(control.realtime_stt),
            "conversation": _safe_lane(control.conversation),
        },
        "invariants": {
            "provider_credentials_in_clients": False,
            "provider_endpoints_in_clients": False,
            "silent_cross_provider_fallback": False,
        },
    }


def safe_capabilities(path: Path | str | None = None) -> dict[str, object]:
    safe = safe_voice_config(path)
    return {
        "schema_version": safe["schema_version"],
        "config_hash": safe["config_hash"],
        "lanes": safe["lanes"],
        "jobs": {
            "version": 1,
            "admission": "/v1/voice/jobs",
            "status_template": "/v1/voice/jobs/{job_id}",
            "completion": "poll",
        },
    }


def require_admin_token(provided: str | None) -> None:
    expected = os.environ.get("WIZARD_VOICE_ADMIN_TOKEN", "").strip()
    if not expected:
        raise VoiceControlError("voice administration is not enabled")
    if not provided or not hmac.compare_digest(provided.strip(), expected):
        raise VoiceControlError("voice administration is unauthorized")


def no_op_patch(
    proposed: dict[str, object],
    *,
    expected_hash: str,
    path: Path | str | None = None,
) -> dict[str, object]:
    """Validate a read-only bridge save; real mutation is intentionally absent.

    The current implementation is safe to expose before any UI exists: an exact
    redacted readback can be submitted as a no-op, while endpoint/secret/profile
    mutation is impossible. Config writes arrive in a later, separately reviewed
    change.
    """
    current = safe_voice_config(path)
    current_hash = str(current["config_hash"])
    if not expected_hash or not hmac.compare_digest(expected_hash, current_hash):
        raise VoiceControlError("voice configuration changed; refresh before saving")
    submitted = dict(proposed)
    submitted.pop("config_hash", None)
    comparable = dict(current)
    comparable.pop("config_hash", None)
    if submitted != comparable:
        raise VoiceControlError("voice configuration writes are not enabled")
    return {"action": "noop", **current}
