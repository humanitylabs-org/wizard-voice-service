from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import yaml

LEGACY_CONFIG_PATH = Path.home() / ".hermes/config.yaml"
SERVICE_CONFIG_PATH = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "wizard-voice/config.yaml"
DEFAULT_CONFIG_PATH = SERVICE_CONFIG_PATH if SERVICE_CONFIG_PATH.exists() else LEGACY_CONFIG_PATH
CORRECTIONS_PATH = Path(__file__).resolve().parent / "corrections.yaml"


def _owned_corrections(path: Path | str | None = None) -> list[dict]:
    """Wizard Voice-owned corrections, appended after the system stt.postprocess rules."""
    target = Path(path or os.environ.get("WIZARD_STT_CORRECTIONS", CORRECTIONS_PATH)).expanduser()
    try:
        raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return []
    rules = raw.get("replacements") if isinstance(raw, dict) else None
    return [r for r in rules if isinstance(r, dict)] if isinstance(rules, list) else []


def _merged_postprocess(raw_postprocess: object) -> dict[str, object]:
    postprocess = dict(raw_postprocess) if isinstance(raw_postprocess, dict) else {}
    system_rules = postprocess.get("replacements")
    system_rules = list(system_rules) if isinstance(system_rules, list) else []
    owned = _owned_corrections()
    if owned:
        postprocess["replacements"] = system_rules + owned
    return postprocess
APPROVED_PROVIDERS = {"tinfoil", "openai-compatible", "parakeet"}
APPROVED_REALTIME_STT_PROVIDERS = {"tinfoil", "openai-compatible"}
APPROVED_CONVERSATION_PROVIDERS = {"tinfoil", "openai-realtime"}


class ProviderConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    model: str
    endpoint: str
    api_key_env: str
    attested: bool
    language: str = "en"
    timeout: int = 180
    postprocess: dict[str, object] = field(default_factory=dict)
    profile: str = ""
    privacy: str = "provider"


@dataclass(frozen=True)
class VoiceLaneConfig:
    name: str
    enabled: bool
    managed: bool
    provider: str
    model: str
    transport: str
    models: dict[str, str]


@dataclass(frozen=True)
class VoiceControlConfig:
    schema_version: int
    config_hash: str
    batch_stt: VoiceLaneConfig
    realtime_stt: VoiceLaneConfig
    conversation: VoiceLaneConfig


def _mapping(value: object, label: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ProviderConfigurationError(f"{label} must be a mapping")
    return value


def _endpoint(base_url: str) -> str:
    return base_url.rstrip("/") + "/v1/audio/transcriptions"


def _read_root(path: Path | str | None = None) -> tuple[Path, dict]:
    config_path = Path(path or os.environ.get("WIZARD_VOICE_CONFIG") or os.environ.get("WIZARD_STT_CONFIG") or DEFAULT_CONFIG_PATH).expanduser()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ProviderConfigurationError(f"cannot read STT config: {exc}") from exc
    return config_path, _mapping(raw, "config")


def _voice_hash(root: dict) -> str:
    relevant = {"stt": root.get("stt") or {}, "voice": root.get("voice") or {}}
    canonical = json.dumps(relevant, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _lane(
    name: str,
    raw: dict,
    *,
    fallback_provider: str = "",
    fallback_model: str = "",
    fallback_models: dict[str, str] | None = None,
    fallback_transport: str = "",
) -> VoiceLaneConfig:
    configured = bool(raw)
    provider = str(raw.get("provider") or fallback_provider).strip().lower()
    model = str(raw.get("model") or fallback_model).strip()
    models = {
        str(key): str(value).strip()
        for key, value in (raw.get("models") or fallback_models or {}).items()
        if str(value).strip()
    }
    return VoiceLaneConfig(
        name=name,
        enabled=bool(raw.get("enabled", bool(provider and (model or models)))),
        managed=configured,
        provider=provider,
        model=model,
        transport=str(raw.get("transport") or fallback_transport).strip(),
        models=models,
    )


def load_voice_control_config(path: Path | str | None = None) -> VoiceControlConfig:
    """Load independent batch, realtime-STT, and conversation lanes.

    Existing installations remain valid: batch STT comes from ``stt.provider``;
    realtime STT temporarily inherits that provider's ``realtime_model`` until
    an explicit ``voice.lanes.realtime_stt`` block is configured. Conversation
    stays visibly unmanaged until its own lane is configured.
    """
    config_path, root = _read_root(path)
    stt = _mapping(root.get("stt"), "stt")
    selected = load_provider_config(config_path, require_enabled=False)
    providers = _mapping(stt.get("providers"), "stt.providers")
    # Batch profiles must never implicitly switch the legacy realtime lane.
    realtime_provider_name = str(stt.get("provider") or selected.name)
    selected_raw = _mapping(providers.get(realtime_provider_name), f"stt.providers.{realtime_provider_name}")
    voice = _mapping(root.get("voice"), "voice")
    lanes = _mapping(voice.get("lanes"), "voice.lanes")

    batch = VoiceLaneConfig(
        name="batch_stt",
        enabled=stt.get("enabled") is not False,
        managed=True,
        provider=selected.name,
        model=selected.model,
        transport="file",
        models={"transcription": selected.model},
    )
    realtime = _lane(
        "realtime_stt",
        _mapping(lanes.get("realtime_stt"), "voice.lanes.realtime_stt"),
        fallback_provider=realtime_provider_name,
        fallback_model=str(selected_raw.get("realtime_model") or ""),
        fallback_transport="websocket",
    )
    conversation = _lane(
        "conversation",
        _mapping(lanes.get("conversation"), "voice.lanes.conversation"),
        fallback_transport="websocket",
    )
    schema_version = int(voice.get("schema_version") or 1)
    if schema_version != 1:
        raise ProviderConfigurationError("voice.schema_version must be 1")
    if realtime.enabled:
        if realtime.provider not in APPROVED_REALTIME_STT_PROVIDERS:
            raise ProviderConfigurationError("realtime STT provider is not approved")
        if not realtime.model:
            raise ProviderConfigurationError("realtime STT model is required when enabled")
        realtime_provider = _mapping(
            providers.get(realtime.provider), f"stt.providers.{realtime.provider}"
        )
        approved_realtime_model = str(realtime_provider.get("realtime_model") or "").strip()
        if not approved_realtime_model or realtime.model != approved_realtime_model:
            raise ProviderConfigurationError("realtime STT model is not approved for the selected provider")
    if conversation.enabled:
        if conversation.provider not in APPROVED_CONVERSATION_PROVIDERS:
            raise ProviderConfigurationError("conversation provider is not approved")
        required_models = {"transcription", "reasoning", "speech", "voice"}
        if required_models - set(conversation.models):
            raise ProviderConfigurationError("conversation requires transcription, reasoning, speech, and voice models")
        if conversation.provider == "tinfoil":
            tinfoil = _mapping(providers.get("tinfoil"), "stt.providers.tinfoil")
            approved_transcription = str(tinfoil.get("realtime_model") or "").strip()
            if conversation.models.get("transcription") != approved_transcription:
                raise ProviderConfigurationError("conversation transcription model is not approved")
    return VoiceControlConfig(
        schema_version=schema_version,
        config_hash=_voice_hash(root),
        batch_stt=batch,
        realtime_stt=realtime,
        conversation=conversation,
    )


def _router_tinfoil(profile: dict) -> dict:
    """Future router projection; no implicit legacy proxy or key fallback."""
    missing = "Tinfoil profile is disabled: no router connection config (WIZARD_VOICE_TINFOIL_CONNECTION)"
    path = os.environ.get("WIZARD_VOICE_TINFOIL_CONNECTION", "")
    if not path:
        raise ProviderConfigurationError(missing)
    try:
        connection = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProviderConfigurationError("Tinfoil router connection config is unavailable or invalid") from exc
    if not isinstance(connection, dict) or (
        connection.get("schema_version") != 1
        or connection.get("provider") != "tinfoil"
        or connection.get("model") != "voxtral-small-24b"
        or connection.get("endpoint") != "http://127.0.0.1:3301/v1/audio/transcriptions"
        or connection.get("attestation_required") is not True
        or connection.get("api_key_env") != "WIZARD_VOICE_TINFOIL_KEY"
    ):
        raise ProviderConfigurationError("Tinfoil connection requires the approved loopback attestation proxy, model and auth binding")
    if not os.environ.get("WIZARD_VOICE_TINFOIL_KEY", "").strip():
        raise ProviderConfigurationError("Tinfoil router connection credential is unavailable")
    return dict(profile, enabled=True, proxy_url="http://127.0.0.1:3301",
                secret_env="WIZARD_VOICE_TINFOIL_KEY")


def is_local_transcriber(config: ProviderConfig) -> bool:
    return (config.name == "openai-compatible" and config.privacy == "local-box"
            and config.endpoint == "http://transcriber:8000/v1/audio/transcriptions"
            and not config.api_key_env)


def load_provider_config(
    path: Path | str | None = None,
    *,
    require_enabled: bool = True,
) -> ProviderConfig:
    _config_path, root = _read_root(path)
    stt = _mapping(root.get("stt"), "stt")
    if require_enabled and stt.get("enabled") is False:
        raise ProviderConfigurationError("system STT is disabled")
    profile = stt.get("active_profile", "")
    profile_raw = None
    if profile:
        profiles = _mapping(stt.get("profiles"), "stt.profiles")
        if not isinstance(profile, str) or profile not in profiles:
            raise ProviderConfigurationError("unknown active profile")
        profile_raw = _mapping(profiles[profile], "selected profile")
    name = str((profile_raw if profile_raw is not None else stt).get("provider") or "").strip().lower()
    if name not in APPROVED_PROVIDERS:
        raise ProviderConfigurationError(
            f"stt.provider must be one of {', '.join(sorted(APPROVED_PROVIDERS))}; got {name or 'unset'}"
        )
    providers = _mapping(stt.get("providers"), "stt.providers")
    selected = profile_raw if profile_raw is not None else _mapping(providers.get(name), f"stt.providers.{name}")
    if selected.get("connection") == "router-tinfoil":
        selected = _router_tinfoil(selected)
    elif selected.get("enabled") is False:
        raise ProviderConfigurationError(f"profile {profile or name} is disabled")
    postprocess = _merged_postprocess(stt.get("postprocess"))
    from voice_rules import RuleError, read_rules
    rule_path = os.environ.get("WIZARD_VOICE_RULES") or _mapping(root.get("voice"), "voice").get("rules_file")
    if rule_path:
        if not isinstance(rule_path, str):
            raise ProviderConfigurationError("voice.rules_file must be a path string")
        rule_path = Path(rule_path).expanduser()
        if not rule_path.is_absolute():
            rule_path = _config_path.parent / rule_path
    try:
        if rule_path:
            postprocess["rules"] = read_rules(rule_path)
    except RuleError as exc:
        raise ProviderConfigurationError(str(exc)) from exc
    model = str(selected.get("model") or "").strip()
    if not model:
        raise ProviderConfigurationError(f"stt.providers.{name}.model is required")
    timeout = int(selected.get("timeout") or 180)
    if timeout < 1 or timeout > 900:
        raise ProviderConfigurationError(f"stt.providers.{name}.timeout must be between 1 and 900")
    language = str(selected.get("language") or stt.get("language") or "en").strip() or "en"

    if name == "parakeet":
        base_url = str(selected.get("base_url") or "").strip()
        parsed = urlparse(base_url)
        # Remote hosts require an owner-managed encrypted tunnel.
        try:
            port = parsed.port
        except ValueError as exc:
            raise ProviderConfigurationError("Parakeet adapter port is invalid") from exc
        if (parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1"}
                or parsed.username or parsed.password or parsed.path not in {"", "/"}
                or parsed.query or parsed.fragment or not port):
            raise ProviderConfigurationError("Parakeet requires an explicit loopback HTTP adapter port (tunnel for remote)")
        if model != "parakeet-tdt-0.6b-v3":
            raise ProviderConfigurationError("Parakeet model must be parakeet-tdt-0.6b-v3")
        privacy = selected.get("privacy")
        if privacy not in {"local-host", "private-remote"}:
            raise ProviderConfigurationError("Parakeet privacy must be local-host or private-remote")
        return ProviderConfig(name=name, model=model, endpoint=_endpoint(base_url),
            api_key_env="", attested=False, language=language, timeout=timeout,
            postprocess=postprocess, profile=profile, privacy=privacy)

    if name == "tinfoil":
        base_url = str(selected.get("proxy_url") or "http://127.0.0.1:3301").strip()
        if model != "voxtral-small-24b":
            raise ProviderConfigurationError("Tinfoil model must be voxtral-small-24b")
        parsed = urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
            raise ProviderConfigurationError("Tinfoil attestation proxy must be loopback HTTP")
        return ProviderConfig(
            name=name,
            model=model,
            endpoint=_endpoint(base_url),
            api_key_env=str(
                selected.get("secret_env")
                or selected.get("api_key_env")
                or "TINFOIL_API_KEY"
            ).strip(),
            attested=True,
            profile=profile,
            privacy="attested-cloud",
            language=language,
            timeout=timeout,
            postprocess=postprocess,
        )

    base_url = str(selected.get("base_url") or "https://api.openai.com/v1").strip()
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ProviderConfigurationError("OpenAI-compatible base_url is invalid")
    local_engine = selected.get("local_engine")
    if local_engine is not None:
        if local_engine != "transcriber" or base_url not in {"http://transcriber:8000", "http://transcriber:8000/v1"}:
            raise ProviderConfigurationError("local_engine must use the installed transcriber at http://transcriber:8000")
    elif parsed.scheme != "https" and parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ProviderConfigurationError("OpenAI-compatible base_url must use HTTPS or loopback HTTP")
    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("/v1"):
        endpoint += "/v1"
    return ProviderConfig(
        name=name,
        model=model,
        endpoint=endpoint + "/audio/transcriptions",
        api_key_env="" if local_engine else str(
            selected.get("secret_env")
            or selected.get("api_key_env")
            or "VOICE_TOOLS_OPENAI_KEY"
        ).strip(),
        attested=False,
        language=language,
        timeout=timeout,
        postprocess=postprocess,
        profile=profile,
        privacy="local-box" if local_engine else "provider",
    )
