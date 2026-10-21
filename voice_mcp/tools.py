"""Narrow Files-relative transcription tools, following Transcriber's MCP contract."""
import os
from pathlib import Path

from fastapi import UploadFile


def transcribe(path, mode="dictation", language="en"):
    from server import MAX_AUDIO_BYTES, _transcribe_upload
    from voice_rules import MODES
    if not isinstance(path, str) or not path or Path(path).is_absolute():
        raise ValueError("path must be relative to the owner's Files root")
    if mode not in MODES or not isinstance(language, str) or len(language) > 12:
        raise ValueError("invalid mode or language")
    root = Path(os.environ.get("WIZARD_VOICE_MEDIA_DIR", "/media")).resolve()
    full = (root / path).resolve()
    if not full.is_relative_to(root) or not full.is_file():
        raise ValueError("file must be inside the owner's Files root")
    if full.stat().st_size > MAX_AUDIO_BYTES:
        raise ValueError("audio is larger than the upload limit")
    with full.open("rb") as source:
        return _transcribe_upload(UploadFile(filename=full.name, file=source), language, mode=mode)


def voice_status():
    from server import health
    from config import load_provider_config

    status = health()
    if status["status"] == "ok":
        cfg = load_provider_config()
        rules = cfg.postprocess.get("rules", [])
        status["personal_rules"] = sum(r.get("scope") == "user" for r in rules)
        status["legacy_replacements"] = len(cfg.postprocess.get("replacements", []))
        status["enabled_style_rules"] = sum(r["kind"] == "style" and r["enabled"] for r in rules)
    status["engine_readiness_checked"] = False
    return status


TOOLS = {
    "transcribe": ({"name": "transcribe", "description": "Transcribe a Files-relative audio file in dictation, meeting or verbatim mode. Local engine by default; no automatic fallback.",
                    "inputSchema": {"type": "object", "additionalProperties": False, "properties": {
                        "path": {"type": "string"}, "mode": {"type": "string", "enum": ["dictation", "meeting", "verbatim"]},
                        "language": {"type": "string", "maxLength": 12}}, "required": ["path"]}}, transcribe),
    "voice_status": ({"name": "voice_status", "description": "Read provider configuration and clean-rule counts; not an engine readiness or attestation check.",
                      "inputSchema": {"type": "object", "additionalProperties": False, "properties": {}}}, voice_status),
}
