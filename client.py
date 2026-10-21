#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import secrets
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_ENDPOINT = os.environ.get(
    "WIZARD_STT_ENDPOINT", "http://127.0.0.1:8864/v1/audio/transcriptions"
).strip()
DEFAULT_HEALTH_ENDPOINT = os.environ.get(
    "WIZARD_STT_HEALTH_ENDPOINT", "http://127.0.0.1:8864/health"
).strip()


class ClientError(RuntimeError):
    pass


def _multipart(audio: Path, language: str) -> tuple[bytes, str]:
    boundary = f"wizard-stt-client-{secrets.token_hex(16)}"
    mime = mimetypes.guess_type(audio.name)[0] or "application/octet-stream"
    body = b"".join(
        [
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\n{language}\r\n".encode(),
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{audio.name}\"\r\nContent-Type: {mime}\r\n\r\n".encode(),
            audio.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return body, f"multipart/form-data; boundary={boundary}"


def _multipart_fields(audio: Path, fields: dict[str, str]) -> tuple[bytes, str]:
    boundary = f"wizard-voice-client-{secrets.token_hex(16)}"
    mime = mimetypes.guess_type(audio.name)[0] or "application/octet-stream"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
        )
    parts.extend(
        [
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{audio.name}\"\r\nContent-Type: {mime}\r\n\r\n".encode(),
            audio.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _json_request(url: str, *, data: bytes | None = None, content_type: str = "", timeout: int = 30) -> dict[str, object]:
    headers = {"Accept": "application/json"}
    if content_type:
        headers["Content-Type"] = content_type
    request = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        try:
            detail_payload = json.loads(exc.read(4096))
            detail = str(detail_payload.get("detail") or detail_payload.get("error") or "")
        except (json.JSONDecodeError, AttributeError):
            detail = ""
        message = f"Wizard Voice returned HTTP {exc.code}"
        if detail:
            message += f": {detail[:400]}"
        raise ClientError(message) from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ClientError(f"Wizard Voice unavailable ({type(exc).__name__})") from exc
    if not isinstance(payload, dict):
        raise ClientError("Wizard Voice returned invalid data")
    return payload


def submit_voice_job(
    path: Path | str,
    *,
    source_app: str,
    device_id: str,
    idempotency_key: str,
    mode: str = "dictation",
    intent: str = "transcribe",
    destination: str = "client.transcript",
    destination_params: dict[str, object] | None = None,
    dispatch: str = "client",
    language: str = "en",
    endpoint: str = "http://127.0.0.1:8864/v1/voice/jobs",
    timeout: int = 30,
) -> dict[str, object]:
    audio = Path(path).expanduser().resolve()
    if not audio.is_file() or audio.stat().st_size == 0:
        raise ClientError("audio file is missing or empty")
    body, content_type = _multipart_fields(
        audio,
        {
            "source_app": source_app,
            "device_id": device_id,
            "idempotency_key": idempotency_key,
            "mode": mode,
            "intent": intent,
            "destination": destination,
            "destination_params": json.dumps(destination_params or {}, ensure_ascii=False, separators=(",", ":")),
            "dispatch": dispatch,
            "language": language,
        },
    )
    payload = _json_request(endpoint, data=body, content_type=content_type, timeout=timeout)
    if not payload.get("job_id") or not payload.get("audio_sha256"):
        raise ClientError("Wizard Voice returned an invalid admission receipt")
    return payload


def get_voice_job(status_url: str, *, base_url: str = "http://127.0.0.1:8864", timeout: int = 10) -> dict[str, object]:
    url = status_url if status_url.startswith("http://") or status_url.startswith("https://") else base_url.rstrip("/") + "/" + status_url.lstrip("/")
    return _json_request(url, timeout=timeout)


def wait_voice_job(
    status_url: str,
    *,
    base_url: str = "http://127.0.0.1:8864",
    timeout: float = 240,
    poll_seconds: float = 0.5,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = get_voice_job(status_url, base_url=base_url)
        state = str(job.get("state") or "")
        if state == "completed":
            return job
        if state == "failed_terminal":
            raise ClientError(str(job.get("error") or "voice job failed"))
        time.sleep(poll_seconds)
    raise ClientError("Wizard Voice job timed out")


def transcribe_file(
    path: Path | str,
    *,
    language: str = "en",
    endpoint: str = DEFAULT_ENDPOINT,
    timeout: int = 240,
) -> dict[str, object]:
    audio = Path(path).expanduser().resolve()
    if not audio.is_file() or audio.stat().st_size == 0:
        raise ClientError("audio file is missing or empty")
    body, content_type = _multipart(audio, language[:12] or "en")
    request = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={"Content-Type": content_type, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
            status = response.status
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read(4096))
            detail = str(payload.get("detail") or payload.get("error") or "")
        except (json.JSONDecodeError, AttributeError):
            detail = ""
        raise ClientError(detail[:500] or f"system STT returned HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ClientError(f"system STT unavailable ({type(exc).__name__})") from exc
    text = str(payload.get("text") if isinstance(payload, dict) else "").strip()
    if status != 200 or not text:
        raise ClientError("system STT returned no transcript")
    return {
        "text": text,
        "transcript": text,
        "provider": str(payload.get("provider") or "configured"),
        "model": str(payload.get("model") or ""),
        "latency_ms": payload.get("latency_ms"),
    }


def get_health(
    *,
    endpoint: str = DEFAULT_HEALTH_ENDPOINT,
    timeout: int = 5,
) -> dict[str, object]:
    request = urllib.request.Request(endpoint, method="GET", headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ClientError(f"system STT health unavailable ({type(exc).__name__})") from exc
    if not isinstance(payload, dict):
        raise ClientError("system STT health returned invalid data")
    return payload


def write_transcript(receipt: dict[str, object], output: Path | str) -> None:
    Path(output).write_text(str(receipt.get("text") or "").strip(), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Transcribe through the WizardOS system STT broker")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--language", default="en")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        receipt = transcribe_file(args.input, language=args.language)
        if args.output:
            write_transcript(receipt, args.output)
        if args.json:
            print(json.dumps(receipt, ensure_ascii=False, separators=(",", ":")))
        elif not args.output:
            print(receipt["text"])
        return 0
    except ClientError as exc:
        print(str(exc), file=__import__("sys").stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
