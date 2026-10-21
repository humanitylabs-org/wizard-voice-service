#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, Header, HTTPException, UploadFile
from starlette.middleware.trustedhost import TrustedHostMiddleware

from config import ProviderConfigurationError, load_provider_config
from control import VoiceControlError, no_op_patch, require_admin_token, safe_capabilities, safe_voice_config
from jobs import DEFAULT_DATA_DIR, VoiceJobConflict, VoiceJobError, VoiceJobStore, VoiceJobWorker
from providers import NoSpeechError, ProviderError, route_transcription

MAX_AUDIO_BYTES = int(os.environ.get("WIZARD_STT_MAX_AUDIO_BYTES", str(32 * 1024 * 1024)))

@asynccontextmanager
async def lifespan(_app: FastAPI):
    _start_jobs()
    try:
        yield
    finally:
        _stop_jobs()


app = FastAPI(title="Wizard Voice Service", version="0.2.1rc2", docs_url=None, redoc_url=None, lifespan=lifespan)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=[
        host.strip()
        for host in os.environ.get("WIZARD_VOICE_ALLOWED_HOSTS", "127.0.0.1,localhost,testserver").split(",")
        if host.strip()
    ],
)
_jobs_lock = threading.Lock()
_job_store: VoiceJobStore | None = None
_job_worker: VoiceJobWorker | None = None


def _jobs() -> tuple[VoiceJobStore, VoiceJobWorker]:
    global _job_store, _job_worker
    with _jobs_lock:
        if _job_store is None:
            data_dir = Path(
                os.environ.get(
                    "WIZARD_VOICE_DATA_DIR",
                    str(DEFAULT_DATA_DIR),
                )
            )
            _job_store = VoiceJobStore(data_dir)
            _job_worker = VoiceJobWorker(_job_store)
        assert _job_worker is not None
        return _job_store, _job_worker


def _start_jobs() -> None:
    _jobs()[1].start()


def _stop_jobs() -> None:
    if _job_worker is not None:
        _job_worker.stop()


@app.get("/")
@app.get("/health")
def health() -> dict[str, object]:
    try:
        selected = load_provider_config()
    except ProviderConfigurationError as exc:
        return {"status": "misconfigured", "service": "wizard-stt", "error": str(exc)}
    return {
        "status": "ok",
        "service": "wizard-stt",
        "product": "Wizard Voice Service",
        "provider": selected.name,
        "model": selected.model,
        "attested": selected.attested,
        "fallback": None,
    }


@app.get("/v1/voice/capabilities")
def voice_capabilities() -> dict[str, object]:
    try:
        return safe_capabilities()
    except ProviderConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)[:500]) from exc


def _admin(token: str | None) -> None:
    try:
        require_admin_token(token)
    except VoiceControlError as exc:
        status = 503 if "not enabled" in str(exc) else 401
        raise HTTPException(status_code=status, detail=str(exc)) from exc


@app.get("/admin/v1/voice/config")
def voice_admin_config(x_wizard_voice_admin: str | None = Header(default=None)) -> dict[str, object]:
    _admin(x_wizard_voice_admin)
    try:
        return safe_voice_config()
    except ProviderConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)[:500]) from exc


@app.patch("/admin/v1/voice/config")
def voice_admin_patch(
    proposed: dict[str, object] = Body(...),
    if_match: str | None = Header(default=None, alias="If-Match"),
    x_wizard_voice_admin: str | None = Header(default=None),
) -> dict[str, object]:
    _admin(x_wizard_voice_admin)
    try:
        return no_op_patch(proposed, expected_hash=(if_match or "").strip('"'))
    except VoiceControlError as exc:
        status = 412 if "changed" in str(exc) else 409
        raise HTTPException(status_code=status, detail=str(exc)) from exc


FAILED_AUDIO_DIR = Path(os.environ.get("WIZARD_STT_FAILED_AUDIO_DIR", str(DEFAULT_DATA_DIR / "failed")))
FAILED_AUDIO_TTL_SECONDS = 48 * 3600


def _retain_failure(temp_path: Path, suffix: str, reason: str) -> None:
    """Keep a private copy of audio that failed transcription so the exact
    recording can be replayed during repair. Owner-only, pruned after 48 h.
    Never stores transcript text."""
    import hashlib
    import shutil
    import sys
    import time as _time

    if os.environ.get("WIZARD_VOICE_RETAIN_FAILED_AUDIO", "0") != "1":
        return
    try:
        FAILED_AUDIO_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
        now = _time.time()
        for old in FAILED_AUDIO_DIR.iterdir():
            if now - old.stat().st_mtime > FAILED_AUDIO_TTL_SECONDS:
                old.unlink(missing_ok=True)
        digest = hashlib.sha256(temp_path.read_bytes()).hexdigest()
        target = FAILED_AUDIO_DIR / f"{digest}{suffix}"
        if not target.exists():
            shutil.copyfile(temp_path, target)
            target.chmod(0o600)
        print(f"wizard-stt failure sha256={digest} reason={reason[:120]!r}", file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001 - diagnostics must never change the response
        pass


def _transcribe_upload(file: UploadFile, language: str, prompt: str = "", mode: str = "dictation") -> dict[str, object]:
    from voice_rules import MODES
    if mode not in MODES:
        raise HTTPException(status_code=400, detail="unknown transcription mode")
    suffix = Path(file.filename or "recording.wav").suffix[:12] or ".wav"
    total = 0
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix="wizard-stt-", suffix=suffix, delete=False) as handle:
            temp_path = Path(handle.name)
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_AUDIO_BYTES:
                    raise HTTPException(status_code=413, detail="audio too large")
                handle.write(chunk)
        if total == 0:
            raise HTTPException(status_code=400, detail="audio is empty")
        try:
            options = {"mode": mode} if mode != "dictation" else {}
            result = route_transcription(temp_path, (language or "en")[:12], prompt, **options)
        except NoSpeechError as exc:
            raise HTTPException(status_code=422, detail="No speech detected") from exc
        except ProviderError as exc:
            _retain_failure(temp_path, suffix, str(exc))
            raise HTTPException(status_code=502, detail=str(exc)[:500]) from exc
        return {
            "text": result.text,
            "provider": result.provider,
            "model": result.model,
            "latency_ms": result.latency_ms,
        }
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


@app.post("/inference")
@app.post("/v1/audio/transcriptions")
def inference(
    file: UploadFile = File(...),
    language: str = Form(default="en"),
    prompt: str = Form(default=""),
    mode: str = Form(default="dictation"),
) -> dict[str, object]:
    return _transcribe_upload(file, language, prompt, mode)


@app.post("/admin/v1/voice/test")
def voice_admin_test(
    file: UploadFile = File(...),
    language: str = Form(default="en"),
    x_wizard_voice_admin: str | None = Header(default=None),
) -> dict[str, object]:
    _admin(x_wizard_voice_admin)
    receipt = _transcribe_upload(file, language)
    receipt["config_hash"] = str(safe_voice_config()["config_hash"])
    return receipt


@app.post("/v1/voice/jobs", status_code=202)
def submit_voice_job(
    file: UploadFile = File(...),
    source_app: str = Form(...),
    device_id: str = Form(...),
    idempotency_key: str = Form(...),
    mode: str = Form(default="dictation"),
    intent: str = Form(default="transcribe"),
    destination: str = Form(default="client.transcript"),
    destination_params: str = Form(default="{}"),
    dispatch: str = Form(default="client"),
    language: str = Form(default="en"),
) -> dict[str, object]:
    suffix = Path(file.filename or "recording.wav").suffix[:12] or ".wav"
    temp_path: Path | None = None
    total = 0
    try:
        with tempfile.NamedTemporaryFile(prefix="wizard-voice-admission-", suffix=suffix, delete=False) as handle:
            temp_path = Path(handle.name)
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_AUDIO_BYTES:
                    raise HTTPException(status_code=413, detail="audio too large")
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        if total == 0:
            raise HTTPException(status_code=400, detail="audio is empty")
        try:
            params = json.loads(destination_params or "{}")
            if not isinstance(params, dict):
                raise ValueError
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="destination_params must be a JSON object")

        store, worker = _jobs()
        try:
            admission = store.accept(
                temp_path,
                source_app=source_app,
                device_id=device_id,
                idempotency_key=idempotency_key,
                mode=mode,
                intent=intent,
                destination=destination,
                destination_params=params,
                dispatch=dispatch,
                language=language,
            )
        except VoiceJobConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except VoiceJobError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        worker.wake()
        return {
            "job_id": admission.job_id,
            "state": admission.state,
            "audio_sha256": admission.audio_sha256,
            "accepted_at": admission.accepted_at,
            "duplicate": admission.duplicate,
            "status_url": f"/v1/voice/jobs/{admission.job_id}",
        }
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


@app.get("/v1/voice/jobs/{job_id}")
def voice_job_status(job_id: str) -> dict[str, object]:
    try:
        job = _jobs()[0].get(job_id)
    except VoiceJobError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if job is None:
        raise HTTPException(status_code=404, detail="voice job not found")
    return job


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("WIZARD_STT_PORT", "8864")))
