from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from config import load_voice_control_config
from providers import NoSpeechError, TranscriptionResult, route_transcription

_LEGACY_DATA_DIR = Path.home() / ".hermes/data/apps/wizard-voice"
DEFAULT_DATA_DIR = _LEGACY_DATA_DIR if _LEGACY_DATA_DIR.exists() else Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share"))) / "wizard-voice"
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_DESTINATION_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}\.[a-z][a-z0-9_-]{0,63}$")
FINAL_STATES = {"completed", "failed_terminal"}


class VoiceJobError(RuntimeError):
    pass


class VoiceJobConflict(VoiceJobError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bounded(value: str, label: str, *, maximum: int, pattern: re.Pattern[str] | None = None) -> str:
    cleaned = str(value or "").strip()
    if len(cleaned) > maximum or (pattern and cleaned and not pattern.fullmatch(cleaned)):
        raise VoiceJobError(f"invalid {label}")
    return cleaned


@dataclass(frozen=True)
class VoiceJobAdmission:
    job_id: str
    state: str
    audio_sha256: str
    accepted_at: str
    duplicate: bool = False


class VoiceJobStore:
    def __init__(self, data_dir: Path | str = DEFAULT_DATA_DIR) -> None:
        self.data_dir = Path(data_dir).expanduser()
        self.audio_dir = self.data_dir / "audio"
        self.db_path = self.data_dir / "jobs.sqlite3"
        self._lock = threading.RLock()
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.data_dir, 0o700)
        os.chmod(self.audio_dir, 0o700)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS voice_jobs (
                    job_id TEXT PRIMARY KEY,
                    source_app TEXT NOT NULL,
                    device_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    destination TEXT NOT NULL,
                    destination_params TEXT NOT NULL,
                    dispatch TEXT NOT NULL,
                    language TEXT NOT NULL,
                    audio_sha256 TEXT NOT NULL,
                    audio_path TEXT NOT NULL,
                    state TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at REAL NOT NULL DEFAULT 0,
                    transcript TEXT,
                    provider TEXT,
                    model TEXT,
                    latency_ms REAL,
                    config_hash TEXT,
                    error TEXT,
                    accepted_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT,
                    UNIQUE(source_app, device_id, idempotency_key)
                );
                CREATE INDEX IF NOT EXISTS voice_jobs_state_idx
                    ON voice_jobs(state, next_attempt_at, accepted_at);
                """
            )
            conn.execute("UPDATE voice_jobs SET state='accepted' WHERE state='processing'")
        os.chmod(self.db_path, 0o600)

    def accept(
        self,
        source: Path,
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
    ) -> VoiceJobAdmission:
        source_app = _bounded(source_app, "source_app", maximum=64, pattern=_ID_RE)
        device_id = _bounded(device_id, "device_id", maximum=128, pattern=_ID_RE)
        idempotency_key = _bounded(idempotency_key, "idempotency_key", maximum=128, pattern=_ID_RE)
        mode = _bounded(mode, "mode", maximum=32, pattern=_ID_RE) or "dictation"
        from voice_rules import MODES
        if mode not in MODES:
            raise VoiceJobError("unknown transcription mode")
        intent = _bounded(intent, "intent", maximum=64, pattern=_ID_RE) or "transcribe"
        destination = _bounded(destination, "destination", maximum=96, pattern=_DESTINATION_RE)
        language = _bounded(language, "language", maximum=12, pattern=_ID_RE) or "en"
        if dispatch not in {"client", "owner"}:
            raise VoiceJobError("invalid dispatch")
        encoded_params = json.dumps(destination_params or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if len(encoded_params.encode("utf-8")) > 4096:
            raise VoiceJobError("destination_params too large")
        if not source.is_file() or source.stat().st_size == 0:
            raise VoiceJobError("audio is missing or empty")

        audio_hash = _sha256(source)
        accepted_at = _now()
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT job_id,state,audio_sha256,accepted_at FROM voice_jobs WHERE source_app=? AND device_id=? AND idempotency_key=?",
                (source_app, device_id, idempotency_key),
            ).fetchone()
            if existing:
                if existing["audio_sha256"] != audio_hash:
                    raise VoiceJobConflict("idempotency key already belongs to different audio")
                return VoiceJobAdmission(existing["job_id"], existing["state"], audio_hash, existing["accepted_at"], True)

            job_id = str(uuid.uuid4())
            suffix = source.suffix[:12] if source.suffix else ".audio"
            target = self.audio_dir / f"{audio_hash}{suffix}"
            if not target.exists():
                temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
                with source.open("rb") as reader, temporary.open("xb") as writer:
                    shutil.copyfileobj(reader, writer, length=1024 * 1024)
                    writer.flush()
                    os.fsync(writer.fileno())
                os.chmod(temporary, 0o600)
                os.replace(temporary, target)
                directory_fd = os.open(self.audio_dir, os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    """
                    INSERT INTO voice_jobs (
                        job_id,source_app,device_id,idempotency_key,mode,intent,destination,
                        destination_params,dispatch,language,audio_sha256,audio_path,state,
                        accepted_at,updated_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        job_id, source_app, device_id, idempotency_key, mode, intent, destination,
                        encoded_params, dispatch, language, audio_hash, str(target), "accepted",
                        accepted_at, accepted_at,
                    ),
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return VoiceJobAdmission(job_id, "accepted", audio_hash, accepted_at)

    def get(self, job_id: str) -> dict[str, object] | None:
        job_id = _bounded(job_id, "job_id", maximum=64, pattern=_ID_RE)
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM voice_jobs WHERE job_id=?", (job_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        result["destination_params"] = json.loads(str(result["destination_params"]))
        result.pop("audio_path", None)
        result.pop("next_attempt_at", None)
        return result

    def claim_for_processing(self) -> dict[str, object] | None:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    """
                    SELECT * FROM voice_jobs
                    WHERE state IN ('accepted','failed_retryable') AND next_attempt_at<=?
                    ORDER BY accepted_at LIMIT 1
                    """,
                    (now,),
                ).fetchone()
                if not row:
                    conn.execute("COMMIT")
                    return None
                conn.execute(
                    "UPDATE voice_jobs SET state='processing', attempts=attempts+1, updated_at=? WHERE job_id=?",
                    (_now(), row["job_id"]),
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        claimed = dict(row)
        claimed["attempts"] = int(claimed["attempts"]) + 1
        return claimed

    def complete(self, job_id: str, result: TranscriptionResult) -> None:
        try:
            config_hash = load_voice_control_config().config_hash
        except Exception:
            config_hash = ""
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE voice_jobs SET state='completed',transcript=?,provider=?,model=?,latency_ms=?,
                    config_hash=?,error=NULL,updated_at=?,completed_at=? WHERE job_id=?
                """,
                (result.text, result.provider, result.model, result.latency_ms, config_hash, now, now, job_id),
            )

    def fail(self, job_id: str, error: str, *, attempts: int, max_attempts: int = 3) -> None:
        terminal = attempts >= max_attempts
        state = "failed_terminal" if terminal else "failed_retryable"
        retry_at = 0 if terminal else time.time() + min(30, 2 ** max(0, attempts - 1))
        with self._connect() as conn:
            conn.execute(
                "UPDATE voice_jobs SET state=?,error=?,next_attempt_at=?,updated_at=? WHERE job_id=?",
                (state, str(error)[:500], retry_at, _now(), job_id),
            )


class VoiceJobWorker:
    def __init__(
        self,
        store: VoiceJobStore,
        *,
        transcriber: Callable[[Path, str], TranscriptionResult] = route_transcription,
        poll_seconds: float = 1.0,
    ) -> None:
        self.store = store
        self.transcriber = transcriber
        self.poll_seconds = poll_seconds
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="wizard-voice-jobs", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout)

    def wake(self) -> None:
        self._wake.set()

    def process_one(self) -> bool:
        job = self.store.claim_for_processing()
        if not job:
            return False
        try:
            # Legacy custom transcribers keep their two-argument interface.
            options = {"mode": str(job["mode"])} if self.transcriber is route_transcription else {}
            result = self.transcriber(Path(str(job["audio_path"])), str(job["language"]), **options)
            self.store.complete(str(job["job_id"]), result)
        except NoSpeechError as exc:
            self.store.fail(str(job["job_id"]), str(exc), attempts=int(job["attempts"]), max_attempts=1)
        except Exception as exc:
            # A single malformed provider response or unexpected adapter bug must
            # become a visible retryable/terminal job state, never kill the worker.
            self.store.fail(str(job["job_id"]), str(exc), attempts=int(job["attempts"]))
        return True

    def _run(self) -> None:
        while not self._stop.is_set():
            worked = self.process_one()
            if not worked:
                self._wake.wait(self.poll_seconds)
                self._wake.clear()
