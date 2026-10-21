"""Private CPU-only Parakeet sidecar. No downloads, credentials or cloud fallback.

Run one worker, bound to loopback. For another host use an SSH tunnel.
The owner must supply already verified model files via WIZARD_PARAKEET_MODEL_DIR.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import threading
import time
import urllib.request
import uuid
from contextlib import asynccontextmanager

MODEL = 'parakeet-tdt-0.6b-v3'
FILES = ('config.json', 'vocab.txt', 'encoder-model.int8.onnx', 'decoder_joint-model.int8.onnx', 'nemo128.onnx')
MAX_BYTES = 16 * 1024 * 1024


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request_transcription(audio, config):
    from providers import ProviderError
    # Deliberately no inherited proxy settings or Authorization header.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    req = urllib.request.Request(config.endpoint, data=audio.read_bytes(), method='POST',
                                 headers={'Content-Type': 'application/octet-stream'})
    try:
        with opener.open(req, timeout=config.timeout) as response:
            data = response.read(1024 * 1024 + 1)
            if len(data) > 1024 * 1024:
                raise ValueError('oversized response')
            payload = json.loads(data)
        if (payload.get('provider') != 'parakeet' or payload.get('model') != MODEL
                or not isinstance(payload.get('text'), str) or not payload['text'].strip()):
            raise ValueError('invalid identity or transcript')
        return payload['text'].strip()
    except Exception as exc:
        # Do not echo response bodies, transcript or private routing addresses.
        raise ProviderError(f'parakeet adapter unavailable ({type(exc).__name__}); no fallback') from exc


class Runtime:
    def __init__(self):
        import onnxruntime as ort
        import onnx_asr
        directory = Path(os.environ['WIZARD_PARAKEET_MODEL_DIR']).resolve(strict=True)
        manifest = {}
        for name in FILES:
            with (directory / name).open('rb') as f:
                manifest[name] = hashlib.file_digest(f, 'sha256').hexdigest()
        threads = int(os.environ.get('WIZARD_PARAKEET_THREADS', '4'))
        if not 1 <= threads <= 8:
            raise ValueError('threads must be 1..8')
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        start = time.monotonic()
        self.model = onnx_asr.load_model('nemo-parakeet-tdt-0.6b-v3', path=str(directory),
            quantization='int8', providers=['CPUExecutionProvider'], sess_options=options)
        sessions = {name: getattr(self.model.asr, name).get_providers()
                    for name in ('_encoder', '_decoder_joint')}
        if any(providers != ['CPUExecutionProvider'] for providers in sessions.values()):
            raise RuntimeError('CPU-only execution required')
        self.lock = threading.Lock()
        self.receipt = {'status': 'ready', 'provider': 'parakeet', 'model': MODEL,
            'instance': str(uuid.uuid4()), 'machine': platform.machine(), 'system': platform.system(),
            'execution_provider': 'CPUExecutionProvider', 'threads': threads,
            'sessions': sessions,
            'model_sha256': manifest, 'model_load_seconds': round(time.monotonic()-start, 3),
            'fallback': None}

    def transcribe(self, audio: bytes):
        import numpy as np
        from fastapi import HTTPException
        if not self.lock.acquire(blocking=False):
            raise HTTPException(429, 'adapter busy; retry original audio')
        start = time.monotonic()
        try:
            with tempfile.TemporaryDirectory(prefix='parakeet-') as directory:
                source = Path(directory) / 'input'
                source.write_bytes(audio)
                proc = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-i', str(source),
                    '-map', '0:a:0', '-ac', '1', '-ar', '16000', '-f', 'f32le', '-t', '121', '-'],
                    capture_output=True, timeout=120, check=False)
                if proc.returncode or not proc.stdout:
                    raise HTTPException(422, 'invalid audio')
                x = np.frombuffer(proc.stdout, dtype=np.float32)
                if len(x) > 120 * 16000:
                    raise HTTPException(413, 'adapter accepts at most 120 seconds per request')
                # Prefer a low-energy boundary near 25 s, with continuous coverage.
                # Hard cuts through a word caused omissions in real saved notes.
                bounds = [0]
                while len(x) - bounds[-1] > 30 * 16000:
                    left = bounds[-1]
                    candidates = np.arange(left + 20*16000, min(left + 30*16000, len(x)-16000)+1, 1600, dtype=int)
                    energy = np.array([np.mean(x[i-800:i+800]**2) for i in candidates])
                    quiet = candidates[energy <= max(0.0001, float(np.min(energy))*1.2)]
                    cut = int(quiet[np.argmin(np.abs(quiet-(left+25*16000)))])
                    bounds.append(cut)
                bounds.append(len(x))
                texts = [self.model.recognize(x[a:b], sample_rate=16000)
                         for a, b in zip(bounds, bounds[1:])]
                text = ' '.join(texts).strip()
                if not text:
                    raise HTTPException(422, 'no transcript; retry original audio')
            return {'text': text, 'provider': 'parakeet', 'model': MODEL,
                    'instance': self.receipt['instance'], 'latency_ms': round((time.monotonic()-start)*1000, 1)}
        finally:
            self.lock.release()


def create_app():
    from fastapi import FastAPI, HTTPException, Request
    from starlette.concurrency import run_in_threadpool
    from starlette.middleware.trustedhost import TrustedHostMiddleware
    @asynccontextmanager
    async def lifespan(app):
        app.state.runtime = Runtime()
        yield
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['localhost', '127.0.0.1', 'testserver'])
    @app.get('/health')
    def health():
        return app.state.runtime.receipt
    # Request annotation must resolve in module globals for FastAPI.
    globals()['Request'] = Request
    @app.post('/v1/audio/transcriptions')
    async def transcribe(request: Request):
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > MAX_BYTES:
                raise HTTPException(413, 'audio too large')
        if not data:
            raise HTTPException(400, 'empty audio')
        return await run_in_threadpool(app.state.runtime.transcribe, bytes(data))
    return app

app = create_app()
