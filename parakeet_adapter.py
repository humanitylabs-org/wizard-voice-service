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
import tempfile
import threading
import time
import urllib.request
import uuid

MODEL = 'parakeet-tdt-0.6b-v3'
FILES = ('config.json', 'vocab.txt', 'encoder-model.int8.onnx', 'decoder_joint-model.int8.onnx', 'nemo128.onnx')


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
            'fallback': None, 'chunk_policy': 'pause30-quietest200ms-last8s'}

    def transcribe(self, audio: bytes):
        from fastapi import HTTPException
        if not self.lock.acquire(blocking=False):
            raise HTTPException(429, 'adapter busy; retry original audio')
        start = time.monotonic()
        try:
            with tempfile.TemporaryDirectory(prefix='parakeet-') as directory:
                source = Path(directory) / 'input'
                source.write_bytes(audio)
                from batch_audio import decode
                try:
                    x = decode(source)
                except ValueError as exc:
                    raise HTTPException(413 if 'exceeds' in str(exc) else 422, str(exc)) from None
                from batch_audio import pause_spans
                spans = pause_spans(x)
                texts = [self.model.recognize(x[a:b], sample_rate=16000)
                         for a, b in spans]
                text = ' '.join(texts).strip()
            return {'text': text, 'provider': 'parakeet', 'model': MODEL,
                    'instance': self.receipt['instance'], 'chunks': [
                        {'start': a/16000, 'end': b/16000} for a, b in spans],
                    'latency_ms': round((time.monotonic()-start)*1000, 1)}
        finally:
            self.lock.release()


def create_app():
    from local_engine_api import create_engine_app
    return create_engine_app(Runtime, MODEL)


app = create_app()
