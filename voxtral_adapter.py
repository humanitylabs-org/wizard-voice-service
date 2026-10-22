"""Voxtral Mini 4B Realtime, pinned voxmlx, used as a local batch engine.

Not Voxtral Small 24B; not Mini 3B. Loads only an existing local model path.
The shared pause30 policy bounds prefill memory and preserves final samples.
"""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile
import threading
import time
import uuid

MODEL = 'voxtral-mini-4b-realtime-6bit'
RUNTIME_COMMIT = 'e6d193e85e84e30f26e370c66973ce287b8a9d57'
MODEL_REVISION = '02eb0caeb9dafb554c17a72b93dbf40cd3736c31'


class Runtime:
    def __init__(self):
        import mlx.core as mx
        import voxmlx
        if platform.system() != 'Darwin' or platform.machine() != 'arm64':
            raise RuntimeError('this local engine requires Apple Silicon / MLX')
        directory = Path(os.environ['WIZARD_VOXTRAL_MODEL_DIR']).resolve(strict=True)
        if not directory.is_dir() or not (directory / 'model.safetensors').is_file():
            raise ValueError('supply preinstalled local model directory; no downloads')
        direct = json.loads(importlib.metadata.distribution('voxmlx').read_text('direct_url.json') or '{}')
        if direct.get('vcs_info', {}).get('commit_id') != RUNTIME_COMMIT:
            raise RuntimeError('voxmlx must be installed at the reviewed runtime commit')
        start = time.monotonic()
        self.model, self.tokenizer, self.config = voxmlx.load_model(str(directory))
        mx.eval(self.model.parameters())
        model_load_seconds = time.monotonic()-start
        verification_start = time.monotonic()
        self.prompt, self.delay = voxmlx._build_prompt_tokens(self.tokenizer)
        self.lock = threading.Lock()
        manifest = {}
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix in {'.json', '.safetensors'}:
                with path.open('rb') as source:
                    manifest[path.name] = hashlib.file_digest(source, 'sha256').hexdigest()
        self.receipt = {'status': 'ready', 'provider': 'voxtral-local', 'model': MODEL,
            'runtime_commit': RUNTIME_COMMIT, 'model_revision_expected': MODEL_REVISION,
            'instance': str(uuid.uuid4()), 'machine': platform.machine(), 'system': platform.system(),
            'execution_provider': 'MLX/Metal', 'model_load_seconds': model_load_seconds,
            'verification_seconds': time.monotonic()-verification_start,
            'mlx_active_bytes': mx.get_active_memory(), 'model_sha256': manifest,
            'fallback': None, 'chunk_policy': 'pause30-quietest200ms-last8s'}

    def transcribe(self, audio: bytes):
        from fastapi import HTTPException
        from batch_audio import decode, pause_spans, write_wav
        from voxmlx.generate import generate
        from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy
        import mlx.core as mx
        if not self.lock.acquire(blocking=False):
            raise HTTPException(429, 'engine busy; retry original audio')
        start = time.monotonic()
        try:
            chunks, texts = [], []
            with tempfile.TemporaryDirectory(prefix='voxtral-batch-') as temporary:
                source = Path(temporary) / 'input'
                source.write_bytes(audio)
                samples = decode(source)
                for index, (a, b) in enumerate(pause_spans(samples)):
                    path = Path(temporary) / f'{index}.wav'
                    write_wav(path, samples[a:b])
                    tick = time.monotonic()
                    tokens = generate(self.model, str(path), self.prompt, n_delay_tokens=self.delay,
                        temperature=0.0, eos_token_id=self.tokenizer.eos_id)
                    texts.append(self.tokenizer.decode(tokens, special_token_policy=SpecialTokenPolicy.IGNORE))
                    mx.synchronize()
                    chunks.append({'start': a/16000, 'end': b/16000,
                                   'seconds': time.monotonic()-tick, 'tokens': len(tokens)})
                    mx.clear_cache()
            self.receipt.update(mlx_peak_bytes=mx.get_peak_memory(), mlx_active_bytes=mx.get_active_memory())
            return {'text': ' '.join(texts).strip(), 'provider': 'voxtral-local', 'model': MODEL,
                'instance': self.receipt['instance'], 'chunks': chunks,
                'mlx_peak_bytes': mx.get_peak_memory(), 'mlx_active_bytes': mx.get_active_memory(),
                'latency_ms': (time.monotonic()-start)*1000}
        finally:
            self.lock.release()


def create_app():
    from local_engine_api import create_engine_app
    return create_engine_app(Runtime, MODEL)


app = create_app()
