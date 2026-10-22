"""OpenAI-compatible local batch-engine boundary; no cloud code or downloads."""
from __future__ import annotations
from contextlib import asynccontextmanager
import os
import platform
import resource

MAX_BYTES = 64 * 1024 * 1024


def resources():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {'peak_rss_bytes': int(usage.ru_maxrss * (1 if platform.system() == 'Darwin' else 1024)),
            'cpu_seconds': usage.ru_utime + usage.ru_stime}


def create_engine_app(runtime_factory, model: str):
    from fastapi import FastAPI, HTTPException, Request
    from starlette.datastructures import UploadFile
    from starlette.concurrency import run_in_threadpool
    from starlette.middleware.trustedhost import TrustedHostMiddleware
    @asynccontextmanager
    async def lifespan(app):
        app.state.runtime = await run_in_threadpool(runtime_factory)
        yield
    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['localhost', '127.0.0.1', 'testserver'])
    @app.get('/health')
    def health():
        return dict(app.state.runtime.receipt, **resources(), pid=os.getpid())
    @app.get('/v1/models')
    def models():
        # A runtime exists only after successful model loading.
        return {'object': 'list', 'data': [{'id': model, 'object': 'model',
                'owned_by': 'local', 'task': 'automatic-speech-recognition'}]}
    globals()['Request'] = Request
    @app.post('/v1/audio/transcriptions')
    async def transcribe(request: Request):
        data = bytearray()
        async for block in request.stream():
            data.extend(block)
            if len(data) > MAX_BYTES:
                raise HTTPException(413, 'audio upload exceeds limit')
        if request.headers.get('content-type', '').startswith('multipart/form-data'):
            # Parse a bounded body, never a second unbounded request stream.
            from starlette.requests import Request as StarletteRequest
            sent = False
            async def receive():
                nonlocal sent
                if sent:
                    return {'type': 'http.request', 'body': b'', 'more_body': False}
                sent = True
                return {'type': 'http.request', 'body': bytes(data), 'more_body': False}
            parsed = StarletteRequest(request.scope, receive)
            async with parsed.form(max_files=1, max_fields=8, max_part_size=MAX_BYTES) as form:
                if form.get('model') != model:
                    raise HTTPException(400, 'wrong or missing model; no fallback')
                if form.get('response_format', 'json') != 'json':
                    raise HTTPException(400, 'only response_format=json is supported')
                if form.get('language', 'en') not in {'en', ''}:
                    raise HTTPException(400, 'only English is validated by this candidate')
                if form.get('prompt', '') or form.get('stream', 'false') not in {'false', ''}:
                    raise HTTPException(400, 'batch engine does not accept prompts or streaming')
                upload = form.get('file')
                if not isinstance(upload, UploadFile):
                    raise HTTPException(400, 'file is required')
                audio = await upload.read(MAX_BYTES + 1)
        elif request.headers.get('content-type', '').startswith('application/octet-stream'):
            # Legacy Parakeet client remains supported.
            audio = bytes(data)
        else:
            raise HTTPException(415, 'multipart/form-data required')
        if not audio:
            raise HTTPException(400, 'empty audio')
        try:
            result = await run_in_threadpool(app.state.runtime.transcribe, audio)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        return dict(result, **resources())
    return app
