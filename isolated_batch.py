"""Opt-in pause30 trial path. Does not modify the packaged default or live config.

Raw engines intentionally see silence: benchmark hallucinations instead of hiding
those behind VAD. Voice converts truly empty output to its existing no-speech code.
"""
from __future__ import annotations
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
from urllib.parse import urlparse
import urllib.request

LOCAL_MODELS = {'parakeet-tdt-0.6b-v3': 'parakeet', 'voxtral-mini-4b-realtime-6bit': 'voxtral-local'}


def request_local(source, config):
    from providers import ProviderError, _multipart, _open_transcription
    parsed = urlparse(config.endpoint)
    if (config.model not in LOCAL_MODELS or config.api_key_env or config.attested
            or parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost'}
            or not parsed.port or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path != '/v1/audio/transcriptions'):
        raise ProviderError('invalid local engine route; no fallback')
    body, content_type = _multipart(config.model, source, config.language)
    req = urllib.request.Request(config.endpoint, data=body, method='POST',
            headers={'Content-Type': content_type, 'Accept': 'application/json'})
    try:
        with _open_transcription(req, timeout=config.timeout) as response:
            raw = response.read(4*1024*1024+1)
            if len(raw) > 4*1024*1024:
                raise ValueError('oversized response')
            payload = json.loads(raw)
        if (payload.get('model') != config.model or payload.get('provider') != LOCAL_MODELS[config.model]
                or not isinstance(payload.get('text'), str) or not payload.get('instance')):
            raise ValueError('engine identity mismatch')
        return payload
    except Exception as exc:
        raise ProviderError(f'local engine unavailable ({type(exc).__name__}); no fallback') from exc


def transcribe_batch(source, config, *, api_key):
    from providers import (ProviderError, NoSpeechError, TranscriptionResult,
                           _transcribe_chunk, postprocess_transcript)
    start = time.monotonic()
    if config.name == 'local-engine':
        if api_key:
            raise ProviderError('credentials forbidden on local route')
        text = request_local(source, config)['text']
    elif (config.name == 'tinfoil' and config.model == 'voxtral-small-24b' and config.attested
          and config.endpoint in {'http://127.0.0.1:3301/v1/audio/transcriptions',
                                  'http://localhost:3301/v1/audio/transcriptions'} and api_key):
        from batch_audio import decode, pause_spans, write_wav
        samples = decode(source)
        texts = []
        with tempfile.TemporaryDirectory(prefix='voice-pause30-') as temporary:
            for index, (a, b) in enumerate(pause_spans(samples)):
                remaining = config.timeout - (time.monotonic() - start)
                if remaining <= 0:
                    raise ProviderError('batch deadline exceeded; no partial success')
                audio = Path(temporary) / f'{index}.wav'
                write_wav(audio, samples[a:b])
                texts.append(_transcribe_chunk(audio, replace(config, timeout=min(45, remaining)), api_key))
        text = ' '.join(texts).strip()
    else:
        raise ProviderError('unapproved pause30 route; no fallback')
    if not text.strip():
        raise NoSpeechError('No speech detected')
    return TranscriptionResult(text=postprocess_transcript(text, config.postprocess),
        provider='tinfoil-attested' if config.attested else LOCAL_MODELS[config.model],
        model=config.model, latency_ms=round((time.monotonic()-start)*1000, 1))
