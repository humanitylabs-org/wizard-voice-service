"""Bounded batch decoding and production-style pause cuts (no speech filtering).

Keep this policy separate from the experimental overlapping/context-reread policy.
Every sample, including the last partial frame, belongs to exactly one chunk.
"""
from __future__ import annotations
import subprocess
from pathlib import Path
import numpy as np

SAMPLE_RATE = 16000
MAX_SECONDS = 900


def decode(source: Path) -> np.ndarray:
    proc = subprocess.run([
        'ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-i', str(source),
        '-map', '0:a:0', '-af', 'aresample=async=1:first_pts=0,asetpts=N/SR/TB',
        '-ac', '1', '-ar', str(SAMPLE_RATE), '-t', str(MAX_SECONDS + 1),
        '-f', 'f32le', '-'], capture_output=True, timeout=180, check=False)
    if proc.returncode or not proc.stdout:
        raise ValueError('invalid audio')
    samples = np.frombuffer(proc.stdout, dtype='<f4')
    if len(samples) > MAX_SECONDS * SAMPLE_RATE:
        raise ValueError('audio exceeds 900 seconds; no truncated transcription')
    if not np.isfinite(samples).all():
        raise ValueError('nonfinite audio')
    return samples


def pause_spans(samples: np.ndarray, chunk_seconds: float = 30) -> list[tuple[int, int]]:
    """Quietest 200ms in last 8s before limit; latest wins ties.

    Matches the deployed pause30 algorithm without borrowing private config.
    The final partial frame is included, unlike frame-only duration estimates.
    """
    if not 16 <= chunk_seconds <= 30:
        raise ValueError('chunk_seconds must be 16..30')
    frame = 320
    count = len(samples)
    if not count:
        return []
    padded = np.pad(samples, (0, (-count) % frame))
    energy = (padded.reshape(-1, frame).astype(np.float64) ** 2).mean(axis=1)
    smooth = np.convolve(energy, np.ones(10) / 10, mode='same')
    limit = int(chunk_seconds * SAMPLE_RATE)
    search = int(min(8, chunk_seconds / 2) * SAMPLE_RATE)
    cuts = [0]
    while count - cuts[-1] > limit:
        hi = (cuts[-1] + limit) // frame
        lo = (cuts[-1] + limit - search) // frame
        window = smooth[lo:hi]
        best = lo + len(window) - 1 - int(np.argmin(window[::-1]))
        cuts.append(best * frame)
    cuts.append(count)
    return list(zip(cuts, cuts[1:]))


def write_wav(path: Path, samples: np.ndarray) -> None:
    import wave
    pcm = (np.clip(samples, -1, 1) * 32767).astype('<i2')
    with wave.open(str(path), 'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(pcm.tobytes())
