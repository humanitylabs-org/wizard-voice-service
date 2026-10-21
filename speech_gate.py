"""Conservative, local speech-presence gate; never generates transcript text.

Silero's ONNX weights are vendored with their MIT licence and pinned by digest.
Anything uncertain is sent to the configured ASR. This is not a word detector.
"""
from __future__ import annotations
from functools import lru_cache
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile

MODEL_SHA256 = '1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3'
SPEECH_POSSIBLE = 0.35
# Two 32ms frames preserve short words; isolated detector spikes are not speech.
MIN_SPEECH_FRAMES = 2
MIN_SAMPLES = 8000

class SpeechGateError(RuntimeError):
    pass

@lru_cache(maxsize=1)
def _session():
    import onnxruntime as ort
    path = Path(__file__).parent / 'assets' / 'silero_vad.onnx'
    if not path.is_file():
        path = Path(sys.prefix) / 'share' / 'wizard-voice-core' / 'silero_vad.onnx'
    if hashlib.sha256(path.read_bytes()).hexdigest() != MODEL_SHA256:
        raise SpeechGateError('speech detector model integrity check failed')
    options = ort.SessionOptions()
    options.inter_op_num_threads = options.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])

def speech_probabilities(audio):
    import numpy as np
    session = _session()
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros((1, 64), dtype=np.float32)
    for offset in range(0, len(audio), 512):
        chunk = audio[offset:offset+512]
        if len(chunk) < 512:
            chunk = np.pad(chunk, (0, 512-len(chunk)))
        frame = np.concatenate([context, chunk.reshape(1, -1)], axis=1)
        output, state = session.run(None, {'input': frame, 'state': state, 'sr': np.array(16000, dtype=np.int64)})
        context = frame[:, -64:]
        yield float(output[0][0])

def max_speech_probability(audio, *, stop_at: float = 1.1) -> float:
    maximum = 0.0
    for probability in speech_probabilities(audio):
        maximum = max(maximum, probability)
        if maximum >= stop_at:
            break
    return maximum

def has_possible_speech(audio) -> bool:
    consecutive = 0
    for probability in speech_probabilities(audio):
        consecutive = consecutive + 1 if probability >= SPEECH_POSSIBLE else 0
        if consecutive >= MIN_SPEECH_FRAMES:
            return True
    return False

def pcm_is_no_speech(pcm: bytes) -> bool:
    import numpy as np
    if len(pcm) % 2:
        raise SpeechGateError('invalid decoded audio')
    audio = np.frombuffer(pcm, dtype='<i2').astype(np.float32) / 32768.0
    if len(audio) == 0:
        return False
    peak = float(np.max(np.abs(audio)))
    if peak == 0:
        # Exact digital silence carries no word at any length. Recordings just
        # past a segment boundary (e.g. 30.66 s with 30 s segments) leave a
        # sub-second all-zero tail; sending it to the ASR returned an empty
        # transcript and failed the whole note on every retry.
        return True
    if len(audio) < MIN_SAMPLES:
        return False  # Preserve short words and cut-off syllables.
    if has_possible_speech(audio):
        return False
    # A second, gain-normalized view protects softly spoken audio from being discarded.
    gain = min(1000.0, max(1.0, 0.8 / peak))
    return not has_possible_speech(audio * gain)

def detect_no_speech(source: Path) -> bool:
    try:
        with tempfile.TemporaryDirectory(prefix='wizard-vad-') as directory:
            output = Path(directory) / 'audio.pcm'
            result = subprocess.run([
                'ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-y', '-i', str(source),
                '-map', '0:a:0', '-ac', '1', '-ar', '16000', '-f', 's16le', str(output),
            ], capture_output=True, timeout=300, check=False)
            if result.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
                raise SpeechGateError('audio speech detection failed')
            return pcm_is_no_speech(output.read_bytes())
    except SpeechGateError:
        raise
    except Exception as exc:
        raise SpeechGateError('audio speech detector unavailable') from exc
