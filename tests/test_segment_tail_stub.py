"""Regression: a recording just short of the 30 s segment boundary.

Live failure: a 29.976 s phone recording left a 748-byte header-only second
segment. The speech gate could not decode it, so every Retry failed with
"chunk 2/2 failed: audio speech detection failed". Uses real ffmpeg.
"""
import shutil
import subprocess

import pytest

import providers

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


def _m4a(path, seconds):
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi",
         "-i", f"sine=frequency=440:sample_rate=48000:duration={seconds}",
         "-c:a", "aac", "-b:a", "64k", str(path)],
        check=True, capture_output=True,
    )
    return path


def test_header_only_tail_segment_is_dropped(tmp_path):
    source = _m4a(tmp_path / "phone.m4a", 29.976)
    out = tmp_path / "out"
    out.mkdir()
    chunks = providers.normalize_audio_chunks(source, out, chunk_seconds=30)
    assert len(chunks) == 1
    assert all(path.exists() for path in chunks)
    assert len(list(out.glob("voice-*.mp3"))) == 1


def test_audio_past_old_boundary_is_kept_in_whole_note(tmp_path):
    source = _m4a(tmp_path / "phone.m4a", 34)
    out = tmp_path / "out"
    out.mkdir()
    chunks = providers.normalize_audio_chunks(source, out, chunk_seconds=30)
    assert len(chunks) == 1
    assert providers._audio_duration(chunks[0]) >= 34


def test_single_chunk_never_dropped(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    only = out / "voice-whole.mp3"
    only.write_bytes(b"x" * 10)
    monkeypatch.setattr(providers, "_is_empty_tail", lambda path: True)
    monkeypatch.setattr(providers, "_audio_duration", lambda path: 0.1)

    def runner(*args, **kwargs):
        return subprocess.CompletedProcess(args, 0, "", "")

    assert providers.normalize_audio_chunks(tmp_path / "in.m4a", out, runner=runner) == [only]
