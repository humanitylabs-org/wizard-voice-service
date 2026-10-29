"""Messenger voice notes from the phone must transcribe even with an empty last Ogg packet.

opus-media-recorder (Android WebView) closes its Ogg stream with an
end-of-stream page carrying a zero-length packet. ffmpeg decodes all real
audio, then rejects that packet; with -xerror the whole note failed as
"upstream_unavailable" in Messenger (71 s note, 26 Oct 2026).
"""
from __future__ import annotations

import json
import shutil
import struct
import subprocess

import pytest

import providers

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


def _crc_table() -> list[int]:
    table = []
    for i in range(256):
        r = i << 24
        for _ in range(8):
            r = ((r << 1) ^ 0x04C11DB7) if r & 0x80000000 else (r << 1)
        table.append(r & 0xFFFFFFFF)
    return table


_TABLE = _crc_table()


def _ogg_crc(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc = ((crc << 8) ^ _TABLE[((crc >> 24) & 0xFF) ^ byte]) & 0xFFFFFFFF
    return crc


def _pages(data: bytes) -> list[bytes]:
    pages, i = [], 0
    while i < len(data):
        assert data[i:i + 4] == b"OggS"
        count = data[i + 26]
        size = 27 + count + sum(data[i + 27:i + 27 + count])
        pages.append(data[i:i + size])
        i += size
    return pages


def _phone_style_note(tmp_path):
    clean = tmp_path / "clean.ogg"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
         "-ac", "1", "-ar", "48000", "-c:a", "libopus", str(clean)],
        check=True,
    )
    pages = _pages(clean.read_bytes())
    last = bytearray(pages[-1])
    last[5] &= ~0x04  # last audio page is no longer end-of-stream
    granule = struct.unpack("<q", last[6:14])[0]
    seq = struct.unpack("<I", last[18:22])[0]
    last[22:26] = b"\0\0\0\0"
    last[22:26] = struct.pack("<I", _ogg_crc(bytes(last)))
    eos = bytearray(b"OggS\x00\x04" + struct.pack("<q", granule) + bytes(last[14:18])
                    + struct.pack("<I", seq + 1) + b"\0\0\0\0" + b"\x01\x00")
    eos[22:26] = struct.pack("<I", _ogg_crc(bytes(eos)))
    note = tmp_path / "note.ogg"
    note.write_bytes(b"".join(pages[:-1]) + bytes(last) + bytes(eos))
    strict = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(note), "-f", "null", "-"],
                            capture_output=True, text=True)
    assert strict.returncode != 0, "fixture must reproduce the phone recorder's rejected end packet"
    return note


def _duration(path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(json.loads(out)["format"]["duration"])


def test_messenger_voice_note_with_empty_end_packet_normalizes(tmp_path):
    note = _phone_style_note(tmp_path)
    out = tmp_path / "out.mp3"
    providers.normalize_audio(note, out)
    assert _duration(out) == pytest.approx(3.0, abs=0.2)


def test_messenger_voice_note_with_empty_end_packet_chunks(tmp_path):
    note = _phone_style_note(tmp_path)
    out = tmp_path / "chunks"
    out.mkdir()
    chunks = providers.normalize_audio_chunks(note, out, chunk_seconds=30)
    assert len(chunks) == 1
    assert _duration(chunks[0]) == pytest.approx(3.0, abs=0.2)


def test_truly_undecodable_audio_still_fails(tmp_path):
    junk = tmp_path / "junk.ogg"
    junk.write_bytes(b"not audio at all" * 64)
    with pytest.raises(providers.ProviderError, match="audio normalization failed"):
        providers.normalize_audio(junk, tmp_path / "o.mp3")
