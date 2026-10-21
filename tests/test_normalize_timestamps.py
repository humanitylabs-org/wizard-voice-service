"""Regression: Chromium (Omarchy) WebM voice with non-monotonic DTS must still normalize."""
from pathlib import Path
import subprocess

from providers import normalize_audio, normalize_audio_chunks


def _capture():
    calls = []

    def runner(cmd, **kwargs):
        calls.append(cmd)
        out = Path(cmd[-1].replace("%03d", "000"))
        out.write_bytes(b"x")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    return calls, runner


def test_both_normalizers_rebuild_timestamps(tmp_path):
    for fn, target in ((normalize_audio, tmp_path / "o.mp3"), (normalize_audio_chunks, tmp_path)):
        calls, runner = _capture()
        try:
            fn(tmp_path / "in.webm", target, runner=runner)
        except Exception:
            pass
        cmd = calls[0]
        assert "-af" in cmd
        assert "asetpts=N/SR/TB" in cmd[cmd.index("-af") + 1]
