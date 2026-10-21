

import pytest as _pytest


@_pytest.fixture(autouse=True)
def _isolate_failed_audio(tmp_path, monkeypatch):
    import server
    monkeypatch.setattr(server, "FAILED_AUDIO_DIR", tmp_path / "failed-audio")


@_pytest.fixture(autouse=True)
def _isolate_owned_corrections(tmp_path, monkeypatch):
    # Owned corrections are opt-in per test (test_owned_corrections clears this to use the real file).
    monkeypatch.setenv("WIZARD_STT_CORRECTIONS", str(tmp_path / "no-owned-corrections.yaml"))
