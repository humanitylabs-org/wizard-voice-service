"""Exercise the installed package from an empty cwd/HOME; no provider credentials."""
from __future__ import annotations
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import wave


def main():
    with tempfile.TemporaryDirectory(prefix="voice-clean-smoke-") as work:
        root = Path(work)
        home = root / "home"
        home.mkdir()
        config = root / "config.yaml"
        config.write_text("stt:\n  enabled: true\n  provider: tinfoil\n  providers:\n    tinfoil:\n      model: voxtral-small-24b\n      api_key_env: TINFOIL_API_KEY\n")
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(home),
               "TMPDIR": str(root), "WIZARD_VOICE_CONFIG": str(config),
               "WIZARD_VOICE_DATA_DIR": str(root / "state"), "LANG": "en_US.UTF-8"}
        # Import must resolve outside the source checkout, including packaged VAD.
        code = "import json, numpy as np, server, speech_gate; print(json.dumps({'module':server.__file__, 'vad':speech_gate.max_speech_probability(np.zeros(16000,dtype=np.float32))}))"
        installed = json.loads(subprocess.check_output([sys.executable, "-c", code], cwd=home, env=env, text=True))
        assert "site-packages" in installed["module"], installed
        processes = []
        def start():
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            sock.listen(128)
            port = sock.getsockname()[1]
            proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "server:app", "--fd", str(sock.fileno()), "--no-access-log"],
                                    cwd=home, env=env, pass_fds=(sock.fileno(),), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            processes.append(proc)
            sock.close()
            url = f"http://127.0.0.1:{port}"
            for _ in range(100):
                assert proc.poll() is None, "service exited"
                try:
                    if get(url + "/health")[0] == 200:
                        return proc, url
                except OSError:
                    pass
                time.sleep(.05)
            raise AssertionError("service never became ready")
        def get(url, data=None, headers=None):
            request = urllib.request.Request(url, data=data, headers=headers or {})
            try:
                with urllib.request.urlopen(request, timeout=10) as response:
                    return response.status, json.loads(response.read())
            except urllib.error.HTTPError as exc:
                return exc.code, json.loads(exc.read())
        def form(fields, audio=b"synthetic-invalid-audio"):
            boundary = "smoke-boundary"
            body = b""
            for key, value in fields.items():
                body += f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode()
            body += f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="synthetic.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode() + audio
            body += f'\r\n--{boundary}--\r\n'.encode()
            return body, {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        try:
            proc, url = start()
            status, health = get(url + "/health")
            assert status == 200 and health["provider"] == "tinfoil" and health["fallback"] is None
            assert get(url + "/v1/voice/capabilities")[0] == 200
            assert get(url + "/admin/v1/voice/config")[0] == 503
            body, headers = form({})
            status, error = get(url + "/inference", body, headers)
            assert status == 502 and "credential" in error["detail"]
            assert not (root / "state/failed").exists()
            body, headers = form({"mode": "unknown"})
            assert get(url + "/v1/audio/transcriptions", body, headers)[0] == 400
            body, headers = form({"source_app": "smoke", "device_id": "synthetic", "idempotency_key": "one", "mode": "meeting"})
            status, admission = get(url + "/v1/voice/jobs", body, headers)
            assert status == 202
            assert get(url + "/v1/voice/jobs", body, headers)[1]["duplicate"] is True
            proc.terminate()
            proc.wait(timeout=10)
            proc, url = start()
            status, job = get(url + admission["status_url"])
            assert status == 200 and job["job_id"] == admission["job_id"] and job["mode"] == "meeting"
            print(json.dumps({"installed_module": installed["module"], "packaged_vad_loaded": True,
                              "health": health, "missing_credential_fails_closed": True,
                              "failed_audio_not_retained": True, "durable_job_restart": True,
                              "idempotent_admission": True, "clean_home": True}, sort_keys=True))
        finally:
            for proc in processes:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=10)


if __name__ == "__main__":
    main()
