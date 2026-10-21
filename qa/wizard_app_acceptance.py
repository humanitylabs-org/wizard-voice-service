"""macOS/Docker acceptance runner. Synthetic speech only; isolated project, always cleanup.

Invoke on the authorized build worker, not on a production voice host.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
TRANSCRIBER_COMMIT = "b8d1170a3f97e1d6eed8abc874bdebdf7e9f21dc"
IMAGE = "wizard-voice-service:0.2.2-rc1.app1"
PHRASE = "The blue notebook is on the table. Please bring it to the meeting tomorrow morning."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--docker-context", default="colima")
    args = parser.parse_args()
    evidence = args.evidence.resolve()
    evidence.mkdir(parents=True, exist_ok=False)
    work = evidence / "work"
    (work / "media").mkdir(parents=True)
    env = dict(os.environ, APP_DATA_DIR=str(work / "app-data"), ROOT_FOLDER_HOST=str(work), APP_PORT="0",
               TRANSCRIBER_MODEL="Systran/faster-whisper-small", TMPDIR=str(work))
    docker = ["docker", "--context", args.docker_context]
    project = "voice-app-" + evidence.name.lower().replace("_", "-")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]+", project):
        raise ValueError("evidence directory basename must be alphanumeric/hyphen")
    summary: dict = {"transcriber_commit": TRANSCRIBER_COMMIT, "synthetic_phrase": PHRASE, "evidence": str(evidence)}

    def run(argv, label, *, check=True, timeout=3600, stdin=None):
        with (evidence / (label + ".log")).open("w") as log:
            if stdin is None:
                result = subprocess.run(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            else:
                with Path(stdin).open("rb") as source:
                    result = subprocess.run(argv, cwd=ROOT, env=env, stdin=source, stdout=log, stderr=subprocess.STDOUT,
                                            timeout=timeout)
        print(label, "exit", result.returncode, flush=True)
        if check and result.returncode:
            raise RuntimeError(label + " failed: see log")
        return result.returncode

    def output(argv):
        return subprocess.check_output(argv, cwd=ROOT, env=env, text=True).strip()

    def fetch(url, payload=None):
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        body = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(url, data=body, headers=headers)
        with urllib.request.urlopen(request, timeout=900) as response:
            return json.load(response)

    def rpc(base, method, params=None):
        response = fetch(base + "/mcp", {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}})
        assert "result" in response, response
        return response["result"]

    def record(name, data):
        (evidence / (name + ".json")).write_text(json.dumps(data, indent=2) + "\n")

    commit = output(["git", "rev-parse", "HEAD"])
    assert not output(["git", "status", "--porcelain"]), "build only a clean pushed checkout"
    summary["commit"] = commit
    base_compose = work / "transcriber.yml"
    with urllib.request.urlopen("https://raw.githubusercontent.com/oscarhenrycollins/transcriber/" + TRANSCRIBER_COMMIT + "/runtipi/docker-compose.yml", timeout=30) as response:
        base_compose.write_bytes(response.read())
    override = work / "acceptance.yml"
    override.write_text('''services:
  transcriber:
    ports: ["127.0.0.1::8000"]
    volumes:
      - media:/media:ro
  transcriber-engine:
    volumes:
      - hf-cache:/home/ubuntu/.cache/huggingface
  wizard-voice-service:
    volumes:
      - voice-data:/data
      - media:/media:ro
volumes:
  hf-cache: {}
  voice-data: {}
  media: {}
''')
    compose = docker + ["compose", "--project-directory", str(ROOT), "-p", project,
                        "-f", str(base_compose), "-f", str(ROOT / "runtipi/docker-compose.yml"), "-f", str(override)]
    cleanup_ok = False
    try:
        run(docker + ["buildx", "build", "--platform", "linux/arm64", "--load", "--build-arg", "VCS_REF=" + commit,
                      "--metadata-file", str(evidence / "arm64-build-metadata.json"), "-t", IMAGE, "."], "build-arm64")
        record("arm64-image", json.loads(output(docker + ["image", "inspect", IMAGE])))
        # The Docker VM need not share this checkout's disk (colima shares only $HOME by default), so the
        # source goes in as a tar stream rather than a bind mount; an empty mount once made pytest collect nothing.
        source_tar = work / "source.tar"
        run(["git", "archive", "--format=tar", "-o", str(source_tar), "HEAD"], "source-archive", timeout=120)
        run(docker + ["run", "--rm", "-i", "--user", "0:0", "--entrypoint", "sh", IMAGE, "-c",
                      "mkdir -p /tmp/source && tar -xf - -C /tmp/source && "
                      "python -m pip install 'httpx==0.28.1' 'pytest==9.0.3' && "
                      "cd /tmp/source && python -m pytest -q -p no:cacheprovider && python -m pip check && "
                      "cd / && python /tmp/source/qa/smoke.py"], "tests", timeout=1800, stdin=source_tar)
        run(compose + ["config"], "compose-config", timeout=60)
        run(compose + ["up", "-d"], "compose-up", timeout=1800)
        def service_id(name):
            return output(compose + ["ps", "-q", name])
        def url(name, port):
            info = json.loads(output(docker + ["inspect", service_id(name)]))[0]
            bind = info["NetworkSettings"]["Ports"][str(port) + "/tcp"][0]
            return "http://127.0.0.1:" + bind["HostPort"]
        voice = url("wizard-voice-service", 8803)
        transcriber = url("transcriber", 8000)
        deadline = time.monotonic() + 900
        while True:
            try:
                health = fetch(voice + "/health")
                models = fetch(transcriber + "/v1/models")
                if health.get("status") == "ok" and any(m["id"] == env["TRANSCRIBER_MODEL"] for m in models.get("data", [])):
                    break
            except (OSError, ValueError):
                pass
            if time.monotonic() > deadline:
                raise RuntimeError("engine or voice did not become ready")
            time.sleep(3)
        record("health", health); record("engine-models", models)
        aiff = work / "synthetic.aiff"
        wav = work / "media/synthetic.wav"
        run(["say", "-v", "Samantha", "-r", "145", "-o", str(aiff), PHRASE], "tts", timeout=120)
        run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav)], "tts-convert", timeout=120)
        # Same reason: the shared media folder is a named volume, filled through a stream.
        run(docker + ["run", "--rm", "-i", "--user", "0:0", "--entrypoint", "sh", "-v", project + "_media:/media", IMAGE,
                      "-c", "cat > /media/synthetic.wav && chmod 0644 /media/synthetic.wav"], "media-copy", timeout=120,
            stdin=wav)
        for mode in ("dictation", "meeting"):
            run(["curl", "--fail-with-body", "--silent", "--show-error", "--max-time", "900", "-F", "file=@" + str(wav),
                 "-F", "mode=" + mode, "-F", "language=en", "-o", str(evidence / (mode + ".json")),
                 voice + "/v1/audio/transcriptions"], mode, timeout=930)
            transcript = json.loads((evidence / (mode + ".json")).read_text())
            words = set(re.findall(r"[a-z]+", transcript["text"].lower()))
            assert {"blue", "notebook", "table", "meeting", "tomorrow"} <= words, transcript
            assert transcript["provider"] == "openai-compatible" and transcript["model"] == env["TRANSCRIBER_MODEL"]
        tools = rpc(voice, "tools/list"); record("mcp-tools", tools)
        assert {t["name"] for t in tools["tools"]} == {"transcribe", "voice_status"}
        initialized = rpc(voice, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "acceptance", "version": "1"}})
        record("mcp-initialize", initialized)
        status = rpc(voice, "tools/call", {"name": "voice_status", "arguments": {}})
        record("mcp-status", status)
        clean = json.loads(status["content"][0]["text"])
        assert clean["personal_rules"] == clean["enabled_style_rules"] == clean["legacy_replacements"] == 0, clean
        transcript = rpc(voice, "tools/call", {"name": "transcribe", "arguments": {"path": "synthetic.wav", "mode": "meeting"}})
        record("mcp-transcribe", transcript)
        assert not transcript.get("isError"), transcript
        assert "notebook" in json.loads(transcript["content"][0]["text"])["text"].lower()
        rc = run(compose + ["exec", "-T", "wizard-voice-service", "wizard-voice-profile", "--config", "/data/config.yaml",
                            "select", "tinfoil-voxtral", "--expected", "local-transcriber"], "tinfoil-refusal", check=False, timeout=30)
        refusal = json.loads((evidence / "tinfoil-refusal.log").read_text())
        assert rc == 2 and "disabled: no router connection config" in refusal["error"], refusal
        assert fetch(voice + "/health")["profile"] == "local-transcriber"
        summary.update(e2e_passed=True, clean_personal_rules=0, enabled_style_rules=0)
        # Separate optional build: never replace the ARM64 image exercised above.
        rc = run(docker + ["buildx", "build", "--platform", "linux/amd64", "--load", "--build-arg", "VCS_REF=" + commit,
                           "--metadata-file", str(evidence / "amd64-build-metadata.json"), "-t", IMAGE + "-amd64", "."],
                 "build-amd64", check=False, timeout=3600)
        summary["amd64_build_exit"] = rc
        if rc == 0:
            record("amd64-image", json.loads(output(docker + ["image", "inspect", IMAGE + "-amd64"])))
    except BaseException as exc:
        summary["error"] = str(exc)
        raise
    finally:
        run(compose + ["logs", "--no-color"], "compose-logs", check=False, timeout=60)
        run(compose + ["down", "--volumes", "--remove-orphans"], "compose-cleanup", check=False, timeout=120)
        containers = output(docker + ["ps", "-aq", "--filter", "label=com.docker.compose.project=" + project])
        volumes = output(docker + ["volume", "ls", "-q", "--filter", "label=com.docker.compose.project=" + project])
        networks = output(docker + ["network", "ls", "-q", "--filter", "label=com.docker.compose.project=" + project])
        cleanup_ok = not (containers or volumes or networks)
        summary["cleanup"] = {"ok": cleanup_ok, "containers": containers, "volumes": volumes, "networks": networks}
        record("summary", summary)
        print(json.dumps(summary, indent=2), flush=True)
        if not cleanup_ok:
            raise RuntimeError("acceptance resources remain; see summary")


if __name__ == "__main__":
    main()
