"""Check a legacy meeting client's actual stt function without importing its service.

Usage: python integrations/verify_meeting_client.py /path/to/meeting/server.py
Run with Wizard Voice Service installed. This is a caller-contract/text-pipeline
check, not an acoustic or external-provider test. Never opens meeting storage.
"""
from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path
import re
import sys

from postprocess import preview_transcript


def load_stt(source):
    tree = ast.parse(Path(source).read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "stt")
    # Postponed annotations avoid importing a client service and its dependencies.
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn], type_ignores=[])
    namespace = {"STT_URL": "http://127.0.0.1:8864/v1/audio/transcriptions", "re": re,
                 "_split_wav": lambda data: [data[:len(data)//2], data[len(data)//2:]]}
    exec(compile(ast.fix_missing_locations(module), "<meeting-stt-contract>", "exec"), namespace)
    return namespace["stt"]


async def check(source):
    stt = load_stt(source)
    rule = {"id": "user.test", "kind": "style", "scope": "user", "provenance": "synthetic regression",
            "enabled": True, "priority": 100, "modes": ["dictation"], "operation": "attach_laughter"}
    calls = []

    class Response:
        status_code = 200
        text = ""
        def raise_for_status(self):
            pass
        def __init__(self, text):
            self.transcript = text
        def json(self):
            return {"text": self.transcript}

    class Client:
        def __init__(self, fail_once=False):
            self.fail_once = fail_once
        async def post(self, url, *, files, data):
            assert data and data.get("mode") == "meeting", data
            calls.append(data)
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("synthetic retry")
            result = preview_transcript("Funny. Ha ha.", {"rules": [rule]}, mode=data["mode"])
            return Response(result["styled_text"])

    for prompt in ("", "Faithful meeting transcript"):
        result = await stt(Client(), b"audio", prompt=prompt)
        assert result == "Funny. Ha ha.", result
        assert calls[-1].get("prompt", "") == prompt
    result = await stt(Client(fail_once=True), b"x" * 20000, prompt="Keep speakers")
    assert result == "Funny. Ha ha. Funny. Ha ha.", result
    assert all(c["prompt"] == "Keep speakers" for c in calls[-3:])
    print(json.dumps({"caller_checks": 3, "multipart_calls": len(calls), "mode": "meeting", "style_preserved": True, "retry_prompt_preserved": True}))


if __name__ == "__main__":
    asyncio.run(check(sys.argv[1]))
