"""Stateless streamable-HTTP MCP front on the existing ASGI port.

Same initialize/tools/list/tools/call pattern as Transcriber, without another
proxy process. No SSE/session state is necessary for these request/response tools.
"""
import hmac
import json
import os
import re

from fastapi import APIRouter, Request
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse, Response
from voice_mcp.tools import TOOLS

router = APIRouter()
MAX_BODY = 1 << 20

# Plain-words causes for the agent. Each is matched against the service's own
# error text (never echoed raw: it can carry provider responses or temp paths).
_CAUSES = (
    ("audio normalization failed", "The audio file could not be decoded: it is not a supported audio type, has no audio track, or is damaged."),
    ("audio speech detection failed", "The audio file could not be decoded: it is not a supported audio type, has no audio track, or is damaged."),
    ("speech detector", "The local speech detector is unavailable on this Wizard; check voice_status."),
    ("route unavailable", "The transcription engine could not be reached. Check that the Transcriber app is installed and running."),
    ("redirect refused", "The transcription engine could not be reached (it answered with a redirect)."),
    ("credential", "The selected transcription engine has no credential; check voice_status."),
    ("deadline exceeded", "Transcription took too long and was stopped; try a shorter recording."),
    ("returned no transcript", "The transcription engine returned no text for this recording."),
    ("invalid transcription response", "The transcription engine returned an answer Voice could not read."),
    ("transcript_seam_unresolved", "The recording could not be joined back together after splitting it; try again or use a shorter recording."),
    ("refusal", "The transcription engine answered instead of transcribing; try again."),
)


def explain(exc):
    """One plain sentence naming the real cause of a failed tool call."""
    status = getattr(exc, "status_code", None)
    detail = str(getattr(exc, "detail", "") or exc)
    if isinstance(exc, PermissionError):
        return "Voice cannot read that file (permission denied)."
    if isinstance(exc, ValueError):
        return detail[:200]  # tools.py raises ValueError only with its own fixed text
    if status == 422 or "no speech" in detail.lower():
        return "No speech was detected in the recording."
    if status == 413:
        return "The recording is larger than the upload limit."
    if status == 400 and "empty" in detail:
        return "The recording is empty."
    m = re.search(r"returned HTTP (\d{3})", detail)
    if m:
        return "The transcription engine returned an error (HTTP " + m.group(1) + "). Check that the Transcriber app is running."
    for needle, message in _CAUSES:
        if needle in detail:
            return message
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return "The transcription engine could not be reached. Check that the Transcriber app is installed and running."
    return "Voice tool failed (" + type(exc).__name__ + (" " + str(status) if status else "") + "); check voice_status and the file/mode."


def error(i, code, message):
    return {"jsonrpc": "2.0", "id": i, "error": {"code": code, "message": message}}


def rpc(msg):
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or not isinstance(msg.get("method"), str):
        return error(None, -32600, "invalid request")
    i, method, params = msg.get("id"), msg["method"], msg.get("params") or {}
    if not isinstance(params, dict):
        return error(i, -32602, "params must be an object")
    if i is None:
        return None
    if method == "initialize":
        result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                  "serverInfo": {"name": "wizard-voice-service", "version": "0.2.2rc1"}}
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": [schema for schema, _ in TOOLS.values()]}
    elif method == "tools/call":
        name, args = params.get("name"), params.get("arguments", {})
        if not isinstance(name, str) or name not in TOOLS or not isinstance(args, dict):
            return error(i, -32602, "unknown tool or invalid arguments")
        schema, fn = TOOLS[name]
        spec = schema["inputSchema"]
        if set(args) - spec["properties"].keys() or set(spec.get("required", [])) - args.keys():
            return error(i, -32602, "missing or unknown arguments")
        try:
            result = {"content": [{"type": "text", "text": json.dumps(fn(**args))}], "isError": False}
        except Exception as exc:
            # Name the real cause in plain words; never echo transcript,
            # filesystem paths, provider response or secrets.
            result = {"content": [{"type": "text", "text": json.dumps({"error": explain(exc)})}], "isError": True}
    else:
        return error(i, -32601, "method not found")
    return {"jsonrpc": "2.0", "id": i, "result": result}


@router.api_route("/mcp", methods=["POST", "GET", "DELETE"])
async def mcp(request: Request):
    origin = request.headers.get("origin")
    allowed_origins = {x.strip() for x in os.environ.get("WIZARD_VOICE_MCP_ORIGINS", "").split(",") if x.strip()}
    if origin and origin not in allowed_origins:
        return JSONResponse({"error": "MCP origin not allowed"}, status_code=403)
    token = os.environ.get("WIZARD_VOICE_MCP_TOKEN", "")
    if token and not hmac.compare_digest(request.headers.get("authorization", "").encode(), ("Bearer " + token).encode()):
        return JSONResponse({"error": "MCP bearer token required"}, status_code=401)
    if request.method != "POST":
        return Response(status_code=405, headers={"Allow": "POST"})
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_BODY:
            return Response(status_code=413)
    try:
        msg = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return JSONResponse(error(None, -32700, "parse error"), status_code=400)
    if isinstance(msg, list):
        # Current MCP streamable HTTP protocol excludes JSON-RPC batches.
        return JSONResponse(error(None, -32600, "batch requests are not supported"), status_code=400)
    result = await run_in_threadpool(rpc, msg)
    return JSONResponse(result) if result is not None else Response(status_code=202)
