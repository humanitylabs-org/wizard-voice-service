"""Stateless streamable-HTTP MCP front on the existing ASGI port.

Same initialize/tools/list/tools/call pattern as Transcriber, without another
proxy process. No SSE/session state is necessary for these request/response tools.
"""
import hashlib
import hmac
import json
import os

from fastapi import APIRouter, Request
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse, Response
from voice_mcp.tools import TOOLS

router = APIRouter()
MAX_BODY = 1 << 20

# Box-peer gate (wizard-os issue 386, card 5), default-off: only with WIZARD_BROWSER_BOUNDARY=1, which App Manager
# writes into this app's user env together with WIZARD_GATEWAY_KEY, a per-installation key it also gives the Agent
# Service gateway (grants/agent-service.json; the gateway sends it as X-Wizard-Grant). Then /mcp answers only that
# gateway: no browser or network-identity markers (Voice has no view and no Serve route), and the key, or a key whose
# SHA-256 is a host:agent-service row of WIZARD_GRANTS_FILE, or the operator's WIZARD_VOICE_MCP_TOKEN. Nothing
# configured = refused (fail closed). Off, /mcp behaves exactly as before.
BROWSER_MARKERS = ("origin", "cookie", "referer", "sec-fetch-site", "sec-fetch-dest", "sec-fetch-user")
NETWORK_MARKERS = ("tailscale-user-login", "tailscale-user-name", "tailscale-user-profile-pic", "tailscale-app-capabilities",
                   "x-forwarded-for", "x-forwarded-host", "x-forwarded-proto", "forwarded", "x-real-ip")


def _gateway_rows():
    path = os.environ.get("WIZARD_GRANTS_FILE", "")
    if not path:
        return []
    try:
        with open(path, "rb") as f:
            doc = json.loads(f.read(65537)[:65536])
        rows = doc.get("incoming") if isinstance(doc, dict) else None
        return [str(r.get("sha256", "")) for r in rows or [] if isinstance(r, dict) and r.get("from") == "host:agent-service"]
    except (OSError, ValueError):
        return []


def gateway_refusal(request: Request):
    """None to serve /mcp, else (status, message). Only with the boundary on."""
    if os.environ.get("WIZARD_BROWSER_BOUNDARY") != "1":
        return None
    headers = request.headers
    marked = [h for h in BROWSER_MARKERS + NETWORK_MARKERS if h in headers]
    if marked:
        return 403, "box-only MCP: refused a request carrying " + marked[0]
    token = os.environ.get("WIZARD_VOICE_MCP_TOKEN", "")
    if token and hmac.compare_digest(headers.get("authorization", "").encode(), ("Bearer " + token).encode()):
        return None
    key = os.environ.get("WIZARD_GATEWAY_KEY", "").strip()
    rows = _gateway_rows()
    if not key and not rows:
        return 503, "MCP gateway key not configured"
    supplied = headers.get("x-wizard-grant", "")
    if supplied and key and hmac.compare_digest(supplied.encode(), key.encode()):
        return None
    digest = hashlib.sha256(supplied.encode()).hexdigest()
    if supplied and any(hmac.compare_digest(r.encode(), digest.encode()) for r in rows):
        return None
    return 403, "MCP answers only the Wizard agent gateway"


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
            # Do not echo transcript, filesystem paths, provider response or secrets.
            result = {"content": [{"type": "text", "text": "Voice tool failed (" + type(exc).__name__ + "); check voice_status and the file/mode."}], "isError": True}
    else:
        return error(i, -32601, "method not found")
    return {"jsonrpc": "2.0", "id": i, "result": result}


@router.api_route("/mcp", methods=["POST", "GET", "DELETE"])
async def mcp(request: Request):
    refused = gateway_refusal(request)
    if refused:
        return JSONResponse({"error": refused[1]}, status_code=refused[0])
    origin = request.headers.get("origin")
    allowed_origins = {x.strip() for x in os.environ.get("WIZARD_VOICE_MCP_ORIGINS", "").split(",") if x.strip()}
    if origin and origin not in allowed_origins:
        return JSONResponse({"error": "MCP origin not allowed"}, status_code=403)
    token = os.environ.get("WIZARD_VOICE_MCP_TOKEN", "")
    if token and os.environ.get("WIZARD_BROWSER_BOUNDARY") != "1" and not hmac.compare_digest(request.headers.get("authorization", "").encode(), ("Bearer " + token).encode()):
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
