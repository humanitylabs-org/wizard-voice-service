# Wizard app candidate

## Layering and default-slot decision

Voice provides **voice**, needs **speech-to-text-engine**, and optionally uses
Files for MCP paths. Transcriber continues to own `transcription` and
`speech-to-text-engine`. Claiming `transcription` here would make an engine slot
resolve back to the dictation layer, risking cycles or replacing timestamp APIs
with a plain-text dictation API. No App Manager defaults are changed. A later
box-release PR should install Voice alongside Transcriber, not replace it.

**Current manifest checker gap:** App Manager's `wam/wizardmd.py` at
`5005f6b76ed9783c29972e9a43730e306de3b515` permits only device needs
(`mic`, `system-audio`, `camera`, `notifications`, `files`) and `credentials:*`.
It rejects the requested `needs: [speech-to-text-engine]`. We retain the explicit
engine dependency rather than misdeclare it as a microphone or omit it. Catalog
admission needs an upstream slot-dependency contract/checker change; it is not
silently worked around here. First-party code is MIT (see LICENSE).

Default engine: `http://transcriber:8000/v1/audio/transcriptions`, using the
existing OpenAI-compatible adapter, `Systran/faster-whisper-small`. This exact
Docker DNS/port is the only non-loopback HTTP exception. It needs no bearer key,
does not read cloud credentials, ignores HTTP proxy env and refuses redirects.
No ASR model is bundled in Voice. The small bundled ONNX file is a speech-presence
gate, not a recognizer. Change the selected model to match Transcriber's installed
model if the owner customizes it. No provider fallback exists.

## Immutable image / release boundary

The recipe pins `ghcr.io/humanitylabs-org/wizard-voice-service` by its immutable
multi-platform index digest. The index contains linux/amd64 and linux/arm64 images
built from the same main commit by the dedicated repository-scoped macOS worker.
No moving `latest` tag or preloaded local image is required.

**Public verification:** [publication run 39659639855](https://github.com/humanitylabs-org/wizard-voice-service/actions/runs/39659639855)
built main commit `510bc89f507f02fd3fd41b64605918b2a1825a82` and its independent
hosted anonymous-pull job passed for both platforms by digest and full-SHA tag,
checking the source revision. The compose pin is that verified index digest.
Future release candidates must pass the same check before recipe promotion.
If a future package is private, an organization administrator must first set it
to Public. The package settings are
https://github.com/orgs/humanitylabs-org/packages/container/wizard-voice-service/settings.
After changing visibility, rerun failed verification jobs only (do not rebuild or
retag the immutable image). See `scripts/ci/README.md` for publication details.

Install Transcriber first, then install this recipe through App Manager. The
public-image gate does not replace the separate catalog/dependency admission
requirements above.

The Runtipi compose's `add_to_main_network` attaches Voice to the shared app
network where `transcriber:8000` is reachable. Port 8803 is loopback-only on the
host; the service is unauthenticated internal infrastructure, **not** an
internet-facing API. Files is read-only; `/data` is the only persistent writable
mount; normalized uploads use a bounded tmpfs. Runtime UID/GID is 1000. Runtipi
must provision app data ownership accordingly (as for other UID-1000 apps).
The default config is copied once with mode 0600 to `/data/config.yaml`; restart
never overwrites owner configuration. Jobs live in `/data/jobs`; failed-upload
retention is off. Rules/corrections are absent unless the owner creates them.
Use only one service process per data directory.

Public image pulls are one gate, **not a zero-touch fleet/default release**.
Index/catalog admission and the explicit box default-install change remain
follow-up gates. Do not edit the existing transcription slot to point at Voice.

## Optional Tinfoil router contract (not implemented in Connection Router)

The packaged `tinfoil-voxtral` profile is disabled, and selecting it refuses
without `WIZARD_VOICE_TINFOIL_CONNECTION`. It cannot use the legacy Tinfoil key
or silently select an unattested provider. Future owner-approved router setup:

1. Project a read-only JSON file and set `WIZARD_VOICE_TINFOIL_CONNECTION` to its
   container path. It must have exactly the required values below (extra fields
   are ignored). Inject `WIZARD_VOICE_TINFOIL_KEY` through the runtime secret
   channel; never store a key in a recipe, image, JSON projection or transcript.
2. Endpoint: `http://127.0.0.1:3301/v1/audio/transcriptions`; model:
   `voxtral-small-24b`; auth: HTTP `Authorization: Bearer` from that env binding.
3. Router must provision a reviewed Tinfoil **attestation-verifying** proxy in
   Voice's network namespace, bound only to loopback 3301. This app does not
   implement cryptography, start a proxy, or treat a JSON boolean as attestation
   proof. The proxy must reject failed/stale attestation before forwarding audio;
   remote plaintext proxy URLs and direct cloud URLs are not accepted.
4. After explicit owner approval, select `tinfoil-voxtral` in `/data/config.yaml`
   with `wizard-voice-profile --config /data/config.yaml select tinfoil-voxtral
   --expected local-transcriber`. Config validation must pass before selection is
   committed; removal of the connection config fails closed, with no fallback.

```json
{
  "schema_version": 1,
  "provider": "tinfoil",
  "model": "voxtral-small-24b",
  "endpoint": "http://127.0.0.1:3301/v1/audio/transcriptions",
  "attestation_required": true,
  "api_key_env": "WIZARD_VOICE_TINFOIL_KEY"
}
```

Health's `attested` field describes the selected route contract; it is not a fresh
cryptographic attestation receipt. Router audio/Tinfoil support is future work;
no router changes or keys are part of this app slice.

## Agent and API

`/mcp` is the Transcriber-style stateless streamable-HTTP JSON-RPC front on the
same ASGI port: initialize, ping, tools/list, tools/call, notification 202.
Tools: `transcribe(path, mode?, language?)` (Files-relative, max 32 MiB), and
`voice_status` (provider and personal/style rule counts). No arbitrary URL,
configuration mutation, secret retrieval or UI. Optional MCP-only bearer auth:
`WIZARD_VOICE_MCP_TOKEN`; it does not secure the REST API. Only the private
box/gateway network is in scope. All request bodies stay out of access logs.
MCP rejects browser `Origin` by default; a future gateway can allow exact origins
with comma-separated `WIZARD_VOICE_MCP_ORIGINS` (never a wildcard).
Meeting mode is not diarization; with no enabled rules it should produce the same
words as dictation. `/health` validates config, not engine readiness.

## Reproducible acceptance

Run `python3 qa/wizard_app_acceptance.py --evidence /absolute/evidence/path` on a
macOS build worker with Docker/buildx, Python 3, macOS `say` and `afconvert`.
It overrides only its isolated project to use a locally built image with
`pull_policy: never`; this is source-build acceptance, not anonymous publication
proof. It builds ARM64 and tries AMD64, runs the full suite and installed-package smoke
inside the ARM64 image, fetches Transcriber's pinned public recipe, starts its
unmodified main+engine plus Voice on an isolated compose network, generates only
synthetic TTS audio and exercises dictation, meeting, MCP listing/call/status,
clean rule counts and Tinfoil refusal. It records OCI digests and the commit.
Cleanup runs in `finally` and removes project containers, networks and volumes,
including downloaded model weights; source and small evidence are retained.
