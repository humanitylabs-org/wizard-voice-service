# Wizard Voice Service

A small standalone voice component for **Wizard OS**, maintained by Humanity
Labs. Our purpose is to improve open voice dictation, transcription, meeting
transcription and diarization while giving people control over recognition
corrections and how their words are styled. Voice data is personal: useful
customization must not silently become shared data.

First-party code is MIT licensed (see [LICENSE](LICENSE)); third-party
components keep their own licenses, listed in [THIRD_PARTY.md](THIRD_PARTY.md).

## What works now

- File transcription through the existing Tinfoil-attested proxy route and
  `voxtral-small-24b`, with **no automatic provider fallback**. The pre-existing
  explicitly selected OpenAI-compatible adapter remains for compatibility;
  this MVP does not select it or silently downgrade attestation.
- Existing `/inference` and `/v1/audio/transcriptions` multipart clients;
  optional `mode=dictation|meeting|verbatim` (default dictation).
- SQLite-backed durable jobs, idempotent admission and polling after restart.
- Conservative local speech-presence detection, chunking and same-provider
  repair guards. These do not guarantee transcript accuracy.
- Separate recognition-correction and preferred-style buckets, stable rule IDs,
  enable/disable, scope/provenance, deterministic ordering, local CLI previews.
  A clean install contains **no personal dictionary** and no enabled style rules.
- Existing configuration/corrections can be reused without rewriting them.

Meeting mode currently means ordinary transcription **without personal style**;
it is not speaker diarization. Realtime/conversation lane metadata is retained
for Wizard OS integration, not an implemented streaming or conversational API.
Speaker labels, a rule-editing GUI and optional consensual learning are roadmap,
not delivered features.

## Install and run (Linux or macOS)

Requires Python 3.11+ (tested on 3.12), `ffmpeg` and `ffprobe` on PATH, and a
separately provisioned Tinfoil attestation proxy on `127.0.0.1:3301` for real
transcription. The service does not install/start a proxy. Health validates
configuration, not proxy reachability or fresh cryptographic attestation.
No Hermes installation, home directory or private files are needed.

```sh
# Clone the reviewed branch/revision from this private repository first.
python3 -m venv .venv
.venv/bin/python -m pip install '.[test]'
mkdir -p "$HOME/.config/wizard-voice"
cp examples/config.yaml "$HOME/.config/wizard-voice/config.yaml"
export WIZARD_VOICE_CONFIG="$HOME/.config/wizard-voice/config.yaml"
export WIZARD_VOICE_DATA_DIR="$HOME/.local/share/wizard-voice"
# Supply TINFOIL_API_KEY via your secret manager/process environment.
.venv/bin/python -m uvicorn server:app --host 127.0.0.1 --port 8864
```

Run a single process/worker per data directory. Do not start a second process
against an active jobs database: startup recovers interrupted `processing` jobs.
Use separate config/data directories and a different port for candidates.

```sh
curl http://127.0.0.1:8864/health
curl -F file=@sample.wav http://127.0.0.1:8864/v1/audio/transcriptions
curl -F file=@sample.wav -F mode=meeting http://127.0.0.1:8864/inference
.venv/bin/python -m pytest -q
.venv/bin/python -m pip check
.venv/bin/python qa/smoke.py
```

The installed-package smoke launches the real service from an empty HOME/cwd on
an exclusively held dynamic socket, loads the packaged ONNX model, checks the
missing-credential failure, verifies no failed-audio retention and proves durable
job admission/deduplication/restart. It makes **no upstream provider call**. Real
audio/provider acceptance is a separate private deployment gate. Unit fixtures
and text previews are not acoustic proof.

## Local rule management

Rule edits are local owner operations, not a remote/untrusted API. They use a
stable file lock and atomic owner-only writes. Config/rule files are reread on
each request; edits need no restart. CLI `--file` must name the same file as
`voice.rules_file` or `WIZARD_VOICE_RULES` for them to affect transcription.

```sh
export WIZARD_VOICE_RULES="$HOME/.config/wizard-voice/rules.yaml"
.venv/bin/wizard-voice-rules --file "$WIZARD_VOICE_RULES" add examples/correction.yaml
.venv/bin/wizard-voice-rules --file "$WIZARD_VOICE_RULES" list
.venv/bin/wizard-voice-rules --file "$WIZARD_VOICE_RULES" enable common.join-laughter
printf '%s' 'Acem says ha ha.' | .venv/bin/wizard-voice-rules --file "$WIZARD_VOICE_RULES" preview
.venv/bin/wizard-voice-rules --file "$WIZARD_VOICE_RULES" disable user.example-name
# Include the effective legacy configuration in list/preview:
.venv/bin/wizard-voice-rules --config "$WIZARD_VOICE_CONFIG" list
```

A missing explicitly selected rules file is a configuration error; create it
with CLI add/enable before pointing the service at it. Without a selected file,
the only built-in rule is disabled `common.join-laughter`.

A rule has `id`, `kind` (`correction`/`style`), `scope` (`common`/`user`),
`provenance`, `enabled`, integer `priority`, `modes` and `operation`.
`regex` needs `pattern`, `replacement`, and optional `case_sensitive`.
`join_laughter` is a dictation-only style: `ha ha` → `haha`, preserving repetition.
`attach_laughter` is also available but only as an explicit **user-scoped**
dictation preference. It joins laughter, lowercases a trailing laughter suffix,
removes its single final period and attaches it across a single sentence stop.
It also handles sentence-internal laughter conservatively: `Funny. Ha ha. Next
sentence.` becomes `Funny haha\nNext sentence.` (here `\n` denotes a newline,
not literal backslash characters). The laughter has no trailing period; the
newline retains the next sentence boundary without altering the following
sentence's text or case. This internal form requires a single period after
laughter, horizontal whitespace, then an unquoted word beginning with a letter.
That separating whitespace becomes one newline. Existing line endings (including
CRLF) are preserved; exclamation marks, question marks, ellipses and ambiguous
boundaries are not reinterpreted. Repetition count is preserved and applying the
rule again leaves its output unchanged. Common `join_laughter` alone never
attaches sentences or inserts a newline; meeting/verbatim modes never apply
these laughter styles.
No owner-specific attachment preference ships enabled. Quotes, code fences,
path-like tokens and literal/spelling-report lines are conservatively protected;
laughter never joins across a newline. In particular, a following quoted/code/
path fragment is not treated as an internal sentence boundary (`Funny. Ha ha.
"Next."` only joins to `Funny. haha. "Next."`). This is a bounded text heuristic,
not semantic understanding. Test your own examples before enabling a rule.

Order: the legacy pipeline (optional time normalization, then system replacements,
then service-owned replacements) → new correction bucket → new style bucket.
The correction-only preview excludes legacy time normalization; emitted dictation
preserves the old time-before-replacement order for compatibility. Inside each
new bucket: common before user, then ascending priority, then stable ID. Files
may override a built-in rule using its ID; duplicate IDs in one file are invalid.
Legacy IDs are read-only content/index fingerprints; existing files are not
migrated or rewritten. New-rule order does not depend on YAML list order.

Preview returns `corrected_text`, `styled_text`, and changed new-rule IDs. It
never sends preview text to the provider or saves it. Prefer stdin to avoid
shell-history retention. Transcript API responses and job records retain their
existing compact shape; no extra raw/corrected transcript is stored by default.
Verbatim bypasses **all** deterministic rules; meeting allows recognition
corrections but disables styles/time normalization. Provider whitespace cleanup
and acoustic/model limitations still apply; “verbatim” is not a byte-exact promise.

Regex is trusted administrator code; Python regex can run slowly. Do not expose
rule creation to untrusted callers. Invalid new-rule files fail configuration
validation. Legacy invalid replacement entries continue to be skipped for
compatibility.

## Configuration and existing deployments

Config precedence: explicit function/CLI path → `WIZARD_VOICE_CONFIG` → legacy
`WIZARD_STT_CONFIG` → existing `$XDG_CONFIG_HOME/wizard-voice/config.yaml`
(default `~/.config`) → legacy `~/.hermes/config.yaml`.
The service config keeps the existing `stt`/`voice` structure. `voice.rules_file`
is relative to that config's directory; `WIZARD_VOICE_RULES` overrides it.
The legacy `WIZARD_STT_CORRECTIONS` path (or adjacent `corrections.yaml` in a
source deployment) continues to append replacements after `stt.postprocess`.
Wheel installs do not bundle that private dictionary: explicitly point to the
existing file when upgrading. Existing disabled/malformed legacy file behavior
is unchanged. No data is imported from a different user.

Credentials: process environment first, then explicit `WIZARD_VOICE_ENV`, legacy
`HERMES_ENV`, or legacy `~/.hermes/.env`. No credential is included in this repo.
For standalone isolation set the config/data/env/corrections paths explicitly.
Default data directory preserves an existing `~/.hermes/data/apps/wizard-voice`;
otherwise it uses `$XDG_DATA_HOME/wizard-voice` (default `~/.local/share`).
`WIZARD_VOICE_DATA_DIR` always wins. Database schema is unchanged.

Other operational options remain `WIZARD_STT_MAX_AUDIO_BYTES` (32 MiB default),
`WIZARD_STT_PROVIDER_CHUNK_SECONDS`, `WIZARD_STT_CHUNK_CONCURRENCY`,
`WIZARD_STT_SEGMENT_TIMEOUT_SECONDS`, `WIZARD_VOICE_ALLOWED_HOSTS`, and
`WIZARD_VOICE_ADMIN_TOKEN`. Admin config remains authenticated and read-only
(exact no-op PATCH); rule management is CLI-only. The optional Linux `reconcile`
module is a legacy realtime-unit adapter, not needed for standalone service use.

## Trust and data retention

### Context-preserving transcription (0.2.1rc2 candidate)

Notes up to 120 seconds are normalized to the same mono 16 kHz / 64 kbps MP3
and attempted whole, even when the legacy chunk override is 30 seconds.
Longer notes use that override as a **fallback target**, clamped to 30–45
seconds. Cuts prefer the nearest energy pause within five seconds and retain
up to four seconds on each side. The existing acoustic speech gate still runs.

Adjacent readings join only on a unique, exact word suffix/prefix (ignoring
case/punctuation), with at least three distinct words. Repeated or conflicting
seams are not guessed. A conflict rereads the union of the two adjacent audio
spans on the same selected provider/model, without supplying either proposed
transcript. A reread must still agree with its neighbours. Each split group
allows at most four union rereads, each at most 120 seconds of audio; exhaustion
or ambiguity returns `transcript_seam_unresolved`, never a partial success.
Repeated speech outside an agreed overlap is not deduplicated.

Each provider attempt is capped by `WIZARD_STT_SEGMENT_TIMEOUT_SECONDS`
(45 seconds by default) and the remaining original `stt.timeout` request budget.
Timeouts can split at most two levels, using the same pause/context/join policy.
Very short stalled segments retain their one neutral-prompt retry. Local
normalization/rendering retains separate bounded subprocess timeouts; the
request deadline prevents further provider calls after its budget is spent,
but is not a hard wall-clock cancellation of local decoding.

This policy trades some latency/provider calls for context and explicit failure
instead of silently contradictory joins. It cannot guarantee acoustic truth:
an agreed overlap can still contain a shared model error, and a whole-note or
join-window reading can omit speech. There is no new model, fallback, glossary,
retention policy, or change to recognition/style/meeting modes. Synchronous
uploads remain temporary unless failure retention was explicitly enabled;
durable jobs retain their original audio and expose retryable/terminal errors.

The candidate also closes a refusal-policy gap exposed by replaying the existing
answer-expansion fixture: “I can't find any audio to transcribe” from a recovery
call is an assistant refusal, not successful dictation. Recovery still uses only
the selected attested model and fails closed if it cannot return speech.

### Existing custody policy

Bind to loopback. Inference/jobs assume a trusted single-user machine and are
**not authenticated** by this service. Host-header filtering is not access
control. Put authentication/authorization in front before exposing it to a
network; never expose the inference bill or job transcripts publicly.
Admin requests use `X-Wizard-Voice-Admin` and are disabled without the token.
Provider errors may include provider diagnostics: keep logs and endpoints private.

Synchronous upload audio is temporary and deleted. Failed-audio retention is
now **off by default**; an operator can explicitly restore the legacy 48-hour
failure spool with `WIZARD_VOICE_RETAIN_FAILED_AUDIO=1` and
`WIZARD_STT_FAILED_AUDIO_DIR`. Pruning is triggered by a later retained failure,
not by a background timer. Durable jobs deliberately retain their audio and
final transcript for recovery, with owner-only files; there is no automatic
job TTL/deletion API yet. Do not use durable jobs where that retention is
inappropriate. This release adds no collection, upload, training or telemetry.

## Wizard OS component relationship

```text
Wizard OS / app clients / meeting capture
                  | stable multipart or durable-job API
                  v
          Wizard Voice Service
          /                  \
local rules + SQLite     existing attested proxy -> ASR
                  |
             transcript -> caller / agent
```

| Sibling repository | Responsibility |
|---|---|
| `wizard-os`, `try-wizard` | Compose and distribute the system |
| `wizard-ux` | Shell/UI and voice API clients |
| `wizard-agent-service` | Consume transcripts, not own this correction pipeline |
| `meeting-recorder` | Separate recording capture and lifecycle |
| `wizard-app-manager`, `wizard-app-store`, `wizard-app-index` | Installation and discovery |

This repository neither replaces those components nor requires a shell/core
update. Existing clients can keep their endpoint and response shape.

Source layout: `server.py`/`client.py` own HTTP edges, `providers.py` the existing
ASR adapter, `jobs.py` durable state, `config.py`/`control.py` configuration,
`postprocess.py`/`voice_rules.py`/`voice_cli.py` local rules, `assets/` the VAD,
`tests/` synthetic regressions, `qa/smoke.py` installed-package acceptance,
`examples/` safe configuration, and `docs/` the inactive contribution schema.

| Main environment variable | Purpose |
|---|---|
| `WIZARD_VOICE_CONFIG` | Explicit service-owned configuration |
| `WIZARD_VOICE_RULES` | Owner-managed versioned rules file |
| `WIZARD_STT_CORRECTIONS` | Existing private legacy dictionary |
| `WIZARD_VOICE_DATA_DIR` | Durable job directory; keep across upgrades |
| `WIZARD_VOICE_ENV`, `TINFOIL_API_KEY` | Secret-manager/env-file reference, provider credential |
| `WIZARD_VOICE_ADMIN_TOKEN` | Optional local admin credential |
| `WIZARD_VOICE_RETAIN_FAILED_AUDIO` | Explicit opt-in for legacy failure spooling |

## Future opt-in seam (inactive)

`docs/contribution-seam.schema.json` distinguishes recognition-correction labels
from style-preference labels, and records rule provenance, consent status and
license status. Collection/upload are explicitly false even when consent is
recorded: this schema is documentation, not an operational submission API.
There are no audio/text fields, collectors, uploads, training jobs or reward
systems. A future extension requires separate consent, privacy/license review
and deletion/revocation design; local customization never implies permission.

## Upgrade and rollback

Keep config, private corrections, rules and durable state outside the checkout.
Pin a reviewed commit. Test a new venv/package with an isolated data directory,
then stop the old process before changing the one service command. Point the
new process at the **same** approved config/corrections/data paths; do not copy
or recreate a live SQLite database. Verify health, a private retained-audio
request and job readback. Roll back by stopping the candidate and restoring the
previous executable/venv with those same paths. No schema migration or automatic
provider switch occurs. A first live activation/restart is operator-owned, not
performed by any install, test or CI command here.
