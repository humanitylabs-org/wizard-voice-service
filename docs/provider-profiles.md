# Selectable batch providers (staged, no production cutover)

Install the supplied wheel with `pip install './wizard_voice_service-0.2.2rc1-py3-none-any.whl[parakeet]'` on adapter hosts.
The broker does not require the Parakeet extra. Python >=3.11 and ffmpeg are required.

Copy `examples/profiles.yaml` to an owner-only **trial** directory. Keep existing
postprocessing and `voice.rules_file` settings in this copy if desired. Keep rules,
state and failed audio separate from production. Profiles affect batch only;
legacy realtime inheritance is unchanged. There is no new streaming service.

```
export WIZARD_VOICE_CONFIG="$TRIAL/config.yaml"
export WIZARD_VOICE_DATA_DIR="$TRIAL/jobs"
export WIZARD_STT_FAILED_AUDIO_DIR="$TRIAL/failed"
export WIZARD_STT_CORRECTIONS="$TRIAL/corrections.yaml"
export WIZARD_VOICE_RULES="$TRIAL/rules.yaml"
export WIZARD_VOICE_ENV="$TRIAL/provider.env"
python -m uvicorn server:app --host 127.0.0.1 --port 18864
```

Start the same sidecar on each authorized host with an already downloaded, verified
`istupakov/parakeet-tdt-0.6b-v3-onnx` int8 directory. No automatic downloads occur.
Pin and record the model revision and file hashes before use. Run one worker only:

```
export WIZARD_PARAKEET_MODEL_DIR="$MODEL_DIR"
export WIZARD_PARAKEET_THREADS=4
python -m uvicorn parakeet_adapter:app --host 127.0.0.1 --port 18865 --no-access-log
```

For the remote adapter, forward its loopback port over the owner's SSH/Tailnet route:

```
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:18866:127.0.0.1:18865 "$MAC_SSH_HOST"
```

Keep this tunnel available only while trialing the remote lane. The sidecar is an
unauthenticated **single-tenant loopback service**, not a public API. Do not bind it
to a public interface or publish its port. SSH authenticates remote access.
Local-host/private-remote labels declare intended deployment; they do not attest
physical location. Verify the actual listening process/tunnel and adapter receipt.

## Explicit switches and rollback

```
wizard-voice-profile --config "$TRIAL/config.yaml" status
wizard-voice-profile --config "$TRIAL/config.yaml" select current-vps-parakeet --expected tinfoil
curl -fsS http://127.0.0.1:18864/health
curl -fsS -F file=@"$AUDIO" http://127.0.0.1:18864/v1/audio/transcriptions
wizard-voice-profile --config "$TRIAL/config.yaml" select mac-mini-parakeet --expected current-vps-parakeet
# Roll back to attested Tinfoil (only this explicit action allows cloud processing):
wizard-voice-profile --config "$TRIAL/config.yaml" select tinfoil --expected mac-mini-parakeet
```

No restart is required: each newly admitted transcription snapshots its profile.
In-flight work stays on its original route. Queued jobs select at execution time;
drain a trial queue before switching when all queued audio must remain on one route.
The CLI locks, validates and atomically writes only the explicitly named config;
it never edits service units or default config paths. Concurrent stale switches fail.
It does not probe a provider or claim availability: always perform a real audio check.

Tinfoil needs its existing attestation proxy on loopback 3301 and server-side
`TINFOIL_API_KEY`. Private lanes never load or forward a cloud credential, follow
redirects or fail over to another provider. Adapter outage returns a recoverable
error. Prompts/context are not sent to Parakeet (this ASR model has no prompting
interface). Corrections/styling remain broker-side and mode-scoped.

The sidecar serializes inference and rejects concurrent requests with 429. Broker
context processing still attempts whole notes then reconciles overlapping long-note
segments; set `WIZARD_STT_CHUNK_CONCURRENCY=1` for Parakeet trials. Sidecar input is
limited to 120 seconds per request, decoded into contiguous pause-aware 20–30-second spans without
omitted audio. This is saved-recording ASR, **not streaming**, and is not a guarantee
of word-level fidelity at model slice seams. Test the actual recordings.

Health describes configuration (broker) or a loaded model (sidecar), not transcript
quality. Record source hash, adapter instance/host, result, latency, errors and
production before/after evidence. Stop only trial processes to roll back the staging
installation; no production changes are necessary or authorized by these examples.

## Current trial limitation (not production approval)

Actual saved-note trials pass on both CPU hosts, but a 343-second meeting take on
x86_64 hit the existing `transcript_seam_unresolved: join window exceeds safe limit`
guard. The native ARM64 Mac and Tinfoil completed that take. The guard remains intact:
no guessed join, partial successful result, or automatic cloud retry is allowed.
Keep the original audio and explicitly retry an owner-chosen route. Local long-form
meeting parity is therefore **not established**; keep this candidate staged until
that limitation is accepted or repaired. Raw ASR name/phrase errors also remain;
speed is not a transcription-quality guarantee.
