# Isolated three-engine batch trial

This is an **opt-in candidate**, not a daily-service update. The packaged Voice
app still defaults to the separately installed Transcriber app. Do not replace
an existing service/config/data directory, enable a launch agent, or expose these
unauthenticated trial endpoints beyond loopback. Use an SSH tunnel if needed.

## Architecture / identities

| Profile | Engine | Execution | API |
|---|---|---|---|
| `tinfoil-voxtral` | Voxtral Small 24B | Existing attested loopback Tinfoil proxy | Existing hosted transcription route |
| `parakeet-local` | Parakeet TDT 0.6B v3 ONNX INT8 | CPU, four threads | OpenAI-compatible multipart `POST /v1/audio/transcriptions` |
| `voxtral-local` | **Voxtral Mini 4B Realtime, 6-bit** | Apple Silicon MLX/Metal, batch use | Same multipart transcription contract |

Local engines expose `/health` (model-load receipt, identity, file hashes,
instance, process RSS/CPU) and `/v1/models`. They require the exact model field;
empty transcripts are valid on silence. Prompts, streaming, unknown models,
non-JSON response formats and non-English requests are rejected rather than
silently interpreted as a supported task. Legacy Parakeet octet-stream input
remains supported. Serve one worker only.

The Voice profile is explicitly `provider: local-engine` so credential loading
and arbitrary OpenAI public endpoints cannot be inherited. Engine requests use
loopback-only HTTP, ignore ambient proxies, refuse redirects and check response
identity. No cloud fallback exists. A failed local engine yields the existing
Voice 502 response; partial batch transcripts are not returned. An explicitly
selected Tinfoil profile uses only its existing approved attested route and
read-only credential sources. Credentials never belong in this repository.

This selection is compatible with the new Wizard app division: Voice consumes
an STT engine endpoint; the local engines can be packaged separately later.
This branch does **not** alter `wizard.md`, `needs`, the default Transcriber
connection, the image, or realtime/conversation capabilities. Engine app images
and daily deployment are outside this trial.

## Batch policy / historical boundary

The candidate opts into `pause30`: decode to mono 16 kHz, cover every sample once,
cut at the quietest 200 ms in the eight seconds before each 30 s limit, and retain
the last partial frame. Every engine uses that same segmentation policy. Local
engines accept up to 900 seconds / 64 MiB and reject over-limit audio, never
truncate it. The existing Voice upload ceiling remains 32 MiB.

The trial path bypasses the existing public-main context/overlap/seam-reread
experiment. It does not cherry-pick old private history or revive rolled-back
context-boundary/rc3 changes. Existing non-trial paths remain untouched. No
personal replacement rules or recordings are included.

For measurement, silence is deliberately **not masked with VAD**; the purpose
is to expose hallucinations, not manufacture clean silence results. Empty local
output is Voice's normal no-speech result. Hosted rejection/empty-response errors
remain failures in the table, not successful silence. This raw acoustic trial is
not an assertion of daily-ready silence/refusal policy parity.

## Install / run local engines

Use a new environment and existing verified model directories. Download models
only on the inference machine, not the control-plane host. `onnx-asr==0.12.0`
is already the optional `parakeet` dependency. Install voxmlx explicitly:

```sh
uv pip install --python "$ENV/bin/python" '.[test,parakeet]' \
  'voxmlx @ git+https://github.com/awni/voxmlx.git@e6d193e85e84e30f26e370c66973ce287b8a9d57'

WIZARD_PARAKEET_MODEL_DIR="$PARAKEET_MODEL" WIZARD_PARAKEET_THREADS=4 \
  "$ENV/bin/python" -m uvicorn parakeet_adapter:app --host 127.0.0.1 --port 19865 --workers 1

WIZARD_VOXTRAL_MODEL_DIR="$VOXTRAL_MODEL" HF_HUB_OFFLINE=1 \
  "$ENV/bin/python" -m uvicorn voxtral_adapter:app --host 127.0.0.1 --port 19866 --workers 1
```

Model pins:
- `istupakov/parakeet-tdt-0.6b-v3-onnx@8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce`:
  config, vocabulary, INT8 encoder/joint decoder, `nemo128.onnx`.
- `mlx-community/Voxtral-Mini-4B-Realtime-6bit@02eb0caeb9dafb554c17a72b93dbf40cd3736c31`:
  local model/config/tokenizer files. The worker pins the download; runtime
  refuses remote IDs, verifies the voxmlx Git commit and records model hashes.
- Mini 3B is an explicitly **different, unimplemented alternative**, never an
  automatic fallback or a result attributed to 4B. If 4B fails, evaluate 3B in a
  separate named lane with its own adapter/pin/results.

Copy `examples/three-engines.yaml` to an isolated work directory. Run an isolated
Voice broker with an explicit config, fresh data dir and distinct port:

```sh
WIZARD_VOICE_CONFIG="$WORK/three-engines.yaml" WIZARD_VOICE_DATA_DIR="$WORK/state" \
  "$ENV/bin/python" -m uvicorn server:app --host 127.0.0.1 --port 19864
```

Use `wizard-voice-profile --config "$WORK/three-engines.yaml" status`, then
`select voxtral-local --expected parakeet-local` to change this candidate only.
For a stable benchmark, the worker uses separate immutable per-engine configs
and starts a fresh broker rather than changing a profile under queued jobs.

## Corpus / scoring

A corpus folder contains audio and same-stem UTF-8 `.txt` references, including
an **empty** `.txt` for true silence/noise. The harness freezes SHA-256 hashes of
audio and references; adding another engine to an output requires exactly the
same manifest. It rejects repeated lanes rather than quietly overwriting failed
attempts. Use a new output directory for a retry. No confidence scores or
synthetic transcripts substitute for actual inference.

```sh
"$ENV/bin/python" benchmarks/three_engines.py \
  --config "$WORK/three-engines.yaml" --corpus "$CORPUS" --output "$RESULTS"
```

`--engines parakeet-local,voxtral-local` runs just those lanes. Running
`--engines tinfoil-voxtral` later with the same output appends the hosted lane.
The selected config does not become the daily configuration. Outputs:

- `corpus.json`: immutable audio/reference manifest.
- `attempts.jsonl`: fsynced actual per-file attempts, text/error, elapsed time,
  RTF, engine health before/after, memory and CPU counters.
- `results.json` and `RESULTS.md`: read-back aggregate and side-by-side table.

WER is case/punctuation-normalized word Levenshtein; no semantic/numeric rewrite.
The table's speech WER includes successful speech files only. Inspect failures
before comparing engines. It reports deletions, trailing omissions, exact last
12 words, silence/noise hallucinated words, latency and RTF separately. Process
peak RSS includes model loading and is a lifetime high-water mark, not per-file
allocation. Hosted remote RAM/CPU are unavailable and remain null/unmeasured.
End-word checks are useful probes, not proof of every boundary's correctness.

Output can contain sensitive transcripts when used with future real audio;
keep the result directory private. Only scripted/non-private fixtures are
admitted to the current worker. The worker may copy earlier `synthetic.wav` and
`silence.wav` only if their manifest explicitly says synthetic and supplies a
reference. It never scans or transcribes private recordings.

## Finite remote worker

Stage this pushed branch on the Apple Silicon worker's external work volume.
`benchmarks/run_apple_worker.py` takes explicit `--source`, new one-use `--work`, `--uv`,
`--parakeet-model-dir` and optional `--prior-synthetic-dir`. It installs a wheel
into a fresh environment, records installed module paths and dependencies,
runs the suite, generates macOS `say` speech including a six-minute-plus note,
downloads the pinned 4B model, then runs local lanes sequentially. Each local
lane includes actual HTTP broker short/long probes and an adapter-outage probe.
Only owned foreground child servers are terminated. All data/caches/scratch
stay under `--work`. Submit this worker through the durable build/test queue;
do not hold an interactive model turn open waiting for inference.

The worker does not copy hosted credentials to the Mac. Replay the identical
frozen corpus through the hosted lane on the control plane, using the existing
read-only credential source and unchanged proxy. Preserve original hashes when
transferring audio. No daily STT service is needed or changed for this replay.
A local-lanes receipt alone is **not** three-engine completion.
