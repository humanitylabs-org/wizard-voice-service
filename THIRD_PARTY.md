# Provenance and licenses

The service is a clean export of the existing Wizard OS standalone voice
service, with portable configuration and rule management added. Original local
history and owner-specific data were deliberately not imported. The private
review handoff records the exact source revision and export hashes. It is not a
rewrite of the ASR provider or a fork of Hermes core. First-party code is MIT
licensed (see LICENSE); upstream components keep the licenses listed below.

| Dependency | Use | License / upstream |
|---|---|---|
| FastAPI | HTTP API | MIT, https://github.com/fastapi/fastapi |
| Uvicorn | ASGI server | BSD-3-Clause, https://github.com/encode/uvicorn |
| python-multipart | upload parsing | Apache-2.0, https://github.com/Kludex/python-multipart |
| python-dotenv | explicit/legacy env file | BSD-3-Clause, https://github.com/theskumar/python-dotenv |
| PyYAML | config | MIT, https://github.com/yaml/pyyaml |
| NumPy | local VAD arrays | BSD-3-Clause, https://github.com/numpy/numpy |
| ONNX Runtime | local VAD runtime | MIT, https://github.com/microsoft/onnxruntime |
| onnx-asr | optional CPU-only Parakeet sidecar | upstream distribution notices, https://github.com/istupakov/onnx-asr |
| Silero VAD | bundled ONNX speech detection | MIT, https://github.com/snakers4/silero-vad |
| FFmpeg / ffprobe | system audio tools, not bundled | LGPL/GPL depending on installed build, https://ffmpeg.org/legal.html |
| pytest / httpx | test-only | MIT / BSD-3-Clause, upstream project distributions |
| setuptools | wheel build | MIT, upstream distribution |

The vendored Silero model is pinned by SHA-256 in `speech_gate.py`; its original
license is included verbatim. Installed transitive dependencies retain their
own distribution metadata/license files. Review the resolved dependency list
and notices for each release; a Python lock is not a license audit. The Tinfoil
proxy and hosted model are external prerequisites and are not bundled or
relicensed here. Health's `attested` flag describes the configured approved
route, not a fresh cryptographic attestation test.

Parakeet weights are not bundled or downloaded by the sidecar. Operators supply
the ONNX conversion from `istupakov/parakeet-tdt-0.6b-v3-onnx`, record its revision
and verify every model-file hash. The staged trial uses revision
`8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce`. Review both the conversion's notices
and NVIDIA's original Parakeet model terms before redistributing weights.
