FROM python:3.12-slim@sha256:2259debdc6af85626058a15a7bf44ad6a68b8c98c9ae82a02cf5cdb92f096d47
ARG VCS_REF=unknown
LABEL org.opencontainers.image.source="https://github.com/humanitylabs-org/wizard-voice-service" \
      org.opencontainers.image.revision=$VCS_REF
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOME=/data \
    WIZARD_VOICE_CONFIG=/data/config.yaml WIZARD_VOICE_DATA_DIR=/data/jobs \
    WIZARD_STT_CORRECTIONS=/data/corrections.yaml WIZARD_VOICE_ENV=/dev/null \
    WIZARD_VOICE_ALLOWED_HOSTS=127.0.0.1,localhost,wizard-voice-service,testserver \
    WIZARD_VOICE_RETAIN_FAILED_AUDIO=0
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/* && mkdir /data && chown 1000:1000 /data
# Explicit allowlist: no test transcripts, recordings, runtime state or owner files.
COPY pyproject.toml LICENSE THIRD_PARTY.md ./
COPY assets/ ./assets/
COPY *.py ./
COPY voice_mcp/ ./voice_mcp/
COPY packaging/ ./packaging/
RUN python -m pip install --no-cache-dir .
USER 1000:1000
EXPOSE 8803
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s CMD ["python", "-c", "import json,urllib.request; assert json.load(urllib.request.urlopen('http://127.0.0.1:8803/health',timeout=3))['status']=='ok'"]
CMD ["python", "packaging/entrypoint.py"]
