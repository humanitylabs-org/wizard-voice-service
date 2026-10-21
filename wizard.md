---
wizard: 0.2
id: wizard-voice-service
name: Voice
type: service
category: system
tags: [audio, dictation]
version: 0.2.2-rc1.app1
license: MIT
summary: Dictation modes, local corrections and durable voice jobs over the installed speech engine.
source: https://github.com/humanitylabs-org/wizard-voice-service
runtime:
  kind: runtipi
  config: runtipi/config.json
  compose: runtipi/docker-compose.yml
mcp:
  path: /mcp
  port: 8803
  transport: streamable-http
skill: SKILL.md
provides: [voice]
needs: [speech-to-text-engine]
uses: [files]
health: /health
---

# Voice

The dictation layer, not another speech engine. Install Transcriber first.
No owner UI. See docs/wizard-app.md for private image preload and rollout gates.
