# Contributing

Keep changes small and tested. Use named feature/fix branches and pull requests.
Default/protected branches are review-only. Run the full suite and installed
package smoke in README before proposing a release.

Never commit real audio, transcripts, personal dictionaries, private hostnames,
credentials, operational logs or user configuration. Use synthetic fixtures.
Scan the tracked tree and every Git ref before publication. Regression evidence
from private recordings belongs outside the repository. Do not change providers,
attestation, fallback behavior, data retention or consent defaults as a drive-by
fix. Correction and style labels must remain separate. Test both positive and
literal/negative examples, mode isolation, idempotence and deterministic order.

Regex rules are trusted administrator code: Python regex can consume excessive
CPU. Do not expose arbitrary rule creation to untrusted callers. The CLI is
local and owner-operated, not a remote multi-user management API. Keep this
private until independent privacy and licensing review approves release.
