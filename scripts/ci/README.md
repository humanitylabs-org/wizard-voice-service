# Public container publication

`Publish public image` runs on every canonical-branch push (including PR merges).
Every commit gets a write-once `sha-<full 40-character commit>` tag. It does not
move `latest` or any release tag. Re-running a SHA reuses its manifest and still
requires the independent public-pull job to pass.

## First publication and releases

1. Merge through a reviewed PR with all required checks green. The repo-scoped
   Actions token has `packages: write`; no personal package token is required.
2. A new GHCR package is initially private. An administrator must set it to
   **Public**, with the linked repository retaining Actions write access.
3. Re-run the failed `anonymous-pull` job. A private package is a release failure,
   never a warning or a skipped test.
4. For a version alias, dispatch on the canonical branch with `release_tag` and
   `confirm_release_tag` equal (for example `0.3.0-candidate.20261012`) and
   `expected_sha` equal to the full selected branch commit. Existing release tags
   fail rather than being reassigned. A dispatch whose build succeeded but public
   check failed should rerun **failed jobs only**, not repeat the alias creation.

Anonymous verification runs in a fresh hosted Linux job, has no package
permission, uses an empty isolated Docker configuration and no inherited auth
variables, obtains only anonymous registry tokens, and actually pulls both
`linux/amd64` and `linux/arm64` by the published index digest and every requested
tag. It verifies child manifest digests and the pulled source commit revision. Registry
propagation is retried for five minutes, then the job fails closed.

## Trusted worker contract

The build job uses a dedicated repo-scoped macOS ARM64 runner with the
`wizard-ghcr-<repository-name>` label. Runner install, worktree, HOME, temporary
Docker credentials, caches, diagnostic logs and listener output must reside on the
external worker SSD, supervised by the control plane's user service over SSH.
Colima's existing containerd image store must advertise
both platforms. Do not replace its VM, switch global contexts, install binfmt,
prune other builders or touch live containers.

Host environment supplies `WIZARD_HEAVY_ROOT` (fallback `WIZARD_WORK_ROOT`) and
`WIZARD_CI_ROOT` beneath that configured root,
`WIZARD_CI_WORKER_LOCK` (the existing shared Android/VM worker lock), and
`DOCKER_HOST` (the explicit existing Colima socket). Builds use `fcntl.flock`,
wait up to one hour and release automatically on exit. The 100-minute job limit
bounds lock plus build time. If another heavy task holds the lock longer, rerun
the failed publication after that task ends; never remove or replace the lock.

Before registering the runner, require approval for **all external fork
contributors**, protect the canonical branch with PR/check gates, and install a
host-owned job-start guard matching repository, ref, event and workflow path.
The workflow never runs PR code on the self-hosted worker; PR contract tests run
on hosted Linux. Repository writers and people approving fork workflows are
trusted: a job-start hook is defense in depth, **not a sandbox** against a
malicious approved workflow using `always()` steps. Do not approve untrusted
workflows that target this runner. Untrusted execution needs separate ephemeral
VM isolation, not this persistent runner.

Temporary publish credentials are written directly with mode 0600 and removed
on normal exit. `docker login` is deliberately not used: it may auto-detect the
macOS keychain and overwrite the owner's credentials. No global Docker config,
keychain, GH CLI identity or personal token is used by the publisher.

## Checks

```sh
python3 -m unittest discover -s scripts/ci -p 'test_*.py' -v
```

Unit tests exercise guards, collision refusal and credential-free pull commands;
only an actual protected-branch Actions run proves the build and public install
contract. The existing Tests workflow remains unchanged. PR verifier smoke now
pulls the first verified public voice image itself; dispatch can verify any exact
voice digest and source revision.

Publication code is adapted from `humanitylabs-org/wizard-agent-service`
(`scripts/ci/ghcr_image.py`, `test_ghcr_image.py` and publication workflows);
the source revision and byte hashes are recorded in `provenance.json`.
