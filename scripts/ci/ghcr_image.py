#!/usr/bin/env python3
"""Publish on the trusted worker; verify from a credential-free consumer job."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

MEDIA = "application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json"
PLATFORMS = {"amd64", "arm64"}


class Registry:
    def __init__(self, image, authenticated=False):
        if not re.fullmatch(r"ghcr.io/[a-z0-9-]+/[a-z0-9-]+", image):
            raise ValueError("unexpected image name")
        self.path = image.removeprefix("ghcr.io/")
        headers = {}
        if authenticated:
            pair = os.environ["GITHUB_ACTOR"] + ":" + os.environ["GHCR_TOKEN"]
            headers["Authorization"] = "Basic " + base64.b64encode(pair.encode()).decode()
        query = urllib.parse.urlencode({"service": "ghcr.io", "scope": f"repository:{self.path}:pull"})
        request = urllib.request.Request("https://ghcr.io/token?" + query, headers=headers)
        with urllib.request.urlopen(request, timeout=30) as response:
            self.token = json.load(response)["token"]

    def manifest(self, reference, missing_ok=False):
        request = urllib.request.Request(
            f"https://ghcr.io/v2/{self.path}/manifests/{reference}",
            headers={"Authorization": "Bearer " + self.token, "Accept": MEDIA},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read()
        except urllib.error.HTTPError as error:
            if error.code == 404 and missing_ok:
                return None
            raise
        return "sha256:" + hashlib.sha256(data).hexdigest(), json.loads(data)


def platform_manifests(registry, digest):
    actual, index = registry.manifest(digest)
    if actual != digest:
        raise RuntimeError("registry returned wrong content digest")
    found = {}
    for entry in index.get("manifests", []):
        platform = entry.get("platform", {})
        arch = platform.get("architecture")
        if platform.get("os") == "linux" and arch in PLATFORMS:
            if arch in found:
                raise RuntimeError("duplicate supported architecture")
            child_digest, child = registry.manifest(entry["digest"])
            if child_digest != entry["digest"]:
                raise RuntimeError("child manifest digest mismatch")
            found[arch] = child["config"]["digest"]
    if set(found) != PLATFORMS:
        raise RuntimeError("image must contain linux/amd64 and linux/arm64")
    return found


def docker(args, env, **kwargs):
    return subprocess.run(["docker", *args], env=env, check=True, **kwargs)


def clean_env(config):
    # No Docker credential helper, DOCKER_AUTH_CONFIG, cloud identity, or gh token.
    # RUNNER_TRACKING_ID is not a credential: the self-hosted runner uses it to reap
    # orphaned docker/buildx clients when a job is cancelled or times out.
    passthrough = ("DOCKER_HOST", "RUNNER_TRACKING_ID")
    return {"PATH": os.environ["PATH"], "HOME": config, "DOCKER_CONFIG": config,
            **{key: os.environ[key] for key in passthrough if key in os.environ}}


def _volume_of(path):
    """The mounted volume holding path, unless that is the system disk ("/").

    Runners that predate WIZARD_HEAVY_ROOT keep WIZARD_CI_ROOT on a dedicated worker volume; accept that volume so the
    portable check keeps the old guarantee (build storage off the control-plane disk) without a machine path in git."""
    p = Path(path).expanduser().resolve()
    while not os.path.ismount(p):
        p = p.parent
    return None if p == Path(p.anchor) else str(p)


def publish(args):
    import fcntl  # macOS/Linux only, after CLI validation
    expected_ref = "refs/heads/" + args.branch
    if os.environ.get("GITHUB_REF") != expected_ref or os.environ.get("GITHUB_EVENT_NAME") not in ("push", "workflow_dispatch"):
        raise RuntimeError("publication is only allowed on the trusted branch")
    if args.release:
        if os.environ["GITHUB_EVENT_NAME"] != "workflow_dispatch" or args.confirm != args.release or args.expected_sha != args.sha:
            raise RuntimeError("release dispatch must confirm tag and exact selected commit")
        if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:-candidate\.[0-9]{8})?", args.release):
            raise ValueError("only semantic versions or dated candidate tags are accepted")
    root = Path(os.environ["WIZARD_CI_ROOT"])
    configured = os.environ.get('WIZARD_HEAVY_ROOT') or os.environ.get('WIZARD_WORK_ROOT')
    if configured and not Path(configured).is_absolute():
        raise RuntimeError('heavy lane not configured: set WIZARD_HEAVY_ROOT')
    lock_path = Path(os.environ["WIZARD_CI_WORKER_LOCK"])
    if configured:
        allowed = Path(configured).resolve()
        if not root.resolve().is_relative_to(allowed) or not lock_path.resolve().is_relative_to(allowed):
            raise RuntimeError("build storage and lock must be under the configured heavy root")
    elif not _volume_of(root):
        # Unconfigured runner: build storage must still sit on a dedicated mounted volume, never the system disk.
        raise RuntimeError('heavy lane not configured: set WIZARD_HEAVY_ROOT')
    print("Waiting for shared heavy-worker lock", flush=True)
    with lock_path.open("a") as lock:
        deadline = time.monotonic() + 3600
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("shared worker lock busy for one hour")
                time.sleep(10)
        with tempfile.TemporaryDirectory(prefix="publish-", dir=root) as config:
            # Explicit plugin path: no owner HOME or global Docker config is inherited.
            # Do not invoke docker login: on macOS it may auto-select the owner's
            # keychain even with an isolated DOCKER_CONFIG. Inline auth is temporary.
            auth = base64.b64encode((os.environ["GITHUB_ACTOR"] + ":" + os.environ["GHCR_TOKEN"]).encode()).decode()
            config_file = Path(config, "config.json")
            with config_file.open("x") as output:
                os.chmod(config_file, 0o600)
                json.dump({"auths": {"ghcr.io": {"auth": auth}},
                           "cliPluginsExtraDirs": ["/opt/homebrew/lib/docker/cli-plugins"]}, output)
            env = clean_env(config)
            registry = Registry(args.image, authenticated=True)
            existing = registry.manifest(args.tag, missing_ok=True)
            # Release names are write-once, including reruns: operator must investigate.
            if args.release and registry.manifest(args.release, missing_ok=True):
                raise RuntimeError("release tag already exists; refusing to overwrite")
            if existing:
                digest = existing[0]
                platform_manifests(registry, digest)
                print("Reusing immutable SHA tag", digest, flush=True)
            else:
                metadata = Path(config, "metadata.json")
                docker(["buildx", "build", "--platform", "linux/amd64,linux/arm64", "--provenance=false",
                        "--label", "org.opencontainers.image.source=https://github.com/" + os.environ["GITHUB_REPOSITORY"],
                        "--label", "org.opencontainers.image.revision=" + args.sha,
                        "--tag", args.image + ":" + args.tag, "--metadata-file", str(metadata), "--push", "."], env)
                digest = json.loads(metadata.read_text())["containerimage.digest"]
                registry = Registry(args.image, authenticated=True)
                if registry.manifest(args.tag)[0] != digest:
                    raise RuntimeError("published SHA tag does not match build digest")
                platform_manifests(registry, digest)
            if args.release:
                docker(["buildx", "imagetools", "create", "--tag", args.image + ":" + args.release, args.image + "@" + digest], env)
                if Registry(args.image, authenticated=True).manifest(args.release)[0] != digest:
                    raise RuntimeError("release alias changed image digest")
            with open(os.environ["GITHUB_OUTPUT"], "a") as output:
                output.write(f"digest={digest}\n")
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as output:
                output.write(f"Published `{args.image}@{digest}` from `{args.sha}`.\n\nTags: `{args.tag}` `{args.release}`. Public pull verification is a separate required job.\n")


def verify(args):
    # Deliberately never reads GITHUB_TOKEN / GHCR_TOKEN, even if accidentally supplied.
    with tempfile.TemporaryDirectory(prefix="anonymous-ghcr-") as config:
        Path(config, "config.json").write_text('{"auths":{}}')
        env = clean_env(config)
        deadline = time.monotonic() + args.retry_seconds
        while True:
            try:
                registry = Registry(args.image)
                platform_manifests(registry, args.digest)
                # Docker 29/containerd keeps both platforms locally; select the
                # platform explicitly. Classic stores replace the local tag on pull.
                store = docker(["info", "--format", "{{json .DriverStatus}}"], env, capture_output=True, text=True)
                containerd = "io.containerd.snapshotter" in store.stdout
                refs = [args.image + "@" + args.digest]
                for tag in [args.tag, args.release]:
                    if tag:
                        if registry.manifest(tag)[0] != args.digest:
                            raise RuntimeError("tag does not match published digest: " + tag)
                        refs.append(args.image + ":" + tag)
                for arch in sorted(PLATFORMS):
                    for ref in refs:
                        docker(["pull", "--platform", "linux/" + arch, ref], env)
                        options = ["--platform", "linux/" + arch] if containerd else []
                        result = docker(["image", "inspect", *options, "--format", "{{.Architecture}} {{index .Config.Labels \"org.opencontainers.image.revision\"}}", ref], env, capture_output=True, text=True)
                        if result.stdout.strip() != f"{arch} {args.sha}":
                            raise RuntimeError("pulled image configuration/revision mismatch: " + result.stdout.strip())
                        # Classic Docker cannot associate one index digest with
                        # two platform-specific local images. This job's daemon is
                        # disposable: remove only the exact tested ref before the
                        # next pull (never prune global caches or other images).
                        docker(["image", "rm", "--force", ref], env)
                print("PASS: fully anonymous digest and tag pulls for linux/amd64 and linux/arm64", flush=True)
                return
            except (urllib.error.URLError, RuntimeError, subprocess.CalledProcessError) as error:
                if time.monotonic() >= deadline:
                    raise RuntimeError("anonymous pull verification failed; package must be Public and all tags available") from error
                print(f"Anonymous verification not ready ({type(error).__name__}); retrying in 20s", flush=True)
                time.sleep(20)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["publish", "verify"])
    parser.add_argument("--image", required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--branch", default="")
    parser.add_argument("--release", default="")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--expected-sha", default="")
    parser.add_argument("--digest", default="")
    parser.add_argument("--retry-seconds", type=int, default=300)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.sha):
        parser.error("a full commit SHA is required")
    if args.mode == "verify" and not re.fullmatch(r"sha256:[0-9a-f]{64}", args.digest):
        parser.error("an exact image digest is required")
    args.tag = "sha-" + args.sha
    {"publish": publish, "verify": verify}[args.mode](args)


if __name__ == "__main__":
    main()
