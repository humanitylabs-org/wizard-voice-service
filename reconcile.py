#!/usr/bin/env python3
"""Reconcile provider-specific realtime services after stt.provider changes."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from config import DEFAULT_CONFIG_PATH, load_voice_control_config

REALTIME_UNIT = "mizel-realtime.service"
STATE_PATH = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")) / "wizard-stt" / "reconcile.json"


def read_selection(path: Path | None = None) -> tuple[str, str]:
    lane = load_voice_control_config(path or DEFAULT_CONFIG_PATH).realtime_stt
    if not lane.enabled:
        return "disabled", ""
    return lane.provider, lane.model


def read_conversation_selection(path: Path | None = None) -> tuple[str, str]:
    lane = load_voice_control_config(path or DEFAULT_CONFIG_PATH).conversation
    if not lane.managed:
        return "unmanaged", ""
    if not lane.enabled:
        return "disabled", ""
    fingerprint = ":".join(lane.models.get(key, "") for key in ("transcription", "reasoning", "speech", "voice"))
    return lane.provider, fingerprint


def reconcile(
    provider: str,
    capability_model: str = "",
    *,
    dry_run: bool = False,
    state_path: Path | None = None,
    run: Callable[..., Any] = subprocess.run,
) -> dict[str, str | bool]:
    fingerprint = f"{provider}:{capability_model}"
    previous = ""
    if state_path and state_path.is_file():
        try:
            previous = str(json.loads(state_path.read_text(encoding="utf-8")).get("fingerprint") or "")
        except (OSError, ValueError, TypeError):
            previous = ""

    if state_path and previous == fingerprint:
        action = "noop"
        command: list[str] | None = None
    elif provider == "tinfoil":
        action = "restart"
        command = ["/usr/bin/systemctl", "--user", "restart", REALTIME_UNIT]
    else:
        action = "stop"
        command = ["/usr/bin/systemctl", "--user", "stop", REALTIME_UNIT]

    if command and not dry_run:
        run(command, check=True, timeout=60)
    if state_path and not dry_run:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"fingerprint": fingerprint}) + "\n", encoding="utf-8")
        os.chmod(tmp, 0o600)
        os.replace(tmp, state_path)

    return {
        "provider": provider,
        "capability_model": capability_model,
        "unit": REALTIME_UNIT,
        "action": action,
        "dry_run": dry_run,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--condition-tinfoil", action="store_true")
    parser.add_argument("--condition-lane", choices=("realtime_stt", "conversation"))
    args = parser.parse_args(argv)

    config_error = ""
    try:
        provider, capability_model = read_selection()
    except Exception as exc:
        provider, capability_model = "invalid", ""
        config_error = type(exc).__name__
    if args.provider:
        provider, capability_model = str(args.provider).strip().lower(), ""
    if args.condition_lane:
        try:
            if args.condition_lane == "conversation":
                provider, _ = read_conversation_selection()
        except Exception:
            return 1
        return 0 if provider == "tinfoil" else 1
    if args.condition_tinfoil:
        return 0 if provider == "tinfoil" else 1

    result = reconcile(
        provider,
        capability_model,
        dry_run=args.dry_run,
        state_path=None if args.dry_run else STATE_PATH,
    )
    if config_error:
        result["config_error"] = config_error
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
