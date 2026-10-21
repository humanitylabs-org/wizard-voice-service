"""Local, owner-operated rule administration. Never sends text to a provider."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import tempfile

import yaml

from config import ProviderConfigurationError, load_provider_config
from postprocess import preview_transcript
from voice_rules import MODES, RuleError, legacy_id, read_rules, validate_rule


def mutate(path: Path, action: str, *, rule: dict | None = None, rule_id: str = "") -> list[dict]:
    path = path.expanduser().absolute()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Stable sidecar lock covers read-modify-write; atomic replace alone loses edits.
    fd = os.open(str(path) + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(fd, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rules = read_rules(path if path.exists() else None)
        if action == "add":
            incoming = validate_rule(rule)
            if any(r["id"] == incoming["id"] for r in rules):
                raise RuleError("id already exists; disable it or choose a new id")
            rules.append(incoming)
        else:
            selected = next((r for r in rules if r["id"] == rule_id), None)
            if selected is None:
                raise RuleError("rule id not found")
            selected["enabled"] = action == "enable"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".rules-", delete=False) as handle:
                temporary = Path(handle.name)
                os.chmod(temporary, 0o600)
                yaml.safe_dump({"schema_version": 1, "rules": rules}, handle, sort_keys=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return rules


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, help="owner rule file (required for writes)")
    parser.add_argument("--config", type=Path, help="include legacy correction layers in list/preview")
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("list")
    add = commands.add_parser("add")
    add.add_argument("rule", type=Path, help="one YAML rule mapping")
    for action in ("enable", "disable"):
        commands.add_parser(action).add_argument("id")
    preview = commands.add_parser("preview")
    preview.add_argument("--mode", choices=sorted(MODES), default="dictation")
    preview.add_argument("--text", help="defaults to stdin; stdin avoids shell history")
    args = parser.parse_args(argv)
    try:
        if args.action in {"add", "enable", "disable"}:
            if args.file is None:
                raise RuleError("writes require --file")
            if args.config:
                raise RuleError("--config is read-only; use --file for writes")
            rule = yaml.safe_load(args.rule.read_text()) if args.action == "add" else None
            output = mutate(args.file, args.action, rule=rule, rule_id=getattr(args, "id", ""))
        else:
            config = dict(load_provider_config(args.config).postprocess) if args.config else {}
            if args.file is not None or "rules" not in config:
                config["rules"] = read_rules(args.file)
            if args.action == "list":
                output = {"rules": config["rules"], "legacy_replacements": [
                    {"id": legacy_id(r, i), "kind": "correction", "scope": "legacy",
                     "provenance": "legacy configuration (unmodified)", "enabled": True, **r}
                    for i, r in enumerate(config.get("replacements", [])) if isinstance(r, dict)
                ]}
            else:
                text = args.text if args.text is not None else sys.stdin.read()
                output = preview_transcript(text, config, mode=args.mode)
        print(json.dumps(output, ensure_ascii=False, sort_keys=True))
        return 0
    except (RuleError, ProviderConfigurationError, OSError, yaml.YAMLError) as exc:
        print(f"wizard-voice-rules: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
