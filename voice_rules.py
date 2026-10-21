"""Versioned, local correction and style rules. No collection or network I/O."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import yaml

MODES = {"dictation", "meeting", "verbatim"}
COMMON_RULES = [{
    "id": "common.join-laughter", "kind": "style", "scope": "common",
    "provenance": "Wizard Voice Service", "enabled": False, "priority": 100,
    "modes": ["dictation"], "operation": "join_laughter",
}]
_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,95}$")
# Protect quoted text, code, URLs and path-like tokens. Apostrophes inside words
# are contractions, not quote delimiters. Never join laughter across a newline.
_PROTECTED = re.compile(r'```[\s\S]*?```|`[^`\n]*`|"[^"\n]*"|“[^”\n]*”|(?<!\w)\x27[^\x27\n]*\x27|‘[^’\n]*’|https?://\S+|[/\\]\S+')
_LAUGHTER = re.compile(r"(?<![\w/\\.\-])ha(?:[ \t]+ha)+(?![\w/\\\-]|\.[\w])", re.I)
_TRAILING = re.compile(r"(?<![\w/\\.\-])(?:ha){2,}(?:\.)?(?=[ \t]*\r?$)", re.I)
# Internal boundaries require one period, horizontal space, then an unquoted
# word. Keep that word untouched; a newline replaces the laughter's stop.
_INTERNAL = re.compile(r"(?<![\w/\\.\-])((?:ha){2,})\.[ \t]+(?=[^\W\d_])", re.I)
_LITERAL = re.compile(r"\b(?:literal|spell(?:ing|ed)?|quote[ds]?|regex|pattern|filename|instead of)\b", re.I)


class RuleError(ValueError):
    pass


def validate_rule(rule: object) -> dict:
    if not isinstance(rule, dict):
        raise RuleError("rule must be a mapping")
    required = {"id", "kind", "scope", "provenance", "enabled", "priority", "modes", "operation"}
    if required - rule.keys() or rule.keys() - (required | {"pattern", "replacement", "case_sensitive"}):
        raise RuleError("rule fields are missing or unknown")
    if not isinstance(rule["id"], str) or not _ID.fullmatch(rule["id"]):
        raise RuleError("invalid rule id")
    if rule["kind"] not in ("correction", "style") or rule["scope"] not in ("common", "user"):
        raise RuleError("invalid kind or scope")
    if not isinstance(rule["provenance"], str) or not 1 <= len(rule["provenance"]) <= 256:
        raise RuleError("provenance must be a short nonempty string")
    if type(rule["enabled"]) is not bool or type(rule["priority"]) is not int or not 0 <= rule["priority"] <= 10000:
        raise RuleError("invalid enabled or priority")
    if not isinstance(rule["modes"], list) or not rule["modes"] or any(not isinstance(m, str) or m not in MODES for m in rule["modes"]):
        raise RuleError("invalid modes")
    op = rule["operation"]
    if op not in ("regex", "join_laughter", "attach_laughter"):
        raise RuleError("unknown operation")
    if op != "regex":
        if rule["kind"] != "style" or rule["modes"] != ["dictation"]:
            raise RuleError("laughter rules are dictation-only styles")
        if op == "attach_laughter" and rule["scope"] != "user":
            raise RuleError("attachment is a user preference, not a common rule")
    else:
        if not isinstance(rule.get("pattern"), str) or not isinstance(rule.get("replacement"), str):
            raise RuleError("regex needs pattern and replacement strings")
        if len(rule["pattern"]) > 2000 or len(rule["replacement"]) > 2000:
            raise RuleError("regex rule too large")
        if "case_sensitive" in rule and type(rule["case_sensitive"]) is not bool:
            raise RuleError("case_sensitive must be boolean")
        try:
            re.compile(rule["pattern"]).sub(rule["replacement"], "")
        except (re.error, IndexError) as exc:
            raise RuleError("invalid regex or replacement") from exc
    return dict(rule)


def read_rules(path: Path | None = None) -> list[dict]:
    rules = {r["id"]: dict(r) for r in COMMON_RULES}
    if path is None:
        return list(rules.values())
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RuleError("cannot read rule file") from exc
    if not isinstance(raw, dict) or type(raw.get("schema_version")) is not int or raw.get("schema_version") != 1 or set(raw) != {"schema_version", "rules"} or not isinstance(raw["rules"], list):
        raise RuleError("expected schema_version: 1 and rules list")
    if len(raw["rules"]) > 1000:
        raise RuleError("too many rules")
    seen = set()
    for entry in raw["rules"]:
        rule = validate_rule(entry)
        if rule["id"] in seen:
            raise RuleError("duplicate rule id")
        seen.add(rule["id"])
        rules[rule["id"]] = rule
    return list(rules.values())


def legacy_id(rule: dict, index: int) -> str:
    digest = hashlib.sha256(json.dumps(rule, sort_keys=True).encode()).hexdigest()[:16]
    return f"legacy.{index}.{digest}"


def _laughter(text: str, *, attach: bool) -> str:
    # Fenced/code-heavy transcripts are literal material, not casual laughter.
    if "```" in text:
        return text
    def plain(fragment: str, *, trailing: bool = True) -> str:
        if _LITERAL.search(fragment):
            return fragment
        joined = _LAUGHTER.sub(lambda m: re.sub(r"[ \t]+", "", m[0]).lower(), fragment)
        if not attach:
            return joined
        pieces = []
        offset = 0
        for match in _INTERNAL.finditer(joined):
            prefix = re.sub(r"(?<!\.)\.[ \t]+$", " ", joined[offset:match.start()])
            pieces.extend((prefix, match[1].lower(), "\n"))
            offset = match.end()
        joined = "".join(pieces) + joined[offset:]
        if not trailing:
            return joined
        match = _TRAILING.search(joined)
        if not match:
            return joined
        prefix = joined[:match.start()]
        # A single sentence stop may attach; ellipses and line breaks do not.
        prefix = re.sub(r"(?<!\.)\.[ \t]+$", " ", prefix)
        return prefix + match[0].rstrip(".").lower() + joined[match.end():]

    # Protect the whole line when it describes literals (including quoted ones).
    lines = []
    for line in text.splitlines(keepends=True):
        if _LITERAL.search(line):
            lines.append(line)
            continue
        pieces = []
        offset = 0
        for match in _PROTECTED.finditer(line):
            # A plain fragment before a quote is not a sentence-ending suffix.
            fragment = line[offset:match.start()]
            pieces.append(plain(fragment, trailing=False))
            pieces.append(match[0])
            offset = match.end()
        pieces.append(plain(line[offset:]))
        lines.append("".join(pieces))
    return "".join(lines)


def apply_rules(text: str, rules: list[dict], *, kind: str, mode: str) -> tuple[str, list[str]]:
    if mode not in MODES:
        raise RuleError("unknown transcription mode")
    if mode == "verbatim" or (kind == "style" and mode != "dictation"):
        return text, []
    applied = []
    # Bucket first; common before user; explicit priority then stable ID.
    ordered = sorted(rules, key=lambda r: (r["scope"] == "user", r["priority"], r["id"]))
    for rule in ordered:
        if not rule["enabled"] or rule["kind"] != kind or mode not in rule["modes"]:
            continue
        if rule["operation"] == "regex":
            updated = re.sub(rule["pattern"], rule["replacement"], text,
                             flags=0 if rule.get("case_sensitive") else re.I)
        else:
            updated = _laughter(text, attach=rule["operation"] == "attach_laughter")
        if updated != text:
            applied.append(rule["id"])
        text = updated
    return text, applied
