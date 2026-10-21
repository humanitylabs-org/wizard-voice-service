from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_SPOKEN_TIME_RE = re.compile(
    r"\b(0?[1-9]|1[0-2])(?::([0-5][0-9]))?\s*([ap])\s*\.?\s*m(?:\.(?=\W|$)|\b)",
    re.IGNORECASE,
)


def _normalize_spoken_times(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        hour = str(int(match.group(1)))
        minutes = match.group(2)
        suffix = match.group(3).lower() + "m"
        if not minutes or minutes == "00":
            return f"{hour}{suffix}"
        return f"{hour}:{minutes}{suffix}"

    return _SPOKEN_TIME_RE.sub(replace, text)


def _legacy_postprocess(text: str, postprocess: object = None) -> str:
    """Apply Wizard Voice-owned deterministic cleanup to a transcript.

    ``postprocess`` accepts the existing ``stt.postprocess`` mapping. Invalid
    optional rules are skipped so cleanup can never discard a valid provider
    transcript. This module deliberately has no Hermes source dependency.
    """
    if not isinstance(text, str) or not text or not isinstance(postprocess, dict):
        return text

    config: dict[str, Any] = postprocess
    result = text
    if config.get("normalize_spoken_times") is True:
        result = _normalize_spoken_times(result)

    replacements = config.get("replacements") or []
    if not isinstance(replacements, list):
        logger.warning("Ignoring stt.postprocess.replacements: expected a list")
        return result

    for index, rule in enumerate(replacements):
        if not isinstance(rule, dict):
            logger.warning(
                "Ignoring stt.postprocess.replacements[%d]: expected a mapping",
                index,
            )
            continue
        pattern = rule.get("pattern")
        replacement = rule.get("replacement")
        if not isinstance(pattern, str) or not isinstance(replacement, str):
            logger.warning(
                "Ignoring stt.postprocess.replacements[%d]: pattern and replacement must be strings",
                index,
            )
            continue
        flags = 0 if rule.get("case_sensitive") is True else re.IGNORECASE
        try:
            result = re.sub(pattern, replacement, result, flags=flags)
        except re.error as exc:
            logger.warning(
                "Ignoring invalid stt.postprocess.replacements[%d] regex: %s",
                index,
                exc,
            )
    return result


def preview_transcript(text: str, postprocess: object = None, *, mode: str = "dictation") -> dict:
    from voice_rules import MODES, RuleError, apply_rules
    if mode not in MODES:
        raise RuleError("unknown transcription mode")
    config = dict(postprocess) if isinstance(postprocess, dict) else {}
    if mode == "verbatim":
        return {"corrected_text": text, "styled_text": text, "applied_rule_ids": []}
    # Legacy replacements retain their order and recognition-correction behavior.
    legacy = dict(config, normalize_spoken_times=False)
    corrected = _legacy_postprocess(text, legacy)
    rules = config.get("rules", [])
    corrected, corrections = apply_rules(corrected, rules, kind="correction", mode=mode)
    styled = corrected
    if mode == "dictation" and config.get("normalize_spoken_times") is True:
        # Preserve legacy time-before-replacements behavior exactly. Preview's
        # corrected_text omits that legacy style, but the emitted text must not
        # break an existing replacement that expects normalized times.
        styled = _legacy_postprocess(text, config)
        styled, styled_corrections = apply_rules(styled, rules, kind="correction", mode=mode)
        corrections = list(dict.fromkeys(corrections + styled_corrections))
    styled, styles = apply_rules(styled, rules, kind="style", mode=mode)
    return {"corrected_text": corrected, "styled_text": styled,
            "applied_rule_ids": corrections + styles}


def apply_transcription_postprocess(text: str, postprocess: object = None) -> str:
    mode = postprocess.get("mode", "dictation") if isinstance(postprocess, dict) else "dictation"
    return preview_transcript(text, postprocess, mode=mode)["styled_text"]
