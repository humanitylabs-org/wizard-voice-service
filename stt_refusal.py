"""Shared, dependency-free ASR refusal policy (v2).

Carried forward from the standalone Wizard OS voice service so independent
installations retain the refusal policy without a cross-repository import.
Match the diagnosed whole-chunk opener families, regardless of suffix; never
retain text in errors. This is a narrow contamination signature, not a general
speech/content filter.

v2 widens v1 after live verification: the same audio chat model also emits
whole-chunk boilerplate that never says "unable to provide". Production
families (aggregate-only survey of live chunk text):
  I'm sorry, but I'm unable to provide ...        (v1, unchanged)
  I'm sorry, but I'm not sure what you're asking. Could you please provide
    more context or clarify ... I'm here to help!
  I'm sorry, but I don't have the (necessary context|ability|tools|context)
    ... feel free to ask
  I'm sorry, but I can't provide a response ...
  I'm sorry for any inconvenience ... Let's try/clarify ...
  I'm not sure what you're asking. Could you please provide more context or
    clarify your question? I'm here to help!   (no apology opener)
A genuine in-conversation apology is NOT matched: it needs a capability verb
(assistant-refusal family) or an offered-help closer, not "I'm sorry" alone.

v3 closes gaps found during the live historical repair:
- "Sorry, ..." without the "I'm" ("Sorry, I'm not sure what you're asking...").
- "I don't have any information about that" and "I'll do my best to help".
- whole-chunk assistant ANSWERS, not just refusals: "Sure, I'd be happy to
  help with that! Here are some steps to get started: ...". Hallmark-
  combination rule: >=2 distinct hallmark phrases anywhere in the chunk, or
  one strong hallmark (assistant-meta: transcribe-the-audio, browse the
  internet, knowledge cutoff) alone.
"""
from __future__ import annotations

import re

REFUSAL_CODE = "assistant_refusal"

_SORRY = r"(?:i(?:['\u2019]m|\s+am)\s+)?sorry"
_WS = r"[\s,]*"
# Assistant-capability verb families that follow an apology.
_INABILITY = (
    r"i(?:['\u2019]m|\s+am)\s+unable\s+to\s+provide\b"
    r"|i(?:['\u2019]m|\s+am)\s+not\s+sure\s+what\s+(?:you|u|your)[-'’\u2019\s]*"
    r"(?:re|r)?\s*(?:asking|referring\s+to|question)"
    r"|i\s+don?['\u2019]?t\s+have\s+(?:the\s+)?(?:necessary\s+)?(?:context|ability|tools|capability|access)\b"
    r"|i\s+can?['\u2019]?t\s+provide\b"
    r"|i\s+can?['\u2019]?t\s+assist\b"
    r"|i\s+can?['\u2019]?t\s+help\s+with\b"
    r"|i\s+can(?:not|['\u2019]?t)\s+find\s+(?:(?:any|the)\s+)?audio\s+to\s+transcribe\b"
    r"|i\s+do(?:n?['\u2019]t|\s+not)\s+have\s+the\s+capability\b"
)
# Offered-help closers that mark whole-chunk assistant boilerplate.
_CLOSER = (
    r"(?:i(?:['\u2019]m|\s+am)\s+here\s+to\s+help"
    r"|feel\s+free\s+to\s+ask"
    r"|let\s+me\s+know\s+if\s+you\s+(?:have|need)"
    r"|is\s+there\s+(?:anything|something\s+else)\s+(?:else\s+)?i\s+can\s+(?:help|assist))"
)
_PATTERN = re.compile(
    rf"(?P<apology>^\s*{_SORRY}{_WS}(?:but{_WS})?)?(?P<core>{_INABILITY})\b",
    re.IGNORECASE,
)
_CORE_PATTERN = re.compile(rf"\b(?:{_INABILITY})\b", re.IGNORECASE)
_CLOSER_PATTERN = re.compile(_CLOSER, re.IGNORECASE)
_APOLOGY_ONLY = re.compile(rf"^\s*{_SORRY}{_WS}for\s+any\s+inconvenience\b", re.IGNORECASE)

# v3: hallmark phrases that mark assistant boilerplate anywhere in a chunk.
_HALLMARKS = (
    r"provide\s+more\s+context",
    r"clarify\s+your\s+(?:question|request|query)",
    r"(?:i(?:['\u2019]m|\s+am)\s+)?here\s+to\s+help",
    r"feel\s+free\s+to\s+ask",
    r"(?:i?['\u2019]?ll\s+)?do\s+my\s+best\s+to\s+help",
    r"i\s+don?['\u2019]?t\s+have\s+any\s+information\b",
    r"(?:some|any)\s+steps\s+to\s+get\s+started",
    r"happy\s+to\s+help",
    r"i(?:['\u2019]m|\s+am)?\s+not\s+sure\s+what\s+you.?['\u2019]?re\s+(?:asking|referring\s+to|looking\s+for)",
    r"could\s+you\s+please\s+(?:provide|clarify|rephrase)",
    r"for\s+any\s+inconvenience",
)
_HALLMARK_RX = tuple(re.compile(p, re.IGNORECASE) for p in _HALLMARKS)
# Strong hallmarks: assistant-meta statements alone prove boilerplate.
_STRONG_RX = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"i(?:['\u2019]m|\s+am)\s+unable\s+to\s+provide\b",
    r"transcrib(?:e|ing)\s+(?:the\s+)?audio",
    r"browse\s+the\s+internet|access\s+(?:specific\s+)?(?:websites|articles|the\s+link)",
    r"knowledge\s+cutoff|trained\s+on\s+the\s+data|up\s+until\s+my\s+training",
    r"i\s+don?['\u2019]?t\s+have\s+the\s+(?:ability|capability|tools|necessary\s+context)",
    r"i\s+can?['\u2019]?t\s+(?:provide|assist\s+with|help\s+with)\b",
    r"i\s+do(?:n?['\u2019]t|\s+not)\s+have\s+the\s+capability\b",
))


def is_assistant_refusal(text: object) -> bool:
    """Reject whole responses/chunks that are assistant boilerplate.

    Rule: a capability-verb opener ("unable to provide", "not sure what
    you're asking", "don't have the ability", "can't provide") matches at
    chunk start. An apology alone never matches unless followed by the
    inconvenience boilerplate family, or unless an offered-help closer
    appears later in the same chunk. Genuine speech ("I'm sorry we are
    late", "I'm sorry, Rob.") stays transcript.
    """
    if not isinstance(text, str):
        return False
    if _APOLOGY_ONLY.match(text):
        return True
    if _PATTERN.match(text):
        return True
    if re.match(rf"^\s*{_SORRY}\b", text, re.IGNORECASE):
        if _CORE_PATTERN.search(text) is not None or _CLOSER_PATTERN.search(text) is not None:
            return True
    # v3: quoted/embedded mentions keep their leading words; a story or
    # quotation prefix means genuine speech regardless of later phrases.
    if re.match(r"^\s*(?:[\[({\"'\u201c\u2018]|they\s+said|he\s+said|she\s+said|and\s+i|but\s+i|so\s+i)\b",
               text, re.IGNORECASE):
        return False
    if any(r.search(text) for r in _STRONG_RX):
        return True
    # v3: two or more distinct hallmark phrases anywhere prove boilerplate.
    return sum(1 for r in _HALLMARK_RX if r.search(text)) >= 2
