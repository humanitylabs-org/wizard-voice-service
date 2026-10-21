from copy import deepcopy

import pytest

from postprocess import preview_transcript
from voice_rules import COMMON_RULES, RuleError, read_rules, validate_rule


def preference():
    return {"id": "user.laughter", "kind": "style", "scope": "user",
            "provenance": "local user", "enabled": True, "priority": 100,
            "modes": ["dictation"], "operation": "attach_laughter"}


@pytest.mark.parametrize("text,want", [
    ("ha ha", "haha"), ("ha ha ha", "hahaha"),
    ("That's funny. Ha ha.", "That's funny haha"),
    ("That's funny! Ha ha!", "That's funny! haha!"),
    ("That's funny? Ha ha.", "That's funny? haha"),
    ("Ha ha ha ha.", "hahahaha"),
    ("already haha", "already haha"),
    ("alpha ha hat", "alpha ha hat"),
    ("ha haiku", "ha haiku"),
    ("ha\nha", "ha\nha"),
    ("That's funny.\nHa ha.", "That's funny.\nhaha"),
    ('She said "Ha ha."', 'She said "Ha ha."'),
    ("She said 'ha ha'.", "She said 'ha ha'."),
    ("She said “Ha ha.”", "She said “Ha ha.”"),
    ("Keep `ha ha` here.", "Keep `ha ha` here."),
    ("```\nha ha\n```", "```\nha ha\n```"),
    ("Spell ha ha instead of haha.", "Spell ha ha instead of haha."),
    ("ha ha.txt", "ha ha.txt"),
    ("/ha/ha ha", "/ha/ha ha"),
    ("ha ha-ha", "ha ha-ha"),
    ("It's fine. Ha ha.", "It's fine haha"),
    ("Wait... Ha ha.", "Wait... haha"),
    ("ha ha, yes", "haha, yes"),
])
def test_laughter(text, want):
    config = {"rules": [preference()]}
    got = preview_transcript(text, config)
    assert got["corrected_text"] == text
    assert got["styled_text"] == want
    assert preview_transcript(want, config)["styled_text"] == want


@pytest.mark.parametrize("text,want", [
    ("Funny. Ha ha. Next sentence.", "Funny haha\nNext sentence."),
    ("Funny. Ha ha. next Sentence!", "Funny haha\nnext Sentence!"),
    ("Funny. HA HA HA. Élan stays.", "Funny hahaha\nÉlan stays."),
    ("Funny. Ha ha.  Next  sentence.", "Funny haha\nNext  sentence."),
    ("Funny. Ha ha.\tNext sentence.", "Funny haha\nNext sentence."),
    ("Funny. Ha ha.\nNext sentence.", "Funny haha\nNext sentence."),
    ("Funny. Ha ha.\r\nNext sentence.", "Funny haha\r\nNext sentence."),
    ("Funny. Ha ha. Next. Ha ha ha. Last.", "Funny haha\nNext hahaha\nLast."),
    ("Funny. Ha ha. Ha ha. Next.", "Funny haha\nhaha\nNext."),
    ("Funny... Ha ha. Next.", "Funny... haha\nNext."),
    ("Funny! Ha ha! Next.", "Funny! haha! Next."),
    ("Funny. Ha ha... Next.", "Funny. haha... Next."),
    ("Funny. Ha ha.Next.", "Funny. Ha ha.Next."),
    ("Funny. Ha ha Next.", "Funny. haha Next."),
    ('Funny. Ha ha. Next says "Ha ha."', 'Funny haha\nNext says "Ha ha."'),
    ('"Funny. Ha ha. Next sentence."', '"Funny. Ha ha. Next sentence."'),
    ("'Funny. Ha ha. Next sentence.'", "'Funny. Ha ha. Next sentence.'"),
    ("“Funny. Ha ha. Next sentence.”", "“Funny. Ha ha. Next sentence.”"),
    ("`Funny. Ha ha. Next sentence.`", "`Funny. Ha ha. Next sentence.`"),
    ("```\nFunny. Ha ha. Next sentence.\n```", "```\nFunny. Ha ha. Next sentence.\n```"),
    ("Literal Funny. Ha ha. Next sentence.", "Literal Funny. Ha ha. Next sentence."),
    ("Funny. /Ha/ha. Next sentence.", "Funny. /Ha/ha. Next sentence."),
    ("Funny. ha ha.txt. Next sentence.", "Funny. ha ha.txt. Next sentence."),
    ('Funny. Ha ha. "Next sentence."', 'Funny. haha. "Next sentence."'),
])
def test_sentence_internal_laughter(text, want):
    config = {"rules": [preference()]}
    got = preview_transcript(text, config)
    assert got["corrected_text"] == text
    assert got["styled_text"] == want
    assert preview_transcript(want, config)["styled_text"] == want


def test_optional_common_rule_not_attachment():
    common = deepcopy(COMMON_RULES)
    assert preview_transcript("Nice. Ha ha.", {"rules": common})["styled_text"] == "Nice. Ha ha."
    common[0]["enabled"] = True
    assert preview_transcript("Nice. Ha ha.", {"rules": common})["styled_text"] == "Nice. haha."
    text = "Funny. Ha ha. Next sentence."
    assert preview_transcript(text, {"rules": common})["styled_text"] == "Funny. haha. Next sentence."
    config = {"rules": common + [preference()]}
    want = "Funny haha\nNext sentence."
    assert preview_transcript(text, config)["styled_text"] == want
    assert preview_transcript(want, config)["styled_text"] == want


@pytest.mark.parametrize("count", [2, 3, 10, 100, 256])
def test_laughter_repetition_count_has_no_fixed_limit(count):
    text = " ".join(["Ha"] * count) + "."
    result = preview_transcript(text, {"rules": [preference()]})["styled_text"]
    assert result == "ha" * count
    assert preview_transcript(result, {"rules": [preference()]})["styled_text"] == result


@pytest.mark.parametrize("mode", ["meeting", "verbatim"])
@pytest.mark.parametrize("text", ["Nice. Ha ha.", "Funny. Ha ha. Next sentence."])
def test_no_personal_styling(mode, text):
    assert preview_transcript(text, {"rules": [preference()]}, mode=mode)["styled_text"] == text


@pytest.mark.parametrize("count", [2, 3, 10, 100, 256])
def test_sentence_internal_repetition(count):
    config = {"rules": [preference()]}
    text = "Funny. " + " ".join(["Ha"] * count) + ". Next sentence."
    want = "Funny " + "ha" * count + "\nNext sentence."
    assert preview_transcript(text, config)["styled_text"] == want
    assert preview_transcript(want, config)["styled_text"] == want


def test_buckets_precedence_and_verbatim():
    correction = dict(preference(), id="user.name", kind="correction", operation="regex",
                      pattern="Acem", replacement="Acme", modes=["dictation", "meeting"])
    config = {"replacements": [{"pattern": "Acm", "replacement": "Acem"}],
              "rules": [preference(), correction], "normalize_spoken_times": True}
    got = preview_transcript("Acm 9 p.m. Ha ha.", config)
    assert got["corrected_text"] == "Acme 9 p.m. Ha ha."
    assert got["styled_text"] == "Acme 9pm haha"
    assert got["applied_rule_ids"] == ["user.name", "user.laughter"]
    assert preview_transcript("Acm 9 p.m. Ha ha.", config, mode="verbatim")["styled_text"] == "Acm 9 p.m. Ha ha."
    assert preview_transcript("Acm 9 p.m. Ha ha.", config, mode="meeting")["styled_text"] == "Acme 9 p.m. Ha ha."


def test_rule_schema(tmp_path):
    import yaml
    rule = preference()
    for key, bad in [("enabled", "yes"), ("scope", "common"), ("modes", ["meeting"]), ("priority", True)]:
        with pytest.raises(RuleError):
            validate_rule(dict(rule, **{key: bad}))
    file = tmp_path / "rules.yaml"
    file.write_text(yaml.safe_dump({"schema_version": 1, "rules": [rule, rule]}))
    with pytest.raises(RuleError, match="duplicate"):
        read_rules(file)
    with pytest.raises(RuleError):
        validate_rule(dict(rule, operation="regex", pattern="(", replacement="a"))
    with pytest.raises(RuleError):
        validate_rule(dict(rule, operation="regex", pattern="a", replacement=r"\9"))


def test_precedence_is_deterministic():
    def rule(id, scope, priority, pattern, replacement):
        return dict(preference(), id=id, kind="correction", scope=scope,
                    priority=priority, operation="regex", pattern=pattern, replacement=replacement)
    rules = [rule("user", "user", 0, "C", "D"), rule("b", "common", 2, "B", "C"),
             rule("a", "common", 1, "A", "B")]
    assert preview_transcript("A", {"rules": rules})["styled_text"] == "D"
    rules[0]["enabled"] = False
    assert preview_transcript("A", {"rules": rules})["styled_text"] == "C"


def test_legacy_time_replacement_order_is_preserved():
    config = {"normalize_spoken_times": True,
              "replacements": [{"pattern": "9pm", "replacement": "evening"}]}
    result = preview_transcript("9 p.m.", config)
    assert result["corrected_text"] == "9 p.m."
    assert result["styled_text"] == "evening"
