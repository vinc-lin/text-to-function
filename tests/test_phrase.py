"""Driver-facing wording — the guards that catch internal vocabulary reaching the cabin.

Twice now an internal identifier has been spoken aloud: a signal address
(`window_position 最高只能到 60`) and an enum value (`氛围灯颜色只支持red/blue/green`).
Both were caught by looking at real output, not by a test, because the tests asserted the
NUMBER and ignored the SENTENCE. These sweep the whole catalog instead.
"""
import re

from t2f.cards import load_catalog
from t2f.phrase import (_ENUM_CN, enum_phrase, limit_phrase, missing_phrase,
                        param_subject, type_phrase)
from t2f.state import primary_numeric_param

CARDS = load_catalog("data/catalog")
_LATIN = re.compile(r"[a-z_]{3,}")          # an identifier, not an acronym like USB or AUX


def _speaks_latin(text: str) -> bool:
    return bool(_LATIN.search(text))


def test_every_enum_value_in_the_catalog_has_a_chinese_label():
    """A value with no label is silently dropped from the spoken options, so a gap here
    quietly shortens what the driver is offered rather than failing loudly."""
    used = {v for c in CARDS for p in c.params if p.type == "enum" and p.enum for v in p.enum}
    assert used - set(_ENUM_CN) == set()


def test_no_enum_phrase_speaks_an_english_identifier():
    leaks = []
    for card in CARDS:
        for param in card.params:
            if param.type == "enum" and param.enum:
                phrase = enum_phrase(card, param)
                if _speaks_latin(phrase):
                    leaks.append((card.name, param.name, phrase))
    assert leaks == [], f"English identifiers spoken to the driver: {leaks}"


def test_no_limit_phrase_speaks_an_internal_name():
    leaks = []
    for card in CARDS:
        param = primary_numeric_param(card)
        if param is None or param.maximum is None:
            continue
        phrase = limit_phrase(card, param, param.maximum, "最高")
        for address in (card.name, param.name):
            if address in phrase:
                leaks.append((card.name, address, phrase))
    assert leaks == [], f"internal addresses spoken to the driver: {leaks}"


def test_no_missing_or_type_phrase_speaks_an_internal_name():
    leaks = []
    for card in CARDS:
        for param in card.params:
            if not param.required:
                continue
            for phrase in (missing_phrase(card, param), type_phrase(card, param)):
                if phrase and (_speaks_latin(phrase) or param.name in phrase):
                    leaks.append((card.name, param.name, phrase))
    assert leaks == [], f"internal names spoken to the driver: {leaks}"


def test_every_required_parameter_can_be_asked_for_by_name():
    """gap 3's guard. 请补充更多信息。 does not tell a driver what is missing; every required
    parameter must produce a question that names the thing it wants."""
    silent = [(c.name, p.name) for c in CARDS for p in c.params
              if p.required and not missing_phrase(c, p)]
    assert silent == [], f"required parameters with no question: {silent}"


def test_a_parenthetical_description_is_cut_before_it_is_spoken():
    """`加热档位，0为关闭` is a note to a developer; only the head reads as a sentence."""
    card = next(c for c in CARDS if c.name == "set_seat_heating")
    param = card.param("level")
    assert param.description.startswith("加热档位，")          # the catalog really is like this
    assert param_subject(card, param) == "加热档位"
