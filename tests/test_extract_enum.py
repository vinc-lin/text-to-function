"""Non-position enum extraction.

Before this existed, `_dispatch` sent every enum whose values were not positions to
`extract_number`, so 13 required parameters could never be filled and their functions
recognised correctly and then always clarified.

The vocabulary is mined from the catalog's own `utterances`, so the catalog is both the
source of the words and the test corpus for them.
"""
import pytest

from t2f.cards import load_catalog
from t2f.lexical import extract_features
from t2f.params.extract import ParameterExtractor
from t2f.phrase import enum_surface_forms

CARDS = load_catalog("data/catalog")
BY = {c.name: c for c in CARDS}
POSITIONS = {"driver", "passenger", "rear", "all", "left", "right"}
EXTRACT = ParameterExtractor()

# The card's own examples that name no value at all, so extraction SHOULD return nothing and
# the driver should be asked. Listed explicitly so a real regression cannot hide among them.
NAMES_NO_VALUE = {
    ("set_air_direction", "切换出风口方向"),
    ("set_equalizer", "音效模式切一下"),
    # the catalog's own example asks for a value its enum does not contain — see the test below
    ("set_route_preference", "路线偏好设成躲避拥堵"),
}


def _enum_params():
    for card in CARDS:
        for param in card.params:
            if param.required and param.type == "enum" and not (set(param.enum or []) & POSITIONS):
                yield card, param


def test_every_catalog_utterance_that_names_a_value_fills_it():
    """The corpus is the catalog's own examples — if the shipped phrasings do not work, the
    vocabulary is wrong no matter how good it looks in isolation."""
    misses = []
    for card, param in _enum_params():
        for utterance in card.utterances:
            got, _ = EXTRACT.extract(utterance, extract_features(utterance), card)
            if not got.get(param.name) and (card.name, utterance) not in NAMES_NO_VALUE:
                misses.append((card.name, utterance))
    assert misses == [], f"utterances naming a value that was not extracted: {misses}"


def test_the_known_misses_really_do_name_no_value():
    """Guards the allow-list above from becoming a dumping ground: each entry must genuinely
    contain no surface form of any of its enum's values."""
    for name, utterance in NAMES_NO_VALUE:
        card = BY[name]
        param = next(p for p in card.params if p.type == "enum" and p.required)
        present = [v for v in param.enum if any(f in utterance for f in enum_surface_forms(v))]
        assert present == [], f"{utterance} does name {present}; it should not be allow-listed"


@pytest.mark.parametrize("utterance,function,param,expected", [
    ("切换到制冷模式", "set_ac_mode", "mode", "cool"),
    ("空调调到自动模式", "set_ac_mode", "mode", "auto"),
    ("空调吹脸", "set_air_direction", "direction", "face"),
    ("打开内循环", "set_air_recirculation", "mode", "internal"),
    ("切换到夜间模式", "set_display_mode", "mode", "night"),
    ("大灯调成远光", "set_headlight_mode", "mode", "high_beam"),
    ("氛围灯调成蓝色", "set_ambient_light_color", "color", "blue"),
    ("音源切到U盘", "set_audio_source", "source", "usb"),
    ("音效调成摇滚", "set_equalizer", "mode", "rock"),
    ("座椅往前移", "move_seat", "direction", "front"),
    ("靠背往后放倒", "adjust_backrest", "direction", "recline"),
    ("靠背立直", "adjust_backrest", "direction", "upright"),
    ("座椅升高一点", "adjust_seat_height", "direction", "up"),
    ("头枕降低一点", "adjust_headrest", "direction", "down"),
])
def test_specific_values(utterance, function, param, expected):
    card = BY[function]
    got, _ = EXTRACT.extract(utterance, extract_features(utterance), card)
    assert got.get(param) == expected


def test_longest_surface_form_wins():
    """避开高速 must not be beaten by a shorter form it contains."""
    card = BY["set_route_preference"]
    for utterance, expected in [("避开高速", "avoid_highway"), ("不走收费的路", "avoid_toll"),
                                ("走最快的路线", "fastest"), ("选最短路线", "shortest")]:
        got, _ = EXTRACT.extract(utterance, extract_features(utterance), card)
        assert got.get("mode") == expected, utterance


def test_a_number_no_longer_becomes_a_bogus_enum_value():
    """The safety half. 空调模式调到3 used to fall through to extract_number and produce
    mode=3, an invalid value the validator then had to catch. The deterministic path can no
    longer fabricate one at all — it asks instead."""
    card = BY["set_ac_mode"]
    got, missing = EXTRACT.extract("空调模式调到3", extract_features("空调模式调到3"), card)
    assert "mode" not in got and missing == ["mode"]


def test_every_enum_value_in_the_catalog_has_at_least_one_surface_form():
    """A value with no way to say it is a value the driver can never select."""
    silent = [(c.name, p.name, v) for c, p in _enum_params()
              for v in (p.enum or []) if not enum_surface_forms(v)]
    assert silent == [], f"enum values with no surface form: {silent}"
