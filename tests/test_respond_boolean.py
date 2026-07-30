"""A confirmation that does not say which way it went is not a confirmation."""
import pytest

from t2f.cards import load_catalog
from t2f.respond import render_response
from t2f.types import ToolCall

# Templates where 打开/关闭 does not read naturally. Listed explicitly so the set cannot grow
# by accident: a new ambiguous card fails test_every_other_boolean_card_states_direction.
KNOWN_AMBIGUOUS = {"spray_washer"}


@pytest.fixture(scope="module")
def cards():
    return {c.name: c for c in load_catalog("data/catalog")}


def _bool_param(card):
    return next((p for p in card.params if p.type == "boolean"), None)


def test_the_child_lock_says_which_way_it_went(cards):
    card = cards["set_window_child_lock"]
    on = render_response(card, ToolCall(card.name, {"enabled": True}))
    off = render_response(card, ToolCall(card.name, {"enabled": False}))
    assert on == "已为您打开车窗儿童锁。"
    assert off == "已为您关闭车窗儿童锁。"


def test_a_fold_card_folds_rather_than_opens(cards):
    """折叠/展开, not 打开/关闭 — 已为您打开后视镜折叠 is not Chinese anyone speaks. The verb
    comes from the function name, the same way sim/mapping.py derives a signal attribute."""
    card = cards["fold_mirror"]
    on = render_response(card, ToolCall(card.name, {"enabled": True}))
    assert "折叠" in on and "打开" not in on


def test_an_is_off_parameter_inverts(cards):
    """turn_off_screen{is_off: true} turned the screen OFF. Reading the raw boolean would
    announce the opposite of what happened."""
    card = cards["turn_off_screen"]
    assert "关闭" in render_response(card, ToolCall(card.name, {"is_off": True}))
    assert "打开" in render_response(card, ToolCall(card.name, {"is_off": False}))


def test_every_other_boolean_card_states_direction(cards):
    for card in cards.values():
        p = _bool_param(card)
        if p is None or card.name in KNOWN_AMBIGUOUS:
            continue
        on = render_response(card, ToolCall(card.name, {p.name: True}))
        off = render_response(card, ToolCall(card.name, {p.name: False}))
        assert on != off, f"{card.name} says the same thing both ways: {on}"


def test_a_non_boolean_card_is_unchanged(cards):
    card = cards["set_temperature"]
    out = render_response(card, ToolCall(card.name, {"temperature": 25.0, "position": "driver"}))
    assert out == "已将主驾温度设置为25°C。"
