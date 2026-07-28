# t2f/params/extractors.py
from __future__ import annotations
import re
from ..types import ParamSpec, LexFeatures
from ..phrase import enum_surface_forms


def _coerce(value: float, spec: ParamSpec):
    return int(round(value)) if spec.type == "integer" else value


def extract_temperature(clause, f: LexFeatures, spec: ParamSpec):
    return _coerce(f.temperatures[0], spec) if f.temperatures else None


def extract_percentage(clause, f: LexFeatures, spec: ParamSpec):
    return _coerce(f.percentages[0], spec) if f.percentages else None


def extract_level(clause, f: LexFeatures, spec: ParamSpec):
    return _coerce(f.levels[0], spec) if f.levels else None


def extract_number(clause, f: LexFeatures, spec: ParamSpec):
    return _coerce(f.numbers[0], spec) if f.numbers else None


def extract_position(clause, f: LexFeatures, spec: ParamSpec):
    if not spec.enum:
        return None
    for p in f.positions:
        if p in spec.enum:
            return p
    return None


def extract_enum(clause, f: LexFeatures, spec: ParamSpec):
    """Match an enum value by the words a driver uses for it.

    Longest surface form first across ALL of the spec's values, so 避开高速 wins over any
    shorter form it contains. Scoped to one card's enum, so `自动` being a value of three
    different enums is not a collision — routing has already chosen the card.
    """
    if not spec.enum:
        return None
    candidates = [(form, value) for value in spec.enum for form in enum_surface_forms(value)]
    for form, value in sorted(candidates, key=lambda c: -len(c[0])):
        if form in clause:
            return value
    return None


# Politeness that precedes the real request, and the connectors that link a trigger to its
# object (导航<到>北京南站, 打电话<给>老婆).
_LEAD = re.compile(r"^(帮我|请|麻烦|我要|我想|那个|给我)+")
_CONNECTOR = re.compile(r"^[到去给为的成了一下个条把]+")

# A remainder containing any of these is a question or a fragment, not the object of the
# request: 附近<有什么>加油站 leaves 有什么加油站, which is not a place name. Extracting it
# would send the driver somewhere that does not exist. Measured on the catalog's own
# examples, this guard turns 8 wrong answers into 8 questions and costs zero correct ones.
_NOT_AN_OBJECT = ("有什么", "在哪", "哪里", "什么", "怎么", "吗", "呢",
                  "一下", "换成", "改成", "可以", "多少")


def extract_string(clause, f: LexFeatures, spec: ParamSpec, card=None):
    """The free-text object of the request — a destination, a contact, a category.

    Deliberately high-precision and low-recall. A wrong destination navigates somewhere
    else and a wrong contact calls the wrong person, whereas declining costs one question
    the reply layer already words well. So this fires only on the shape it can read —
    trigger followed by its object — and returns None on anything it cannot.
    """
    if card is None or not spec.required:
        # One clause carries ONE free-text object, so it may fill one slot. Without this,
        # 发短信给妈妈 filled send_message's `contact` AND its optional `content` with 妈妈.
        # An optional free-text slot cannot be separated from the required one by this rule,
        # so it is left to the LLM path rather than guessed at.
        return None
    rest = _LEAD.sub("", clause)
    best = None
    for alias in sorted(card.aliases, key=len, reverse=True):
        index = rest.find(alias)
        if index == -1:
            continue
        candidate = _CONNECTOR.sub("", rest[index + len(alias):]).strip("，。 ")
        if candidate and (best is None or len(candidate) < len(best)):
            best = candidate
    if not best or len(best) < 2 or any(m in best for m in _NOT_AN_OBJECT):
        return None
    return best


def extract_boolean(clause, f: LexFeatures, spec: ParamSpec):
    return f.on_off
