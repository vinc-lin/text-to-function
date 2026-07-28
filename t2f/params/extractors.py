# t2f/params/extractors.py
from __future__ import annotations
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


def extract_boolean(clause, f: LexFeatures, spec: ParamSpec):
    return f.on_off
