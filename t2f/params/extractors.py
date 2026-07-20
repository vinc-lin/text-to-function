# t2f/params/extractors.py
from __future__ import annotations
from ..types import ParamSpec, LexFeatures


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


def extract_boolean(clause, f: LexFeatures, spec: ParamSpec):
    return f.on_off
