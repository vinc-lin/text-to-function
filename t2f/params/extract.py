# t2f/params/extract.py
from __future__ import annotations
from ..types import FunctionCard, LexFeatures
from . import extractors as ex

_POSITION_ENUM = {"driver", "passenger", "rear", "all", "left", "right"}


def _dispatch(clause, features, spec, card=None):
    if spec.unit == "celsius":
        return ex.extract_temperature(clause, features, spec)
    if spec.unit == "percent":
        return ex.extract_percentage(clause, features, spec)
    if spec.unit == "level":
        return ex.extract_level(clause, features, spec)
    if spec.type == "enum" and spec.enum and set(spec.enum) & _POSITION_ENUM:
        return ex.extract_position(clause, features, spec)
    if spec.type == "enum" and spec.enum:
        return ex.extract_enum(clause, features, spec)
    if spec.type == "boolean":
        return ex.extract_boolean(clause, features, spec)
    if spec.type == "string":
        return ex.extract_string(clause, features, spec, card)
    return ex.extract_number(clause, features, spec)


class ParameterExtractor:
    def extract(self, clause: str, features: LexFeatures, card: FunctionCard):
        params: dict = {}
        missing: list[str] = []
        for spec in card.params:
            val = _dispatch(clause, features, spec, card)
            if val is not None:
                params[spec.name] = val
            elif spec.required:
                missing.append(spec.name)
        return params, missing
