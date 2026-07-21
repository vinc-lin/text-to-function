from __future__ import annotations
from ..types import FunctionCard, ParamSpec


def _param_schema(p: ParamSpec) -> dict:
    if p.type in ("number", "integer"):
        s: dict = {"type": "number" if p.type == "number" else "integer"}
        if p.minimum is not None:
            s["minimum"] = p.minimum
        if p.maximum is not None:
            s["maximum"] = p.maximum
        return s
    if p.type == "boolean":
        return {"type": "boolean"}
    if p.type == "enum":
        return {"enum": list(p.enum or [])}
    return {"type": "string"}


def _card_schema(card: FunctionCard) -> dict:
    props = {p.name: _param_schema(p) for p in card.params}
    required = [p.name for p in card.params if p.required]
    return {
        "type": "object",
        "properties": {
            "name": {"const": card.name},
            "parameters": {"type": "object", "properties": props,
                           "required": required, "additionalProperties": False},
        },
        "required": ["name", "parameters"],
        "additionalProperties": False,
    }


def candidates_to_json_schema(cards: list[FunctionCard]) -> dict:
    options = [_card_schema(c) for c in cards]
    return options[0] if len(options) == 1 else {"oneOf": options}
