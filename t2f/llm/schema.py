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


REJECT_NAME = "__reject__"

_REJECT_OPTION = {
    "type": "object",
    "properties": {"name": {"const": REJECT_NAME}},
    "required": ["name"],
    "additionalProperties": False,
}


def candidates_to_json_schema(cards: list[FunctionCard], allow_reject: bool = False) -> dict:
    """JSON schema constraining output to one candidate tool-call.

    When allow_reject is set, an extra `{"name": "__reject__"}` option is added so the model can
    decline when no candidate fits (out-of-domain / unsupported) instead of being forced to emit a
    wrong call — the key safety escape hatch for a constrained decoder.
    """
    options = [_card_schema(c) for c in cards]
    if allow_reject:
        options.append(_REJECT_OPTION)
    return options[0] if len(options) == 1 else {"oneOf": options}
