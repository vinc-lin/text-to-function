# t2f/validate.py
from __future__ import annotations
from .types import FunctionCard, ToolCall, ValidationError
from .phrase import limit_phrase, enum_phrase, type_phrase


def validate_tool_call(name: str, params: dict, cards_by_name: dict[str, FunctionCard],
                       candidate_names: list[str]):
    errs: list[ValidationError] = []
    if name not in candidate_names:
        return None, [ValidationError("not_in_candidates", f"{name} not in candidate set")]
    card = cards_by_name.get(name)
    if card is None:
        return None, [ValidationError("unknown_function", f"{name} not in catalog")]

    known = set(card.param_names)
    for k in params:
        if k not in known:
            errs.append(ValidationError("unknown_param", f"unknown param {k}"))
    for req in card.required_params:
        if req not in params:
            errs.append(ValidationError("missing_required", f"missing required {req}"))

    for k, v in params.items():
        spec = card.param(k)
        if spec is None:
            continue
        if spec.type in ("number", "integer"):
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                errs.append(ValidationError("type_mismatch", f"{k} must be numeric",
                                            type_phrase(card, spec)))
                continue
            if spec.type == "integer" and float(v) != int(v):
                errs.append(ValidationError("type_mismatch", f"{k} must be integer",
                                            type_phrase(card, spec)))
            if spec.minimum is not None and v < spec.minimum:
                errs.append(ValidationError("out_of_range", f"{k} < {spec.minimum}",
                                            limit_phrase(card, spec, spec.minimum, "最低")))
            if spec.maximum is not None and v > spec.maximum:
                errs.append(ValidationError("out_of_range", f"{k} > {spec.maximum}",
                                            limit_phrase(card, spec, spec.maximum, "最高")))
        elif spec.type == "boolean":
            if not isinstance(v, bool):
                errs.append(ValidationError("type_mismatch", f"{k} must be boolean",
                                            type_phrase(card, spec)))
        elif spec.type == "enum":
            if spec.enum and v not in spec.enum:
                errs.append(ValidationError("bad_enum", f"{k}={v} not in {spec.enum}",
                                            enum_phrase(card, spec)))
        elif spec.type == "string":
            if not isinstance(v, str):
                errs.append(ValidationError("type_mismatch", f"{k} must be string",
                                            type_phrase(card, spec)))

    if errs:
        return None, errs
    return ToolCall(name=name, parameters=params), []
