"""function + params  ->  the signals it writes.

entity    = "<domain>.<position|all>"
attribute = the FUNCTION's feature name — its name with a leading verb stripped.

The attribute is derived from the function, not from the parameter, because parameter names
are generic: 30-odd cards take `enabled`, four seat cards take `level`, three door cards take
`is_open`. Keying on the parameter would put every light on one `light.all.enabled` row, so
switching the hazards would erase the headlights. Keying on the function gives each physical
feature its own row.

`fold_` is deliberately NOT a stripped verb: `fold_mirror` and `adjust_mirror` are different
signals (folded-ness vs angle) and stripping both verbs would merge them.

_OVERRIDES exists ONLY for genuine aliases — two functions that address ONE physical thing.
The attribute there must be qualified (`window_position`, not `position`) because the sunroof
functions live in the `window` domain too, so a bare `position` would make
open_window{position: "all"} and the sunroof share a row. Add an entry when two functions
really are the same actuator — never to paper over a collision.
"""
from __future__ import annotations
import re
from t2f.types import FunctionCard, ToolCall
from t2f.state import primary_numeric_param

_VERB = re.compile(r"^(set|open|adjust|move|vent|spray)_")

# function -> (attribute, transform)   transform: raw param value -> stored signal value
_OVERRIDES = {
    "open_window":          ("window_position",  lambda v: 100 if v else 0),
    "set_window_position":  ("window_position",  None),
    "open_sunroof":         ("sunroof_position", lambda v: 100 if v else 0),
    "set_sunroof_position": ("sunroof_position", None),
}


def _primary_param(card: FunctionCard) -> str | None:
    """Which parameter carries the value being written."""
    p = primary_numeric_param(card)
    if p is not None:
        return p.name
    req = [x for x in card.required_params if x != "position"]
    return req[0] if req else None


def _attribute(card: FunctionCard) -> tuple[str, object]:
    return _OVERRIDES.get(card.name) or (_VERB.sub("", card.name), None)


def _entity(card: FunctionCard, params: dict) -> str:
    return f"{card.domain}.{params.get('position') or 'all'}"


def resolve_writes(card: FunctionCard, tool_call: ToolCall) -> list[tuple]:
    """[(entity, attribute, value)] for this call. Empty when nothing is addressable."""
    params = tool_call.parameters
    source = _primary_param(card)
    if source is None or source not in params:
        return []
    attribute, transform = _attribute(card)
    value = params[source]
    return [(_entity(card, params), attribute, transform(value) if transform else value)]


def signal_for_function(card: FunctionCard, position: str | None = None) -> tuple | None:
    """Reverse lookup so StateResolver can read the current value back."""
    if _primary_param(card) is None:
        return None
    attribute, _ = _attribute(card)
    return (f"{card.domain}.{position or 'all'}", attribute)
