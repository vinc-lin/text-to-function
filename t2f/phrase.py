"""Driver-facing wording for parameters and their limits.

One module so the validator and the vehicle simulator cannot drift into saying the same
thing two different ways. The rule everywhere: **speak the catalog's words, never an
internal address.** `window_position` is how the code addresses a signal; `车窗开度` is
what a driver calls it, and the catalog already holds it.

Nothing here decides *whether* to speak — only *how*. Callers author a sentence and the
reply layer speaks it verbatim.
"""
from __future__ import annotations
from typing import Optional
from .types import FunctionCard, ParamSpec

POSITION_CN = {"driver": "主驾", "passenger": "副驾", "rear": "后排", "all": "全车",
               "left": "左侧", "right": "右侧"}

_UNIT_CN = {"percent": "%", "celsius": "度", "level": "档"}


def fmt_num(value) -> str:
    """25.0 -> 25, 22.5 -> 22.5. Limits are stored as REAL and must not be spoken as floats."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _first_clause(text: str) -> str:
    """Catalog descriptions carry parentheticals — `加热档位，0为关闭`. Only the head of the
    phrase reads as a sentence when embedded mid-utterance."""
    return (text or "").split("，")[0].strip()


def param_subject(card: FunctionCard, param: Optional[ParamSpec]) -> str:
    """What the driver calls the quantity this parameter controls."""
    return _first_clause(param.description if param else "") or card_subject(card)


def card_subject(card: FunctionCard) -> str:
    """What the driver calls the thing this function controls.

    Aliases first: they are, by construction, the words drivers actually use for it.
    """
    if card.aliases:
        return card.aliases[0]
    return _first_clause(card.description) or card.name


def with_unit(value, param: Optional[ParamSpec]) -> str:
    return f"{fmt_num(value)}{_UNIT_CN.get(param.unit if param else None, '')}"


def limit_phrase(card: FunctionCard, param: Optional[ParamSpec], bound, direction: str,
                 low=None, high=None) -> str:
    """State the whole permitted range when it is known, not just the bound that was broken.

    `目标温度只能设置在16到32度之间` beats `目标温度最高只能到32度`: a driver who said 99度
    can act on the first and has to guess after the second. Falls back to the one-sided form
    when only a single bound exists (a physical limit read off a jammed actuator, say).
    """
    subject = param_subject(card, param)
    if low is None and param is not None:
        low = param.minimum
    if high is None and param is not None:
        high = param.maximum
    if low is not None and high is not None:
        return f"{subject}只能设置在{fmt_num(low)}到{with_unit(high, param)}之间"
    return f"{subject}{direction}只能到{with_unit(bound, param)}"


def enum_phrase(card: FunctionCard, param: Optional[ParamSpec]) -> str:
    """`温区只支持主驾/副驾/后排/全车`."""
    options = "/".join(POSITION_CN.get(e, e) for e in ((param.enum if param else None) or []))
    return f"{param_subject(card, param)}只支持{options}" if options else ""


def type_phrase(card: FunctionCard, param: Optional[ParamSpec]) -> str:
    kind = (param.type if param else "") or ""
    if kind in ("number", "integer"):
        return f"{param_subject(card, param)}需要一个数值"
    if kind == "boolean":
        return f"{param_subject(card, param)}只能是开或关"
    return ""


def missing_phrase(card: FunctionCard, param: Optional[ParamSpec]) -> str:
    """The question asked when a required parameter could not be extracted.

    Asking beats stating here — the driver can answer a question, and cannot do anything
    with 请补充更多信息。
    """
    kind = (param.type if param else "") or ""
    if kind == "boolean":
        return f"您想打开还是关闭{card_subject(card)}？"
    if kind == "enum":
        options = "/".join(POSITION_CN.get(e, e) for e in ((param.enum if param else None) or []))
        return f"您想设置{param_subject(card, param)}的哪个选项？（{options}）" if options else ""
    if kind in ("number", "integer"):
        return f"您想把{param_subject(card, param)}设置成多少？"
    return ""
