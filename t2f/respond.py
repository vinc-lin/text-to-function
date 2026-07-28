# t2f/respond.py
from __future__ import annotations
from .types import FunctionCard, ToolCall, ClarificationRequest
from .phrase import POSITION_CN as _POSITION_CN, missing_phrase

_CLARIFY = {"position": "您想调整哪个区域？（主驾/副驾/后排）",
            "temperature": "您想设置到多少度？",
            "level": "您想调到几档？"}


class _SafeDict(dict):
    def __missing__(self, key):
        return ""


def _fmt_num(v):
    """Render integral floats without a trailing ".0" (25.0 -> 25)."""
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def render_response(card: FunctionCard, tool_call: ToolCall) -> str:
    if not card.response_template:
        return f"已执行{card.name}。"
    params = {k: _fmt_num(v) for k, v in tool_call.parameters.items()}
    if "position" in params:
        params["position"] = _POSITION_CN.get(params["position"], params["position"])
    elif card.param("position"):
        params.setdefault("position", "当前区域")
    return card.response_template.format_map(_SafeDict(params))


def build_clarification(card: FunctionCard, missing: list[str]) -> ClarificationRequest:
    """Ask for the missing parameter BY NAME.

    The three hand-written questions stay — they read better than anything generated. Every
    other required parameter used to fall through to 请补充更多信息。, which does not tell the
    driver what is missing; the catalog's own `description` answers that for all 17 of them.
    """
    first = missing[0] if missing else ""
    question = _CLARIFY.get(first) or missing_phrase(card, card.param(first)) or "请补充更多信息。"
    return ClarificationRequest(question=question)


def build_low_confidence_clarification() -> ClarificationRequest:
    """Clarification for LOW-band / out-of-scope requests where no function is chosen."""
    return ClarificationRequest(question="抱歉，我不太确定您的意思，可以换个说法吗？")


def build_plan_clarification(pending) -> ClarificationRequest:
    """One question covering all unresolved actions in a multi-action plan."""
    spans = "」「".join(a.span for a in pending)
    return ClarificationRequest(question=f"关于「{spans}」我还需要确认一下，请补充信息。")
