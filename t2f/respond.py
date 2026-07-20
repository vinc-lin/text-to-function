# t2f/respond.py
from __future__ import annotations
from .types import FunctionCard, ToolCall, ClarificationRequest, PendingState

_POSITION_CN = {"driver": "主驾", "passenger": "副驾", "rear": "后排", "all": "全车",
                "left": "左侧", "right": "右侧"}
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
    first = missing[0] if missing else ""
    question = _CLARIFY.get(first, "请补充更多信息。")
    pending = PendingState(pending_function=card.name, known_parameters={}, missing_parameters=missing)
    return ClarificationRequest(question=question, pending=pending)


def build_low_confidence_clarification() -> ClarificationRequest:
    """Clarification for LOW-band / out-of-scope requests where no function is chosen."""
    return ClarificationRequest(question="抱歉，我不太确定您的意思，可以换个说法吗？", pending=None)
