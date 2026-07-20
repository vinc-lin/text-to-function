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


def render_response(card: FunctionCard, tool_call: ToolCall) -> str:
    if not card.response_template:
        return f"已执行{card.name}。"
    params = dict(tool_call.parameters)
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
