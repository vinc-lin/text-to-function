from __future__ import annotations
import json
from ..types import FunctionCard, ParamSpec

_SYS = ("你是车载语音指令解析器。从给定候选功能中选择唯一一个，"
        "输出一个JSON工具调用：{\"name\": 功能名, \"parameters\": {...}}。"
        "只能使用候选中的功能名，不要输出候选之外的功能，不要解释，不要输出多余文本。")


def _param_str(p: ParamSpec) -> str:
    if p.type == "enum":
        rng = "|".join(p.enum or [])
    elif p.minimum is not None or p.maximum is not None:
        rng = f"{p.minimum}-{p.maximum}"
    else:
        rng = p.type
    req = "*" if p.required else ""
    return f"{p.name}{req}({rng})"


def compact_schema(card: FunctionCard) -> str:
    params = ", ".join(_param_str(p) for p in card.params) or "无参数"
    return f"- {card.name}: {card.description} | 参数: {params}"


def build_prompt(clause: str, cards: list[FunctionCard], extracted_params: dict) -> list[dict]:
    tools = "\n".join(compact_schema(c) for c in cards)
    user = (f"用户指令：{clause}\n候选功能：\n{tools}\n"
            f"已提取参数：{json.dumps(extracted_params, ensure_ascii=False)}\n"
            "请输出JSON工具调用。")
    return [{"role": "system", "content": _SYS}, {"role": "user", "content": user}]
