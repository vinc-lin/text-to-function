from __future__ import annotations
import json
from ..types import FunctionCard, ParamSpec

_SYS = ("你是车载语音指令解析器。从给定候选功能中选择唯一一个，"
        "输出一个JSON工具调用：{\"name\": 功能名, \"parameters\": {...}}。"
        "只能使用候选中的功能名，不要输出候选之外的功能，不要解释，不要输出多余文本。")


def _param_str(p: ParamSpec) -> str:
    if p.type == "enum":
        rng = "|".join(p.enum or [])
    elif p.minimum is not None and p.maximum is not None:
        rng = f"{p.minimum}-{p.maximum}"
    elif p.minimum is not None:
        rng = f">={p.minimum}"
    elif p.maximum is not None:
        rng = f"<={p.maximum}"
    else:
        rng = p.type
    req = "*" if p.required else ""
    return f"{p.name}{req}({rng})"


def compact_schema(card: FunctionCard) -> str:
    params = ", ".join(_param_str(p) for p in card.params) or "无参数"
    return f"- {card.name}: {card.description} | 参数: {params}"


def build_prompt(clause: str, cards: list[FunctionCard], extracted_params: dict,
                 allow_reject: bool = False) -> list[dict]:
    tools = "\n".join(compact_schema(c) for c in cards)
    reject = ("\n如果用户指令与上述候选功能都不匹配（例如闲聊或不支持的请求），"
              "请输出 {\"name\": \"__reject__\"}。" if allow_reject else "")
    user = (f"用户指令：{clause}\n候选功能：\n{tools}\n"
            f"已提取参数：{json.dumps(extracted_params, ensure_ascii=False)}{reject}\n"
            "请输出JSON工具调用。")
    return [{"role": "system", "content": _SYS}, {"role": "user", "content": user}]


_PLAN_SYS = ("你是车载语音指令解析器。用户的一句话可能包含多个操作以及说明背景的话。"
             "只为真正的操作生成工具调用，忽略仅说明背景的内容（不要为背景生成调用）。"
             "按出现顺序输出JSON：{\"actions\": [{\"name\": 功能名, \"parameters\": {...}}, ...]}。"
             "只能使用候选功能名。与候选都不匹配的操作用 {\"name\": \"__reject__\"}。不要解释。")


def build_plan_prompt(utterance: str, action_spans: list[str],
                      candidate_cards: list[FunctionCard]) -> list[dict]:
    tools = "\n".join(compact_schema(c) for c in candidate_cards)
    spans = "\n".join(f"{i+1}. {t}" for i, t in enumerate(action_spans))
    user = (f"用户原话：{utterance}\n识别到的操作片段：\n{spans}\n候选功能：\n{tools}\n"
            "请输出JSON操作计划。")
    return [{"role": "system", "content": _PLAN_SYS}, {"role": "user", "content": user}]
