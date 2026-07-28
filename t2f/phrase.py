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

# Every enum value the catalog uses, in the driver's language. Enum values are English
# identifiers; speaking them aloud is the same defect as speaking a signal address, and it
# reached a driver as 氛围灯颜色只支持red/blue/green. A value missing from this table is
# omitted from the spoken options rather than read out in English -- see enum_phrase.
_ENUM_CN = {
    # zones / directions
    "driver": "主驾", "passenger": "副驾", "rear": "后排", "all": "全车",
    "left": "左侧", "right": "右侧", "up": "上", "down": "下",
    "front": "前", "back": "后", "face": "吹面", "feet": "吹脚",
    "windshield": "吹窗", "mix": "混合", "recline": "放倒", "upright": "竖直",
    # climate
    "cool": "制冷", "heat": "制热", "auto": "自动", "dry": "除湿", "fan": "送风",
    "internal": "内循环", "external": "外循环",
    # colours
    "red": "红色", "blue": "蓝色", "green": "绿色", "white": "白色",
    "purple": "紫色", "orange": "橙色", "cyan": "青色",
    # light
    "low_beam": "近光", "high_beam": "远光", "day": "日间", "night": "夜间",
    # media
    "radio": "收音机", "bluetooth": "蓝牙", "usb": "USB", "aux": "AUX",
    "fm": "调频", "am": "调幅", "local": "本地",
    "standard": "标准", "rock": "摇滚", "pop": "流行", "jazz": "爵士",
    "classical": "古典", "vocal": "人声",
    # navigation
    "fastest": "最快", "shortest": "最短", "avoid_toll": "避开收费",
    "avoid_highway": "避开高速",
}


# Surface forms a driver actually uses for each enum value, mined from the catalog's own
# `utterances` rather than invented — 空调吹脸 and 风向调成吹面 are both `face`. The label in
# _ENUM_CN is always included, so this table only carries the ADDITIONAL ways of saying it.
_ENUM_ALIASES = {
    "cool": ["制冷", "冷风"], "heat": ["制热", "暖风", "热风"], "dry": ["除湿"],
    "fan": ["送风", "吹风"],
    "face": ["吹面", "吹脸"], "feet": ["吹脚"], "windshield": ["挡风玻璃", "吹窗", "除雾"],
    "mix": ["混合", "上下都吹"],
    "internal": ["内循环"], "external": ["外循环"],
    "day": ["日间", "白天"], "night": ["夜间", "夜晚"],
    "low_beam": ["近光"], "high_beam": ["远光"],
    "red": ["红"], "blue": ["蓝"], "green": ["绿"], "white": ["白"],
    "purple": ["紫"], "orange": ["橙"], "cyan": ["青"],
    "bluetooth": ["蓝牙"], "usb": ["USB", "usb", "U盘"], "radio": ["收音机", "电台"],
    "aux": ["AUX", "aux"], "local": ["本地"],
    "fm": ["FM", "fm", "调频"], "am": ["AM", "am", "调幅"],
    # 音源切到U盘 and 走最省时间的路 are the catalog's own examples; the words are the driver's.
    "pop": ["流行"], "rock": ["摇滚"], "classical": ["古典"], "jazz": ["爵士"],
    "vocal": ["人声"], "standard": ["标准"],
    "fastest": ["最快", "最省时间"], "shortest": ["最短"],
    "avoid_highway": ["避开高速", "别上高速", "不上高速"],
    "avoid_toll": ["避开收费", "不走收费", "躲避收费"],
    "front": ["往前", "向前", "前移", "前挪"],
    "back": ["往后", "向后", "后移", "后退", "后调"],
    "recline": ["放倒", "往后躺", "往后调", "放平", "后躺"],
    "upright": ["立起来", "立直", "调直", "竖直"],
    "up": ["升高", "往上", "调高", "上升", "抬高"],
    "down": ["降低", "往下", "调低", "下降", "降下来"],
}


def enum_surface_forms(value: str) -> list[str]:
    """Every way a driver might say this enum value, longest first."""
    forms = list(_ENUM_ALIASES.get(value, []))
    label = _ENUM_CN.get(value)
    if label and label not in forms:
        forms.append(label)
    return sorted(forms, key=len, reverse=True)


def fmt_num(value) -> str:
    """25.0 -> 25, 22.5 -> 22.5. Limits are stored as REAL and must not be spoken as floats."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _first_clause(text: str) -> str:
    """Catalog descriptions carry parentheticals — `加热档位，0为关闭`. Only the head of the
    phrase reads as a sentence when embedded mid-utterance."""
    return (text or "").split("，")[0].strip()


_IDENTIFIER = __import__("re").compile(r"[A-Za-z_]{3,}")


def param_subject(card: FunctionCard, param: Optional[ParamSpec]) -> str:
    """What the driver calls the quantity this parameter controls.

    Some descriptions are notes to a developer rather than words for a driver — a boolean's
    reads `开启为true，关闭为false`, whose head is `开启为true`. Any description carrying a
    latin identifier is rejected in favour of the card's own subject, which comes from the
    aliases drivers actually say. Booleans are rejected outright: there is no useful noun in
    "enabled", only in the thing being enabled.
    """
    text = _first_clause(param.description if param else "")
    if not text or _IDENTIFIER.search(text) or (param is not None and param.type == "boolean"):
        return card_subject(card)
    return text


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


def enum_options(param: Optional[ParamSpec]) -> str:
    """The supported values, in Chinese. A value with no Chinese label is DROPPED rather than
    spoken in English — a driver can act on a partial list and cannot act on `avoid_toll`."""
    labels = [_ENUM_CN[e] for e in ((param.enum if param else None) or []) if e in _ENUM_CN]
    return "/".join(labels)


def enum_phrase(card: FunctionCard, param: Optional[ParamSpec]) -> str:
    """`温区只支持主驾/副驾/后排/全车`."""
    options = enum_options(param)
    return f"{param_subject(card, param)}只支持{options}" if options else ""


def type_phrase(card: FunctionCard, param: Optional[ParamSpec]) -> str:
    kind = (param.type if param else "") or ""
    if kind in ("number", "integer"):
        return f"{param_subject(card, param)}需要一个数值"
    if kind == "boolean":
        return f"{param_subject(card, param)}只能是开或关"
    if kind == "string":
        return f"{param_subject(card, param)}需要一段文字"
    if kind == "enum":
        return enum_phrase(card, param)
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
        options = enum_options(param)
        # Options first so the sentence ends on the question mark; a trailing parenthetical
        # makes the reply layer append 。 after ）, which reads as a stumble.
        return f"{param_subject(card, param)}支持{options}，您想设置成哪个？" if options else ""
    if kind in ("number", "integer"):
        return f"您想把{param_subject(card, param)}设置成多少？"
    if kind == "string":
        return f"请告诉我{param_subject(card, param)}。"
    return ""
