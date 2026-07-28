import re
from .types import LexFeatures
from .params.numerals import find_numbers, parse_number, parse_fraction_percent

_POSITION = [
    ("driver", ["主驾驶", "主驾", "驾驶位", "驾驶座", "左前"]),
    ("passenger", ["副驾驶", "副驾", "右前"]),
    ("rear", ["后排", "后座", "后面"]),
    ("all", ["全车", "所有", "整车"]),
    ("left", ["左边", "左侧", "左"]),
    ("right", ["右边", "右侧", "右"]),
]
_ON = ["打开", "开启", "开一下", "启动", "开"]
_OFF = ["关闭", "关掉", "关上", "关一下", "关"]
# A negated instruction is NOT the opposite instruction. 别关车窗 does not mean "open the
# window" — the driver may want it left exactly where it is — so a negated cue yields None
# and the missing-parameter path asks 您想打开还是关闭车窗？ Guessing the inverse would be a
# second way to act against what was said.
_NEGATE = ["别", "不要", "不用", "甭", "勿", "先别"]
_MAX = ["最大", "最高", "开到最大", "拉满"]
_MIN = ["最小", "最低"]
_INC = ["调高", "升高", "大一点", "高一点", "增大", "提高", "热一点", "升"]
_DEC = ["调低", "降低", "小一点", "低一点", "减小", "凉一点", "降"]

_AMT_SMALL = ["一点点", "一点", "点儿", "一些", "些", "稍微", "稍稍", "略微"]
_AMT_LARGE = ["多一些", "大幅", "很多", "好多"]
_REL_INC = ["开", "大", "高", "升", "加", "多", "亮", "热"]
_REL_DEC = ["关", "小", "低", "降", "减", "少", "暗", "凉"]

_TEMP = re.compile(r"(\d+(?:\.\d+)?|[零〇一二两俩三四五六七八九十百千]+)\s*度")
_PCT = re.compile(r"(\d+(?:\.\d+)?|[零〇一二两俩三四五六七八九十百千]+)\s*%|百分之\s*(\d+|[零〇一二两俩三四五六七八九十百千]+)")
_LEVEL = re.compile(r"(\d+|[零〇一二两俩三四五六七八九十百千]+)\s*[档级挡]")


def _match_positions(text: str) -> list[str]:
    # Find each category's best (longest) variant match independently, then resolve
    # cross-category overlaps by leftmost-match-first (ties broken by longest span).
    # A fixed category-priority order (as originally drafted) can let an earlier
    # category's shorter match block a later category's longer, more specific match
    # when their spans overlap (e.g. driver's "驾驶座" vs. passenger's "副驾驶" both
    # occurring inside "副驾驶座位").
    matches: list[tuple[int, int, str]] = []  # (start_idx, length, norm)
    for norm, variants in _POSITION:
        for v in sorted(variants, key=len, reverse=True):
            idx = text.find(v)
            if idx != -1:
                matches.append((idx, len(v), norm))
                break
    matches.sort(key=lambda m: (m[0], -m[1]))
    found, used = [], [False] * len(text)
    for idx, length, norm in matches:
        if not any(used[idx:idx + length]):
            found.append(norm)
            for i in range(idx, idx + length):
                used[i] = True
    return found


def _polarity(clause: str):
    """on / off / unknown, decided by POSITION rather than by table order.

    The old rule tested every _OFF form before any _ON form, so the 关 inside 关窗 beat a
    leading 打开 and 打开下雨自动关窗 disabled the feature it asked to enable. The governing
    cue in a Chinese imperative is the leading verb, so the earliest match wins, and the
    longest match wins a tie so 打开 is preferred over the 开 inside it.
    """
    hits = [(clause.find(k), False, k) for k in _OFF if k in clause]
    hits += [(clause.find(k), True, k) for k in _ON if k in clause]
    if not hits:
        return None
    position, polarity, _ = min(hits, key=lambda h: (h[0], -len(h[2])))
    for marker in _NEGATE:
        if clause[max(0, position - len(marker)):position] == marker:
            return None                      # negated: refuse to guess, ask instead
    return polarity


def extract_features(clause: str) -> LexFeatures:
    f = LexFeatures(raw=clause)
    f.numbers = find_numbers(clause)
    f.temperatures = [parse_number(m.group(1)) for m in _TEMP.finditer(clause)]
    for m in _PCT.finditer(clause):
        f.percentages.append(parse_number(m.group(1) or m.group(2)))
    f.levels = [int(parse_number(m.group(1))) for m in _LEVEL.finditer(clause)]
    f.positions = _match_positions(clause)
    if any(k in clause for k in _MAX):
        f.operation = "max"
    elif any(k in clause for k in _MIN):
        f.operation = "min"
    elif any(k in clause for k in _INC):
        f.operation = "increase"
    elif any(k in clause for k in _DEC):
        f.operation = "decrease"
    f.on_off = _polarity(clause)

    # fraction words -> percent (e.g. 一半 -> 50); only when no explicit % was found
    if not f.percentages:
        frac = parse_fraction_percent(clause)
        if frac is not None:
            f.percentages.append(float(frac))

    # relative amount + operation (e.g. 再开一点 -> increase/small, 调小一点 -> decrease/small)
    small = any(k in clause for k in _AMT_SMALL)
    large = any(k in clause for k in _AMT_LARGE)
    if small or large or f.operation in ("increase", "decrease"):
        f.amount = "large" if large else ("small" if small else "medium")
        if f.operation not in ("increase", "decrease"):
            inc = any(k in clause for k in _REL_INC)
            dec = any(k in clause for k in _REL_DEC)
            if dec and not inc:
                f.operation = "decrease"
            elif inc and not dec:
                f.operation = "increase"
            else:
                f.amount = None  # ambiguous direction -> not a usable relative op
    return f
