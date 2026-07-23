import re

_CN_DIGIT = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "俩": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNIT = {"十": 10, "百": 100, "千": 1000}
_CN_CHARS = set(_CN_DIGIT) | set(_CN_UNIT)


def _parse_cn(s: str) -> float | None:
    if not s or any(ch not in _CN_CHARS for ch in s):
        return None
    total, section, number = 0, 0, 0
    for ch in s:
        if ch in _CN_DIGIT:
            number = _CN_DIGIT[ch]
        else:  # unit
            unit = _CN_UNIT[ch]
            if number == 0:
                number = 1  # e.g. 十 == 一十
            section += number * unit
            number = 0
    return float(total + section + number)


def parse_number(s: str) -> float | None:
    s = s.strip()
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s)
    return _parse_cn(s)


_NUM_SPAN = re.compile(r"\d+(?:\.\d+)?|[零〇一二两俩三四五六七八九十百千]+")


def find_numbers(text: str) -> list[float]:
    out = []
    for m in _NUM_SPAN.finditer(text):
        v = parse_number(m.group())
        if v is not None:
            out.append(v)
    return out


# "一半" and "N分之M" -> integer percent (0-100), else None.
# 百 is excluded from the denominator group so "百分之N" is not treated as a
# fraction here; it is an explicit percentage handled elsewhere.
_HALF = ("一半", "半")
_FRAC = re.compile(r"([零〇一二两俩三四五六七八九十千]+|\d+)分之([零〇一二两俩三四五六七八九十百千]+|\d+)")


def parse_fraction_percent(text: str) -> int | None:
    m = _FRAC.search(text)
    if m:
        denom = parse_number(m.group(1))
        numer = parse_number(m.group(2))
        if denom and numer is not None and denom != 0:
            return int(round(numer / denom * 100))
    if any(h in text for h in _HALF):
        return 50
    return None
