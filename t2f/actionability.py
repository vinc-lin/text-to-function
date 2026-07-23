from __future__ import annotations
from .types import FunctionCard, LexFeatures, SpanRole
from .lexical import _MAX, _MIN, _INC, _DEC

# operation/polarity/value verbs that mark an imperative control (not narration).
_OP_CUES = [
    "打开", "关闭", "关掉", "关上", "开启", "调到", "调成", "调至", "设为", "设成", "设定",
    "开到", "开大", "开小", "升到", "降到", "调高", "调低", "锁上", "锁定", "解锁", "取消",
    "拉开", "拉上", "收起", "翘起", "开一下", "关一下", "打开到", "定在", "弄到", "留个缝",
    "留一条缝",
]
_CONNECTOR_ONLY = {"然后", "还有", "并且", "同时", "接着", "并", "而且", "以及"}
_EXPLICIT_OPERATION_VERBS = _MAX + _MIN + _INC + _DEC


def build_alias_index(cards: list[FunctionCard], domain_keywords: dict | None = None) -> list[str]:
    """Target-word index = every card alias UNION the config domain_keywords (longest-first).
    Domain keywords (e.g. 窗户/玻璃) catch common synonyms that curated aliases miss, so genuine
    commands aren't dropped as context. Context suppression is unaffected because an ACTION still
    additionally requires an operation cue."""
    targets: set[str] = set()
    for c in cards:
        targets.update(c.aliases)
    for words in (domain_keywords or {}).values():
        targets.update(words)
    return sorted(targets, key=len, reverse=True)


def _has_target(text: str, alias_index: list[str]) -> bool:
    return any(a in text for a in alias_index)


def _has_operation(text: str, feats: LexFeatures) -> bool:
    if feats.on_off is not None:        # 开/关 style polarity
        return True
    if feats.operation is not None:     # increase/decrease/max/min (incl. relative)
        # Must have explicit operation verb, not just relative amount inference
        if any(verb in text for verb in _EXPLICIT_OPERATION_VERBS):
            return True
        return False
    if feats.percentages or feats.temperatures or feats.levels:  # explicit value
        return True
    return any(cue in text for cue in _OP_CUES)


def classify_span(text: str, feats: LexFeatures, alias_index: list[str]) -> SpanRole:
    """ACTION iff (a target word) AND (an operation/polarity/value cue). Else CONNECTOR (pure
    conjunction residue) or CONTEXT (narration). Fails safe: implied-desire narration like
    '我有点冷' has no operation cue -> CONTEXT, never a silent action."""
    stripped = text.strip()
    if stripped in _CONNECTOR_ONLY or not stripped:
        return SpanRole.CONNECTOR
    if _has_target(stripped, alias_index) and _has_operation(stripped, feats):
        return SpanRole.ACTION
    return SpanRole.CONTEXT
