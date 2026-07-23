import re
from .types import Span, SpanRole
from .lexical import extract_features
from .actionability import build_alias_index, classify_span

# delimiter punctuation always splits; a "." between digits (decimals, "FM101.7") is NOT a delimiter
# includes both ASCII (,;) and Chinese fullwidth punctuation (，；)
_PUNCT = re.compile(r"[,;，；]|(?<!\d)\.(?!\d)")
# conjunctions that split when surrounded by non-trivial text
_CONJ = ["然后", "还有", "并且", "同时", "接着", "并"]
_MIN_FRAG = 2  # a fragment shorter than this is not a standalone intent

def _split_conjunctions(seg: str) -> list[str]:
    for conj in _CONJ:
        if conj in seg:
            parts = [p.strip() for p in seg.split(conj)]
            if all(len(p) >= _MIN_FRAG for p in parts) and len(parts) > 1:
                out: list[str] = []
                for p in parts:
                    out.extend(_split_conjunctions(p))
                return out
    return [seg]

def split(text: str) -> list[str]:
    raw = [s.strip() for s in _PUNCT.split(text) if s.strip()]
    out: list[str] = []
    for seg in raw:
        out.extend(_split_conjunctions(seg))
    return [s for s in out if s] or [text]


def segment(text, cards, domain_keywords=None) -> list[Span]:
    """Split into fragments, label each ACTION/CONTEXT/CONNECTOR, and attach each CONTEXT
    fragment to the nearest following ACTION (or the previous ACTION if it is trailing)."""
    alias_index = build_alias_index(cards, domain_keywords)
    raw = split(text)
    spans = [Span(text=t, role=classify_span(t, extract_features(t), alias_index)) for t in raw]

    actions = [s for s in spans if s.role == SpanRole.ACTION]
    if not actions:
        return spans

    for i, s in enumerate(spans):
        if s.role != SpanRole.CONTEXT:
            continue
        following = next((a for a in spans[i + 1:] if a.role == SpanRole.ACTION), None)
        target = following or next((a for a in reversed(spans[:i]) if a.role == SpanRole.ACTION), None)
        if target is not None:
            target.attached_context.append(s.text)
    return spans
