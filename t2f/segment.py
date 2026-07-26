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
    """Split into fragments and label each ACTION/CONTEXT/CONNECTOR.

    CONTEXT spans are *suppressed* — the caller routes only ACTION spans. Narration used to be
    attached to the nearest action span as well, but nothing ever read it, so the carrier field
    and the attachment pass were removed. If context attribution is wanted later (using the
    narration to fill a slot), it needs a consumer designed alongside it, not a dead field.
    """
    alias_index = build_alias_index(cards, domain_keywords)
    raw = split(text)
    return [Span(text=t, role=classify_span(t, extract_features(t), alias_index)) for t in raw]
