# t2f/reply.py
"""Utterance-level reply composition.

Sits one altitude above `respond.py`: that module renders a sentence for a single
(card, tool_call); this one composes a whole RouteResult into the single string a
voice assistant speaks. Pure — no cards, no state, no I/O.
"""
from __future__ import annotations
from .types import RouteResult

_TERMINATORS = "。！？"
_ACK = "好的。"


def _sentence(text: str) -> str:
    """Ensure a rendered fragment ends with a sentence terminator."""
    if not text:
        return ""
    return text if text[-1] in _TERMINATORS else text + "。"


def _confirmations(clauses) -> list[str]:
    """Non-empty responses in clause order, exact duplicates dropped."""
    out: list[str] = []
    for cl in clauses:
        text = (cl.response or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def compose_reply(result: RouteResult) -> str:
    """Compose the utterance-level reply. Runs AFTER execution, so it must never raise."""
    clauses = result.clauses or []
    parts = [_sentence(t) for t in _confirmations(clauses)]
    return "".join(parts) if parts else _ACK
