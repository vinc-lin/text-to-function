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
_FAILURE = "抱歉，这个操作没能完成。"


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


def _questions(clauses) -> list[str]:
    """Non-empty clarification questions in clause order, exact duplicates dropped."""
    out: list[str] = []
    for cl in clauses:
        clar = cl.clarification
        text = ((clar.question if clar else "") or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def _has_failure(clauses) -> bool:
    """A clause that failed validation and has nothing else to say."""
    for cl in clauses:
        spoke = bool((cl.response or "").strip())
        asked = bool(cl.clarification and (cl.clarification.question or "").strip())
        if cl.validation_errors and not spoke and not asked:
            return True
    return False


def compose_reply(result: RouteResult) -> str:
    """Compose the utterance-level reply. Runs AFTER execution, so it must never raise."""
    clauses = result.clauses or []
    parts = [_sentence(t) for t in _confirmations(clauses)]
    questions = _questions(clauses)
    if questions:                      # at most ONE question per reply
        parts.append(_sentence(questions[0]))
    elif _has_failure(clauses):
        parts.append(_FAILURE)
    return "".join(parts) if parts else _ACK
