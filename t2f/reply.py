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


def _question_text(clause) -> str:
    """The clause's clarification question, or '' when absent or blank."""
    clar = clause.clarification
    return ((clar.question if clar else "") or "").strip()


def _questions(clauses) -> list[str]:
    """Non-empty clarification questions in clause order, exact duplicates dropped."""
    out: list[str] = []
    for cl in clauses:
        text = _question_text(cl)
        if text and text not in out:
            out.append(text)
    return out


def _has_failure(clauses) -> bool:
    """A clause that failed validation and has nothing else to say."""
    for cl in clauses:
        spoke = bool((cl.response or "").strip())
        asked = bool(_question_text(cl))
        # The 'asked' guard is currently redundant (compose_reply only reaches here in the
        # elif, i.e. when no clause has a question) but is kept so this helper stays correct
        # in isolation.
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
