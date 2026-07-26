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


def _exec_failures(clauses) -> list[str]:
    """Vehicle-reported refusals, in clause order, de-duplicated. The detail is authored by
    the simulator for the driver, so it is spoken verbatim.

    This is requirement 4b's THIRD failure branch — "I understood you, I tried, and the car
    said no" — as opposed to "I didn't understand" (a clarification question) and "a parameter
    is unusable" (the validation table, still unspoken). Only the executor's own detail is
    spoken; an exec_error with a blank detail has nothing driver-usable in it and falls back
    to the generic line via `_has_failure`.
    """
    out: list[str] = []
    for cl in clauses:
        err = getattr(cl, "exec_error", None)      # defensive: compose_reply must never raise
        detail = ((err.message if err else "") or "").strip()
        if detail and detail not in out:
            out.append(detail)
    return out


def _has_failure(clauses) -> bool:
    """A clause that produced nothing the driver can hear.

    This covers both an outright validation failure and an UNRESOLVED clause — e.g. a
    MEDIUM-band span with no LLM configured, which reaches the reply layer with no response,
    no clarification and no errors (t2f/pipeline.py:200-201 and NullMediumResolver). Such a
    clause must never be reported as success: saying 好的。 when nothing happened is a
    falsely affirmative confirmation, the worst failure mode for a vehicle assistant.
    """
    for cl in clauses:
        spoke = bool((cl.response or "").strip())
        asked = bool(_question_text(cl))
        if not spoke and not asked:
            return True
    return False


def compose_reply(result: RouteResult) -> str:
    """Compose the utterance-level reply. Runs AFTER execution, so it must never raise.

    好的。 is spoken ONLY when there were no clauses at all. A clause that produced nothing
    is an unresolved request and yields the failure line instead — never a false affirmation.

    Refusals follow the confirmations, so an utterance where one action succeeded and another
    was refused reports BOTH, with what actually happened first. A spoken refusal replaces the
    generic failure line — repeating it after a specific cause says nothing extra.
    """
    clauses = result.clauses or []
    parts = [_sentence(t) for t in _confirmations(clauses)]
    refusals = _exec_failures(clauses)
    parts += [_sentence(t) for t in refusals]
    questions = _questions(clauses)
    if questions:                      # at most ONE question per reply
        parts.append(_sentence(questions[0]))
    elif _has_failure(clauses) and not refusals:
        parts.append(_FAILURE)
    return "".join(parts) if parts else _ACK
