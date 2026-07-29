"""A Turn becomes the text a person reads. Pure — no pipeline, no car, no I/O.

Every branch here has to produce a line. A span that renders as nothing reads as a bug in
the car, and an exception raised while rendering would kill the session — which costs a
60-second model reload — after the work of the turn is already done.
"""
from __future__ import annotations

from .session import Turn


def _params(parameters: dict) -> str:
    if not parameters:
        return ""
    inner = ", ".join(f"{k}: {v}" for k, v in parameters.items())
    return "{" + inner + "}"


def _outcome_lines(span) -> list[str]:
    if span.outcome == "executed":
        if span.deltas:
            return [f"  executed     {d.entity}/{d.attribute}   {d.before} → {d.after}"
                    for d in span.deltas]
        # "The A/C was already on" and "this function has no state" are different facts and
        # must not read the same. 打开空调 against an on A/C really did succeed.
        return ["  executed     (no change — already at that value)"] if span.writes_signals \
            else ["  executed     (this function holds no state)"]
    if span.outcome == "rejected":
        return [f"  rejected     validation · {span.detail} · never reached the car"]
    if span.outcome == "refused":
        return [f"  refused      vehicle · {span.detail} · nothing changed"]
    if span.outcome == "asked":
        return [f"  asked        {span.detail}"]
    return ["  unresolved   medium band, no model attached"]


def render(turn: Turn) -> str:
    if turn.error:
        return f"  error        {turn.error}\n"
    lines = []
    for span in turn.spans:
        band = f"band={(span.band or '?').upper()}"
        # Two independent guards against the same lie. Session makes `escalated` honest (it
        # means a model actually saw the span, not that one was wanted — NullMediumResolver
        # sets needs_llm with no model attached), and this refuses to claim resolution for a
        # span that resolved nothing. Neither a wrong flag nor a hand-built Turn can print
        # "resolved by LLM" above "unresolved".
        if span.escalated and span.band == "medium" and span.outcome != "unresolved":
            band += "  → resolved by LLM"
        lines.append(f"  recognised   {span.function or '—'}{_params(span.parameters)}    {band}")
        lines.extend(_outcome_lines(span))
    lines.append(f"  reply        {turn.reply}")
    return "\n".join(lines) + "\n"
