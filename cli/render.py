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
        return ["  executed     (no signal for this function)"]
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
        # `escalated` means "the medium band handed this over", which the null resolver also
        # sets when there is no model to hand it to. Claiming the model resolved a span that
        # ended `unresolved` would contradict that span's own next line.
        if span.escalated and span.band == "medium" and span.outcome != "unresolved":
            band += "  → resolved by LLM"
        lines.append(f"  recognised   {span.function or '—'}{_params(span.parameters)}    {band}")
        lines.extend(_outcome_lines(span))
    lines.append(f"  reply        {turn.reply}")
    return "\n".join(lines) + "\n"
