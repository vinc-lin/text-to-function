"""A Turn becomes the text a person reads. Pure — no pipeline, no car, no I/O.

Every branch here has to produce a line, with one stated exception: a question that IS the
reply is left to the reply line rather than printed twice (see `_outcome_lines`). Otherwise
a span that renders as nothing reads as a bug in the car, and an exception raised while
rendering would kill the session — which costs a 60-second model reload — after the work of
the turn is already done.

A scene turn is the third shape, alongside an error and a routed utterance: it names the
scene and shows what moved, and it has no spans because nothing was recognised — nobody
said anything. Its commonest correct outcome is silence, so the same rule applies with more
force: every rule the engine evaluated gets a line saying what it decided and, if it matched
and still said nothing, what stopped it. Without those lines `scene —` is one blank string
covering a weak detection, a settled signal, a running cooldown and an open question.
"""
from __future__ import annotations

from .session import Turn


def _params(parameters: dict) -> str:
    if not parameters:
        return ""
    inner = ", ".join(f"{k}: {v}" for k, v in parameters.items())
    return "{" + inner + "}"


def _is_the_reply(question: str, reply: str) -> bool:
    """Whether this span's question is exactly what the driver hears.

    Terminators are stripped because compose_reply appends one (t2f/reply.py::_sentence), so
    a question authored without 。 arrives in the reply wearing one and byte equality would
    miss the duplicate.
    """
    return bool(question) and question.rstrip("。！？") == reply.rstrip("。！？")


def _outcome_lines(span, reply: str) -> list[str]:
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
        # The only branch that can say nothing the reply does not. An out-of-scope utterance
        # asks the driver to rephrase and that question is the whole reply, so printing it
        # here too put the same sentence on two lines and read as two separate questions.
        # compose_reply speaks at most ONE question per utterance, so a second span's
        # question exists nowhere else and still needs this line.
        if _is_the_reply(span.detail, reply):
            return []
        return [f"  asked        {span.detail}".rstrip()]
    return ["  unresolved   medium band, no model attached"]


def _rule_line(report) -> str:
    """One evaluated rule: what it decided, why, and what stopped it if anything did.

    The verdict and the reason are always both printed. `match` alone next to a silent reply
    reads as a bug in the car, and `near_miss` alone does not say which observation was weak —
    which is the only fact that tells someone what to type next.
    """
    suppressed = f"  · suppressed: {report.suppressed_by}" if report.suppressed_by else ""
    # Padding plus an explicit separator, not padding alone: `not_applicable` is wider than
    # any column that would still align the common cases, and a value that fills its field
    # exactly ran straight into the reason — "not_applicableno live observation for ...".
    return (f"  rule         {report.rule_id:22s}  {report.verdict:10s}  "
            f"{report.reason}{suppressed}")


def render(turn: Turn) -> str:
    if turn.error:
        return f"  error        {turn.error}\n"
    if turn.scene:
        lines = [f"  scene        {turn.scene}"]
        lines += [_rule_line(r) for r in turn.rules]
        if turn.fallback:
            # Printed verbatim. The engine merges "what the fallback did" and "why it was
            # skipped" into one note on purpose, and the renderer cannot tell which it is
            # holding — guessing a "not consulted" prefix would eventually label a decode
            # that really was spent as one that never happened.
            lines.append(f"  fallback     {turn.fallback}")
        lines += [f"  executed     {d.entity}/{d.attribute}   {d.before} → {d.after}"
                  for d in turn.deltas]
        # Staying quiet is the commonest correct outcome here, so it has to read as a decision
        # rather than as a blank the renderer forgot to fill. An empty reply line looks like a
        # bug in the car; "nothing spoken" looks like the engine declining to speak.
        spoken = turn.reply or "—  (nothing spoken)"
        return "\n".join(lines + [f"  reply        {spoken}"]) + "\n"
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
        lines.extend(_outcome_lines(span, turn.reply))
    lines.append(f"  reply        {turn.reply}")
    return "\n".join(lines) + "\n"
