"""Scene metrics. Every one reports its denominator so a vacuous score is visible.

`scene_false_speech_rate` is the number this design optimises for: the proactive analogue of
`ood_false_execution_rate`, counting the times the system spoke when gold says it should have
kept quiet. A proactive system's worst failure is not being wrong — it is being uninvited.
"""
from __future__ import annotations


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def scene_false_speech_rate(rows) -> float:
    silent = [r for r in rows if not r.get("expect")]
    return _rate(sum(1 for r in silent if r.get("actual")), len(silent))


def scene_recall(rows) -> float:
    should = [r for r in rows if r.get("expect")]
    return _rate(sum(1 for r in should if r.get("actual") == r["expect"]), len(should))


def scene_false_consent_rate(rows) -> float:
    must_not = [r for r in rows if r.get("expect_consent") is False]
    return _rate(sum(1 for r in must_not if r.get("consented")), len(must_not))


def avg_llm_calls_per_event(rows) -> float:
    return _rate(sum(r.get("llm_calls", 0) for r in rows), len(rows))
