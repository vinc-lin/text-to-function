from __future__ import annotations
import numpy as np


def _single_clause_rows(records):
    return [r for r in records if len(r["ranked_per_clause"]) == 1]


def recall_at_k(records, k: int) -> float:
    rows = [r for r in records if r["row"].get("type") in ("single", "ambiguous")
            and r["row"].get("expected_functions")]
    if not rows:
        return 0.0
    hit = 0
    for r in rows:
        gold = r["row"]["expected_functions"][0]
        if gold in r["ranked_per_clause"][0][:k]:
            hit += 1
    return hit / len(rows)


def multi_intent_set_recall(records) -> float:
    rows = [r for r in records if r["row"].get("type") == "multi_intent"]
    if not rows:
        return 0.0
    tot = 0.0
    for r in rows:
        gold = set(r["row"]["expected_functions"])
        pred = set(r["predicted_functions"])
        tot += len(gold & pred) / len(gold)
    return tot / len(rows)


def param_exact_match(records) -> float:
    rows, hit = 0, 0
    for r in records:
        exp = r["row"].get("expected_params") or {}
        for i, fn in enumerate(r["predicted_functions"]):
            if fn in exp:
                rows += 1
                got = r.get("params_per_clause", [{}] * len(r["predicted_functions"]))[i]
                if got == exp[fn]:
                    hit += 1
    return hit / rows if rows else 0.0


def schema_valid_rate(records) -> float:
    """Fraction of HIGH-band (attempted-execution) clauses that produced a valid tool_call.

    Denominator = number of clauses whose band is "high" (i.e. attempted execution).
    Numerator = those high-band clauses whose tool_calls[i] is not None.
    Returns 1.0 when there are no high-band clauses (vacuously valid).
    Tolerates records missing a "tool_calls" key (treated as []).
    """
    denom = 0
    numer = 0
    for r in records:
        tool_calls = r.get("tool_calls", [])
        for i, band in enumerate(r["bands"]):
            if band == "high":
                denom += 1
                tc = tool_calls[i] if i < len(tool_calls) else None
                if tc is not None:
                    numer += 1
    return numer / denom if denom else 1.0


def e2e_executable_accuracy(records, mode: str = "deterministic") -> float:
    rows = [r for r in records if r["row"].get("type") in ("single", "multi_intent")]
    if not rows:
        return 0.0
    ok = 0
    for r in rows:
        clause_ok = []
        for i, band in enumerate(r["bands"]):
            correct = r["exec_correct"][i]
            if mode == "deterministic":
                clause_ok.append(band == "high" and correct)
            else:  # ceiling: medium credited if gold in top-3
                if band == "high":
                    clause_ok.append(correct)
                elif band == "medium":
                    gold = r["row"]["expected_functions"]
                    clause_ok.append(any(g in r["ranked_per_clause"][i][:3] for g in gold))
                else:
                    clause_ok.append(False)
        if clause_ok and all(clause_ok):
            ok += 1
    return ok / len(rows)


def ood_false_execution_rate(records) -> float:
    rows = [r for r in records if r["row"].get("type") == "ood"]
    if not rows:
        return 0.0
    bad = sum(1 for r in rows if any(r["executed"]))
    return bad / len(rows)


def incorrect_execution_rate(records) -> float:
    executed_clauses = [(r, i) for r in records for i, e in enumerate(r["executed"]) if e]
    if not executed_clauses:
        return 0.0
    wrong = sum(1 for r, i in executed_clauses if not r["exec_correct"][i])
    return wrong / len(executed_clauses)


def clarification_rate(records) -> float:
    total = sum(len(r["bands"]) for r in records)
    clar = sum(1 for r in records for b in r["bands"] if b == "low")
    return clar / total if total else 0.0


def avg_llm_calls(records) -> float:
    single = [r for r in records if r["row"].get("type") == "single"]
    if not single:
        return 0.0
    return sum(sum(1 for n in r["needs_llm"] if n) for r in single) / len(single)


def latency_percentiles(latencies, ps=(50, 95)) -> dict:
    if not latencies:
        return {p: 0.0 for p in ps}
    arr = np.array(latencies, dtype=float)
    return {p: float(np.percentile(arr, p, method="linear")) for p in ps}
