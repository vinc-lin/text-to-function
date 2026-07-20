from __future__ import annotations
from t2f.pipeline import Pipeline
from t2f.score import Scorer, EmbeddingOnlyScorer
from t2f.gate import ConfidenceGate


def build_arm_c(cards, embedder, config) -> Pipeline:
    return Pipeline(cards, embedder, Scorer(config.weights, config.domain_keywords),
                    ConfidenceGate(config.thresholds), config)


def build_arm_c_baseline(cards, embedder, config) -> Pipeline:
    return Pipeline(cards, embedder, EmbeddingOnlyScorer(),
                    ConfidenceGate(config.thresholds), config)


def _params_match(got: dict, exp: dict | None) -> bool:
    if not exp:
        return True
    return all(got.get(k) == v for k, v in exp.items())


def predict(pipeline: Pipeline, row: dict) -> dict:
    res = pipeline.route(row["utterance"])
    gold = row.get("expected_functions", [])
    exp_params = row.get("expected_params", {})
    ranked, preds, bands, tcs, executed, needs, params, exec_ok = [], [], [], [], [], [], [], []
    verrs = []
    for cl in res.clauses:
        names = [c.function for c in cl.decision.candidates]
        ranked.append(names)
        top1 = names[0] if names else None
        preds.append(top1)
        bands.append(cl.decision.band.value)
        tcs.append(cl.tool_call)
        executed.append(cl.tool_call is not None and cl.response is not None)
        needs.append(cl.needs_llm)
        p = cl.tool_call.parameters if cl.tool_call else {}
        params.append(p)
        verrs.append(list(cl.validation_errors))
        ok = (top1 in gold) and _params_match(p, exp_params.get(top1))
        exec_ok.append(ok)
    return {"row": row, "ranked_per_clause": ranked, "predicted_functions": preds,
            "bands": bands, "tool_calls": tcs, "executed": executed, "needs_llm": needs,
            "params_per_clause": params, "exec_correct": exec_ok, "val_errors": verrs,
            "latencies": [cl.latency_ms for cl in res.clauses]}
