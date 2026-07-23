from __future__ import annotations
from t2f.pipeline import Pipeline, LLMResolver
from t2f.score import Scorer, EmbeddingOnlyScorer
from t2f.gate import ConfidenceGate
from t2f.classify.source import ClassifierCandidateSource


def build_arm_c(cards, embedder, config) -> Pipeline:
    return Pipeline(cards, embedder, Scorer(config.weights, config.domain_keywords),
                    ConfidenceGate(config.thresholds), config)


def build_arm_c_baseline(cards, embedder, config) -> Pipeline:
    return Pipeline(cards, embedder, EmbeddingOnlyScorer(),
                    ConfidenceGate(config.thresholds), config)


def build_arm_c_llm(cards, embedder, config, llm_client, ood_texts=None) -> Pipeline:
    medium = LLMResolver(llm_client, max_candidates=config.llm.get("max_candidates", 3),
                         max_retries=config.llm.get("max_retries", 1))
    pipe = Pipeline(cards, embedder, Scorer(config.weights, config.domain_keywords),
                    ConfidenceGate(config.thresholds), config, medium_resolver=medium,
                    ood_texts=ood_texts)
    pipe.llm_client = llm_client
    return pipe


def build_arm_d(cards, embedder, config, llm_client, classifier, ood_texts=None) -> Pipeline:
    medium = LLMResolver(llm_client, max_candidates=config.llm.get("max_candidates", 3),
                         max_retries=config.llm.get("max_retries", 1))
    source = ClassifierCandidateSource(classifier, config.classifier.get("topk", 3))
    pipe = Pipeline(cards, embedder, Scorer(config.weights, config.domain_keywords),
                    ConfidenceGate(config.thresholds), config, medium_resolver=medium,
                    classifier_source=source, ood_texts=ood_texts)
    pipe.llm_client = llm_client
    return pipe


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
    llm_json_ok = []
    for cl in res.clauses:
        names = [c.function for c in cl.decision.candidates]
        ranked.append(names)          # retrieval order — recall metrics read this
        top1 = names[0] if names else None
        tc = cl.tool_call
        # the function actually ACTED ON: the executed/validated tool_call (which the LLM may pick
        # from the top-2/3, not necessarily top1), else the retrieval top1. Correctness/params must
        # be scored against this, not top1 — otherwise a wrong LLM execution is mis-scored.
        acted = tc.name if tc is not None else top1
        preds.append(acted)
        bands.append(cl.decision.band.value)
        tcs.append(tc)
        executed.append(tc is not None and cl.response is not None)
        needs.append(cl.needs_llm)
        p = tc.parameters if tc else {}
        params.append(p)
        verrs.append(list(cl.validation_errors))
        ok = (acted in gold) and _params_match(p, exp_params.get(acted))
        exec_ok.append(ok)
        llm_json_ok.append(cl.needs_llm and tc is not None)
    return {"row": row, "ranked_per_clause": ranked, "predicted_functions": preds,
            "bands": bands, "tool_calls": tcs, "executed": executed, "needs_llm": needs,
            "params_per_clause": params, "exec_correct": exec_ok, "val_errors": verrs,
            "llm_json_ok": llm_json_ok,
            "latencies": [cl.latency_ms for cl in res.clauses]}
