from __future__ import annotations
import argparse, json
from pathlib import Path
from t2f.cards import load_catalog, load_ood_prototypes
from t2f.config import Config
from t2f.gate import Thresholds, ConfidenceGate
from eval.calibrate import calibrate_gate
from t2f.embed import FakeEmbedder
from t2f.types import LLMResult, ToolCall
from t2f.llm.client import FakeLLMClient, TransformersXGrammarClient
from t2f.classify.classifiers import CharNgramLRClassifier
from eval.dataset import load_dataset
from eval import arms as A
from eval import metrics as M


def _embedder(config, fake: bool, backend: str = "transformers"):
    if fake:
        return FakeEmbedder(256)
    if backend == "st":
        from t2f.embed import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder(config.model_id, mrl_dim=config.mrl_dim)
    from t2f.embed import TransformersEmbedder
    return TransformersEmbedder(config.model_id, mrl_dim=config.mrl_dim)


def _llm_client(config, fake: bool, permissive: bool):
    if fake or permissive:
        return FakeLLMClient(default=LLMResult(tool_call=ToolCall(name="noop", parameters={})))
    return TransformersXGrammarClient(model_id=config.llm.get("model_id", "Qwen/Qwen3-0.6B"),
                                      max_new_tokens=config.llm.get("max_new_tokens", 128))


def _classifier(config):
    """Load the char-ngram classifier for Arm D; skip (return None) if it hasn't been trained yet."""
    path = config.classifier.get("char_ngram_path")
    if not path or not Path(path).exists():
        return None
    return CharNgramLRClassifier.load(path)


def run(arm="C", dataset="data/eval/gold.jsonl", catalog="data/catalog",
        config="config.yaml", fake=False, calibrate=False, permissive=False,
        backend="transformers", embedder=None) -> dict:
    cfg = Config.load(config) if not permissive else Config.default()
    if permissive:
        cfg.thresholds = Thresholds(0.2, 0.0, 0.05)
    cards = load_catalog(catalog)
    if embedder is None:
        embedder = _embedder(cfg, fake, backend)
    rows = load_dataset(dataset)
    ctx_path = Path("data/eval/context_negatives.jsonl")
    ctx_rows = load_dataset(ctx_path) if ctx_path.exists() else []
    # Scored in its own file, never appended to gold: eight metrics plus the latency
    # percentiles have no row-type filter, so extending gold.jsonl would move numbers
    # throughout RESULTS.md.
    e2e_path = Path("data/eval/e2e_cases.jsonl")
    e2e_rows = load_dataset(e2e_path) if e2e_path.exists() else []

    if arm == "C":
        pipe = A.build_arm_c(cards, embedder, cfg)  # builds the prototype store once
    elif arm == "baseline":
        pipe = A.build_arm_c_baseline(cards, embedder, cfg)
    elif arm == "C_llm":
        ood = load_ood_prototypes(cfg.ood_prototypes)
        pipe = A.build_arm_c_llm(cards, embedder, cfg, _llm_client(cfg, fake, permissive), ood_texts=ood)
    elif arm == "D":
        ood = load_ood_prototypes(cfg.ood_prototypes)
        classifier = _classifier(cfg)
        client = _llm_client(cfg, fake, permissive)
        if classifier is not None:
            pipe = A.build_arm_d(cards, embedder, cfg, client, classifier, ood_texts=ood)
        else:  # classifier not trained yet — fall back to C+LLM (no candidate augmentation)
            pipe = A.build_arm_c_llm(cards, embedder, cfg, client, ood_texts=ood)
    else:
        raise ValueError(f"unknown arm: {arm}")

    if calibrate:
        dev = [r for r in rows if r.get("split") == "dev"]
        if dev:
            def route_top(utt):
                return pipe.route(utt).clauses[0].decision.candidates
            cfg.thresholds = calibrate_gate(dev, route_top, execute_medium=(arm in ("C_llm", "D")))
            # swap only the gate — do NOT rebuild the pipeline (avoids re-encoding prototypes)
            pipe.gate = ConfidenceGate(cfg.thresholds)
        rows = [r for r in rows if r.get("split") != "dev"]  # report on test only
        ctx_rows = [r for r in ctx_rows if r.get("split") != "dev"]
        e2e_rows = [r for r in e2e_rows if r.get("split") != "dev"]

    records = [A.predict(pipe, r) for r in rows]
    ctx_records = [A.predict(pipe, r) for r in ctx_rows]
    e2e_records = [A.predict(pipe, r) for r in e2e_rows]
    latencies = [lat for rec in records for lat in rec["latencies"]]
    lp = M.latency_percentiles(latencies, (50, 95))
    metrics = {
        "recall@1": M.recall_at_k(records, 1),
        "recall@3": M.recall_at_k(records, 3),
        "multi_intent_set_recall": M.multi_intent_set_recall(records),
        "param_exact_match": M.param_exact_match(records),
        "schema_valid_rate": M.schema_valid_rate(records),
        "e2e_deterministic": M.e2e_executable_accuracy(records, "deterministic"),
        "e2e_ceiling": M.e2e_executable_accuracy(records, "ceiling"),
        "ood_false_execution_rate": M.ood_false_execution_rate(records),
        "context_false_action_rate": M.context_false_action_rate(ctx_records),
        "n_context_rows": len(ctx_records),
        "incorrect_execution_rate": M.incorrect_execution_rate(records),
        "clarification_rate": M.clarification_rate(records),
        "avg_llm_calls_single": M.avg_llm_calls(records),
        "json_valid_rate": M.json_valid_rate(records),
        "reply_action_coverage": M.reply_action_coverage(records),
        "reply_single_question": M.reply_single_question(records),
        "reply_nonempty_rate": M.reply_nonempty_rate(records),
        "reply_question_drop_rate": M.reply_question_drop_rate(records),
        # e2e slices — scored over their own file. reply_* span both files because an
        # expected_reply may be annotated on a gold row too; they are scoped by the
        # presence of the field, not by which file the row came from.
        "invalid_no_execution_rate": M.invalid_no_execution_rate(e2e_records),
        "reply_exact_match": M.reply_exact_match(records + e2e_records),
        "n_reply_annotated": M.n_reply_annotated(records + e2e_records),
        "n_e2e_rows": len(e2e_records),
        "candidate_gen_recall@3": M.candidate_gen_recall(records, 3),
        "p50_latency_ms": lp[50], "p95_latency_ms": lp[95],
        "n_rows": len(rows),
    }
    report = {"arm": arm, "dataset": dataset, "fake": fake,
              "backend": "fake" if fake else backend,
              "thresholds": vars(cfg.thresholds), "metrics": metrics}
    _print_markdown(report)
    with open(f"eval_report_{arm}.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    return report


def _print_markdown(report: dict) -> None:
    print(f"\n## Eval — Arm {report['arm']} ({'fake-emb' if report['fake'] else 'real-emb'})\n")
    print("| metric | value |\n|---|---|")
    for k, v in report["metrics"].items():
        print(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="C", choices=["C", "baseline", "C_llm", "D"])
    ap.add_argument("--dataset", default="data/eval/gold.jsonl")
    ap.add_argument("--catalog", default="data/catalog")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--permissive", action="store_true")
    ap.add_argument("--backend", default="transformers", choices=["transformers", "st"])
    a = ap.parse_args()
    run(a.arm, a.dataset, a.catalog, a.config, a.fake, a.calibrate, a.permissive, a.backend)


if __name__ == "__main__":
    main()
