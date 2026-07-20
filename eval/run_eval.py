from __future__ import annotations
import argparse, json
from t2f.cards import load_catalog
from t2f.config import Config
from t2f.gate import Thresholds, calibrate_gate
from t2f.embed import FakeEmbedder
from eval.dataset import load_dataset
from eval import arms as A
from eval import metrics as M


def _embedder(config, fake: bool):
    if fake:
        return FakeEmbedder(256)
    from t2f.embed import SentenceTransformerEmbedder
    return SentenceTransformerEmbedder(config.model_id, mrl_dim=config.mrl_dim)


def run(arm="C", dataset="data/eval/gold.jsonl", catalog="data/catalog",
        config="config.yaml", fake=False, calibrate=False, permissive=False) -> dict:
    cfg = Config.load(config) if not permissive else Config.default()
    if permissive:
        cfg.thresholds = Thresholds(0.2, 0.0, 0.05)
    cards = load_catalog(catalog)
    embedder = _embedder(cfg, fake)
    rows = load_dataset(dataset)

    build = A.build_arm_c if arm == "C" else A.build_arm_c_baseline
    pipe = build(cards, embedder, cfg)

    if calibrate:
        dev = [r for r in rows if r.get("split") == "dev"]
        if dev:
            def route_top(utt):
                return pipe.route(utt).clauses[0].decision.candidates
            cfg.thresholds = calibrate_gate(dev, route_top)
            pipe = build(cards, embedder, cfg)
        rows = [r for r in rows if r.get("split") != "dev"]  # report on test only

    records = [A.predict(pipe, r) for r in rows]
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
        "incorrect_execution_rate": M.incorrect_execution_rate(records),
        "clarification_rate": M.clarification_rate(records),
        "avg_llm_calls_single": M.avg_llm_calls(records),
        "p50_latency_ms": lp[50], "p95_latency_ms": lp[95],
        "n_rows": len(rows),
    }
    report = {"arm": arm, "dataset": dataset, "fake": fake, "metrics": metrics}
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
    ap.add_argument("--arm", default="C", choices=["C", "baseline"])
    ap.add_argument("--dataset", default="data/eval/gold.jsonl")
    ap.add_argument("--catalog", default="data/catalog")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--permissive", action="store_true")
    a = ap.parse_args()
    run(a.arm, a.dataset, a.catalog, a.config, a.fake, a.calibrate, a.permissive)


if __name__ == "__main__":
    main()
