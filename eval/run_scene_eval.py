"""Scene evaluation: two arms, four metrics, one row at a time against a real simulated car.

`--fake` costs this evaluation nothing. The scene path uses NO embedder: perception arrives as
structured events, rules are declarative, and consent is an exact-membership test over a closed
lexicon (scene/consent.py). `Session.build(fake=True)` swaps the embedder the ROUTER uses, and
the router is not on the path any of these numbers measure — which is why these figures are
trustworthy in a way the router's own `--fake` numbers are not.

The two arms differ in exactly one thing, the client handed to `Session.build(scene_llm=...)`:

  S      no scene LLM  — what the deterministic rules alone do
  S_llm  TransformersSceneLLM attached — what the constrained fallback adds, and what it costs

The headline is `scene_false_speech_rate`, the proactive analogue of `ood_false_execution_rate`.
"""
from __future__ import annotations
import argparse, json, sys

from cli.session import Session
from eval.dataset import load_dataset
from eval import scene_metrics as M


def _scene_llm(arm: str):
    """Arm S_llm's only difference from arm S. Built lazily and loudly.

    Falling back to `None` here would run arm S and file the result under arm S_llm's name —
    a report claiming a model was evaluated when none was loaded. The caller exits instead.
    """
    if arm != "S_llm":
        return None
    from scene.llm import TransformersSceneLLM
    return TransformersSceneLLM()


def predict(row: dict, *, catalog: str, config_path: str, scene_llm) -> dict:
    """One row, one fresh car.

    A shared session would let rows contaminate each other twice over: the 120s rule cooldown
    from an earlier row would silence a later one, and a child lock left open by `consent_yes`
    would make the rule's Signal condition REJECT for every row after it. Both would be scored
    as correct silence.
    """
    session = Session.build(fake=True, llm=False, gate="permissive",
                            catalog=catalog, config_path=config_path, scene_llm=scene_llm)
    # A delta, because the client is shared across rows (see run()) and its counter is
    # cumulative. Reading it raw would charge row 13 with all twelve preceding decodes and
    # turn a per-row average into a running total.
    before_calls = getattr(session.scene.llm, "calls", 0)
    spoken = ""
    for ev in row.get("events", []):
        reply = session.observe(ev["key"], ev["value"], ev.get("conf", 0.9)).reply
        if reply:
            spoken = reply       # last non-empty wins: a suppressed repeat must not blank it

    consented, routed = False, ""
    if "utterance" in row:
        turn = session.handle(row["utterance"])
        # `turn.scene == "consent"` is exact, and a signal diff is not. The fake embedder
        # misroutes 把窗户关上 into an unrelated function, so "did anything change?" would
        # report false consent on a row where the scene never resolved anything at all.
        consented = turn.scene == "consent"
        # `actual` is what the SCENE said on the utterance turn — the reply when the scene
        # owned the turn, and "" when it did not, because then the words came from the router.
        # Crediting the router's reply here would make the headline metric measure the fake
        # embedder: all four `expect_consent: false` rows get a non-empty routed reply
        # (已为您截屏 for 把窗户关上), and scene_false_speech_rate would read 0.444 without a
        # single uninvited scene sentence having been spoken.
        spoken = turn.reply if consented else ""
        # Recorded, not scored. `consent_oblique_yes` exists to make the closed lexicon's cost
        # countable rather than arguable: 开吧 IS a yes a person would say, and dropping it
        # sends the words to the router instead. This field is where that lands.
        routed = "" if consented else turn.reply

    return {"id": row.get("id", ""),
            "expect": row.get("expect", ""),
            "actual": spoken,
            "routed_reply": routed,
            "expect_consent": row.get("expect_consent"),
            "consented": consented,
            "llm_calls": getattr(session.scene.llm, "calls", 0) - before_calls}


def run(arm: str = "S", dataset: str = "data/eval/scenes.jsonl", catalog: str = "data/catalog",
        config: str = "config.yaml", scene_llm=None) -> dict:
    gold = load_dataset(dataset)
    # One client, shared across rows on purpose: it holds no per-row state, only a call
    # counter, and loading a 0.6B once per row would dominate the run. Everything that COULD
    # leak between rows — cooldowns, pending consent, the car — lives in the session, which is
    # rebuilt per row.
    if arm == "S_llm" and scene_llm is None:
        scene_llm = _scene_llm(arm)
    rows = [predict(r, catalog=catalog, config_path=config, scene_llm=scene_llm) for r in gold]

    metrics = {
        "scene_false_speech_rate": M.scene_false_speech_rate(rows),
        "n_silent_rows": sum(1 for r in rows if not r.get("expect")),
        "scene_recall": M.scene_recall(rows),
        "n_speaking_rows": sum(1 for r in rows if r.get("expect")),
        "scene_false_consent_rate": M.scene_false_consent_rate(rows),
        "n_must_not_consent_rows": sum(1 for r in rows if r.get("expect_consent") is False),
        "avg_llm_calls_per_event": M.avg_llm_calls_per_event(rows),
        "n_rows": len(rows),
    }
    misses = [r["id"] for r in rows if r.get("expect") and r["actual"] != r["expect"]]
    false_speech = [r["id"] for r in rows if not r.get("expect") and r["actual"]]
    false_consent = [r["id"] for r in rows
                     if r.get("expect_consent") is False and r["consented"]]
    report = {"arm": arm, "dataset": dataset, "fake": True, "backend": "none (no embedder)",
              "metrics": metrics,
              "false_speech_rows": false_speech,
              "false_consent_rows": false_consent,
              "recall_misses": misses,
              "rows": rows}
    _print_markdown(report)
    with open(f"eval_report_scene_{arm}.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    return report


def _print_markdown(report: dict) -> None:
    print(f"\n## Scene eval — Arm {report['arm']} (no embedder on this path)\n")
    print("| metric | value |\n|---|---|")
    for k, v in report["metrics"].items():
        print(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |")
    for label, ids in (("false speech", report["false_speech_rows"]),
                       ("false consent", report["false_consent_rows"]),
                       ("recall misses", report["recall_misses"])):
        # Printed even when empty: "which rows failed: none" is a claim, and a silent
        # section leaves the reader to assume it.
        print(f"\n{label}: {', '.join(ids) if ids else 'none'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="S", choices=["S", "S_llm"])
    ap.add_argument("--dataset", default="data/eval/scenes.jsonl")
    ap.add_argument("--catalog", default="data/catalog")
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    try:
        scene_llm = _scene_llm(a.arm)
    except Exception as exc:
        # Loudly, and non-zero. A silent fallback to None would run arm S and file the result
        # under arm S_llm's name — the one failure mode a two-arm comparison cannot survive.
        print(f"Arm {a.arm} could not run on this machine: {type(exc).__name__}: {exc}\n"
              f"It needs transformers + xgrammar (and a working torch install) to load "
              f"TransformersSceneLLM. Refusing to report arm S's numbers under "
              f"arm {a.arm}'s name.", file=sys.stderr)
        sys.exit(1)
    run(a.arm, a.dataset, a.catalog, a.config, scene_llm)


if __name__ == "__main__":
    main()
