# tests/test_reply_e2e.py
"""End-to-end: utterance -> route() -> reply, with the deterministic FakeEmbedder
and the reduced fixture catalog. Expected strings are measured, not assumed."""
from pathlib import Path
from t2f.cards import load_catalog
from t2f.embed import FakeEmbedder
from t2f.score import Scorer
from t2f.gate import ConfidenceGate, Thresholds
from t2f.pipeline import Pipeline
from t2f.config import Config

FIX = Path(__file__).parent / "fixtures" / "catalog"

WINDOW = "已为您调整当前区域车窗状态。"
TEMP25 = "已将当前区域温度设置为25°C。"
REJECT = "抱歉，我不太确定您的意思，可以换个说法吗？"


def _pipeline():
    cards = load_catalog(FIX)
    cfg = Config.default()
    cfg.thresholds = Thresholds(high_top1=0.2, high_margin=0.0, low_top1=0.05)
    return Pipeline(cards, FakeEmbedder(256), Scorer(cfg.weights, cfg.domain_keywords),
                    ConfidenceGate(cfg.thresholds), cfg)


def test_e2e_single_intent():                                            # E1
    res = _pipeline().route("把空调调到25度")
    assert res.reply == TEMP25
    assert res.reply == res.clauses[0].response


def test_e2e_two_actions_sentence_joined():                              # E2
    res = _pipeline().route("开车窗,温度调到25度")
    assert res.plan is not None
    assert res.reply == WINDOW + TEMP25
    assert res.reply.count(WINDOW) == 1


def test_e2e_narration_absent_from_reply():                              # E3
    res = _pipeline().route("今天天气怎么样，开车窗")
    assert res.plan is not None
    assert res.reply == WINDOW
    assert "天气" not in res.reply


def test_e2e_low_confidence_reject():                                    # E4
    assert _pipeline().route("今天天气怎么样").reply == REJECT


def test_e2e_partial_failure_one_question():                             # E5
    """开车窗 executes; 温度调高 is LOW-band -> exactly one question, after the confirmation."""
    res = _pipeline().route("开车窗，温度调高")
    assert res.reply == WINDOW + REJECT
    assert res.reply.startswith(WINDOW)
    assert res.reply.count(REJECT) == 1


def test_e2e_reply_is_always_speakable():                                # E6
    p = _pipeline()
    for utterance in ["把空调调到25度", "开车窗,温度调到25度", "今天天气怎么样",
                      "今天天气怎么样，开车窗", "开车窗，温度调高", "外面在下雨，把车窗关上",
                      "", "   ", "。", "，，，"]:
        reply = p.route(utterance).reply
        assert isinstance(reply, str) and reply.strip(), utterance


def test_e2e_every_executed_confirmation_appears():                      # coverage invariant
    p = _pipeline()
    for utterance in ["把空调调到25度", "开车窗,温度调到25度", "外面在下雨，把车窗关上"]:
        res = p.route(utterance)
        for cl in res.clauses:
            if cl.response:
                assert cl.response in res.reply, (utterance, cl.response)


def test_e2e_hard_failure_line():                                        # E8
    """An out-of-range value fails validation and now states its own cause.

    Spec 5 pinned the generic 抱歉，这个操作没能完成。 here, because the cause was computed
    and discarded. It is now spoken, so this test pins the improvement rather than the
    limitation. The generic line still covers causes a driver can do nothing about.
    """
    res = _pipeline().route("把空调调到99度")
    assert res.reply == "目标温度只能设置在16到32度之间。"


def _medium_pipeline():
    """Thresholds that force the MEDIUM band, where arm C has no LLM to resolve the span."""
    cards = load_catalog(FIX)
    cfg = Config.default()
    cfg.thresholds = Thresholds(high_top1=0.9, high_margin=0.5, low_top1=0.01)
    return Pipeline(cards, FakeEmbedder(256), Scorer(cfg.weights, cfg.domain_keywords),
                    ConfidenceGate(cfg.thresholds), cfg)


def test_e2e_medium_band_never_falsely_confirms():                       # E9
    """Regression: a MEDIUM span with no LLM produced response=None/clarification=None/errors=[]
    and used to yield 好的。 — an affirmative reply for work that never happened."""
    p = _medium_pipeline()
    for utterance in ["把空调调到25度", "后排小孩老去按车窗，温度调到25度"]:
        reply = p.route(utterance).reply
        assert reply == "抱歉，这个操作没能完成。", (utterance, reply)


def test_e2e_medium_band_partial_execution_is_honest():                  # E10
    res = _medium_pipeline().route("开车窗，温度调到25度")
    assert res.reply == "已为您调整当前区域车窗状态。抱歉，这个操作没能完成。"


def test_compose_reply_called_exactly_once_per_route(monkeypatch):
    """Guards the single-composition-point invariant: no path may compose twice or zero times."""
    import t2f.pipeline as pipeline_mod
    calls = []
    real = pipeline_mod.compose_reply

    def counting(result):
        calls.append(result)
        return real(result)

    monkeypatch.setattr(pipeline_mod, "compose_reply", counting)
    p = _pipeline()
    for utterance in ["把空调调到25度", "开车窗,温度调到25度", "今天天气怎么样"]:
        calls.clear()
        res = p.route(utterance)
        assert len(calls) == 1, (utterance, len(calls))
        assert calls[0] is res
