"""Consent is exact membership, never substring. This file is the safety case for that."""
import pytest

from scene.consent import Answer, PendingConsent, classify
from t2f.types import ToolCall


@pytest.mark.parametrize("text", ["好", "好的", "好吧", "可以", "行", "嗯", "是的", "对",
                                  "没问题", "麻烦你了", "好的。", "好！"])
def test_affirmative_forms_are_consent(text):
    assert classify(text) is Answer.YES


@pytest.mark.parametrize("text", ["不用", "不要", "不必", "算了", "不了", "没事", "不需要", "不用。"])
def test_negative_forms_decline(text):
    assert classify(text) is Answer.NO


def test_a_sentence_containing_a_yes_is_not_a_yes():
    """好像有点热 contains 好. A substring test would read it as consent and lock the
    windows because the driver mentioned the temperature. This single assertion is the
    difference between consent and a guess."""
    assert classify("好像有点热") is Answer.NOT_AN_ANSWER


@pytest.mark.parametrize("text", ["把窗户关上", "后排太热了", "导航去公司", "开车窗"])
def test_a_command_is_never_consent(text):
    assert classify(text) is Answer.NOT_AN_ANSWER


def test_an_empty_utterance_is_not_an_answer():
    assert classify("") is Answer.NOT_AN_ANSWER


def test_pending_consent_expires_at_its_own_boundary():
    p = PendingConsent("s", ToolCall("f", {}), asked_at=100.0, expires_after=30.0)
    assert p.is_live(now=130.0)
    assert not p.is_live(now=130.1)
