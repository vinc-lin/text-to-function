"""On/off polarity — the one place the router could act against an instruction.

Two independent defects lived in four lines here. There was no negation awareness at all,
so 别关车窗 closed the window and 别开空调 turned the A/C on; and every _OFF form was tested
before any _ON form regardless of position, so the 关 inside 关窗 beat a leading 打开 and
打开下雨自动关窗 disabled the feature it asked to enable.

Every other defect in this project has been a failure to act or a failure to explain. These
were the system doing the opposite of what it was told, which is why the cases are here in
their own file rather than folded into the feature tests.
"""
import pytest

from t2f.lexical import extract_features


def polarity(utterance: str):
    return extract_features(utterance).on_off


@pytest.mark.parametrize("utterance", [
    "打开车窗", "开一下空调", "开启座椅加热", "启动空调", "把车窗打开",
])
def test_plain_on(utterance):
    assert polarity(utterance) is True


@pytest.mark.parametrize("utterance", [
    "关闭车窗", "关掉空调", "把车窗关上", "关一下车窗", "关车窗",
])
def test_plain_off(utterance):
    assert polarity(utterance) is False


@pytest.mark.parametrize("utterance", [
    "别关车窗", "不要关车窗", "不用关空调", "先别关窗户", "勿关车窗",
    "别开空调", "不要开车窗",
])
def test_negation_yields_unknown_not_the_opposite(utterance):
    """The design call, asserted so it cannot be quietly changed.

    "Don't close the window" does not mean "open it" — the driver may want it left exactly
    as it is. None routes to the missing-parameter question; inverting would be a second way
    to act against what was said.
    """
    assert polarity(utterance) is None


@pytest.mark.parametrize("utterance,expected", [
    ("打开下雨自动关窗", True),      # 关 appears inside the FUNCTION NAME, after the verb
    ("关闭自动落锁", False),
    ("打开自动关窗功能", True),
])
def test_the_leading_verb_governs_not_the_first_table_hit(utterance, expected):
    assert polarity(utterance) is expected


def test_longest_cue_wins_a_tie():
    """打开 and the 开 inside it start at the same index; the longer one is the real cue."""
    assert polarity("打开空调") is True


def test_no_cue_at_all_is_unknown():
    assert polarity("空调") is None
    assert polarity("温度调到25度") is None
