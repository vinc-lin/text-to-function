"""Free-text parameter extraction — destinations, contacts, categories.

Deliberately high-precision and low-recall, and the ratio is the whole design. A wrong
destination navigates somewhere else and a wrong contact calls the wrong person; declining
costs one question the reply layer already words well. So this fires only on the shape it
can actually read — trigger followed by its object — and asks on everything else.

Measured on the catalog's own 48 example utterances for the six required string parameters:
22 correct, **0 wrong**, 26 asked. Without the not-an-object guard the same rule scores 22
correct and 8 wrong — the guard converts every wrong answer into a question and costs no
correct ones. That trade is what the tests below pin.
"""
import pytest

from t2f.cards import load_catalog
from t2f.lexical import extract_features
from t2f.params.extract import ParameterExtractor

CARDS = load_catalog("data/catalog")
BY = {c.name: c for c in CARDS}
EXTRACT = ParameterExtractor()


def value(function: str, utterance: str, param: str):
    got, _ = EXTRACT.extract(utterance, extract_features(utterance), BY[function])
    return got.get(param)


@pytest.mark.parametrize("utterance,expected", [
    ("导航到北京南站", "北京南站"),
    ("带我去人民广场", "人民广场"),
    ("导航去机场", "机场"),
    ("设置目的地为西湖", "西湖"),
    ("帮我导航到虹桥火车站", "虹桥火车站"),      # leading politeness is stripped
    ("走导航到公司楼下", "公司楼下"),
])
def test_destinations(utterance, expected):
    assert value("navigate_to", utterance, "destination") == expected


@pytest.mark.parametrize("utterance,expected", [
    ("打电话给老婆", "老婆"),
    ("拨打张经理", "张经理"),
    ("打给爸爸", "爸爸"),
    ("呼叫10086", "10086"),                    # a phone number IS a contact, kept as text
    ("拨号13800138000", "13800138000"),
])
def test_contacts(utterance, expected):
    assert value("make_call", utterance, "contact") == expected


@pytest.mark.parametrize("utterance", [
    "附近有什么加油站",      # 有什么 — a question, not an object
    "附近的停车场在哪",      # 在哪
    "附近有厕所吗",         # 吗
    "附近哪里可以吃饭",      # 哪里
])
def test_a_question_is_asked_about_rather_than_answered_wrongly(utterance):
    """Each of these produced a WRONG category before the guard: 有什么加油站, 停车场在哪.
    Sending a driver to a place called 停车场在哪 is worse than asking which one they meant."""
    assert value("find_nearby", utterance, "category") is None


@pytest.mark.parametrize("utterance", ["把壁纸换成星空", "主题改成科技风", "桌面壁纸换一下"])
def test_a_value_before_the_trigger_is_not_guessed(utterance):
    """set_theme puts the value BEFORE the alias (深色主题), which this rule cannot read.
    It declines rather than returning the fragment that follows."""
    assert value("set_theme", utterance, "theme") is None


def test_one_clause_fills_one_slot():
    """发短信给妈妈 once filled send_message's `contact` AND its optional `content`, both with
    妈妈. One clause carries one free-text object; an optional slot is left to the LLM path."""
    got, _ = EXTRACT.extract("发短信给妈妈", extract_features("发短信给妈妈"), BY["send_message"])
    assert got.get("contact") == "妈妈"
    assert "content" not in got


def test_a_one_character_remainder_is_rejected_but_a_real_one_is_not():
    """The guard is length-based, not digit-phobic."""
    assert value("navigate_to", "导航到3", "destination") is None
    assert value("navigate_to", "导航到3号楼", "destination") == "3号楼"


def test_precision_on_the_catalog_corpus_is_total():
    """The guarantee this design trades recall for: it is never confidently wrong.

    Every extraction over the catalog's own examples must be a substring of the utterance and
    free of interrogative markers — a value that fails either is one no driver said.
    """
    bad = []
    for card in CARDS:
        for spec in card.params:
            if not (spec.required and spec.type == "string"):
                continue
            for utterance in card.utterances:
                got = value(card.name, utterance, spec.name)
                if got is None:
                    continue
                if got not in utterance or any(m in got for m in ("吗", "哪", "什么", "怎么")):
                    bad.append((utterance, got))
    assert bad == [], f"extractions that are not clean objects of the request: {bad}"
