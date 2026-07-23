from t2f.actionability import build_alias_index, classify_span
from t2f.lexical import extract_features
from t2f.types import SpanRole
from t2f.cards import load_catalog
from t2f.config import Config

CARDS = load_catalog("data/catalog")
CFG = Config.load("config.yaml")
IDX = build_alias_index(CARDS, CFG.domain_keywords)   # 窗户 comes from domain_keywords, not aliases

def role(text):
    return classify_span(text, extract_features(text), IDX)

def test_context_spans_not_actions():
    for t in ["后排小孩老去按车窗", "副驾说有点热", "后备箱东西很多", "孩子在后面睡觉"]:
        assert role(t) == SpanRole.CONTEXT, t

def test_action_spans_are_actions():
    for t in ["把车窗锁打开", "天窗开到一半", "开空调", "把温度调到22度",
              "主驾这边窗户再开一点", "音量调小一点"]:
        assert role(t) == SpanRole.ACTION, t

def test_connector_only():
    assert role("然后") == SpanRole.CONNECTOR
