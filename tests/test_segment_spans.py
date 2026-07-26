from t2f.segment import split, segment
from t2f.types import SpanRole
from t2f.cards import load_catalog

CARDS = load_catalog("data/catalog")

def test_split_backcompat():
    assert split("开空调，音量调小一点") == ["开空调", "音量调小一点"]

def test_segment_labels_and_attaches_context():
    spans = segment("后排小孩老去按车窗，把车窗锁打开", CARDS)
    roles = [(s.text, s.role) for s in spans]
    actions = [s for s in spans if s.role == SpanRole.ACTION]
    assert len(actions) == 1 and actions[0].text == "把车窗锁打开"

def test_segment_multi_action():
    spans = segment("把车窗锁打开，天窗开到一半", CARDS)
    assert [s.role for s in spans] == [SpanRole.ACTION, SpanRole.ACTION]
