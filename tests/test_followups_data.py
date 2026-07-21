from eval.followups import load_followups
from t2f.cards import load_catalog

def test_followups_wellformed():
    rows = load_followups("data/eval/followups.jsonl")
    assert len(rows) >= 40
    names = {c.name for c in load_catalog("data/catalog")}
    for r in rows:
        assert "initial_utterance" in r and "followup_reply" in r
        if not r.get("new_query"):
            assert r["expected_tool_call"]["name"] in names
    assert sum(1 for r in rows if r.get("new_query")) >= 5
