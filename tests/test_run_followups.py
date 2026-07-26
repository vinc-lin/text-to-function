from pathlib import Path
from t2f.cards import load_catalog
from research.dialog import FollowUpResolver
from research.dialog import SessionState, PendingState
from t2f.lexical import extract_features
from eval.run_followups import run_followups

FIX = Path(__file__).parent / "fixtures" / "catalog"

def test_followup_completion_via_resolver():
    cards = {c.name: c for c in load_catalog(FIX)}
    R = FollowUpResolver(cards)
    sess = SessionState(pending=PendingState("set_temperature", {"temperature": 25}, ["position"]))
    res, _ = R.resolve(sess, "副驾", extract_features("副驾"))
    got = {"name": res.tool_call.name, "parameters": res.tool_call.parameters} if res.tool_call else None
    assert got == {"name": "set_temperature", "parameters": {"temperature": 25, "position": "passenger"}}

def test_run_followups_completes_via_pending_state():
    cards = {c.name: c for c in load_catalog(FIX)}
    exp = {"name": "set_temperature", "parameters": {"temperature": 25, "position": "passenger"}}
    rows = [{"initial_utterance": "把空调温度调到25度", "missing_param": "position",
             "followup_reply": "副驾", "expected_tool_call": exp}]
    results = run_followups(cards, rows)
    assert results == [{"expected": exp, "got": exp}]
