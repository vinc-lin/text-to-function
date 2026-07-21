from __future__ import annotations
from t2f.dialog import FollowUpResolver
from t2f.types import SessionState, PendingState
from t2f.lexical import extract_features
from t2f.normalize import normalize


def run_followups(cards_by_name, rows, llm_client=None, max_turns=2):
    R = FollowUpResolver(cards_by_name, llm_client=llm_client, max_turns=max_turns)
    results = []
    for row in rows:
        if row.get("new_query"):
            results.append({"expected": None, "got": None, "new_query": True})
            continue
        exp = row["expected_tool_call"]
        known = {k: v for k, v in exp["parameters"].items() if k != row["missing_param"]}
        sess = SessionState(pending=PendingState(exp["name"], known, [row["missing_param"]]))
        reply = normalize(row["followup_reply"])
        res, _ = R.resolve(sess, reply, extract_features(reply))
        got = {"name": res.tool_call.name, "parameters": res.tool_call.parameters} if res.tool_call else None
        results.append({"expected": exp, "got": got})
    return results
