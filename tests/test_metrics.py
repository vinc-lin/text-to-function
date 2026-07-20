from eval.metrics import (recall_at_k, multi_intent_set_recall, ood_false_execution_rate,
                          avg_llm_calls, latency_percentiles, e2e_executable_accuracy)

def _rec(expected, ranked, bands, executed, exec_correct=True, typ="single"):
    return {"row": {"expected_functions": expected, "type": typ},
            "ranked_per_clause": ranked, "predicted_functions": [r[0] for r in ranked],
            "bands": bands, "executed": executed, "needs_llm": [b == "medium" for b in bands],
            "exec_correct": [exec_correct]}

def test_recall_at_1_and_3():
    recs = [_rec(["a"], [["a", "b", "c"]], ["high"], [True]),
            _rec(["a"], [["b", "a", "c"]], ["high"], [True])]
    assert recall_at_k(recs, 1) == 0.5
    assert recall_at_k(recs, 3) == 1.0

def test_multi_intent_set_recall():
    recs = [{"row": {"expected_functions": ["a", "b"], "type": "multi_intent"},
             "predicted_functions": ["a", "b"], "ranked_per_clause": [["a"], ["b"]],
             "bands": ["high", "high"], "executed": [True, True], "needs_llm": [False, False],
             "exec_correct": [True, True]}]
    assert multi_intent_set_recall(recs) == 1.0

def test_ood_false_execution():
    recs = [{"row": {"expected_functions": [], "type": "ood"}, "predicted_functions": ["a"],
             "ranked_per_clause": [["a"]], "bands": ["high"], "executed": [True],
             "needs_llm": [False], "exec_correct": [False]}]
    assert ood_false_execution_rate(recs) == 1.0

def test_avg_llm_calls_and_latency():
    recs = [{"row": {"type": "single"}, "bands": ["medium"], "needs_llm": [True],
             "executed": [False], "predicted_functions": ["a"], "ranked_per_clause": [["a"]],
             "exec_correct": [False]}]
    assert avg_llm_calls(recs) == 1.0
    assert latency_percentiles([10, 20, 30, 40], (50, 95))[50] == 25.0
