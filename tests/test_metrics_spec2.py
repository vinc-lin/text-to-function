from eval.metrics import candidate_gen_recall, clarification_followup_success

def test_candidate_gen_recall():
    recs = [{"row": {"type": "single", "expected_functions": ["a"]}, "ranked_per_clause": [["b", "a", "c"]]},
            {"row": {"type": "single", "expected_functions": ["a"]}, "ranked_per_clause": [["b", "c", "d"]]}]
    assert candidate_gen_recall(recs, 3) == 0.5

def test_clarification_followup_success():
    results = [{"expected": {"name": "f", "parameters": {"x": 1}}, "got": {"name": "f", "parameters": {"x": 1}}},
               {"expected": {"name": "f", "parameters": {"x": 1}}, "got": None}]
    assert clarification_followup_success(results) == 0.5
