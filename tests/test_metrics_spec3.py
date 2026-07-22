from eval.metrics import coverage

def _rec(typ, executed):
    return {"row": {"type": typ}, "executed": executed, "bands": ["high"] * len(executed),
            "exec_correct": [True] * len(executed)}

def test_coverage_counts_fully_executed_rows():
    recs = [_rec("single", [True]), _rec("single", [False]),
            _rec("multi_intent", [True, False]), _rec("ood", [False])]
    # in-scope = 3 rows (single,single,multi); fully executed = 1
    assert abs(coverage(recs) - (1 / 3)) < 1e-9
