from eval.metrics import context_false_action_rate


def test_context_false_action_rate():
    recs = [
        {"row": {"type": "context"}, "executed": [False]},
        {"row": {"type": "context"}, "executed": [True]},   # a false action
        {"row": {"type": "single"}, "executed": [True]},    # ignored
    ]
    assert context_false_action_rate(recs) == 0.5
