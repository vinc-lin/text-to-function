from eval.tools.mine_hard_negatives import mine_confusions, summarize

def test_mine_and_summarize():
    rows = [{"utterance": "u1", "expected_functions": ["set_temperature"], "type": "single"},
            {"utterance": "u2", "expected_functions": ["set_temperature"], "type": "single"},
            {"utterance": "u3", "expected_functions": ["open_window"], "type": "single"}]
    ranked = {"u1": ["set_fan_speed", "set_temperature"],   # gold below distractor
              "u2": ["set_fan_speed", "set_temperature"],
              "u3": ["open_window"]}                          # correct top1, not a confusion
    conf = mine_confusions(rows, lambda u: ranked[u])
    assert len(conf) == 2 and all(c["distractor"] == "set_fan_speed" for c in conf)
    top = summarize(conf)
    assert top[0][0] == ("set_temperature", "set_fan_speed") and top[0][1] == 2
