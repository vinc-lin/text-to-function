from t2f.llm.schema import plan_to_json_schema
from t2f.llm.client import FakePlanClient
from t2f.cards import load_catalog

CARDS = {c.name: c for c in load_catalog("data/catalog")}

def test_plan_schema_is_array_of_actions():
    cards = [CARDS["set_window_child_lock"], CARDS["set_sunroof_position"]]
    schema = plan_to_json_schema(cards, allow_reject=True)
    assert schema["type"] == "object"
    assert schema["properties"]["actions"]["type"] == "array"

def test_fake_plan_client_returns_actions():
    client = FakePlanClient(actions=[
        {"name": "set_window_child_lock", "parameters": {"enabled": True}},
        {"name": "set_sunroof_position", "parameters": {"percent": 50}},
    ])
    out = client.complete_plan("把车窗锁打开，天窗开到一半", [], [])
    assert [a.name for a in out] == ["set_window_child_lock", "set_sunroof_position"]
