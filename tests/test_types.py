from t2f.types import (ParamSpec, FunctionCard, Candidate, Band, Decision,
                       ToolCall, ValidationError, LexFeatures)

def test_functioncard_param_lookup():
    card = FunctionCard(
        name="set_temperature", domain="climate", description="set AC temp",
        params=[ParamSpec(name="temperature", type="number", required=True,
                          minimum=16, maximum=32, unit="celsius"),
                ParamSpec(name="position", type="enum", enum=["driver", "passenger"])])
    assert card.param("temperature").maximum == 32
    assert set(card.param_names) == {"temperature", "position"}
    assert card.param("missing") is None

def test_band_and_lexfeatures_defaults():
    assert Band.HIGH.value == "high"
    f = LexFeatures()
    assert f.numbers == [] and f.positions == [] and f.on_off is None
