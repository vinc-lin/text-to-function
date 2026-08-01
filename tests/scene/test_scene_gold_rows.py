"""The gold rows that need a world, not just a camera.

`animal_ahead` conditions on the car MOVING, and no observation can express that — speed is a
signal the car senses, not something a camera says. So the row format grew `signals`, `bus` and
`clock`, and these tests are what stop those three fields becoming decoration: a runner that
silently ignored them would score every animal row against a parked car and report a tidy
0.0000 while measuring nothing.
"""
import json

import pytest

from eval.run_scene_eval import predict

GOLD = "data/eval/scenes.jsonl"


def _row(row_id):
    for line in open(GOLD, encoding="utf-8"):
        if line.strip() and json.loads(line).get("id") == row_id:
            return json.loads(line)
    raise AssertionError(f"no gold row {row_id!r}")


def _predict(row_id):
    return predict(_row(row_id), catalog="data/catalog", config_path="config.yaml",
                   scene_llm=None)


@pytest.mark.parametrize("row_id", [
    "animal_moving", "animal_stopped", "animal_walking_pace", "animal_stale_bus",
    "animal_near_miss", "animal_too_weak", "not_an_animal",
])
def test_every_animal_row_matches_its_gold(row_id):
    out = _predict(row_id)
    assert out["actual"] == out["expect"], row_id


def test_the_moving_row_actually_warns():
    """The one animal row that speaks. If `signals` were ignored the car would be parked, this
    would fall silent, and the suite would still be green on six of seven rows — which is how a
    rule with no coverage looks exactly like a rule that works."""
    assert _predict("animal_moving")["actual"] == "前方有动物，请注意。"


def test_the_stopped_row_is_silent_for_the_right_reason():
    """Not merely silent — silent because the car is not moving. A row that fell silent because
    `signals` never applied would pass this too, which is why the moving row above is the one
    carrying the weight."""
    assert _predict("animal_stopped")["actual"] == ""


def test_the_stale_bus_row_gives_staleness_its_first_measurement():
    """Speed set, bus stopped, clock past the 2s max age. Until this row existed the freshness
    discipline was tested but never *measured* — no eval number moved if it stopped working."""
    row = _row("animal_stale_bus")
    assert row["bus"] is False and row["clock"] > 2.0
    assert _predict("animal_stale_bus")["actual"] == ""


def test_both_shipped_rules_are_exercised_by_gold():
    """The gap this file closes. The rule set doubled and the gold did not, so scene_recall
    1.000 was measured over rows that touched one of the two rules."""
    from scene.rules import RULES
    # Normalised the way Session.observe does it: a key with no namespace gets the cabin, so
    # gold's bare `rear_occupant` is the rule's `inside.rear_occupant`. Comparing the raw
    # strings would report the child-lock rule as uncovered when thirteen rows exercise it.
    keys = {k if "." in k else f"inside.{k}"
            for line in open(GOLD, encoding="utf-8") if line.strip()
            for k in (e["key"] for e in json.loads(line).get("events", []))}
    for rule in RULES:
        assert any(k in keys for k in rule.observed_keys), \
            f"{rule.id} has no gold row exercising it"
