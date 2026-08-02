"""A Session turns one utterance into a Turn. No I/O, so it is testable like anything else."""
from pathlib import Path

import pytest

from cli.session import Session, Turn

FIX = str(Path(__file__).parent.parent / "fixtures" / "catalog")


@pytest.fixture
def session():
    """Fake embedder + permissive gate + the 3-card FIXTURE catalog.

    NOT the real 92-card catalog: FakeEmbedder has no semantics and misroutes badly over 92
    cards, so a unit test on it would be asserting the harness, not the session.
    """
    return Session.build(fake=True, llm=False, gate="permissive", catalog=FIX)


def test_an_executed_turn_reports_the_signal_change(session):
    turn = session.handle("把空调调到25度")
    assert turn.reply == "已将当前区域温度设置为25°C。"
    assert len(turn.spans) == 1
    span = turn.spans[0]
    assert span.function == "set_temperature"
    assert span.outcome == "executed"
    assert ("climate.all", "temperature") in {(d.entity, d.attribute) for d in span.deltas}


def test_a_validation_failure_never_reaches_the_car(session):
    turn = session.handle("把空调调到99度")
    span = turn.spans[0]
    assert span.outcome == "rejected"
    assert span.deltas == []
    assert "16" in turn.reply and "32" in turn.reply


def test_state_persists_across_turns(session):
    session.handle("把空调调到25度")
    assert session.car.get_signal("climate.all", "temperature") == 25


def test_reset_restores_the_seeded_car(session):
    session.handle("把空调调到25度")
    session.reset()
    assert session.car.get_signal("climate.all", "temperature") != 25


def test_changed_signals_reports_only_what_moved(session):
    assert session.changed_signals() == []
    session.handle("把空调调到25度")
    assert any(a == "temperature" for _, a, _ in session.changed_signals())


def test_a_raising_turn_is_caught_and_reported(session, monkeypatch):
    """A crash costs a 60-second model reload; the session must survive one bad turn."""
    monkeypatch.setattr(session.pipeline, "route",
                        lambda _: (_ for _ in ()).throw(RuntimeError("boom")))
    turn = session.handle("把空调调到25度")
    assert turn.error is not None and "boom" in turn.error
    assert turn.spans == []


def test_the_gate_switch_actually_changes_the_thresholds():
    """The switch must reach the gate, not just the label — a mode indicator that lies is
    worse than no switch at all."""
    session = Session.build(fake=True, llm=False, gate="permissive", catalog=FIX)
    loose = session.pipeline.gate.t
    session.rebuild(gate="shipped")
    assert session.pipeline.gate.t.high_margin > loose.high_margin
    assert "shipped" in session.mode_label()


def test_switching_a_mode_keeps_the_car():
    """The point of the switch is typing the same words twice against the SAME vehicle."""
    session = Session.build(fake=True, llm=False, gate="permissive", catalog=FIX)
    session.handle("把空调调到25度")
    before = session.car.get_signal("climate.all", "temperature")
    session.rebuild(gate="shipped")
    assert session.car.get_signal("climate.all", "temperature") == before


def test_mode_label_states_every_switch():
    s = Session.build(fake=True, llm=False, gate="shipped", catalog=FIX)
    assert "shipped" in s.mode_label() and "FAKE" in s.mode_label() and "C_llm" not in s.mode_label()


def test_escalated_means_a_model_actually_saw_it():
    """NullMediumResolver sets needs_llm=True with no model attached, so the raw flag means
    "wanted a model", not "got one". Rendering the difference wrongly told the driver
    "resolved by LLM" on a session with /llm off."""
    session = Session.build(fake=True, llm=False, gate="shipped", catalog=FIX)
    turn = session.handle("温度调高一点")
    assert all(not span.escalated for span in turn.spans)


# --- the privacy switch, end to end ----------------------------------------------------------

def _rows(session, sql):
    return session.car.conn.execute(sql).fetchall()


def test_no_raw_capture_records_the_decision_and_never_the_words():
    """The whole switch, through the real door. What was decided survives; what was said does
    not exist anywhere in the file."""
    session = Session.build(fake=True, llm=False, gate="permissive", catalog=FIX,
                            raw_capture=False)
    turn = session.handle("把空调调到25度")
    assert turn.reply == "已将当前区域温度设置为25°C。"       # the car still answers

    raw = _rows(session, "SELECT * FROM observation_raw WHERE source='mic'")
    assert len(raw) == 1, "the fact that something arrived is still a fact"
    assert raw[0]["payload"] == ""
    assert _rows(session, "SELECT text FROM utterance")[0]["text"] is None
    decisions = _rows(session, "SELECT * FROM decision d JOIN turn t ON t.id = d.turn_id "
                               "WHERE t.kind = 'route'")
    assert decisions and all(d["subject"] == "" for d in decisions)
    # And the parse is all still there: which function, at what band, and what was said back.
    assert {d["chosen"] for d in decisions} == {"set_temperature"}
    assert _rows(session, "SELECT reply FROM turn")[0]["reply"] == "已将当前区域温度设置为25°C。"
    # No column anywhere holds the sentence.
    for table, column in (("observation_raw", "payload"), ("utterance", "text"),
                          ("decision", "subject"), ("turn", "reply")):
        found = _rows(session, f"SELECT {column} AS c FROM {table}")
        assert not any((r["c"] or "").startswith("把空调") for r in found), f"{table}.{column}"


def test_capture_on_writes_the_words_down():
    """The other half: the default is a store that CAN answer what was said."""
    session = Session.build(fake=True, llm=False, gate="permissive", catalog=FIX)
    session.handle("把空调调到25度")
    assert _rows(session, "SELECT payload FROM observation_raw WHERE source='mic'"
                 )[0]["payload"] == '{"text": "把空调调到25度"}'
    assert _rows(session, "SELECT text FROM utterance")[0]["text"] == "把空调调到25度"


def test_the_switch_is_on_the_prompt_when_it_is_off():
    """A store with no transcripts in it must be explicable as a setting rather than a fault."""
    off = Session.build(fake=True, llm=False, gate="permissive", catalog=FIX, raw_capture=False)
    on = Session.build(fake=True, llm=False, gate="permissive", catalog=FIX)
    assert "no-raw" in off.mode_label() and "no-raw" not in on.mode_label()


def test_a_perception_survives_a_session_with_no_raw_capture():
    """The parse is stored. A belief is the parse."""
    session = Session.build(fake=True, llm=False, gate="permissive", catalog=FIX,
                            raw_capture=False)
    session.observe("rear_occupant", "child")
    assert session.scene.context.get("inside.rear_occupant", session._now()).value == "child"
