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
