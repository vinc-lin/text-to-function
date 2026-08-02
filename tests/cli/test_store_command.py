"""`/store` — the only way to look at what the system wrote down.

Two halves are being tested. `Session.recent_turns` joins the store's turns to the car's
operations, which is the join nothing else in the repo makes; and the typed command prints it
without ever raising out of the loop, because the session holds the car and the models.
"""
import pytest

from cli.__main__ import HELP, _command
from cli.session import Session


@pytest.fixture
def session():
    return Session.build(fake=True, llm=False, gate="permissive")


def _run(session, line):
    return _command(session, line)


def test_the_command_is_in_the_help():
    """Undiscoverable is the same as absent for a command that exists to be typed by hand."""
    assert "/store" in HELP


@pytest.mark.parametrize("line", ["/store x", "/store 0", "/store -3", "/store 1.5"])
def test_a_malformed_argument_prints_usage_and_survives(session, capsys, line):
    """`/store 0` is refused rather than run: `LIMIT 0` prints the same nothing an empty store
    does, and answering "there is no history" to a typo is worse than refusing."""
    assert _run(session, line) is True
    assert "usage:" in capsys.readouterr().out


def test_an_empty_store_says_so_rather_than_printing_nothing(session, capsys):
    """A blank and a broken command must not look the same — the reason /context is never
    blank either. A signal frame is the honest case: it writes a raw row and opens no turn."""
    _run(session, "/signal vehicle.all/speed_kph=45")
    capsys.readouterr()
    _run(session, "/store")
    assert "nothing recorded yet" in capsys.readouterr().out


def test_a_voice_turn_prints_what_was_heard_decided_done_and_said(session, capsys):
    session.handle("开车窗")
    _run(session, "/store")
    out = capsys.readouterr().out
    assert "heard    开车窗" in out
    assert "decided  开车窗" in out and "open_window" in out
    assert "did      open_window  executed" in out
    assert "said     已为您打开当前区域车窗。" in out


def test_silence_is_printed_as_a_decision(session, capsys):
    """The commonest correct outcome of the scene subsystem. A missing line cannot be told
    from a turn that has not finished."""
    session.observe("driver_state", "drowsy", 0.9)
    _run(session, "/store")
    out = capsys.readouterr().out
    assert "said     —  (nothing spoken)" in out
    assert "not_applicable" in out


def test_the_operations_are_joined_to_the_turn_that_caused_them(session):
    """The store owns the turn, the car owns `operation_log`, and `turn_id` is the seam. This
    is the only place the two are put back together."""
    session.observe("rear_occupant", "child", 0.9)
    session.handle("好")
    consent = next(t for t in session.recent_turns() if t["kind"] == "consent")
    assert [op["function"] for op in consent["operations"]] == ["set_window_child_lock"]
    assert consent["operations"][0]["outcome"] == "executed"

    scene = next(t for t in session.recent_turns() if t["kind"] == "scene")
    assert scene["operations"] == [], "the engine asked; it did not act"


def test_a_refusal_keeps_its_cause(session, capsys):
    """"Why did the car do that" is answered by the operation, not by the reply alone: the
    reply says the lock is on, the operation says which check stopped the write."""
    session.observe("rear_occupant", "child", 0.9)
    session.handle("好")
    session.handle("开车窗")
    _run(session, "/store")
    out = capsys.readouterr().out
    assert "did      open_window  refused · precondition_failed · 车窗儿童锁已开启" in out


def test_the_newest_turn_is_first(session):
    session.handle("开车窗")
    session.observe("rear_occupant", "child", 0.9)
    assert [t["kind"] for t in session.recent_turns()][:2] == ["scene", "route"]


def test_a_no_raw_capture_session_records_the_decision_and_not_the_words(session, capsys):
    """The switch has to be visible here or it is not visible anywhere: this command is where
    someone finds out what a `--db` file is holding."""
    quiet = Session.build(fake=True, llm=False, gate="permissive", raw_capture=False)
    quiet.handle("开车窗")
    _run(quiet, "/store")
    out = capsys.readouterr().out
    assert "开车窗" not in out
    assert "heard    —  (not recorded)" in out
    assert "open_window" in out and "已为您打开当前区域车窗。" in out


def test_the_store_outlives_the_words_and_the_command_says_so(session, capsys):
    """An hour on, retention has taken the transcript and the clause text. The turn, the band,
    the function and the reply are still there — which is the whole claim of the raw/parsed
    split, and this is where a person can check it."""
    session.handle("开车窗")
    session.advance_clock(3700.0)
    _run(session, "/store")
    out = capsys.readouterr().out
    assert "开车窗" not in out
    assert "heard    —  (not recorded)" in out
    assert "decided  —" in out and "open_window" in out
