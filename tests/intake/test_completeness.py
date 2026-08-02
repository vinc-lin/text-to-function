"""The database is the record — asserted, not claimed.

Design §12.4 names this as the risk the store carries: *"the database is the record" is a
property, not a promise*. A docstring saying every path leaves a row is worth nothing, because
the way it stops being true is silent — a new path, a refactor that moves a write, a branch that
returns early. Nothing raises. The store simply has a hole in it, and the hole is only visible
to whoever goes looking for a turn that was never written.

So every path is driven here and the rows it left are read back from the tables directly, never
through an accessor: the claim is about the record itself, and a helper that read it for us
could be wrong in the same direction as the writer.

**The negative is the one that matters.** `test_nothing_reaches_the_car_without_a_turn` asserts
that no `operation_log` row is untagged after every path has run. An execution that cannot be
traced back to a decision is exactly what this store exists to prevent, and it is not a
hypothetical: the consent path executed with `turn_id` NULL until this file was written, because
consent is decided in `cli/session.py`, ahead of the door that does all the other recording.
`Session._record_consent` closed it. The test is what keeps it closed.

**Two harnesses, for a reason.** The fixture catalog's three cards route deterministically under
`FakeEmbedder` — `tests/cli/test_session.py` already depends on that — so executed, refused and
rejected can each be produced on demand. The scene and consent paths need
`set_window_child_lock`, which only the real catalog has, so those rows are driven against it
exactly as `tests/scene/test_scene_e2e.py` drives them.
"""
from pathlib import Path

import pytest

from cli.session import Session

ROOT = Path(__file__).resolve().parents[2]
FIX = str(ROOT / "tests" / "fixtures" / "catalog")


@pytest.fixture
def routing():
    """The 3-card fixture catalog. Not the real 92 — `FakeEmbedder` has no semantics, and a
    test that asserted an outcome over 92 cards would be asserting the harness."""
    return Session.build(fake=True, llm=False, gate="permissive", catalog=FIX)


@pytest.fixture
def session():
    """The real catalog, because the child-lock rule proposes a card only it has."""
    return Session.build(fake=True, llm=False, gate="permissive")


def _rows(session, table, **where):
    sql = f"SELECT * FROM {table}"
    if where:
        sql += " WHERE " + " AND ".join(f"{k} = ?" for k in where)
    return [dict(r) for r in session.car.conn.execute(sql + " ORDER BY id",
                                                      tuple(where.values()))]


def _turn(session, kind):
    """The one turn of this kind, or a failure naming what was there instead.

    Asserting `len(...) == 1` here rather than taking `[0]` and hoping: a second turn of the
    same kind means the path ran twice, which is the shape of the double-dispatch bug
    `test_an_input_that_came_through_the_door_is_not_left_in_the_queue` guards from the other
    side.
    """
    turns = _rows(session, "turn", kind=kind)
    assert len(turns) == 1, f"expected one {kind} turn, got {[t['kind'] for t in turns]}"
    return turns[0]


# --- what the driver said ---------------------------------------------------------------------

def test_a_routed_utterance_that_executes_leaves_the_whole_chain(routing):
    """Five rows across four tables, and each one answers a different question.

    What arrived (`observation_raw`), what was said (`utterance`), what the turn was and what
    the driver heard (`turn`), what the router decided about it (`decision`), and what the car
    then did (`operation_log`). Any one of them missing leaves a question the store cannot
    answer about a command that moved the vehicle.
    """
    turn = routing.handle("把空调调到25度")
    assert turn.spans[0].outcome == "executed"

    raw = _rows(routing, "observation_raw", source="mic")
    assert len(raw) == 1 and raw[0]["processed_at"] is not None and raw[0]["error"] is None

    said = _rows(routing, "utterance")
    assert [r["text"] for r in said] == ["把空调调到25度"]
    assert said[0]["raw_id"] == raw[0]["id"]

    recorded = _turn(routing, "route")
    assert recorded["raw_id"] == raw[0]["id"]
    assert recorded["reply"] == turn.reply

    decisions = _rows(routing, "decision", turn_id=recorded["id"])
    assert [(d["subject"], d["verdict"], d["chosen"]) for d in decisions] == \
        [("把空调调到25度", "high", "set_temperature")]

    ops = _rows(routing, "operation_log")
    assert [(o["function"], o["outcome"], o["turn_id"]) for o in ops] == \
        [("set_temperature", "executed", recorded["id"])]


def test_a_command_the_car_refuses_is_recorded_as_an_attempt(routing):
    """A refusal is a thing that happened to the vehicle, not an absence of one.

    `operation_log` is the only place "we tried and the car said no" survives, and it is worth
    exactly as much as its `turn_id`: without one, a refused row is an attempt by nobody.
    """
    routing.car.set_device("climate.all", False, "空调控制器离线")
    turn = routing.handle("把空调调到25度")
    assert turn.spans[0].outcome == "refused"

    recorded = _turn(routing, "route")
    ops = _rows(routing, "operation_log")
    assert [(o["outcome"], o["error"], o["turn_id"]) for o in ops] == \
        [("refused", "device_unavailable", recorded["id"])]
    # The cause is on the decision as well as on the operation, and the two are written by
    # different modules — `Intake._why` from the clause, `SqliteExecutor._refuse` from the car.
    # A store that agreed with itself only by accident would drift here first.
    assert "空调控制器离线" in _rows(routing, "decision", turn_id=recorded["id"])[0]["reason"]


def test_a_command_rejected_by_validation_never_reaches_the_car(routing):
    """The path with no `operation_log` row at all, which is the point of testing it.

    Nothing was attempted, so nothing may be logged as attempted — an empty log here is the
    correct record and a row would be a false one. That makes `decision.reason` the ONLY place
    the cause survives: an unusable parameter never reaches the executor, so the fault exists
    nowhere else but the sentence the driver heard.
    """
    turn = routing.handle("把空调调到99度")
    assert turn.spans[0].outcome == "rejected"

    recorded = _turn(routing, "route")
    assert recorded["reply"] == turn.reply
    assert _rows(routing, "operation_log") == [], "validation must not reach the car"

    decision = _rows(routing, "decision", turn_id=recorded["id"])[0]
    assert decision["chosen"] == "set_temperature"
    assert decision["reason"], "the only surviving trace of why this was refused"


def test_an_input_that_raises_is_still_on_the_record(routing, monkeypatch):
    """A crash must not be a gap. The turn opened before the work still exists, and the error
    is written against the row that carried the input — so "nothing happened" and "something
    went wrong" are different states of the store rather than the same absence."""
    monkeypatch.setattr(routing.pipeline, "route",
                        lambda _: (_ for _ in ()).throw(RuntimeError("boom")))
    turn = routing.handle("把空调调到25度")
    assert turn.error and "boom" in turn.error

    raw = _rows(routing, "observation_raw", source="mic")[0]
    assert raw["processed_at"] is not None and "boom" in raw["error"]
    # Opened before the work and closed after it, so a turn that died mid-way still has a row.
    # Recording only the turns that finished records precisely the set nobody investigates.
    recorded = _turn(routing, "route")
    assert recorded["raw_id"] == raw["id"] and recorded["reply"] == ""


# --- what the cameras saw ----------------------------------------------------------------------

def test_a_scene_that_warns_writes_its_turn_and_every_rule_s_verdict(session):
    """Every rule, not only the one that spoke.

    The rules that stayed quiet are how you answer "why did it warn about that and not this",
    and they are gone the moment the next observation arrives — `_last_reports` is overwritten.
    Writing only the winner would keep the one verdict nobody needs explained.
    """
    assert session.observe("rear_occupant", "child", 0.9).reply == \
        "后排有小孩，要打开儿童锁吗？"

    raw = _rows(session, "observation_raw", source="cabin_cam")
    assert len(raw) == 1
    beliefs = _rows(session, "perception")
    assert [(b["key"], b["source"]) for b in beliefs] == [("inside.rear_occupant", "cabin_cam")]

    recorded = _turn(session, "scene")
    assert recorded["raw_id"] == raw[0]["id"]
    assert recorded["reply"] == "后排有小孩，要打开儿童锁吗？"

    decisions = _rows(session, "decision", turn_id=recorded["id"])
    from scene.rules import RULES
    assert {d["subject"] for d in decisions} == {r.id for r in RULES}, \
        "every rule that ran, not only the one that spoke"
    spoke = [d for d in decisions if d["chosen"]]
    assert [(d["subject"], d["verdict"]) for d in spoke] == \
        [("rear_child_window_lock", "match")]
    # A proposal is not an action: the car is asked, not commanded, so nothing is logged yet.
    assert _rows(session, "operation_log") == []


def test_a_scene_that_stays_silent_still_writes_its_turn(session):
    """Silence is the safe default, and a default that leaves no trace is indistinguishable
    from the system being switched off.

    So the turn is written with an empty reply — `''` is a fact, not a missing value — and
    every rule's reason for saying nothing goes with it. This row is the whole answer to the
    question a proactive system gets asked most often: why didn't it say anything?
    """
    assert session.observe("rear_occupant", "child", 0.30).reply == ""

    recorded = _turn(session, "scene")
    assert recorded["reply"] == ""

    decisions = _rows(session, "decision", turn_id=recorded["id"])
    assert decisions, "a silent turn with no decisions cannot explain its own silence"
    assert all(d["chosen"] is None for d in decisions)
    quiet = next(d for d in decisions if d["subject"] == "rear_child_window_lock")
    assert "0.30" in quiet["reason"] and "0.50" in quiet["reason"]


# --- what the driver answered --------------------------------------------------------------------

def test_a_consent_that_executes_is_tied_to_the_words_that_authorised_it(session):
    """The path that was untraceable, and the reason this file exists.

    Consent is resolved in `cli/session.py`, ahead of the door, so none of `intake`'s recording
    ran for it: the child lock opened and `operation_log` said `turn_id` NULL. The chain
    asserted here — operation → consent turn → raw row holding 好 — is what makes an
    uninvited action answerable for itself.
    """
    session.observe("rear_occupant", "child", 0.9)
    assert session.handle("好").reply == "已为您打开车窗儿童锁。"

    recorded = _turn(session, "consent")
    raw = [r for r in _rows(session, "observation_raw", source="mic")
           if r["id"] == recorded["raw_id"]]
    assert len(raw) == 1 and "好" in raw[0]["payload"]
    assert raw[0]["processed_at"] is not None, "a handled row left pending would run twice"
    assert recorded["reply"] == "已为您打开车窗儿童锁。"

    decision = _rows(session, "decision", turn_id=recorded["id"])[0]
    assert (decision["subject"], decision["verdict"], decision["chosen"]) == \
        ("好", "yes", "set_window_child_lock")

    ops = _rows(session, "operation_log")
    assert [(o["function"], o["outcome"], o["turn_id"]) for o in ops] == \
        [("set_window_child_lock", "executed", recorded["id"])]


def test_a_consent_declined_is_recorded_and_touches_nothing(session):
    """A no has to be as visible as a yes.

    Otherwise the record of a declined proposal is identical to the record of a question the
    driver never answered at all — and those differ in the only way that matters, which is
    whether the car was told not to.
    """
    session.observe("rear_occupant", "child", 0.9)
    assert session.handle("不用").reply == "好的。"

    recorded = _turn(session, "consent")
    assert recorded["reply"] == "好的。"
    decision = _rows(session, "decision", turn_id=recorded["id"])[0]
    assert (decision["subject"], decision["verdict"], decision["chosen"]) == ("不用", "no", None)
    assert decision["reason"] == "", "the driver said no; there is nothing to explain"

    assert _rows(session, "operation_log") == []
    assert session.changed_signals() == []


# --- the negative -----------------------------------------------------------------------------

def test_nothing_reaches_the_car_without_a_turn(session, monkeypatch):
    """**The assertion this whole store exists to support.**

    Every other test here says a path leaves the rows it should. This one says the thing that
    cannot be checked path by path: that there is no path which reaches the car and leaves
    nothing. An execution with a NULL `turn_id` is an action in the vehicle's log with no
    recorded reason, no input behind it and nobody to attribute it to — and it is reachable by
    accident, because `executor.execute` is called from four places and only the door writes
    turns.

    Driven as one run against one car so the rows accumulate the way they would in a session:
    a command that acts, a scene that proposes, a consent that acts, the same command now
    refused because of it, a scene that says nothing, and an input that raises.

    The chain is followed all the way down — operation → turn → `observation_raw` — because a
    `turn_id` pointing at a turn with no input is traceability that stops one row short of the
    thing that caused it.
    """
    session.handle("开车窗")                          # routes and acts on a car with no lock
    session.observe("rear_occupant", "child", 0.9)    # proposes, acts on nothing
    session.handle("好")                              # consent, which actuates
    refused = session.handle("开车窗")                # the same words, now blocked by the lock
    assert "车窗儿童锁已开启" in refused.reply
    session.observe("rear_occupant", "child", 0.30)   # a scene that stays silent
    monkeypatch.setattr(session.pipeline, "route",
                        lambda _: (_ for _ in ()).throw(RuntimeError("boom")))
    assert session.handle("开车窗").error

    ops = _rows(session, "operation_log")
    assert ops, "no operations means this test asserts nothing at all"
    assert {o["outcome"] for o in ops} == {"executed", "refused"}, \
        "both halves of what the car can answer"
    assert [o for o in ops if o["turn_id"] is None] == [], \
        "an execution the store cannot explain"

    turns = {t["id"]: t for t in _rows(session, "turn")}
    raw_ids = {r["id"] for r in _rows(session, "observation_raw")}
    for op in ops:
        turn = turns.get(op["turn_id"])
        assert turn is not None, f"{op['function']} points at turn {op['turn_id']}, which is gone"
        assert turn["raw_id"] in raw_ids, \
            f"{op['function']} traces to a turn with no input behind it"

    # Not vacuous on any of the three: a regression that stopped writing consent turns entirely
    # would satisfy every assertion above, because the operation it failed to attribute would
    # not be there either.
    assert {t["kind"] for t in turns.values()} == {"route", "scene", "consent"}


def test_only_the_modules_that_open_a_turn_may_reach_the_executor():
    """The same guard `ui/actions.py` keeps over its two disjoint tables, for the same failure.

    A second path to the car does not get built deliberately; it gets built by a module that
    needs to actuate something and reaches for the seam that actuates. Every caller below is
    covered by a turn — the three in `t2f/` because `Pipeline.route` runs behind
    `Intake._route`, and `scene/engine.py` because `Session._record_consent` wraps the only
    caller of `SceneEngine.resolve`. A fifth is not automatically wrong; it is unrecorded until
    somebody says where its turn comes from, and this test is where that gets said.

    Scoped to the packaged tree (`pyproject.toml` ships these four). `cli/` and `ui/` reach the
    car only through the executor these modules own, which `ui/actions.py`'s own test pins.
    """
    callers = {p.relative_to(ROOT).as_posix()
               for pkg in ("t2f", "scene", "sim", "intake")
               for p in (ROOT / pkg).rglob("*.py")
               if "executor.execute(" in p.read_text(encoding="utf-8")}
    assert callers == {"t2f/plan.py", "t2f/pipeline.py", "scene/engine.py"}
