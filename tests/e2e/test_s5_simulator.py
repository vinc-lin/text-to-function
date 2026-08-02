"""S5 — the four-step business workflow against a SQLite-simulated car.

    utterance → segmented recognition → the operation really happens → the driver is told.

Steps 1-2 are proved elsewhere (tests/e2e/test_s2_recognition.py). What only this file can
prove is step 3 and step 4 *together against a car that can say no*: a signal in the database
moves, or it does not move and the reply carries the vehicle's own reason.

Unlike the rest of tests/e2e/, these use the REAL 92-card catalog, because the point is that
operations change a real vehicle's state, not that routing works on three fixture cards.
Under the `fake` profile the router is `FakeEmbedder`, a hashed-n-gram stand-in with no
semantics, so every utterance below was probed against the full catalog first and kept only
because it reaches the intended function; none of these assertions were relaxed to fit what
routing happened to do. Under `-m model` the same bodies run again on the real embedder, which
is what makes that selection something other than the only witness.

Signal addresses are the ones `sim.mapping.resolve_writes` really produces
(`climate.driver`/`temperature`) — see tests/sim/test_mapping.py.
"""
from __future__ import annotations
import json

from t2f.cards import load_catalog
from t2f.config import Config
from t2f.gate import ConfidenceGate, PERMISSIVE
from t2f.pipeline import Pipeline, DeterministicResolver
from t2f.score import Scorer

from sim.executor import SqliteExecutor
from sim.seed import seed_from_catalog
from sim.vehicle import SqliteVehicle

CARDS = load_catalog("data/catalog")
BY = {c.name: c for c in CARDS}

TEMP25 = "已将主驾温度设置为25°C。"          # what success sounds like, for the refusal cases
AC_OFF = "空调尚未开启"                      # sim/seed.py::_PRECONDITIONS, authored for the driver


def _pipeline(profile):
    """(pipeline, executor) over a freshly seeded car, at the permissive gate — the shipped
    mode of tests/e2e/conftest.py, not a threshold invented here. Nothing else is tuned. The
    embedder and the fusion weights both come from the fixture, and come together: see
    `Profile` in conftest.py for why they cannot be chosen apart."""
    car = SqliteVehicle(":memory:")
    car.init_schema()
    seed_from_catalog(car, CARDS)
    ex = SqliteExecutor(car, BY)
    cfg = Config.default()
    cfg.thresholds = PERMISSIVE
    cfg.weights = dict(profile.weights)           # copied: the profile is session-scoped, and
                                                  # one shared dict would let a debugging edit
                                                  # here leak into every later test in the run
    pipe = Pipeline(CARDS, profile.embedder, Scorer(cfg.weights, cfg.domain_keywords),
                    ConfidenceGate(cfg.thresholds), cfg,
                    resolver=DeterministicResolver(BY, executor=ex))
    return pipe, ex


# --- step 3: the operation actually happens ----------------------------------------------

def test_step3_an_operation_changes_the_car(profile):
    """The whole chain in one line: spoken Chinese moves a row in the vehicle database.

    Asserted as an absolute value, not `after != before` — a test that only demands *some*
    change would pass if the pipeline wrote the wrong temperature.
    """
    pipe, ex = _pipeline(profile)
    before = ex.car.get_signal("climate.driver", "temperature")
    assert before != 25, "the seeded car must not already hold the value under test"

    pipe.route("把主驾温度调到25度")

    assert ex.car.get_signal("climate.driver", "temperature") == 25


def test_step4a_a_successful_operation_is_confirmed_by_name(profile):
    """Step 4's easy half: the confirmation names the action and the value, not just 'OK'."""
    pipe, _ = _pipeline(profile)
    assert pipe.route("把主驾温度调到25度").reply == TEMP25


# --- step 4b: the car refuses, and the driver is told why ---------------------------------

def test_step4b_a_refused_operation_is_not_confirmed(profile):
    """The case this whole build exists for.

    The A/C is off, so the vehicle refuses a temperature change for a reason no amount of
    catalog validation could see: 25 is a perfectly legal value. The driver must hear the
    cause, and must NOT hear the confirmation for something that did not happen.
    """
    pipe, ex = _pipeline(profile)
    ex.car.set_signal("climate.all", "ac_power", False)

    result = pipe.route("把主驾温度调到25度")

    assert AC_OFF in result.reply
    assert TEMP25 not in result.reply
    assert "已将" not in result.reply                      # no confirmation of any kind
    assert all(cr.response is None for cr in result.clauses)


def test_a_refusal_leaves_the_car_untouched(profile):
    """A refusal is not a partial write: the signal the operation targeted is exactly as it
    was, so a later relative command resolves against the real car and not a fiction."""
    pipe, ex = _pipeline(profile)
    ex.car.set_signal("climate.all", "ac_power", False)
    before = ex.car.get_signal("climate.driver", "temperature")

    pipe.route("把主驾温度调到25度")

    assert ex.car.get_signal("climate.driver", "temperature") == before


# --- the operation log: what was attempted, and how it went -------------------------------

def test_every_attempt_reaches_the_operation_log(profile):
    """Both outcomes, in order, with the cause attached to the refusal.

    The log is the only place from which "we tried and the car said no" can be recovered
    after the fact, so it must record the refusal as a refusal — not silently drop it, and
    not record it as executed.
    """
    pipe, ex = _pipeline(profile)
    pipe.route("把主驾温度调到25度")                        # accepted
    ex.car.set_signal("climate.all", "ac_power", False)
    pipe.route("把主驾温度调到25度")                        # same words, now refused

    log = ex.car.recent_operations()                       # newest first
    assert [(r["function"], r["outcome"], r["error"]) for r in log] == [
        ("set_temperature", "refused", "precondition_failed"),
        ("set_temperature", "executed", None)]
    assert log[0]["detail"] == AC_OFF
    assert json.loads(log[0]["parameters"]) == {"temperature": 25.0, "position": "driver"}


# --- the live state layer finally has a producer ------------------------------------------

def test_snapshot_gives_the_live_state_layer_a_producer(profile):
    """`VehicleState.live` was a store nothing ever filled. `SqliteExecutor.snapshot()` is
    its first producer: it reads the car back keyed the way `state_key()` expects, so what
    the state layer believes and what the car holds are the same number."""
    pipe, ex = _pipeline(profile)
    pipe.route("把主驾温度调到25度")

    pipe.state.reset(live=ex.snapshot())

    assert pipe.state.get("set_temperature/driver") == 25
    assert pipe.state.get("set_temperature/driver") == ex.car.get_signal("climate.driver",
                                                                        "temperature")


def test_a_relative_command_resolves_against_the_real_car(profile):
    """Producer to consumer: `StateResolver` turns 调高一点 into an absolute value read out
    of the car, and the car then moves to it.

    18 + one step (10, `Config.default().relative_steps` is empty) = 28, which the seeded
    24 could not produce (24 + 10 clamps to the card's max of 32) — so 28 can only have come
    from the snapshot.

    KNOWN LIMITATION, and why the utterance is a two-clause one: `StateResolver` is reached
    only from the plan path, which `Pipeline.route` takes for multi-intent. A bare
    「主驾温度调高一点」 goes down the legacy single-clause path, never consults state, and
    asks a clarifying question instead. The control below pins that the resolution really is
    state-driven rather than a coincidence of the utterance.
    """
    pipe, ex = _pipeline(profile)
    ex.car.set_signal("climate.driver", "temperature", 18)
    pipe.state.reset(live=ex.snapshot())

    result = pipe.route("开车窗,主驾温度调高一点")

    assert ex.car.get_signal("climate.driver", "temperature") == 28
    assert "已将主驾温度设置为28°C。" in result.reply

    # control: the identical utterance against a car whose state was never snapshotted
    # cannot resolve at all — it is the producer, not the wording, doing the work.
    starved, other = _pipeline(profile)
    other.car.set_signal("climate.driver", "temperature", 18)
    starved_result = starved.route("开车窗,主驾温度调高一点")
    assert other.car.get_signal("climate.driver", "temperature") == 18
    assert "已将主驾温度" not in starved_result.reply
