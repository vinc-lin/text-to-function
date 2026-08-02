"""A world that writes rows: plausible input for all three sources, from a seed.

A `Scenario` says what happens over a span of simulated time — what the car is doing, what the
cameras see, what the driver says and when — and `frames()` turns it into the rows a producer
would have written. Nothing here parses anything: the rows go into `observation_raw` and
`Intake.process_pending` does the rest, so a simulated input and a real one travel the identical
dispatch. A generator that wrote `perception` or `turn` rows directly would be exercising a path
nothing else in the system uses, and every number measured over it would be about the generator.

**Generated data encodes our beliefs, not the world.** Every confidence, every arrival delay and
every noise figure below is a guess by the people who wrote the rules being tested — so a
scenario that fires a rule is evidence that our model of a camera agrees with our model of a
child in a back seat, and nothing more. Twenty hand-written rows in `data/eval/scenes.jsonl`
carry that caveat visibly; ten thousand generated ones do not, and they read as measurement. Any
figure computed over this module's output is labelled agreement-with-our-own-model or it is
misleading. `GOLD_SCENARIOS` exists so that at least the generator itself is checked against
something a human agreed to: `tests/sim/test_environment.py` drives each one through
`process_pending` and requires the outcome the hand-authored row states.

**Deterministic given a seed, via `random.Random(seed)` and never the global RNG.** A failure at
volume is worth nothing if it cannot be run again, and a module that reached for `random.random()`
would make every run of a suite a different world — including the runs that passed.

**Sources are spelled here, not imported.** `intake/sources.py` is the registry, but `sim` does
not import `intake` (that edge runs the other way, and inverting it is how the composition root
stops being one). So is the payload field naming: a `Frame`'s dict has to match what
`intake.envelope` declares, and nothing in this module can check that. What keeps the two in step
is `test_every_generated_row_is_a_valid_input`, which decodes every generated row back through
`intake.ingest.input_from_row` — the same call `process_pending` makes, so a drift here fails
there rather than in production.
"""
from __future__ import annotations
import json
import random
from dataclasses import dataclass
from typing import Any, Optional

# The declared source names, from `intake/sources.py`. See the module docstring for why they are
# copied rather than imported, and for what stops the copy from rotting.
BUS = "can0"
CABIN_CAM = "cabin_cam"
FRONT_CAM = "front_cam"
MIC = "mic"

# The one signal a scene rule conditions on and no function writes. It must be a signal the car
# declares SENSED — an actuated signal never goes stale, so publishing a bus frame for one would
# model a decay that does not exist. `test_the_bus_publishes_a_signal_the_car_declares_sensed`
# checks this pair against `sim.seed.sensed_signals`, which is the declaration that owns it.
SPEED = ("vehicle.all", "speed_kph")

# --- how much the world is allowed to wobble ------------------------------------------------
#
# Three delays, because three sources are late for three different reasons and by three
# different amounts. All ONE-SIDED: a reading arrives after the thing it measures, never before,
# and a symmetric jitter would generate rows claiming a camera saw something before it happened.
#
# Uniform rather than a queueing distribution, and stated as crude on purpose: none of these is
# measured, and a lognormal would look like it was. What they are for is making sure nothing
# downstream depends on frames landing on exact tenths.
BUS_DELAY = 0.02        # CAN arbitration and the read loop
CAMERA_DELAY = 0.05     # VLM inference plus transport off the accelerator
ASR_DELAY = 0.30        # recognition finishing some time after the words did

# Wheel-speed error as a FRACTION of the reading, not an absolute figure, for two reasons. It is
# how the sensor actually behaves — a tick-counting wheel encoder's error scales with the rate
# — and it keeps a stopped car reading exactly 0.0. "Stopped" is a state two rules distinguish
# (`animal_stopped` against `animal_moving`), and an absolute jitter would have a parked car
# reporting -0.14 kph, which is not a slow car but a broken sensor.
SPEED_NOISE = 0.01

# Confidence is written as the scenario states it, with NO noise, and that is a decision rather
# than an omission. A rule bands on confidence — `rear_child_window_lock` has floor 0.50 and
# threshold 0.80 — and `data/eval/scenes.jsonl` deliberately places rows AT those edges
# (`child_weak` at 0.55, `animal_near_miss` at 0.55). Jittering it would turn the comparison
# against gold into a coin flip at exactly the rows that were written to pin a boundary, and a
# generator whose agreement with the gold depends on the seed is not checking anything. A study
# of how a rule behaves under uncertain perception is a scenario that states the confidences it
# wants, not a global knob that quietly moves every other scenario too.


@dataclass(frozen=True)
class Motion:
    """What the car is doing, as the bus reports it.

    `until` is when the bus goes QUIET, in offsets from the start of the scenario; `None` means
    it publishes for the whole span. That is the honest way to express a stale signal in this
    system: a value does not decay because time passed, it decays because nothing said it again
    (`intake/hub.py`). So `Motion(45.0, until=0.0)` is a car doing 45 kph whose bus then stops,
    and any rule reading speed forty seconds later rejects — which is `animal_stale_bus`,
    reproduced by the mechanism rather than by a flag.
    """
    speed_kph: float
    until: Optional[float] = None
    hz: float = 10.0


@dataclass(frozen=True)
class Sighting:
    """What a camera reports, and how sure it says it is.

    `source=None` picks the camera by the key's namespace, because the namespace already says
    which one could have seen it: `outside.` is the forward camera and everything else is the
    cabin. Naming a source explicitly overrides that, which is how a scenario states a camera
    reporting something it should not be able to see.
    """
    at: float
    key: str
    value: Any
    confidence: float
    ttl: float = 300.0
    source: Optional[str] = None


@dataclass(frozen=True)
class Speech:
    """What the driver said, and when."""
    at: float
    text: str


@dataclass(frozen=True)
class Scenario:
    id: str
    span: float                                  # seconds of simulated time this covers
    motion: Optional[Motion] = None              # None = this scenario says nothing about the bus
    sees: tuple = ()
    says: tuple = ()

    def __post_init__(self):
        # An event outside the span is a typo with a silent consequence: the bus stops at `span`
        # by default, so a sighting stated after it would be read against a signal that went
        # quiet, and the scenario would test staleness while claiming to test a rule.
        for item in tuple(self.sees) + tuple(self.says):
            if not 0.0 <= item.at <= self.span:
                raise ValueError(
                    f"{self.id}: an event at {item.at}s falls outside the {self.span}s span")


@dataclass(frozen=True)
class Frame:
    """One row-to-be: which source produced it, when, and the payload's own fields.

    A dict rather than an `intake.envelope` payload, because `sim` does not import `intake`. The
    keys must match what that module's dataclasses declare and nothing here can check it; see
    the module docstring for what does.
    """
    source: str
    at: float
    payload: dict


def _camera(key: str) -> str:
    return FRONT_CAM if key.startswith("outside.") else CABIN_CAM


def frames(scenario: Scenario, *, seed: int, start: float = 0.0) -> list:
    """One scenario, as the rows a producer would have written. Sorted by time.

    `start` is the epoch the scenario's offsets are measured from, and a caller driving a real
    car should hand it a real one. It defaults to 0.0 rather than to `time.time()` so this
    function is a pure function of its arguments — a default that read a clock would make
    "the same seed produces the same rows" false, which is the property the seed exists for.

    Sorted so that row ids come out in the order the world happened. `process_pending` orders by
    `(at, id)` and does not need it, so this is for whoever reads the table: a store whose ids
    run backwards through time is hard to read and hides nothing useful. Out-of-order ARRIVAL is
    a real thing that happens and it is worth testing — but as a property a test states on
    purpose, not as something the generator does to it at random.
    """
    rnd = random.Random(seed)
    out: list[Frame] = []

    m = scenario.motion
    if m is not None:
        until = scenario.span if m.until is None else m.until
        step = 1.0 / m.hz
        # `int(...) + 1` so a scenario whose bus stops at 0.0 still publishes one frame: a car
        # doing 45 kph whose bus then dies has said 45 once, and generating nothing would model
        # a bus that was never running, which rejects for a different reason.
        for k in range(int(until / step) + 1):
            out.append(Frame(BUS, start + k * step + rnd.uniform(0, BUS_DELAY),
                             {"entity": SPEED[0], "attribute": SPEED[1],
                              "value": m.speed_kph * (1.0 + rnd.gauss(0.0, SPEED_NOISE))}))

    for s in scenario.sees:
        out.append(Frame(s.source or _camera(s.key),
                         start + s.at + rnd.uniform(0, CAMERA_DELAY),
                         {"key": s.key, "value": s.value,
                          "confidence": s.confidence, "ttl": s.ttl}))

    for utterance in scenario.says:
        out.append(Frame(MIC, start + utterance.at + rnd.uniform(0, ASR_DELAY),
                         {"text": utterance.text}))

    return sorted(out, key=lambda f: f.at)


def write(store, frames_: list) -> list:
    """Put the frames in the raw layer and make them durable. Returns the row ids.

    Through `store.put_raw` rather than an INSERT of its own, so generated rows get the same
    retention window and the same privacy switch as rows a microphone produced. A generator that
    reached past the store would be writing rows the policy does not apply to — immortal
    transcripts in a file that exists to be filled with them.

    Duck-typed on the store, which is what keeps this module out of `intake`: `put_raw` and
    `commit` are the whole interface. One commit for the whole scenario, because a scenario is
    the unit here the way an input is the unit in `Intake.ingest` — half a world in the store is
    a worse artifact than none.
    """
    ids = [store.put_raw(f.source, f.at, json.dumps(f.payload, ensure_ascii=False))
           for f in frames_]
    store.commit()
    return ids


# --- the scenarios that stand in for the hand-authored gold -----------------------------------
#
# One per row of `data/eval/scenes.jsonl`, sharing its id, so the generator can be checked
# against something a person agreed to rather than against itself. `test_every_gold_row_has_a_
# scenario` asserts the two sets are equal, which is what makes a row added to the gold a
# failure here rather than a silent gap in coverage.
#
# The inputs mirror the gold row's inputs and nothing more. Where a row states no signals this
# states no motion — a real parked car does publish 0.0 kph continuously, and saying so would be
# more plausible, but it would also be this module quietly adding a fact to a row a person wrote
# and then reporting agreement with it.
#
# **Six of these cannot be checked against their gold outcome, and the reason is worth knowing.**
# The `consent_*` rows turn on whether the SCENE owned the utterance turn, and consent is
# resolved in `cli/session.py`, above the door — so a 好 written as a row is routed against the
# catalog like any other sentence. The rows are still generated, still valid, and still driven
# by the tests that do not depend on the outcome; `test_the_queue_routes_a_consent_answer_
# instead_of_resolving_it` states the divergence rather than leaving it to be discovered.
GOLD_SCENARIOS: tuple = (
    # --- the cabin camera and the child-lock question ---
    Scenario("child_clear", span=2.0,
             sees=(Sighting(1.0, "inside.rear_occupant", "child", 0.90),)),
    Scenario("child_weak", span=2.0,
             sees=(Sighting(1.0, "inside.rear_occupant", "child", 0.55),)),
    Scenario("child_too_weak", span=2.0,
             sees=(Sighting(1.0, "inside.rear_occupant", "child", 0.30),)),
    Scenario("adult_rear", span=2.0,
             sees=(Sighting(1.0, "inside.rear_occupant", "adult", 0.95),)),
    Scenario("empty_rear", span=2.0,
             sees=(Sighting(1.0, "inside.rear_occupant", "none", 0.99),)),
    # A key no rule names. The gold row spells it bare and `Session.observe` prefixes the cabin
    # namespace; a producer writing a row has no such convenience, so it is spelled in full.
    Scenario("unknown_key", span=2.0,
             sees=(Sighting(1.0, "inside.driver_attention", "drowsy", 0.90),)),
    Scenario("repeat_suppressed", span=2.0,
             sees=(Sighting(1.0, "inside.rear_occupant", "child", 0.90),
                   Sighting(1.5, "inside.rear_occupant", "child", 0.92))),

    # --- the same question, answered ---
    Scenario("consent_yes", span=3.0,
             sees=(Sighting(1.0, "inside.rear_occupant", "child", 0.90),),
             says=(Speech(2.0, "好"),)),
    Scenario("consent_no", span=3.0,
             sees=(Sighting(1.0, "inside.rear_occupant", "child", 0.90),),
             says=(Speech(2.0, "不用"),)),
    Scenario("consent_lookalike", span=3.0,
             sees=(Sighting(1.0, "inside.rear_occupant", "child", 0.90),),
             says=(Speech(2.0, "好像有点热"),)),
    Scenario("consent_command", span=3.0,
             sees=(Sighting(1.0, "inside.rear_occupant", "child", 0.90),),
             says=(Speech(2.0, "把窗户关上"),)),
    Scenario("consent_unrelated", span=3.0,
             sees=(Sighting(1.0, "inside.rear_occupant", "child", 0.90),),
             says=(Speech(2.0, "后排太热了"),)),
    Scenario("consent_oblique_yes", span=3.0,
             sees=(Sighting(1.0, "inside.rear_occupant", "child", 0.90),),
             says=(Speech(2.0, "开吧"),)),

    # --- the forward camera and the moving car ---
    Scenario("animal_moving", span=2.0, motion=Motion(45.0),
             sees=(Sighting(1.0, "outside.front_object", "animal", 0.90),)),
    Scenario("animal_stopped", span=2.0, motion=Motion(0.0),
             sees=(Sighting(1.0, "outside.front_object", "animal", 0.90),)),
    Scenario("animal_walking_pace", span=2.0, motion=Motion(3.0),
             sees=(Sighting(1.0, "outside.front_object", "animal", 0.90),)),
    # The bus says 45 once and then stops. Forty seconds later the camera reports an animal and
    # the rule rejects, because the speed it would need is no longer a fact — the gold row spells
    # this `bus: false, clock: 40.0`, which is the session's way of arranging the same silence.
    Scenario("animal_stale_bus", span=40.0, motion=Motion(45.0, until=0.0),
             sees=(Sighting(40.0, "outside.front_object", "animal", 0.90),)),
    Scenario("animal_near_miss", span=2.0, motion=Motion(45.0),
             sees=(Sighting(1.0, "outside.front_object", "animal", 0.55),)),
    Scenario("animal_too_weak", span=2.0, motion=Motion(45.0),
             sees=(Sighting(1.0, "outside.front_object", "animal", 0.30),)),
    Scenario("not_an_animal", span=2.0, motion=Motion(45.0),
             sees=(Sighting(1.0, "outside.front_object", "cyclist", 0.95),)),
)

SCENARIOS: dict = {s.id: s for s in GOLD_SCENARIOS}
