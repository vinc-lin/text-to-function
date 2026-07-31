# Sensed Signals and the Animal-Ahead Scene — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** give the car a category of signals it *knows* but nothing *commands*, starting with speed; let both doors set them; show them in the UI; and build the `animal_ahead` scene on top.

**Why now:** the animal scene is the first thing to need vehicle state that no function produces. `sim/` models exactly what the Central Model can change, which was right until a rule needed to read something it cannot change.

---

## Three design decisions, made before any code

### 1. Sensed signals are a declared category, not a loosened guard

`tests/sim/test_seed.py:134` asserts the car holds **exactly** the signals the 92 cards can write, in both directions. A speed row breaks it, and the correct response is not to weaken the assertion — it is to widen what it asserts:

> the car holds exactly the writable signals **plus the declared sensed ones**

Both directions stay enforced, and a sensed signal nobody declared is still a failure. The guard survives; only its definition of "legitimate" grows.

### 2. "Vehicle moving" needs a third condition form

`Signal` compares with `equals`. Motion is `speed > 0`, and there is no equality that expresses it. Rather than give `Signal` an operator — which reopens the closed vocabulary the contract sweep depends on — add one more closed form:

```python
SignalAbove("vehicle.all", "speed_kph", above=5.0)
```

Three condition types, each trivially inspectable, no expressions. `above=5.0` rather than `> 0` because a warning at walking pace is noise.

### 3. Setting a sensed signal is a **simulator control**, not a Central Model action

The executor is the seam for operations the Central Model performs and must remain the only one. Telling the simulator the car is now doing 45 is the *world* changing — the same category as the camera seeing a child, or `/reset` re-seeding the vehicle.

So it lives **outside** `ACTIONS`, in a separate `CONTROLS` table, and the UI shows it in a visually separate place. The existing test asserting `ACTIONS` has exactly five entries stays exactly as it is; a new test asserts `CONTROLS` and that the two tables are disjoint. Anyone later looking for "how does the page reach the car" finds two lists with different names and different justifications, rather than one list with a quiet sixth entry.

---

## Task 1: sensed signals in the simulator

**Files:** modify `sim/seed.py`, `tests/sim/test_seed.py`

- [ ] **Step 1 — write the failing tests** (append to `tests/sim/test_seed.py`)

```python
def test_speed_is_seeded_and_stationary(car):
    """A car nobody has driven is not moving. 0.0 is a meaningful value, not a missing one."""
    assert car.get_signal("vehicle.all", "speed_kph") == 0.0


def test_a_sensed_signal_carries_limits(car):
    lo, hi = car.limits_of("vehicle.all", "speed_kph")
    assert lo == 0.0 and hi == 240.0


def test_no_function_writes_a_sensed_signal():
    """That is what makes it sensed. If a card ever gains one, this is the wrong category
    for it and the seeder should be deriving it like everything else."""
    from sim.seed import sensed_signals
    writable = set(_every_signal())
    assert {(e, a) for e, a, *_ in sensed_signals()} & writable == set()


def test_the_car_holds_exactly_what_it_can_write_plus_what_it_senses(car):
    """The original guard, widened rather than weakened — both directions still enforced,
    and a sensed signal nobody declared is still a failure."""
    from sim.seed import sensed_signals
    seeded = {(r["entity"], r["attribute"]) for r in _signal_rows(car)}
    legitimate = set(_every_signal()) | {(e, a) for e, a, *_ in sensed_signals()}
    assert seeded - legitimate == set(), "seeded rows nothing can write or sense"
    assert legitimate - seeded == set(), "legitimate signals the car does not have"
```

Then **delete** `test_the_car_holds_exactly_the_signals_the_catalog_can_write`, which the last test replaces. Deleting it is correct — leaving both means one of them must fail.

- [ ] **Step 2** — run, confirm the new tests fail and the old one now fails too (proving the widening is necessary, not cosmetic).

- [ ] **Step 3 — implement.** In `sim/seed.py`, beside `_PRECONDITIONS`:

```python
# (entity, attribute, resting value, unit, min, max) — signals the car KNOWS and nothing
# COMMANDS. Everything else in this file is derived from a function card, because until now
# the simulator modelled exactly what the Central Model can change. A real vehicle bus is
# mostly signals like these; ours had none until a scene needed to condition on one.
#
# They are declared here rather than derived precisely because no card can produce them, and
# tests/sim/test_seed.py asserts no function writes one — if a card ever gains it, this is
# the wrong category and the seeder should derive it like everything else.
_SENSED = [
    ("vehicle.all", "speed_kph", 0.0, "kph", 0.0, 240.0),
]


def sensed_signals() -> list[tuple]:
    """The declared sensed signals. Public so the seed guard can consult one definition."""
    return list(_SENSED)
```

`seed_from_catalog` writes them after the card-derived signals, with their limits.

- [ ] **Step 4** — `python3 -m pytest tests/sim -q`, then the full suite.
- [ ] **Step 5** — commit: `feat(sim): signals the car knows and nothing commands`

---

## Task 2: `SignalAbove`, and the animal-ahead scene

**Files:** modify `scene/rules.py`, `scene/speech.py`, `tests/scene/test_rules.py`, `tests/scene/test_speech.py`

- [ ] **Step 1 — write the failing tests** (append to `tests/scene/test_rules.py`)

```python
from scene.rules import SignalAbove


def _moving(kph):
    return FakeFacts(**{"vehicle.all/speed_kph": kph})


ANIMAL = Rule(
    id="a", description="d",
    when=(Observed("outside.front_object", equals="animal"),
          SignalAbove("vehicle.all", "speed_kph", above=5.0)),
    threshold=0.70, floor=0.40, persist_for=0.0, priority=90, cooldown=30.0,
    intent="notify_animal_ahead", proposes=None)


def _animal_ctx(confidence=0.9, at=100.0):
    ctx = SceneContext()
    ctx.update(Observation("outside.front_object", "animal", confidence, "front_cam", at, 300.0))
    return ctx


def test_a_moving_car_satisfies_signal_above():
    assert evaluate(ANIMAL, _animal_ctx(), _moving(45.0), now=100.0) is Verdict.MATCH


def test_a_stationary_car_rejects_it():
    """Standing still, an animal ahead is not a warning worth making."""
    assert evaluate(ANIMAL, _animal_ctx(), _moving(0.0), now=100.0) is Verdict.REJECT


def test_the_boundary_is_strictly_above():
    assert evaluate(ANIMAL, _animal_ctx(), _moving(5.0), now=100.0) is Verdict.REJECT
    assert evaluate(ANIMAL, _animal_ctx(), _moving(5.1), now=100.0) is Verdict.MATCH


def test_an_absent_sensed_signal_rejects_rather_than_raises():
    """A rule naming a signal the car does not hold must fall silent, not explode — and the
    contract sweep separately guarantees no shipped rule can be in that state."""
    assert evaluate(ANIMAL, _animal_ctx(), FakeFacts(), now=100.0) is Verdict.REJECT


def test_signal_above_explains_itself():
    verdict, why = evaluate_explained(ANIMAL, _animal_ctx(), _moving(0.0), now=100.0)
    assert verdict is Verdict.REJECT
    assert "speed_kph" in why and "0.0" in why and "5.0" in why


def test_the_animal_rule_outranks_the_child_lock_question():
    """A warning beats a convenience question, and this is the first time two shipped rules
    can contend at all."""
    from scene.rules import ANIMAL_AHEAD, REAR_CHILD_WINDOW_LOCK
    assert ANIMAL_AHEAD.priority > REAR_CHILD_WINDOW_LOCK.priority


def test_the_animal_rule_only_warns():
    """A warning proposes nothing: there is no vehicle action that makes an animal safe, and
    the driver is the one who has to act."""
    from scene.rules import ANIMAL_AHEAD
    assert ANIMAL_AHEAD.proposes is None


def test_the_animal_rule_is_readier_to_fire_than_the_question():
    """A deliberate asymmetry. A missed animal is worse than a spurious warning; a spurious
    child-lock question is merely annoying, so it demands more confidence."""
    from scene.rules import ANIMAL_AHEAD, REAR_CHILD_WINDOW_LOCK
    assert ANIMAL_AHEAD.threshold < REAR_CHILD_WINDOW_LOCK.threshold
```

Append to `tests/scene/test_speech.py`:

```python
def test_the_animal_warning_exists_and_names_no_action():
    """It tells the driver to look, not to press anything — nothing in the cabin makes an
    animal in the road safe."""
    from scene.speech import SPEECH
    assert "notify_animal_ahead" in SPEECH
```

- [ ] **Step 2** — run, confirm `ImportError: cannot import name 'SignalAbove'`.

- [ ] **Step 3 — implement.**

`scene/rules.py`:

```python
@dataclass(frozen=True)
class SignalAbove:
    """A vehicle fact that must exceed a threshold. Strictly above.

    A third condition form rather than an operator on `Signal`, because the closed vocabulary
    is what lets the contract sweep walk every rule and assert properties over all of them.
    Three simple shapes stay inspectable; one shape with a comparator does not.
    """
    entity: str
    attribute: str
    above: float
```

`Condition` becomes the three-way union. `evaluate_explained` gains a branch, checked with the other signal conditions **before** any observation — a stationary car settles the question and must not cost a model call. A missing signal (`None`) is a REJECT, never a TypeError.

Reasons: `f"{entity}/{attribute} is {actual}, not above {above}"`, and for a missing one `f"{entity}/{attribute} is not a signal this car holds"`.

The rule:

```python
ANIMAL_AHEAD = Rule(
    id="animal_ahead",
    description="前方检测到动物且车辆行驶中",
    when=(Observed("outside.front_object", equals="animal"),
          SignalAbove("vehicle.all", "speed_kph", above=5.0)),
    # Readier to fire than the child-lock question, deliberately: a missed animal is worse
    # than a spurious warning, while a spurious question is merely annoying.
    threshold=0.70,
    floor=0.40,
    persist_for=0.0,
    # Outranks the question. This is the first pair of shipped rules that can contend, so it
    # is also the first time the arbitration code has anything real to arbitrate.
    priority=90,
    cooldown=30.0,
    intent="notify_animal_ahead",
    # Warns and proposes nothing. No vehicle function makes an animal in the road safe, and
    # the driver is the one who has to act — so there is nothing for consent to authorise.
    proposes=None,
)

RULES: tuple = (ANIMAL_AHEAD, REAR_CHILD_WINDOW_LOCK)
```

`scene/speech.py`: `"notify_animal_ahead": "前方有动物，请注意。"`

- [ ] **Step 4** — `python3 -m pytest -q`. The contract sweep now runs over **two** rules; every one of its properties must hold for both. If any fails, that is a real finding — report it rather than exempting the rule.
- [ ] **Step 5** — commit: `feat(scene): the animal-ahead warning, and a condition for motion`

---

## Task 3: setting a sensed signal, from both doors

**Files:** modify `cli/session.py`, `cli/__main__.py`, `ui/actions.py`, `ui/state.py`; tests alongside

- [ ] `Session.set_signal(entity, attribute, value)` — writes through `SqliteVehicle.set_signal`, refuses anything not in `sensed_signals()` with a clear message. **Refusing is the point:** this is not a way to command the car, and an actuated signal poked directly would bypass every precondition and limit the executor exists to enforce.
- [ ] `/signal <entity>/<attribute>=<value>` in the CLI, listed in `/help`, with the same malformed-input discipline as `/scene`: refuse and print usage, never raise.
- [ ] `ui/actions.py` gains a **separate** `CONTROLS` table with `set_signal` in it. `ACTIONS` keeps its five entries and its test unchanged. New tests: `CONTROLS` has exactly one entry, and `set(ACTIONS) & set(CONTROLS) == set()`.
- [ ] `POST /control/<name>` routes to `CONTROLS`, distinct from `/action/<name>`.
- [ ] `ui/state.py` snapshot gains `sensed: [{entity, attribute, value, unit, min, max}]`, read from the car for the declared sensed signals. The existing `car` pane keeps showing only what changed — sensed signals are always visible, actuated ones appear when they move, and those are different questions.
- [ ] commit: `feat: a simulator control for the signals nothing commands`

---

## Task 4: the UI panes

**Files:** modify `ui/page.html`

- [ ] A **Vehicle** section above the car pane: each sensed signal with its value, unit, and a slider or number input bounded by its limits, posting to `/control/set_signal`. Styled distinctly from everything else on the page — this is the world being set, not the car being commanded, and the page should not let anyone confuse the two.
- [ ] The car pane keeps its current meaning and gains a heading that says so ("changed from seeded").
- [ ] commit: `feat(ui): the vehicle pane`

---

## Task 5: verification and docs

- [ ] Drive the full animal scene by hand in both doors and paste real transcripts: set speed to 45, inject the animal, get the warning; drop speed to 0 and confirm the rule rejects with its reason.
- [ ] Confirm the two-rule contention is real: stage a car where **both** rules match and show that the animal warning wins and the child-lock question reports `outranked by animal_ahead`.
- [ ] `docs/TRYING_IT.md`: `/signal` in the commands table, and the animal scene as a worked example.
- [ ] Re-run `python3 -m eval.run_scene_eval --arm S`. **It will change** — the gold file's `unknown_key` row uses `driver_attention`, but a second rule now exists and the denominators shift. Report the before and after and say what moved and why; do not adjust gold to keep a number stable.

## Done criteria

- `/signal vehicle.all/speed_kph=45` works in the CLI and the UI, and refuses an actuated signal
- an animal ahead of a moving car warns; ahead of a stationary one is silent, with a reason
- both rules present in the contract sweep, every property holding for both
- `python3 -m pytest -q` green, no new dependency
