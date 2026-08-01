# Sensed Signals and the Animal-Ahead Scene — Design

**Date:** 2026-07-31
**Goal:** give the car a category of signals it *knows* but nothing *commands*, starting with
speed; let both doors set them; and build the `animal_ahead` scene on top.

> **Provenance.** This records the design decisions that were originally written into the
> implementation plan for this work. The plans directory was removed on 2026-08-01 as one-use
> scaffolding; this is the part of it that was not scaffolding. The measured outcome is in
> [`RESULTS.md`](../RESULTS.md) under *After Spec 9*.

---

## Why now

The animal scene is the first thing to need vehicle state that no function produces. `sim/`
modelled exactly what the Central Model can *change* — which was right until a rule needed to
read something it cannot change.

## 1. Sensed signals are a declared category, not a loosened guard

`tests/sim/test_seed.py` asserted the car holds **exactly** the signals the 92 cards can write,
in both directions. A speed row breaks that, and the correct response was not to weaken the
assertion but to widen what it asserts:

> the car holds exactly the writable signals **plus the declared sensed ones**

Both directions stay enforced, and a sensed signal nobody declared is still a failure. The
guard survives; only its definition of "legitimate" grows.

**Actuated signals never go stale, and that asymmetry is the point.** A window position holds
until something commands it otherwise. A speed is a continuous measurement, and its absence
means the bus stopped. So `max_age` lives only on sensed rows.

## 2. "Vehicle moving" needs a third condition form

`Signal` compares with `equals`. Motion is `speed > 0`, and no equality expresses it. Rather
than give `Signal` an operator — which would reopen the closed vocabulary the contract sweep
depends on — one more closed form:

```python
SignalAbove("vehicle.all", "speed_kph", above=5.0)
```

Three condition types, each trivially inspectable, no expressions. `above=5.0` rather than
`> 0` because a warning at walking pace is noise.

**This is also what exposed the sweep silently shrinking to fit.** Every gate in
`tests/scene/test_contract_sweep.py` spelled a vehicle condition `isinstance(c, Signal)`, so
the moment a rule used `SignalAbove` two properties skipped themselves on a rule that has
exactly the condition they test, and three more passed vacuously — including the one whose own
docstring warns that a misspelled rule is "indistinguishable from working correctly". Adding a
condition form is therefore a moment to re-verify the sweep by mutation, not to trust green.

## 3. Setting a sensed signal is a simulator control, not a Central Model action

The executor is the seam for operations the Central Model performs and must remain the only
one. Telling the simulator the car is now doing 45 is the *world* changing — the same category
as the camera seeing a child, or `/reset` re-seeding the vehicle.

So it lives **outside** `ACTIONS`, in a separate `CONTROLS` table, and the UI shows it in a
visually separate place. `ACTIONS` keeps its exactly-five-entries test unchanged; a new test
asserts `CONTROLS` and that the two tables are disjoint. Anyone later asking "how does the page
reach the car" finds two lists with different names and different justifications, rather than
one list with a quiet sixth entry.

`Session.set_signal` refuses anything not declared sensed. Refusing is the point: poking an
actuated signal directly would bypass every availability check, precondition and physical limit
that `SqliteExecutor` exists to enforce.

## The rule

`animal_ahead` is the first **notify-only** rule — it warns and proposes nothing, because no
vehicle function makes an animal in the road safe and there is nothing for consent to
authorise. It is also the first rule that can **outrank** another, which is the first time the
arbitration code has had anything real to arbitrate.

It is deliberately **readier to fire** than the child-lock question (threshold 0.70 against
0.80): a missed animal is worse than a spurious warning, while a spurious question is merely
annoying.

## What this did not do

- **`max_age = 2.0` for speed is a guess.** No measurement supports it; it is a plausible
  number for a 10 Hz signal, and one constant in one declaration.
- **`can0` is a declared source with no hardware behind it.** There is still no vehicle bus.
- **Only one sensed signal exists.** Gear, doors-ajar and the rest belong in the same category
  when something needs them; guessing at a set now would be inventing requirements.
