# Intake and WorldView — Design

**Date:** 2026-08-01
**Goal:** one door for everything entering the system, and one view of everything currently true.

> **Scope.** This restructures how inputs arrive and how facts are read. It changes no routing
> decision and no rule outcome — both are proof obligations with byte-identical eval reports, not
> aspirations. Caption parsing and cross-modal reasoning stay deferred; §9 says so explicitly.

---

## 1. What is wrong today

Three inputs, three shapes, three disciplines:

| input | door | carries | staleness | confidence | provenance |
|---|---|---|---|---|---|
| voice text | `Pipeline.route(str)` | the string | — | via gate bands | none |
| VLM output | `Session.observe(...)` → `Observation` | key, value, conf, source, `at`, `ttl` | read-time expiry | explicit | `source` string |
| CAN signal | `Session.set_signal(...)` → a SQLite row | value, `updated_at` | **none** | none | none |

**One asymmetry is a defect.** `updated_at` has been written on every signal write since Spec 7 and
is **read by nothing** — a repo-wide grep finds it only in a schema line and two comments. So
`SignalAbove("vehicle.all", "speed_kph", above=5.0)` fires the animal warning off a speed frozen ten
minutes ago exactly as readily as off a live one. **A dead bus and a stationary car are
indistinguishable to every rule.** Perception received precisely this discipline —
`Observation.is_live`, read-time expiry, no sweeper — and the car never did.

Three more, in descending order of bite:

- **`source` is decorative.** Everything defaults to `"cabin_cam"`, including `outside.*` and
  `vehicle.*` observations. Nothing validates it; only the display reads it.
- **Voice text has no envelope**, so nothing can relate it to anything else in time.
- **Two stores separated by a string prefix.** `inside.` / `outside.` / `vehicle.` is a convention,
  not a type. Before speed became a signal, `vehicle.speed_kph` was expressible as an `Observation`
  and nothing prevented it.

And underneath all of it, a structural fact the dependency map makes plain:

```
t2f   → (nothing)          the router core
scene → t2f
sim   → t2f
cli   → t2f, scene, sim    ← the only place that knows all three
ui    → cli, scene
```

**The composition root is in the dev tool.** `Session` is the sole thing that assembles router +
scene engine + car, which is why it accumulated `route()`, `observe()` and `set_signal()` as three
unrelated methods — and `cli/` is deliberately not packaged. A real vehicle integration would have to
reimplement wiring the CLI already worked out.

## 2. The decisions

| Decision | Chosen | Consequence |
|---|---|---|
| Shape | one envelope, one dispatch point | provenance and timing captured once, at the edge |
| Depth | existing handlers unchanged underneath | `Pipeline.route()` untouched; router numbers cannot move |
| Stale signals | read as absent; `max_age` per **signal**, not per rule | no rule can forget to ask |
| Placement | a new **packaged** `intake/` | the composition root ships instead of living in a tool |
| Hub | `WorldView`, read-through, owns nothing | one liveness question over two stores |
| Bus | a pumped publisher | a live bus is fresh whenever you look; stopping it is what makes it stale |

## 3. The envelope

```python
@dataclass(frozen=True)
class Utterance:
    text: str

@dataclass(frozen=True)
class Percept:
    key: str
    value: Any
    confidence: float
    ttl: float

@dataclass(frozen=True)
class SignalWrite:
    entity: str
    attribute: str
    value: Any

@dataclass(frozen=True)
class Input:
    source: str
    at: float
    payload: Utterance | Percept | SignalWrite
```

**There is no `kind` field.** The payload's type *is* the kind, so there is nothing that can disagree
with itself. A `kind` beside a payload is two statements about one fact, and eventually they differ.

## 4. Sources are declared

```python
SOURCES = {
    "mic":       Source("mic",       accepts=Utterance),
    "cabin_cam": Source("cabin_cam", accepts=Percept),
    "front_cam": Source("front_cam", accepts=Percept),
    "can0":      Source("can0",      accepts=SignalWrite, publishes=True),
}
```

`Input` refuses at construction if the source is unknown, or if it does not accept that payload type.
`Input(source="cabin_cam", payload=SignalWrite(...))` cannot be built.

This is what turns `source` from decoration into a checkable claim. Today every observation defaults
to `"cabin_cam"` — including vehicle-namespace ones — and nothing notices.

## 5. `WorldView` — the hub

```python
class WorldView:
    def observation(self, key: str, now: float) -> Optional[Observation]: ...
    def signal(self, entity: str, attribute: str, now: float) -> Optional[Any]: ...
    def live_facts(self, now: float) -> dict: ...
```

**It owns nothing.** It knows who to ask, not the answer: perception comes from `SceneContext`,
vehicle facts from the car. A hub that *stored* would recreate exactly the problem signal-keyed state
was built to prevent — `open_window` and `set_window_position` are keyed by signal so they cannot hold
two contradictory beliefs about one window, and a hub holding a copy of `window_child_lock` rebuilds
that one level up, with its own staleness, so the car and the hub can disagree about a lock.

It is **read-through, never a cache.** The moment it holds a value the two-beliefs problem is back.

`VehicleFacts` is absorbed: its entire job was the read-only car port. Its safety property comes with
it — a test walks the instance and asserts nothing reachable exposes `set_signal`, because
"read-only" in a docstring is not an enforcement mechanism.

**What this unlocks:** the fallback's prompt is built from `live_facts()` and therefore sees vehicle
state for the first time. Its prompt today is built from `context.live()` alone, so
"complex relationships between multiple context states" — named as a fallback job in the original
scene-engine brief — has been quietly impossible.

Rules take `(rule, world, now)` instead of `(rule, context, facts, now)`.

## 6. Staleness

`_SENSED` rows gain a `max_age`. `updated_at` is finally read. A signal past its age reads as `None`
— identical to a signal the car does not hold — so every condition on it rejects and the engine falls
silent **with the ages named**: `vehicle.all/speed_kph is stale (4.2s > 2.0s)`.

**Actuated signals never go stale, and the asymmetry is the point.** A window position holds until
something commands it otherwise. A speed is a continuous measurement, and its absence means the bus
stopped. So `max_age` lives only on sensed signals — where the declaration already is.

Freshness is declared on the **signal**, not the rule, because how fast a value decays is a property
of the source: speed at 10 Hz is dead after a second, a door-ajar flag is fine for a minute. One
declaration, every rule inherits it, and **no rule can forget to ask**. A per-rule `max_age` would
put the burden on each new rule, and the one that forgets reads stale values with nothing to catch it
— which is the failure mode this whole design exists to remove.

## 7. The bus is pumped, not threaded

A publishing source re-stamps its held values when pumped. `intake.pump(now)` is called by the
existing loops — the UI on each 400 ms poll, the CLI on each command.

**This is forced, not chosen.** `sim/vehicle.py` opens SQLite with default thread affinity, which is
why the UI server is single-threaded; a background republish thread would hit exactly that, and it
would not raise — `ui/state.py` wraps each pane defensively, so it would serve a snapshot with an
empty car while everything else rendered fine. It would lie rather than break.

The semantics that fall out are the right ones anyway:

- **A live bus is fresh whenever you look at it** — true of a real bus too.
- **`/bus off` stops the pump**, and only then does age accumulate.
- **`/clock` cannot manufacture staleness while the bus is on**, because the pump stamps at the
  offset clock as well.

## 8. What moves

| | |
|---|---|
| `t2f/` | **untouched** |
| `scene/rules.py` | `evaluate(rule, world, now)`; the three condition shapes unchanged |
| `scene/engine.py` | takes a `WorldView` |
| `scene/facts.py` | **absorbed** into `WorldView` |
| `sim/seed.py`, `sim/vehicle.py` | `max_age` on sensed rows; `updated_at` readable |
| `cli/session.py` | a consumer of `intake`; `handle`/`observe`/`set_signal` build `Input`s |
| `ui/` | snapshot from `WorldView`; the bus control |
| `intake/` | new, packaged |

`pyproject.toml` gains `intake*`. `cli/` and `ui/` stay out, as they are.

## 9. Proof obligations

Two, and they are the reason this is safe to do at all:

- `python3 -m eval.run_eval --arm C --dataset data/eval/gold.jsonl --fake --permissive` —
  **byte-identical** before and after. Any routing metric that moves means the change reached `t2f/`.
- `python3 -m eval.run_scene_eval --arm S` — **byte-identical**. The rules must decide exactly what
  they decided before; only the plumbing beneath them changed.

## 10. Testing

- a source can only produce what it declares — refused at construction
- `WorldView` holds nothing writable, asserted by walking the instance
- a stale signal reads as absent, and the rejection names both ages
- a live bus stays fresh; a stopped one does not
- every declared source round-trips through `ingest`
- the contract sweep still walks every rule, now against a `WorldView`

**The migration diff is broad**: every test constructing `SceneEngine(cards, facts, executor)`
changes, and the sweep's fact stubs change shape. That is the same churn `SignalAbove` caused — and
that churn is what exposed the sweep silently shrinking to fit a new condition type. A broad diff
here is an opportunity to check, not a cost to minimise.

## 11. Deferred, and named as deferred

- **Caption parsing.** "VLM output" is still a structured `Percept`, not a sentence. The envelope
  makes the eventual parser a source that emits `Percept`s, rather than a new door.
- **Cross-modal reasoning.** A rule still cannot condition on what the driver said. The envelope
  makes it expressible later; nothing here attempts it, and there is no gold data for it.
- **Dialogue in the hub.** Pending consent stays on the engine, the transcript on the session.
  Neither is a fact about the world, and putting them in the hub is how it becomes the thing every
  module imports.
- **A real CAN adapter.** `can0` is a declared source with no hardware behind it.

## 12. Open risks

1. **`intake/` becomes the new god object.** It is the composition root, so everything is reachable
   from it. The guard is that it holds no logic of its own: dispatch and provenance only, with every
   decision still made by the module that owns it.
2. **`max_age = 2.0` for speed is a guess.** No measurement supports it; it is a plausible number for
   a 10 Hz signal. It is one constant in one declaration and should be treated as provisional.
3. **The pump is only as good as its callers.** A future consumer that forgets to pump sees
   everything stale. Better than the reverse — stale reads as absent, so the failure is silence
   rather than a wrong action — but it is a real footgun and belongs in the module docstring.
