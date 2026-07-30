# Scene Engine — Design

**Date:** 2026-07-30
**Goal:** a proactive layer that turns structured perception into *at most* a spoken question, and
lets the driver's explicit consent be the only thing that ever moves the car.

> **Scope.** This is a vertical slice carrying **one** scene end to end: a child detected in the
> rear with the window child lock off. The engine, the rule format, the arbitration, the LLM
> fallback and the consent loop are all built for real, but only far enough to carry that scene.
> Vision-text parsing, autonomous execution, and scene-aware reference resolution are named in §13
> as deferred, not as done.

---

## 1. Why this is a separate subsystem

`Pipeline.route(utterance: str)` is the only entry point in the shipped runtime, and it is stateless
per call. There is no tick, no clock, no event loop. A proactive engine is therefore a **second
top-level entry**, not a branch inside the first.

Placing it beside `t2f/` rather than inside it is a deliberate protection. Every measured number
this project reports — `recall@1 0.8644`, `ood_false_execution 0.000`, `incorrect_execution 0.0000`
— is produced by code inside `t2f/`. A scene engine that cannot reach the router cannot regress
them, and the existing eval harness stays a valid witness for both subsystems.

```
                         ┌─ t2f/  Pipeline.route(utterance) ─┐
driver speech ──────────▶│  normalize → … → validate         │──┐
                         └───────────────────────────────────┘  │
                                                                ▼
                         ┌─ scene/  SceneEngine.observe(evt) ─┐  executor.execute(ToolCall)
scene event ────────────▶│  context → rules → arbitration     │──┤   → sim/ SqliteVehicle
                         └────────────────────────────────────┘  │
                                    consent ──────────────────────┘
```

The two subsystems meet at exactly one place: `execute(ToolCall) -> ExecResult`. A scene-generated
call gets the same validation, the same preconditions, the same physical-limit checks and the same
operation-log entry as one the driver asked for.

## 2. The decisions this design rests on

| Decision | Chosen | Consequence |
|---|---|---|
| Deliverable | one scene, end to end | the hard safety questions get answered on a small surface |
| First scene | child in rear + child lock off → Ask | cabin perception, no native car feature to collide with |
| Input seam | structured `SceneEvent` only | no caption parsing; a real perception stack emits detections anyway |
| Consent | closed yes/no lexicon, exact match | a command can never be mistaken for consent |
| Autonomous execution | **not permitted** | `execute` is not a rule outcome; it is what consent causes |
| Architecture | sibling `scene/` package | `t2f/` routing untouched; measured numbers protected |
| Fallback trigger | near-misses **and** unconsumed observations | bounded by a budget, not by hoping rules are exhaustive |
| Reply text | templates; the model picks an intent | every sentence the car can say is enumerable and testable |

## 3. Scene Context — perception only

```python
@dataclass(frozen=True)
class Observation:
    key: str            # "inside.rear_occupant"
    value: Any          # "child"
    confidence: float   # 0.0 – 1.0
    source: str         # "cabin_cam"
    at: float           # seconds, caller's clock
    ttl: float          # seconds until stale
```

`SceneContext` keeps the newest observation per key. **Staleness is evaluated at read time** —
`get(key, now)` returns `None` once `at + ttl < now` — rather than by a sweeper thread. There is no
clock to run on the SoC, and tests pass `now` in explicitly instead of sleeping.

### Vehicle state is NOT copied into Scene Context

Rules read vehicle facts **live** from the car through a read-only port over
`SqliteVehicle.get_signal`. Scene Context holds perception and nothing else.

This follows a lesson the repo already paid for. Signals are keyed `(entity, attribute)` rather
than by function precisely so that `open_window` and `set_window_position` cannot hold two
contradictory beliefs about one window (`sim/mapping.py:1-20`). Copying `window_child_lock` into
Scene Context would recreate that at a higher altitude — a second belief about one lock, with its
own independent staleness. The car is the authority; the engine asks it.

## 4. Rules

```python
@dataclass(frozen=True)
class Rule:
    id: str
    description: str                # one line, shown to the fallback so it knows what exists
    when: tuple[Condition, ...]     # ALL must hold
    threshold: float                # fire at or above this observation confidence
    floor: float                    # below this, not even a near-miss
    persist_for: float              # seconds the observation must have held
    priority: int                   # higher wins contention
    cooldown: float                 # seconds before this rule may speak again
    intent: str                     # selects the speech template
    proposes: Optional[ToolCall]    # what consent would execute; None for a pure notify
```

Conditions come in exactly two forms, and there are no others:

```python
Observed("inside.rear_occupant", equals="child")
Signal("window.all", "window_child_lock", equals=False)
```

No arbitrary predicates and no embedded expressions. A closed vocabulary keeps every rule
inspectable, and lets a contract test walk the whole rule set and assert properties over all of it.

Rules are Python dataclasses in `scene/rules.py`, not YAML. One rule does not justify a loader; the
shape is data-only, so a YAML front end is a later addition rather than a rewrite.

### The one rule this slice ships

```python
REAR_CHILD_WINDOW_LOCK = Rule(
    id="rear_child_window_lock",
    description="后排检测到儿童且车窗儿童锁未开启",
    when=(Observed("inside.rear_occupant", equals="child"),
          Signal("window.all", "window_child_lock", equals=False)),
    threshold=0.80,
    floor=0.50,
    persist_for=0.0,
    priority=50,
    cooldown=120.0,
    intent="ask_rear_child_lock",
    proposes=ToolCall("set_window_child_lock", {"enabled": True}),
)
```

`window.all/window_child_lock` seeds to `False` (`sim/seed.py:64` — booleans rest at false), so a
freshly seeded car satisfies the `Signal` condition.

**`persist_for` is 0.0 on this rule deliberately.** A child seen in the rear does not become more
real by being seen for longer, and a non-zero value would make the rule unfireable from a single
`/scene` event — the only way a person can drive this by hand. The mechanism is still built and
unit-tested with explicit `now` values, so the field is real code rather than a stub; this rule
just does not need it.

### Evaluation, per rule

| Outcome | When |
|---|---|
| **reject** | any `Signal` condition is false — checked **first**, being the cheapest and most definitive |
| **match** | every condition holds, every observation ≥ `threshold`, held for ≥ `persist_for` |
| **near-miss** | conditions hold but an observation sits in `[floor, threshold)`, or `persist_for` is not yet met |
| **not applicable** | the rule references a key with no live observation — silent, and **not** a near-miss |

The distinction in the last row matters: absence of evidence is not ambiguity, and routing it to
the model would make the fallback fire on an empty context.

## 5. Arbitration

Runs in a fixed order. The order is what enforces "the LLM never overrides the rules" — by control
flow, not by asking the prompt nicely.

1. Drop rules inside their `cooldown` — measured from the last time that rule produced speech.
2. Drop a rule whose `(scene, proposal)` matches the currently pending consent, so the engine
   cannot ask a question it is already waiting on an answer to.
3. Among **matches**, highest `priority` wins; ties break by declaration order — deterministic, no
   clock involved.
4. **Only if no rule matched** do near-misses and unconsumed observations reach the fallback, and
   only if the budget allows.

## 6. What the engine returns

```python
@dataclass
class SceneOutcome:
    kind: str                       # "notify" | "ask" | "no_action"
    scene: str                      # rule id, or the scene the fallback named
    speech: str                     # "" when no_action
    proposal: Optional[ToolCall]    # only on ask — what consent would run
    source: str                     # "rule" | "llm"
    reason: str                     # diagnostic, never spoken
```

**An `ask` carries the ToolCall it would execute, and that call is validated before the question is
spoken.** If `validate_tool_call` rejects the proposal, the outcome degrades to `no_action` and the
driver hears nothing. Asking a question and discovering afterwards that the answer cannot be
honoured is the proactive form of the falsely-affirmative reply this project treats as its worst
failure mode (`docs/superpowers/RESULTS.md:266`).

## 7. LLM fallback

Reached only when no rule matched, and only for near-misses and unconsumed observations.

**Input:** the live (non-stale) context, the vehicle facts the near-miss rules actually referenced,
and every rule's `id` + `description`. Never accumulated history and never raw caption text —
"do not continuously append raw vision text to the LLM prompt" is made structural rather than left
as a guideline.

**Output**, decoded under an xgrammar JSON-schema constraint built the same way
`t2f/llm/schema.py` builds tool-call schemas:

```json
{"decision": "notify | ask | no_action",
 "scene":    "<enum: every rule id, plus \"unmatched\">",
 "reason":   "<free text, diagnostic only, never spoken>",
 "reply_intent": "<enum: every intent in the speech table>"}
```

Three properties follow from the schema itself rather than from a check that could be forgotten:

- **The decision vocabulary contains no `execute`.** The model cannot ask for the car to move; the
  most it can do is propose a question that still needs consent.
- **`scene` and `reply_intent` are enums.** The model selects from what exists; it cannot invent a
  scene name or author a sentence. Speech comes from the intent's template.
- **`no_action` is the escape hatch.** A constrained decoder always emits something legal, which is
  exactly why `REJECT_NAME` exists on the tool-call path (`t2f/llm/schema.py:34-41`). Without a
  legal way to decline, the model declines by picking something — the mechanism behind the
  99°→16° substitution.

`"unmatched"` is what the model returns for an unconsumed observation — perception reported
something no rule anticipated. One rule governs it, and it preserves §6's invariant on the model
path too:

> **An `ask` is only legal when `scene` names a real rule that carries a `proposes`.** Any other
> `ask` — including every `ask` on `"unmatched"` — degrades to `no_action`.

An unmatched scene has no proposal, so there is nothing consent could authorise; the most the model
can do with one is speak a notification.

**Budget:** `FALLBACK_COOLDOWN = 30.0` seconds — a single module-level constant, distinct from any
rule's own `cooldown`. At most one call per window, and only when the context actually changed.
Budget spent → `no_action`.

**Every failure degrades to `no_action`:** schema-invalid output, timeout, absent model, or any
exception. Silence is the safe default for a system nobody asked to speak.

## 7a. The speech table

Every sentence this subsystem can utter, in one place (`scene/speech.py`):

| intent | speech | reached by |
|---|---|---|
| `ask_rear_child_lock` | 后排有小孩，要打开儿童锁吗？ | the rule; and the fallback on a near-miss |
| `notify_driver_fatigue` | 您看起来有些疲劳，请注意休息。 | the fallback, on an unmatched cabin observation |
| `ack_declined` | 好的。 | consent declined |

A test asserts the fallback schema's `reply_intent` enum is **exactly** this table's keys, so the
two cannot drift apart.

**The confirmation after consent is not in this table.** It comes from `render_response` on the
executed card, so a scene-initiated action confirms with the same sentence a driver-initiated one
would — which is the whole reason §10 fixes the template rather than giving the scene its own.

## 8. Consent

```python
@dataclass
class PendingConsent:
    scene: str
    proposal: ToolCall
    asked_at: float
    expires_after: float
```

A single slot, with the same read-time expiry discipline as observations.

Resolution on the driver's next utterance, against the **normalised whole string**:

| Utterance | Result |
|---|---|
| in the affirmative set | execute the proposal |
| in the negative set | clear the slot, acknowledge |
| anything else | **drop the slot silently**, route the utterance as an ordinary command |
| (nothing, until `expires_after`) | expires on its own |

**Matching is exact set membership on the normalised utterance, never substring.** `好` is in the
affirmative set and `好像有点热` is not a yes — a substring test would make it one. This is the
single most safety-relevant line in the subsystem.

Affirmative: `好` `好的` `好吧` `可以` `行` `嗯` `嗯嗯` `是` `是的` `对` `没问题` `麻烦你了`
Negative: `不用` `不要` `不必` `算了` `不了` `不` `没事` `不需要`

### Consent authorises an action, not an outcome

The proposal is **re-validated and executed at consent time**, not at ask time. The gap between
`要打开儿童锁吗？` and `好` is a real window: the driver may have locked it manually, the module may
have gone offline. So consent re-runs `validate_tool_call` and dispatches through the same
`executor.execute`, and a refusal is spoken with its own cause through the existing 4b path.

## 9. Two speakers

**The driver's own turn always finishes first.** A scene outcome arriving mid-turn is queued and
spoken after. The session appends it — `compose_reply` is not involved, so `t2f/reply.py` and its
four contract metrics are unchanged.

**At most one open question across both systems.** If the router already asked a clarification, a
scene `ask` is dropped rather than queued. Two questions in flight would make `好` ambiguous about
which one it answers, and the entire consent design rests on `好` being unambiguous.

## 10. The boolean-confirmation fix

`set_window_child_lock` confirms with `已为您调整车窗儿童锁状态。` — identical whether the lock goes
on or off (`data/catalog/window.yaml:140`). That is `test_s4a_07_boolean_action_states_on_or_off`,
the repo's last red case, and it is worse in a proactive flow: the driver said `好` to a question
and gets back a sentence that does not say what happened.

`render_response` will humanise boolean parameters, so any boolean card states its direction:

```
已为您打开车窗儿童锁。     enabled=true
已为您关闭车窗儿童锁。     enabled=false
```

This is a change inside `t2f/` — to reply text, not to routing. Expected replies change across the
e2e suite and `reply_exact_match` must be re-measured. **No routing metric should move**, and that
is an assertion to check, not an assumption.

## 11. A consequence worth demonstrating

`sim/seed.py:39` already declares a precondition: `open_window` requires
`window.all/window_child_lock == False`, and refuses with `车窗儿童锁已开启`.

So the slice produces a genuine three-step interaction, and the e2e case asserts all of it:

```
/scene rear_occupant=child conf=0.85
  asked      后排有小孩，要打开儿童锁吗？
好
  executed   window.all/window_child_lock   false → true
  reply      已为您打开车窗儿童锁。
开车窗
  refused    vehicle · 车窗儿童锁已开启 · nothing changed
  reply      车窗儿童锁已开启。
```

A proactive action changing what a later driver-initiated command is allowed to do, with the
refusal explained — all four workflow steps, both entry points, one car.

## 12. Testing

The whole deterministic path runs with no embedder, no LLM and no GPU — milliseconds, the way
`tests/e2e/` already runs on `FakeEmbedder`. **Time is a parameter (`now: float`) everywhere**; no
logic calls `time.time()`, so TTL, persistence and cooldown are tested without sleeping.

### Contract sweep over every rule

In the style of `tests/e2e/test_s8_contract_sweep.py`, which is the strongest thing in the current
suite. For **all** rules:

1. no rule match ever produces a ToolCall on its own — consent is the only path to the car
2. every `ask` carries a proposal that validates
3. no outcome speaks an ASCII identifier or an internal signal name
4. a rule never fires when its `Signal` condition already holds — never ask for what is already true
5. cooldown is never bypassed

### Negatives that must hold

- a command (`把窗户关上`) never reads as consent
- `好像有点热` never reads as consent — the substring trap
- a second `ask` while one is pending is dropped
- an expired observation does not fire a rule
- a below-`floor` confidence never reaches the model
- a rule whose `Signal` condition is false produces silence, not a near-miss

The fallback is tested against a `FakeSceneLLM` returning scripted decisions, plus one
`@model`-marked test with the real Qwen3-0.6B, matching the three model-marked tests that exist.

## 13. Evaluation

Two arms mirroring C and C_llm, differing only in whether the fallback client is attached:
**`S`** (rules only) and **`S_llm`** (rules + fallback).

| Metric | What it catches |
|---|---|
| `scene_false_speech_rate` | spoke when gold says silence — the proactive analogue of OOD false-execution, and the number this design optimises for |
| `scene_recall` | fired when gold says it should have |
| `scene_false_consent_rate` | treated a non-consent utterance as consent; **0.000 by construction**, measured so a later change cannot quietly break it |
| `avg_llm_calls_per_event` | whether the budget holds |

Gold lives in `data/eval/scenes.jsonl` as event sequences with expected outcomes. **It is
hand-authored, so it encodes our beliefs about perception, not measured perception** — the same
caveat `docs/TEST_REPORT.md:196` already applies to the `asr_noise` rows.

## 14. Error handling

| Condition | Result |
|---|---|
| any exception inside the engine | `no_action` + a log line; never propagates |
| unknown observation key | unconsumed → fallback path; not an error |
| model absent, slow, or schema-invalid | `no_action` |
| executor refuses at consent time | spoken with its cause; car unchanged |
| proposal fails validation at ask time | `no_action`; the question is never asked |

The engine must never kill the session — in this repo a crash costs a 60-second model reload
(`cli/session.py:115`).

## 15. Files

```
scene/
  context.py     Observation, SceneContext, read-time staleness
  facts.py       read-only vehicle port over SqliteVehicle
  rules.py       Condition, Rule, evaluation, the rule set
  engine.py      SceneEngine.observe() -> SceneOutcome, arbitration
  consent.py     PendingConsent, the lexicon, resolution
  llm.py         SceneLLM protocol, constrained schema, FakeSceneLLM
  speech.py      intent -> template
tests/scene/     unit per module, e2e, contract sweep
cli/             session.py gains observe(); __main__.py gains /scene
t2f/respond.py   boolean humanisation (§10)
eval/            scene arm + the four metrics
data/eval/scenes.jsonl
```

## 16. Deferred, and named as deferred

- **Vision-text → event parsing.** The natural follow-on is to route captions through the existing
  embedder against scene prototypes — this project's own retrieval-first thesis applied to
  perception. It needs its own prototype set, calibration and gold data.
- **Autonomous execution.** A new outcome type, not a redesign.
- **Scene-aware reference resolution** (`把它关上`, `太吵了`). This is the *router* reading Scene
  Context, which touches `t2f/` and carries its own risk.
- **Free-text reply generation.**
- **More than one pending consent.**
- **On-device benchmarking of the fallback.** No figure here is measured on SA8797, consistent with
  everything else in this repo.

## 17. Open risks

1. **The gold file is authored, not observed.** `scene_recall` measures agreement with our own
   beliefs about what perception would report. It is a contract test wearing a metric's clothes,
   and §13 says so.
2. **`persist_for` ships unused.** The shipped rule sets it to 0.0 for the reason given in §4, so
   the mechanism is covered by unit tests but by no end-to-end case. It is real code with no real
   consumer until a second rule needs it.
3. **The consent lexicon is closed and therefore incomplete.** `开吧` is a yes that this design
   deliberately drops. `scene_recall` will show the cost; widening the lexicon is a measured
   decision, not a guess.
