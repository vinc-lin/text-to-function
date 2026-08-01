# End-to-End Test Report

**Date:** 2026-07-28
**Scope:** the 131 end-to-end cases covering the Central Model's business workflow
**Suite when this was written:** 456 passed · 1 xfailed · 3 deselected. **The current figure lives in
the newest update section and nowhere else — §11.**
**Updated 2026-07-29** after the extractor and negation fixes — see §8.
**Updated 2026-07-30** with the Scene Engine and the last red case — see §10.
**Updated 2026-08-01** with sensed signals, staleness, a second rule, and intake — see §11.
**Evaluation:** arm C (deterministic, zero LLM), real embedder, gold test split n=192

> **This document is the home of the repository's current measured numbers.** `README.md` summarises
> and links here; `docs/superpowers/RESULTS.md` records what each spec measured *when it shipped* and
> is not updated afterwards. A figure quoted anywhere else without a date should be checked against
> the newest section here.

---

## 1. What "end-to-end" means here, and where it stops

An end-to-end case drives the real `Pipeline.route()` and asserts on **both** ends of the workflow:
what was actually dispatched to the vehicle, and what the driver actually hears.

```
 ┌───────┐  text   ┌──────── Central Model ────────┐  text  ┌───────┐
 │  ASR  │────────▶│ route(utterance) -> reply     │───────▶│  TTS  │
 └───────┘         └───────────────┬───────────────┘        └───────┘
  not tested                       │ execute(ToolCall)       not tested
  (upstream)                       ▼
                          SQLite simulated car
```

Step 1 (the user speaking) and the voicing of the reply are outside the boundary — there is no ASR
or TTS in this repo. Everything between is covered.

**131 of the 398 tests are end-to-end.** The rest are unit tests of the components they compose.

---

## 2. Coverage

| File | Cases | What it establishes |
|---|---|---|
| `test_s2_recognition.py` | 11 | segmentation and intent recognition; narration suppressed |
| `test_s3_execution.py` | 10 | dispatch, the plan barrier, and the executor-result contract |
| `test_s4a_confirmation.py` | 5 | success confirmations name the action |
| `test_s4b_failure_cause.py` | 13 | failure causes are explained, not collapsed |
| `test_s5_simulator.py` | 7 | the four-step workflow against a real simulated car |
| `test_s6_success_matrix.py` | **24** | 22 utterances across **9 of 10 domains**, full chain each |
| `test_s7_failure_matrix.py` | **26** | every way an operation can fail |
| `test_s8_contract_sweep.py` | **24** | invariants that hold for *every* utterance |
| `test_reply_e2e.py` | 11 | the original Spec-5 reply cases |
| **Total** | **131** | |

### By workflow step

| Step | Cases | Verdict |
|---|---|---|
| 2 — segmented intent recognition | ~35 | **covered** |
| 3 — execute | ~45 | **covered in simulation** |
| 4a — confirm success | ~30 | **covered** |
| 4b — explain failure | ~40 | **covered** for every cause the system can distinguish |

(Categories overlap: most cases assert across several steps at once, which is the point of an
end-to-end case.)

---

## 3. What a driver actually hears

Every string below is measured from a passing test, not composed for this document.

**Success**

| Utterance | Reply | Car |
|---|---|---|
| 副驾温度调到26度 | 已将副驾温度设置为26°C。 | `climate.passenger/temperature` 24.0 → 26.0 |
| 主驾座椅按摩开到五档 | 已将主驾座椅按摩设置为5档。 | `seat.driver/seat_massage` 2 → 5 |
| 打开车窗，风速调到3档 | 已为您打开当前区域车窗。已将当前区域风速设置为3档。 | both signals move |

**Failure — the driver is told which of three things went wrong**

| Category | Utterance | Reply |
|---|---|---|
| didn't understand | 帮我算一下房贷 | 抱歉，我不太确定您的意思，可以换个说法吗？ |
| value unusable | 把空调调到99度 | 目标温度只能设置在16到32度之间。 |
| value unusable | 空调模式调到3 | 空调模式只支持制冷/制热/自动/除湿/送风。 |
| value unusable | 导航到3 | 目的地名称或地址需要一段文字。 |
| parameter missing | 车窗 | 您想打开还是关闭车窗？ |
| parameter missing | 屏幕亮度调一下 | 您想把屏幕亮度设置成多少？ |
| **the car refused** | 把主驾温度调到25度 *(A/C off)* | 空调尚未开启。 |
| **the car refused** | 把主驾温度调到25度 *(module offline)* | 空调控制模块离线。 |
| **the car refused** | 屏幕亮度调到80% *(panel capped at 60)* | 屏幕亮度只能设置在0到60%之间。 |

The last row is the one no amount of schema validation can produce: the **card** accepts 80 with zero
errors, and the **car** refuses it.

**Mixed**

| Utterance | Reply |
|---|---|
| 开车窗,把主驾温度调到25度 *(A/C off)* | 已为您打开当前区域车窗。空调尚未开启。 |
| 开车窗，风速调到20档，屏幕亮度调到200% | 已为您打开当前区域车窗。风速档位只能设置在1到7档之间。屏幕亮度只能设置在0到100%之间。 |

Confirmation first, then each cause. The second row is a case that until today produced one vague
question covering both bad values.

---

## 4. Invariants — the strongest result in the suite

`test_s8_contract_sweep.py` asserts five properties that must hold for **every** utterance:

1. the reply is never empty — there is no silent path
2. at most **one** question, however many clauses failed
3. every dispatched action has a confirmation — nothing is actuated silently
4. no false affirmation — the reply never claims something happened that did not
5. a confirmation and a refusal for the same action never both appear

The committed sweep routes 38 corpus utterances against a freshly simulated car in two conditions.
Beyond that, the author swept **all 382 rows of both eval datasets under six configurations** —
healthy car, refusing car, MEDIUM-band thresholds, all-LOW thresholds, snapshot-seeded state, and an
executor returning a blank refusal detail. **Zero violations in every configuration**, including 192
rows that really actuated the car and 10 the car really refused.

Two design choices make that result worth something:

- The sweep reads what was dispatched from the **car's own operation log**, not from the
  `RouteResult`. A pipeline that forgot to report an execution cannot hide it.
- The no-false-affirmation check is a **greedy decomposition**: the reply must be exactly the
  concatenation of fragments its clauses authored, plus the two known constants. Invented text fails.

And critically, **the invariants were shown to fire.** A green sweep proves nothing on its own, so
each was mutation-tested against the regression it guards:

| mutation injected | caught by |
|---|---|
| reply always `好的。` | confirmation-coverage |
| speak every clause's question | at-most-one-question |
| confirm dispatched-but-refused calls | refusal-never-also-confirmed |
| fabricate one extra confirmation | no-false-affirmation |
| drop the last confirmation | confirmation-coverage |
| resolver ignores `ExecResult.ok` *(the pre-Spec-5 bug)* | no-confirmation-without-execution |

---

## 5. Measured results

Arm C, real embedder, gold test split (n=192), plus the 54-row e2e slice:

| Metric | Value | Reading |
|---|---|---|
| `reply_nonempty_rate` | **1.0000** | no silent path |
| `reply_action_coverage` | **1.0000** | nothing actuated silently |
| `reply_single_question` | **1.0000** | contract holds |
| `invalid_no_execution_rate` | **1.0000** (22) | no unusable value reached the vehicle |
| `reply_cause_coverage` | **1.0000** (15) | the cause is conveyed in every annotated row |
| `reply_exact_match` | 0.0811 (37) | **measures wording, not capability — see §6** |
| recall@1 | 0.8644 | unchanged — extraction does not affect routing |
| multi-intent set-recall | 0.8194 | unchanged |
| OOD / context false-action | **0.0000** | unchanged |
| **incorrect execution** | **0.0000** | was 0.0312 |
| **param exact-match** | **0.4133** | was 0.2733 |
| **e2e deterministic** | **0.1333** | was 0.1067 |
| **schema-valid rate** | **0.6212** | was 0.5000 |

Every routing metric is identical to the pre-change baseline. One metric moved and is explained in §6.

---

## 6. Two numbers that need reading carefully

**`reply_exact_match` 0.0811 is not evidence of a capability gap.** The 37 annotations are free-form
Chinese written before any implementation existed, each phrased differently. The system now emits
systematic wording conveying the same facts:

```
annotated  温度只能设置在16到32度之间。
actual     目标温度只能设置在16到32度之间。
```

Rewriting the annotations to match the output would make the metric a tautology. `reply_cause_coverage`
was added instead: rows declare the **facts** a reply must convey (`["16","32"]`), not the sentence.

**`schema_valid_rate` moved 0.5079 → 0.5000.** This is a denominator change, not a behaviour change.
The metric counts a clause as having attempted a call when it produced either a tool call or
validation errors. Plan-path clauses ran validation all along but discarded the errors, so they were
never counted; they now report them, and the denominator grew to include attempts that were always
happening. No routing metric moved.

---

## 7. What is NOT covered — read before trusting the number

- **`FakeEmbedder` is not the real embedder.** The deterministic suite uses a hashed-n-gram stand-in
  so it can run without a GPU. Roughly half of the case authors' first-choice utterances misrouted
  under it and were replaced — never accommodated. This suite proves the *mechanism*; the eval
  harness measures *accuracy*.
- **The `phone` domain has no success case.** Not an oversight — see finding 1 below.
- **`type_mismatch` on a boolean is unreachable**: `extract_boolean` returns `bool` or `None`.
- **`unknown_param` / `not_in_candidates` / `unknown_function` / `llm_no_toolcall`** deliberately carry
  no driver-facing detail and fall to the generic line. A driver can do nothing about them.
- **No ASR, no TTS, no real vehicle bus.** Out of scope by design.
- **`asr_noise` rows encode our belief** about ASR errors, not measured misrecognitions.

### The two red cases — both now closed

Both were `xfail(strict=True)`, so closing either made the suite say so. Both have since said so,
and the repo has **no red case left**:

| Case | Defect | Closed |
|---|---|---|
| `test_s2_11_negation_must_not_invert_the_action` | `别关车窗` ("don't close the window") dispatched `is_open=False` and closed it | **2026-07-29** — polarity is positional now, and a negated cue yields *unknown* and asks, because "don't close it" is not "open it" (§8 finding 3) |
| `test_s4a_07_boolean_action_states_on_or_off` | opening and closing produced byte-identical confirmations | **2026-07-30** — `render_response` humanises the boolean, so 38 of the 39 boolean cards state their direction (§10) |

The first was the more serious: the system executed the **inverse** of the instruction.

A red count of zero is not a claim of accuracy. What §9 says still holds — these cases encoded gaps
in the *business workflow*, and closing them says nothing about `param_exact_match 0.4133`.

---

## 8. Findings

**1. String and non-position enum parameters cannot be extracted at all. — FIXED 2026-07-29.**
`t2f/params/extract.py:9-20` has no branch for `type: string`, and none for an enum whose values are
not positions — both fall through to `extract_number`. So `make_call`, `send_message`, `navigate_to`,
`find_nearby`, `add_waypoint` and ~11 enum-driven functions (`set_ac_mode`, `set_audio_source`,
`set_ambient_light_color`, `move_seat`, …) **recognise correctly and then always clarify**, because
their required parameter can never be filled. `呼叫10086` is worse: it fills `contact` with the float
`10086.0` and is rejected as a type mismatch.

This is one missing dispatch branch, and it is a large part of why `param_exact_match` sits at 0.2733
and `e2e_deterministic` at 0.1067 — the router is being blamed for an extractor gap. **Highest-value
open item.**

> **Closed.** Enum extraction now reads the driver's vocabulary, mined from the catalog's own
> `utterances` (101 of its 104 examples fill their parameter). Free-text extraction reads one shape —
> trigger, connector, object — at **100% precision on the catalog corpus**: 22 correct, 0 wrong, 26
> asked, because a wrong destination navigates somewhere else while declining costs one question.
> `param_exact_match` 0.2733 → **0.4133**, `e2e_deterministic` 0.1067 → **0.1333**,
> `schema_valid_rate` 0.5000 → **0.6212**, and `incorrect_execution_rate` 0.0312 → **0.0000**.
>
> Unblocking execution briefly *raised* incorrect-execution to 0.0556, which turned out to be two
> genuine parameter bugs rather than a safety regression — zero wrong-function executions throughout.
> `收音机调到FM101.7` dropped `band` because the vocabulary had 调频 but not `FM`, and 车窗儿童锁解开
> read 解开 as "on" via the 开 inside it. Both fixed.

**2. Correct recognition does not reach execution under production thresholds.**
With the shipped gate (`high_top1 0.35 / high_margin 0.12`), utterances route to the **correct**
function at top1 0.74–0.80 but fall under the margin, landing in MEDIUM — which the zero-LLM arm never
executes. Nine consecutive utterances in a live demo produced nine generic apologies. That is
`e2e_deterministic 0.1067` made visible. Widening the margin is a safety trade, and it is entangled
with the arm decision that is still unrecorded anywhere.

**3. A polarity miss. — FIXED 2026-07-29.** `打开下雨自动关窗` extracted `enabled=False` from a 打开
utterance, because every `_OFF` form was tested before any `_ON` form and the 关 inside 关窗 outranked
the leading verb. Polarity is now positional. The negation defect went with it: 别关车窗 no longer
closes the window and 别开空调 no longer turns the A/C on — a negated cue yields *unknown* and asks,
because "don't close it" is not "open it".

**4. Duplicate actions are de-duplicated at the reply layer only.** `主驾温度调到25度` twice dispatches
twice and confirms once. Outside the stated contract, so it is documented rather than judged.

---

## 9. How much of this to trust

The honest summary: **the mechanisms are well covered and the invariants are genuinely proven; the
accuracy is not what this suite measures.**

What would be misleading to conclude from "393 passed":
- not that the router is accurate — that is `recall@1 0.8644` and `param_exact_match 0.2733`
- not that the vehicle integration works — there is no vehicle, only a simulator
- not that it runs on the 87 platform — nothing here has been measured on target hardware

What it does support: for every utterance the suite has tried, the system either does the right thing
or says something true about why it did not, and never claims to have done something it did not do.
That property is mutation-tested rather than asserted, which is the part I would stand behind.

---

## 10. Update — 2026-07-30: the Scene Engine, and the last red case

Everything above is the report as it stood on 2026-07-28, amended on 2026-07-29. Two things have
landed since, and this section is the only place that reflects them.

**Suite: 624 passed · 0 xfailed · 3 deselected** (`python3 -m pytest -q`, re-run 2026-07-30). The 131
end-to-end cases of §2 are unchanged in number and composition. 111 of the total are new tests under
`tests/scene/`, of which 5 are end-to-end chains and 8 are a contract sweep in the style of §4.

### Boolean confirmations state their direction

`test_s4a_07_boolean_action_states_on_or_off` is closed, and with it the last red case in the repo
(§7). `render_response` now derives a state word from the card's own verb — 打开/关闭, or 折叠/展开
for `fold_mirror` and `fold_rear_seat` — and 38 of the 39 boolean cards interpolate it:

```
已为您打开车窗儿童锁。     enabled=true
已为您关闭车窗儿童锁。     enabled=false
```

**Three literals in §3 were updated to match**: the two-signal success row and both mixed rows now
read `已为您打开当前区域车窗。` where they read `已为您调整当前区域车窗状态。`. They are presented as
measured output, so they had to be re-measured; all three were, against the same `FakeEmbedder`
pipeline the e2e suite uses.

10 of the 92 cards still confirm without naming the value they set — `set_ac_mode` says
`已切换空调模式。` without saying which mode. Nine are enum switches. The tenth is `spray_washer`,
left alone deliberately: its `enabled` is a momentary trigger rather than a state, and
`已为您关闭玻璃水。` is not a sentence anyone would say. The enum nine are a smaller version of the
same defect, and no red case covers them.

**Arm C was re-measured after the change and nothing moved.** Same command as §5
(`--arm C --dataset data/eval/gold.jsonl --calibrate`, real embedder, n=192): `recall@1` 0.8644,
`multi_intent_set_recall` 0.8194, `param_exact_match` 0.4133, `e2e_deterministic` 0.1333,
`schema_valid_rate` 0.6212, OOD / context / incorrect execution all 0.0000 — every figure in §5
identical, **including `reply_exact_match` 0.0811 (37)**. The design predicted that one would move;
it did not, because none of the 37 free-form annotations covers a boolean confirmation. A reply-text
change with no routing consequence was the assertion to check, and it checks out.

### The Scene Engine — a second subsystem, measured separately

`scene/` is a **proactive** layer: it takes a structured perception event and turns it into *at most*
a spoken question. It is not a router change and not a new arm of the router. It cannot reach
`Pipeline.route()`; the two subsystems meet only at `execute(ToolCall)`, so a scene-initiated call
gets the same validation, preconditions, physical limits and operation-log entry as a driver's.
**Consent is the only path from a scene to the car** — `execute` is not an outcome the engine can
produce, and `好` must be the whole normalised utterance, matched by exact set membership rather than
substring, so a command can never be read as a yes.
([design](superpowers/specs/2026-07-30-scene-engine-design.md))

Two arms, mirroring C and C_llm, differing only in whether the constrained fallback client is
attached. Measured 2026-07-30 over all 13 rows of `data/eval/scenes.jsonl`:

| Metric | Arm S (rules only) | Arm S_llm (+ fallback) | n |
|---|---|---|---|
| `scene_false_speech_rate` | **0.0000** | **0.0000** | 9 silent rows |
| `scene_recall` | **1.0000** | **1.0000** | 4 speaking rows |
| `scene_false_consent_rate` | **0.0000** | **0.0000** | 4 must-not-consent rows |
| `avg_llm_calls_per_event` | **0.0000** | 0.1538 | 13 rows |

`avg_llm_calls_per_event` 0.1538 is 2 decodes over 13 rows: the fallback is reached only when no rule
matched, and only for near-misses and unconsumed observations, under a 30-second budget.
`scene_false_consent_rate` is 0.000 **by construction** — it is measured so that a later change to the
lexicon cannot quietly break it, not because the measurement discovered anything.

### What these four numbers do NOT establish

- **`data/eval/scenes.jsonl` is hand-authored, so it encodes our beliefs about perception rather
  than measured perception** — the same caveat §7 already applies to the `asr_noise` rows. There is
  no camera, no vision model and no recorded cabin data anywhere in this repo. `scene_recall 1.000`
  is agreement with what we decided a cabin camera would report; it is a contract test wearing a
  metric's clothes, and the design says so in §13.
- **Denominators of 4 and 9.** Thirteen rows is a vertical slice, not a distribution. One rule ships
  (a child in the rear with the window child lock off), so `scene_recall` has exactly one scene to
  recall.
- **`persist_for` ships with no end-to-end case.** The shipped rule sets it to 0.0, so the mechanism
  is covered by unit tests and by nothing else.
- **The fallback has no on-device figure**, consistent with everything else here.
- **The interactive session attached no scene fallback** when this was written, so `python3 -m cli`
  exercised arm S only. Superseded on 2026-07-30: `/scene-llm on` and `--scene-llm` attach it, and
  both arms are now reachable by hand. Under `--fake` it refuses rather than attaching a scripted
  stand-in, because a fake would fabricate the behaviour someone is trying to observe.

The scene contract sweep is the part worth the same trust as §4: eight properties, each asserted over
**every** rule in the set — no rule match ever produces a ToolCall on its own, every `ask` carries a
proposal that validates, no outcome speaks an ASCII identifier or an internal signal name, a rule
never fires when its `Signal` condition already holds, cooldown is never bypassed, and every rule
yields to a router question already open. With one rule in the set those are cheap to satisfy; they
are written to stay true of the tenth.

---

## 11. Update — 2026-08-01: sensed signals, staleness, a second rule, and intake

**Suite: 860 passed · 1 skipped · 5 deselected**, ~32 s (`python3 -m pytest -q`, run 2026-08-01 at
HEAD). 866 tests collected; the 5 deselected are the `model` marker, which needs a GPU.

The one skip is legitimate and permanent while the rule set is what it is:
`test_every_proposal_validates[animal_ahead]` in `tests/scene/test_contract_sweep.py` — the property
is "every `ask` carries a proposal that validates before the question is spoken", and `animal_ahead`
is notify-only, so there is no proposal to validate. The sweep parametrises over the rule set, so a
notify rule skips that one property and is asserted by the other seven. **It skips rather than passes
vacuously**, which matters: §4's counterpart failure mode is a sweep that silently shrinks to fit, and
a vacuous pass is exactly how that looks.

Where the tests are:

| area | tests | |
|---|---|---|
| `tests/scene/` | 154 | + the 1 skip above |
| `tests/cli/` | 120 | |
| `tests/ui/` | 57 | |
| `tests/sim/` | 56 | |
| `tests/intake/` | 47 | new |
| `tests/` root + `tests/e2e/` | the balance | the router core and the e2e suite |

**The 131 end-to-end cases of §2 are unchanged in number and composition.** §1 says "131 of the 398
tests" — the 131 is still right, the 398 is the denominator as it stood on 2026-07-28 and is now 861
non-model tests. The ratio moved because the suite grew around the e2e cases, not because any e2e case
changed.

### A second rule, and a third condition form

`scene/rules.py` ships two rules now: `animal_ahead` and `rear_child_window_lock`. The first fires on
an animal detected ahead of a car that is **moving**, which needed a condition form the closed
vocabulary did not have — `SignalAbove("vehicle.all", "speed_kph", above=5.0)`, a third shape rather
than a comparator on `Signal`, because it is the closed vocabulary that lets the contract sweep walk
every rule and assert properties over all of them.

`animal_ahead` proposes nothing. No vehicle function makes an animal in the road safe, so there is
nothing for consent to authorise, and it is the first rule whose only outcome is to speak. It also
outranks the child-lock question (priority 90 vs 50) and is readier to fire (threshold 0.70 vs 0.80) —
a missed animal is worse than a spurious warning, while a spurious question is merely annoying. **This
is the first time two shipped rules can contend**, so it is the first time arbitration has been
exercised by anything other than a test fixture: the suppressed rule reports `outranked by
animal_ahead` and does not spend its own cooldown.

The eight sweep properties now run over both rules. That is what turned up the skip above — with one
rule, "every ask has a valid proposal" was a property of the only rule there was.

### Staleness: `updated_at` is finally read

`updated_at` had been written on every signal write since Spec 7 and **read by nothing**. A rule
conditioning on speed therefore fired off a value frozen ten minutes ago exactly as readily as off a
live one: **a dead bus and a stationary car were indistinguishable to every rule.** Perception had
received this discipline in Spec 9 — `Observation.is_live`, read-time expiry, no sweeper — and the car
never had.

Sensed signals now declare a `max_age` (speed: 2.0 s). Past it the signal reads as `None` — identical
to a signal the car does not hold — so every condition on it rejects and the engine falls silent
**with both ages named**: `vehicle.all/speed_kph is stale (40.0s > 2.0s)`. Silence that explains
itself, which is the same posture as everywhere else here.

**Actuated signals never expire, and the asymmetry is deliberate.** A window position holds until
something commands it otherwise; a speed is a continuous measurement whose absence means the bus
stopped. So `max_age` lives only on sensed signals. And it is declared on the **signal**, not the
rule, so no rule can forget to ask — a per-rule setting puts the burden on every new rule, and the one
that forgets reads stale values with nothing to catch it.

Two findings from wiring it, both of which would have shipped looking correct:

- **Staleness failed open.** The session ran on a monotonic clock (~756 thousand) while the car stamps
  `time.time()` (~1.79 billion), so the computed age was about **minus 1.78 billion seconds** — under
  every `max_age`, so everything read as fresh. The discipline had tests passing around it and did
  nothing. The session moved to wall clock, because `--db` persists the car and a monotonic stamp in a
  file outliving its process means nothing on the next run.
- **The seeded speed kept its seed stamp**, so two seconds into any session the animal rule blamed the
  bus for a car that was simply parked.

### Intake — one envelope, and a view that owns nothing

Every input now arrives as one `Input(source, at, payload)` whose payload type *is* its kind (there is
no `kind` field to disagree with the payload), from a source that declares what it may produce —
`Input(source="cabin_cam", payload=SignalWrite(...))` cannot be constructed. `WorldView` is one
read-through view over perception **and** the car; it is asserted to hold nothing writable by walking
the instance, because "read-only" in a docstring is not an enforcement mechanism.

`intake/` is packaged. It is the composition root — the thing that assembles router, scene engine and
car — which until now existed only inside `cli/session.py`, a dev tool `pyproject` deliberately does
not ship, so a real integration had to reimplement wiring the CLI had already worked out.

The bus is **pumped, not threaded**: `sim/vehicle.py` opens SQLite with default thread affinity, and a
background republish thread would not even fail loudly — `ui/state.py` wraps each pane defensively, so
it would serve a snapshot with an empty car while everything else rendered fine. The semantics that
fall out are the right ones anyway: a live bus is fresh whenever you look, and `/bus off` is what makes
age accumulate.

### Nothing measured moved

Both proof obligations hold. `python3 -m eval.run_scene_eval --arm S` was re-run at HEAD on 2026-08-01
and is **byte-identical** to §10's arm S column — `scene_false_speech_rate` 0.0000 (9 silent rows),
`scene_recall` 1.0000 (4 speaking rows), `scene_false_consent_rate` 0.0000 (4 rows),
`avg_llm_calls_per_event` 0.0000 (13 rows) — **with two rules in the set instead of one**. Arm C is
byte-identical apart from its `p50/p95_latency_ms` lines, which are wall-clock jitter larger than most
real changes. `t2f/` was not touched, so every routing figure in §5 stands as written.

### What this update does NOT establish

- **`data/eval/scenes.jsonl` is still hand-authored, so it encodes our beliefs about perception rather
  than measured perception.** This is the caveat that matters most here, and adding a second rule did
  not weaken it — it doubled the rule set without adding a single observed row. There is no camera, no
  vision model and no recorded cabin data anywhere in this repo. `scene_recall 1.000` is agreement
  with what we decided a cabin camera and a front camera would report; it is a contract test wearing a
  metric's clothes. The denominators are still 4 and 9, and the 13 rows are still a vertical slice.
- **`animal_ahead` has no gold row of its own.** The scene gold predates it and was deliberately not
  rewritten to keep a number stable. So the rule is covered by unit tests and the contract sweep, and
  by nothing in the measured column.
- **`max_age = 2.0` for speed is a guess.** No measurement supports it; it is a plausible number for a
  10 Hz signal. It is one constant in one declaration and should be read as provisional.
- **The pump is only as good as its callers.** A consumer that forgets to pump sees everything stale.
  That is the safe direction — stale reads as absent, so the failure is silence rather than a wrong
  action — but it is a real footgun.
- **`can0` is a declared source with no hardware behind it.** There is still no vehicle bus, and no
  ASR and no TTS.
- **`live_facts()` covers both stores but is not yet wired into the fallback prompt.** Doing so would
  move arm S_llm, which neither proof obligation covers, so it needs its own measurement rather than
  being slipped in under a byte-identical arm S.
- **Arm S_llm has not been re-measured since 2026-07-30**, when the rule set had one rule. Its column
  in §10 predates the second rule.
