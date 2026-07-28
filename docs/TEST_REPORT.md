# End-to-End Test Report

**Date:** 2026-07-28
**Scope:** the 131 end-to-end cases covering the Central Model's business workflow
**Suite:** 456 passed · 1 xfailed · 3 deselected
**Updated 2026-07-29** after the extractor and negation fixes — see §8.
**Evaluation:** arm C (deterministic, zero LLM), real embedder, gold test split n=192

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
| 打开车窗，风速调到3档 | 已为您调整当前区域车窗状态。已将当前区域风速设置为3档。 | both signals move |

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
| 开车窗,把主驾温度调到25度 *(A/C off)* | 已为您调整当前区域车窗状态。空调尚未开启。 |
| 开车窗，风速调到20档，屏幕亮度调到200% | 已为您调整当前区域车窗状态。风速档位只能设置在1到7档之间。屏幕亮度只能设置在0到100%之间。 |

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

### The two remaining red cases

Both are `xfail(strict=True)`, so closing either makes the suite say so:

| Case | Defect |
|---|---|
| `test_s2_11_negation_must_not_invert_the_action` | `别关车窗` ("don't close the window") dispatches `is_open=False` and closes it |
| `test_s4a_07_boolean_action_states_on_or_off` | opening and closing produce byte-identical confirmations |

The first is the more serious: the system executes the **inverse** of the instruction.

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
