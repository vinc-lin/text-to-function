# Interactive Session — Design

**Date:** 2026-07-29
**Goal:** a terminal session where a person types Chinese and watches the whole Central Model
workflow run against a live simulated car.

---

## 1. Why

**There is currently no way for a human to try this system.** The repo has 456 tests, an eval
harness and a simulated vehicle, and only two `__main__` blocks: the batch eval runner and the
classifier trainer. Every `Pipeline` in the codebase is constructed inside `eval/arms.py`.

That is gap 6 of the [Central Model system design](2026-07-25-central-model-system-design.md) —
"no serving host; production wiring lives in the eval package" — and it means the only way to form
a view about the system is to read numbers someone else produced.

This closes that. Not with a product UI, but with the smallest honest thing: type an utterance, see
what the car did and what the driver would hear.

## 2. What one turn looks like

Every turn exercises steps 2, 3 and 4 and shows each of them. The strings below are measured from
the current build, not invented.

**Executed:**

```
[C_llm · shipped] > 把主驾温度调到25度

  recognised   set_temperature{temperature: 25.0, position: driver}    band=HIGH
  executed     climate.driver/temperature   24.0 → 25.0
  reply        已将主驾温度设置为25°C。
```

**Escalated to the model** — the case the shipped gate exists for:

```
[C_llm · shipped] > 把空调打开

  recognised   set_ac_power{enabled: true}    band=MEDIUM  → resolved by LLM
  executed     climate.all/ac_power   false → true
  reply        已为您调整空调开关状态。
```

**Refused by the car** — the branch no amount of validation can produce:

```
[C_llm · shipped] > 把主驾温度调到25度        (A/C is off)

  recognised   set_temperature{temperature: 25.0, position: driver}    band=HIGH
  refused      vehicle · precondition_failed · nothing changed
  reply        空调尚未开启。
```

**Rejected before the car** — validation, which never reaches the vehicle at all:

```
[C_llm · shipped] > 把主驾温度调到99度

  recognised   set_temperature{temperature: 99.0, position: driver}    band=HIGH
  rejected     validation · out_of_range · never reached the car
  reply        目标温度只能设置在16到32度之间。
```

**Multi-intent** prints one block per action span, then the single composed reply — because one
utterance produces many actions and exactly one thing the driver hears:

```
[C_llm · shipped] > 开车窗，风速调到20档

  recognised   open_window{is_open: true}          band=HIGH
  executed     window.all/window_position   50 → 100
  recognised   set_fan_speed{level: 20}            band=HIGH
  rejected     validation · out_of_range · never reached the car
  reply        已为您调整当前区域车窗状态。风速档位只能设置在1到7档之间。
```

The distinction between rejected and refused is deliberate and is the thing this tool makes visible:
a validation failure never touches the vehicle, a refusal reaches it and is logged.

## 3. Configuration and the two switches

**Default: arm C_llm on the shipped gate.** A real candidate build, not a loosened one, and it
executes on a normal turn — `e2e_deterministic` 0.62 against arm C's 0.13.

Two orthogonal switches, because they answer different questions:

| switch | options | what it changes |
|---|---|---|
| **LLM** | on *(default)* / off | whether the MEDIUM band is resolved by Qwen3-0.6B or is a dead end |
| **gate** | shipped *(default)* / permissive | how confident the router must be before it may act |

| | shipped gate | permissive gate |
|---|---|---|
| **LLM on** | **default** — executes ~62%, MEDIUM escalates to the model | rarely useful |
| **LLM off** | the safe build — executes ~13%, MEDIUM is a dead end | deterministic with the brakes off — an experiment, not a product |

The thresholds themselves:

| | `high_top1` | `high_margin` | `low_top1` |
|---|---|---|---|
| shipped (`config.yaml`) | 0.35 | **0.12** | 0.15 |
| permissive | 0.20 | **0.00** | 0.05 |

`high_margin` is the one that decides things. Measured: `把主驾温度调到25度` scores top1 **0.801**
with margin **0.107** — far above either `high_top1`, and still MEDIUM, because the runner-up is
close behind. The catalog contains genuinely similar functions, so thin margins are normal and the
shipped gate reads them as "I know roughly what you want, not precisely which function".

**The mode is never ambiguous.** It is printed at startup and shown in the prompt itself
(`[C_llm · shipped] >`), because the whole point of a switch is undermined if you forget which way
it is set.

## 4. The car

A `SqliteVehicle` seeded from the 92-card catalog, **persisting for the whole session**, so a
conversation behaves like one:

```
> 开车窗                 window.all/window_position  50 → 100
> 主驾温度调高一点         climate.driver/temperature  24 → 26     (resolved against real state)
```

Relative commands are the reason this matters. Without persistent state `再开一点` has nothing to
resolve against, and step 3 cannot be tested honestly.

In-memory by default (fresh car each session). `--db <path>` keeps it on disk across runs, which is
also how you would inspect it with `sqlite3` afterwards.

## 5. Commands

Seven, and no more:

| command | |
|---|---|
| `/llm on\|off` | attach or detach the fallback model |
| `/gate shipped\|permissive` | switch thresholds |
| `/car` | current state of every signal that differs from its seeded value |
| `/log` | recent operations — what was attempted, what the car said |
| `/reset` | fresh car |
| `/help`, `/quit` | |

`/log` earns its place because a refusal is the one outcome where "what did it try" is not visible
from the reply alone.

## 6. Structure

```
t2f/build.py        # build_pipeline(...) — assembles the product. NEW.
cli/
  __main__.py       # the I/O loop: read, print, handle commands. Thin.
  session.py        # Session.handle(utterance) -> Turn. The logic. Testable.
  render.py         # Turn -> the text above. Pure.
```

`python3 -m cli` runs it.

**Why a factory.** The session must build a `Pipeline`, and the only code that does that today is
`eval/arms.py`. Copying it would duplicate the wiring; importing the eval package from a runtime
tool would repeat the layering violation the simplification pass removed. So `t2f/build.py`
assembles the standard pipeline (hybrid scorer, confidence gate, optional LLM, injectable executor)
and both the session and `eval/arms.py`'s arms C and C_llm use it. Arms `baseline` and `D` stay in
`eval/arms.py` — the factory builds the product, the eval package builds experiments.

That is ~15 lines and it closes gap 6 as a by-product rather than as new scope.

**Why the split.** `Session.handle` takes a string and returns a structured `Turn` — recognised
calls, execution outcomes, signal deltas, reply. It touches no stdin and no stdout, so it is
testable the same way everything else here is. `__main__.py` is the only part that cannot be
tested by a unit test, and it is deliberately trivial.

**Not in `t2f/`.** A session is not reachable from `route()`, so principle 6 says it is not runtime.
`t2f/build.py` is the exception, and belongs there precisely because it *is* the product's wiring.

## 7. Startup

The real embedder and Qwen3-0.6B both load — roughly **60 seconds**, once per session. The session
prints what it is loading rather than sitting silent.

`--fake` starts instantly with `FakeEmbedder` and a scripted LLM. It exists for checking the plumbing
after a change, and it is **labelled loudly in the prompt** (`[C_llm · shipped · FAKE] >`), because the fake embedder
has no semantics and misroutes badly over 92 cards — `关闭空调` routes to `play_music`. A session on
it would teach you things about the harness, not about the system.

## 8. When something goes wrong

A session must not die on one bad turn. `Session.handle` catches anything the pipeline raises,
prints the exception with the utterance that caused it, and returns to the prompt — a crash costs a
60-second reload, which would make people stop using the tool rather than report the bug.

An empty line re-prompts. An unknown `/command` prints the help rather than being routed as an
utterance.

## 9. Non-goals

- **No ASR, no TTS, no real vehicle.** Same boundary as everywhere else in the project.
- **Not a product UI.** No web page, no packaging for anyone but a developer.
- **Not multi-turn dialogue.** Clarifications are shown, but answering one does not resume the
  pending request — `research/dialog.py` is not wired into `route()` and wiring it is a separate
  decision.
- **No new metrics.** This tool produces impressions, not measurements. The eval harness measures.

## 10. Risks, stated rather than mitigated away

**Arm C_llm acts on out-of-scope input about a third of the time.** Its OOD false-execution is
**0.32** — the constrained decoder is forced to emit one of the candidates it is shown. Type
something unrelated to a car and it may well do something. This is the central finding of Spec 2 and
the reason the arm decision is still open; a session is the fastest way to develop a view about it,
which is an argument for the default rather than against it. `/llm off` shows the contrast
immediately (arm C's OOD false-execution is 0.000).

**The permissive gate breaks refusal.** `low_top1` drops to 0.05, so almost nothing reaches LOW. In
an earlier probe `今天天气怎么样` came back as a parameter question instead of a refusal. That is a
property of the switch, not a defect to fix.

**Impressions are not measurements.** A dozen utterances that work say nothing about `recall@1
0.8644` or `param_exact_match 0.4133`. The tool is for forming hypotheses; the harness tests them.

## 11. Testing

`Session` is covered like anything else here:

- a turn that executes reports the right signal delta
- a turn refused by the car reports no delta and the vehicle's cause
- a turn rejected by validation reports that it never reached the car
- switching the gate changes the band for a known utterance (the 0.107-margin case above)
- switching the LLM off makes a MEDIUM utterance stop escalating
- state persists across turns, so a relative command resolves against the previous one
- `/reset` restores the seeded car

Model-dependent cases are marked `model` and excluded from the default run, matching the existing
convention. The renderer is pure and tested against fixed `Turn` values.

## 12. Success criteria

1. `python3 -m cli`, type `把主驾温度调到25度`, see a signal change and a confirmation.
2. `/llm off`, type the same words, see it stop at MEDIUM — the coverage problem, felt rather than read.
3. `/gate permissive`, type it again, see it execute deterministically.
4. Type something out of scope and see what arm C_llm does with it.
5. No production behaviour changes; the 456 tests and every eval metric are unmoved.
