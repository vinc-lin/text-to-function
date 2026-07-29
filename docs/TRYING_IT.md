# Trying it yourself

A terminal session where you type Chinese and watch the in-vehicle workflow run against a
simulated car. This page stands alone — you do not need to read anything else first.

```bash
python3 -m cli
```

First start takes about a minute: it loads the embedding model and the fallback LLM. They load
once, then every turn is fast.

Add `--no-llm` to start without the fallback, `--gate permissive` to loosen the confidence gate, or
`--fake` for an instant start with no models at all. `--fake` is for checking the plumbing after a
code change — its embedder has no semantics and misroutes badly, so do not judge the system by it.

---

## What you are looking at

Type a command a driver might say. Every turn shows all four steps of the workflow:

```
[C_llm · shipped] > 把主驾温度调到25度

  recognised   set_temperature{temperature: 25.0, position: driver}    band=MEDIUM  → resolved by LLM
  executed     climate.driver/temperature   24.0 → 25.0
  reply        已将主驾温度设置为25°C。
```

| line | |
|---|---|
| `recognised` | which function it matched, with what parameters, and how confident it was |
| `executed` | **the actual row that moved in the car's database** |
| `reply` | what the driver would hear |

The prompt always states which system you are talking to, so you never have to remember.

### The five things a turn can do

| | meaning |
|---|---|
| `executed` | it happened; the signal that moved is shown |
| `rejected` | a parameter was unusable — **it never reached the car** |
| `refused` | it reached the car and the car said no |
| `asked` | it needs something from you before it can act |
| `unresolved` | the medium band with no model attached — see the gate section |

`rejected` versus `refused` is the distinction worth watching. A rejection is caught by validation
and the vehicle never hears about it; a refusal means the car was asked and declined, and it is
recorded in `/log`.

A span with **no outcome line at all** is `asked` where the question is the whole reply — the line
below says it, and printing the same sentence twice reads as two separate questions. An utterance
that produces several questions shows each one, because only the first is ever spoken.

---

## The car is real and it remembers

The vehicle is a SQLite database seeded from the 92-function catalog, and it **persists for the
whole session**. That is what makes a conversation work:

```
> 把主驾温度调到25度        climate.driver/temperature   24.0 → 25.0
> 主驾温度再调高一点         climate.driver/temperature   25.0 → 26.0
```

The second command has nothing to resolve against unless the first one really changed something.
`/car` shows everything that differs from the seeded state; `/reset` gives you a fresh vehicle.

---

## Commands

| | |
|---|---|
| `/llm on\|off` | attach or detach the fallback model |
| `/gate shipped\|permissive` | switch the confidence thresholds |
| `/car` | every signal that differs from the seeded car |
| `/log` | recent operations, and what the car said about each |
| `/reset` | fresh car |
| `/help`, `/quit` | |

Switching a mode **keeps the car exactly as it is**, so you can type the same words twice and
compare the two systems against the same vehicle.

---

## The two switches, and why they matter

**`/llm`** decides who resolves the middle ground. The router sorts each recognition into three
bands: confident enough to act (HIGH), too weak to touch (LOW), and everything between (MEDIUM).
With the LLM attached, MEDIUM is handed to Qwen3-0.6B to pick and fill. Without it, MEDIUM is a
dead end — you will see `unresolved` a lot, which is the honest behaviour of the zero-LLM build.

**`/gate`** decides how confident the router must be. The shipped thresholds require the best match
to beat the runner-up by a margin; the permissive ones do not. Utterances routinely match the
*correct* function at a high score and still land in MEDIUM because a similar function scored close
behind — `把主驾温度调到25度` scores 0.801 with a margin of 0.107, against a required 0.12.

Try the same sentence four ways. The differences are the whole point of the tool.

---

## Four things worth trying first

### 1. Something ordinary

```
> 把主驾温度调到25度
  executed     climate.driver/temperature   24.0 → 25.0
  reply        已将主驾温度设置为25°C。
```

### 2. Two things at once

```
> 开车窗，风速调到3档
  recognised   open_window{is_open: True}    band=MEDIUM  → resolved by LLM
  executed     window.all/window_position   50 → 100
  recognised   set_fan_speed{level: 3}    band=HIGH
  executed     climate.all/fan_speed   4 → 3
  reply        已为您调整当前区域车窗状态。已将当前区域风速设置为3档。
```

Two actions, two signals, and exactly one thing said to the driver.

### 3. Something out of scope

```
> 帮我讲个笑话
  recognised   —    band=LOW
  reply        抱歉，我不太确定您的意思，可以换个说法吗？
```

### 4. Something impossible — and this is the one to pay attention to

Ask for a temperature the car cannot reach, first with the LLM off:

```
[C · shipped] > 把主驾温度调到99度
  rejected     validation · 目标温度只能设置在16到32度之间 · never reached the car
  reply        目标温度只能设置在16到32度之间。
```

Now the same words with the LLM on:

```
[C_llm · shipped] > 把主驾温度调到99度
  recognised   set_temperature{temperature: 16.0, position: driver}    band=MEDIUM  → resolved by LLM
  executed     climate.driver/temperature   32 → 16.0
  reply        已将主驾温度设置为16°C。
```

**You asked for 99 and the car went to 16 — the opposite extreme — and told you it succeeded.**

This is not a bug in the session; it is the system, and the session is how you find it. The
fallback model decodes under a JSON schema that only permits values in range, so it *cannot* emit
99 and it *cannot* say "impossible". It emits something legal. Two more, measured in the same run:

| you say | LLM emits | what actually happens |
|---|---|---|
| `风速调到20档` | `level: 2` | fan set to 2, announced as success. Arm C rejects: `风速档位只能设置在1到7档之间` |
| `主驾温度再调高一点` | `temperature: 32` | jumps 25 → 32 rather than one step |

Arm C is silent-substitution-free by construction: it either acts on what you said or tells you why
it cannot. Arm C_llm does far more (`e2e` 0.62 against 0.13) and sometimes does something you did
not ask for. **That trade is the open question in this project, and four minutes here will tell you
more about it than the metrics will.**

---

## What this does not tell you

**Impressions are not measurements.** A dozen utterances that work say nothing about
`recall@1 0.8644` or `param_exact_match 0.4133`. Use `python3 -m eval.run_eval` for numbers.

**The default acts on out-of-scope input about a third of the time.** Arm C_llm's OOD
false-execution is 0.32 — the decoder is forced to choose among the candidates it is shown. Arm C's
is 0.000. `/llm off` shows the contrast immediately.

**The permissive gate breaks refusal.** It drops the floor low enough that almost nothing reaches
LOW, so out-of-scope input gets a parameter question instead of a refusal.

**The car is a simulation.** `sim/` is a SQLite stand-in behind the same seam a real vehicle bus
adapter would use. Preconditions and physical limits are modelled; a real car is not.

**There is no speech.** Input is typed and output is text. ASR and TTS belong to the surrounding
voice stack, not to this repo.

---

## If something goes wrong

A turn that raises prints the error and returns you to the prompt rather than killing the session —
a crash would cost you the model load. `/reset` clears a car you have got into a strange state.
`--db car.sqlite` keeps the vehicle on disk so you can inspect it with `sqlite3` afterwards.
