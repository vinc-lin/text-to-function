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

A turn the car starts by itself looks different — a `scene` line instead of `recognised`, because
nobody said anything. That is `/scene`, and it has [its own section](#the-fifth-thing-let-the-car-start-the-conversation).

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
| `/scene <key>=<value> [conf=]` | one perception event, as if the cabin camera saw it — see below |
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
  reply        已为您打开当前区域车窗。已将当前区域风速设置为3档。
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
99 and it *cannot* say "impossible". It emits something legal. Two more of the same shape, measured:

| you say | LLM emits | what actually happens |
|---|---|---|
| `屏幕亮度调到200%` | `percent: 20` | brightness set to 20%, announced as success. Arm C rejects: `屏幕亮度只能设置在0到100%之间` |
| `主驾温度再调高一点` | `temperature: 32` | jumps 25 → 32 rather than one step |

`风速调到20档` used to belong in that table and no longer does: it now matches at HIGH confidence, so
the deterministic extractor fills it and validation rejects the 20 before any model is consulted.
Both arms answer `风速档位只能设置在1到7档之间。`. The substitution only happens in the MEDIUM band —
which is exactly the band the LLM exists to resolve.

Arm C is silent-substitution-free by construction: it either acts on what you said or tells you why
it cannot. Arm C_llm does far more (`e2e` 0.62 against 0.13) and sometimes does something you did
not ask for. **That trade is the open question in this project, and four minutes here will tell you
more about it than the metrics will.**

---

## The fifth thing: let the car start the conversation

Everything above begins with you typing a command. `/scene` starts from the other end — it hands the
system one **perception event**, the kind a cabin camera would produce, and lets it decide whether
that is worth saying anything about. Nothing else in this session is proactive.

Type it in three beats. This is a real transcript, models loaded, fresh car:

```
[C_llm · shipped] > /scene rear_occupant=child conf=0.9
  scene        rear_child_window_lock
  reply        后排有小孩，要打开儿童锁吗？

[C_llm · shipped] > 好

  scene        consent
  executed     window.all/window_child_lock   False → True
  reply        已为您打开车窗儿童锁。

[C_llm · shipped] > 开车窗

  recognised   open_window{is_open: True}    band=MEDIUM  → resolved by LLM
  refused      vehicle · 车窗儿童锁已开启 · nothing changed
  reply        车窗儿童锁已开启。
```

**Read the third beat again.** You asked to open a window and the car refused — because of something
*it* did, two turns earlier, after asking your permission. A proactive action changed what a later
driver command is allowed to do, and the refusal came back with its reason rather than a shrug. Both
entry points, one car:

```
[C_llm · shipped] > /car
  window.all/window_child_lock = True
[C_llm · shipped] > /log
  set_window_child_lock    executed
  open_window              refused · precondition_failed · 车窗儿童锁已开启
```

The scene-initiated write is in the same operation log as the one you typed, because it went through
the same executor.

Four things worth knowing while you play with it:

- **The engine never moves the car on its own.** The most a scene can do is ask. `好` is what
  executes; without it nothing happens, and the question expires by itself after 30 seconds.
- **`好` has to be the whole utterance.** Consent is exact membership in a closed list, never a
  substring test. Type `好像有点热` after the question and it is routed as an ordinary command — the
  pending question is dropped and the lock stays where it was:

  ```
  [C_llm · shipped] > 好像有点热

    recognised   set_temperature    band=MEDIUM  → resolved by LLM
    reply        抱歉，我不太确定您的意思，可以换个说法吗？

  [C_llm · shipped] > /car
    (the car is as it was seeded)
  ```

  A command must never be mistakable for consent, and `好` is inside `好像` — a substring test would
  have opened the lock here.
- **`conf=` is the perception confidence**, and this rule fires at 0.80. `/scene rear_occupant=child
  conf=0.6` prints `scene —` and `reply —  (nothing spoken)`: below the threshold but above the
  floor, it is a *near-miss*, which is the fallback model's business and never the car's.
- **Silence is the normal answer.** The session attaches no scene fallback model, so a near-miss or
  an observation no rule anticipated — `/scene driver_state=drowsy conf=0.9` — is silence here too.
  It prints as a decision rather than a blank line so you can tell it apart from a bug.

Only one scene exists so far — a child in the rear with the window child lock off — so
`rear_occupant=child` is the single event with anything to say. That is the whole shipped rule set,
not a sample of it.

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

**And so is the perception.** `/scene` is you typing what a cabin camera would have reported. There
is no camera and no vision model here; the scene gold file is hand-authored the same way, so
`scene_recall 1.000` measures agreement with our own beliefs about what perception would say.

**There is no speech.** Input is typed and output is text. ASR and TTS belong to the surrounding
voice stack, not to this repo.

---

## If something goes wrong

A turn that raises prints the error and returns you to the prompt rather than killing the session —
a crash would cost you the model load. `/reset` clears a car you have got into a strange state.
`--db car.sqlite` keeps the vehicle on disk so you can inspect it with `sqlite3` afterwards.
