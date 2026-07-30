# Trying it yourself

A terminal session where you type Chinese and watch the in-vehicle workflow run against a
simulated car. This page stands alone — you do not need to read anything else first.

```bash
python3 -m cli
```

First start takes about a minute: it loads the embedding model and the fallback LLM. They load
once, then every turn is fast.

Add `--no-llm` to start without the fallback, `--gate permissive` to loosen the confidence gate,
`--scene-llm` to attach the [scene fallback](#the-second-model-scene-llm-on) as well, or `--fake`
for an instant start with no models at all. `--fake` is for checking the plumbing after a code
change — its embedder has no semantics and misroutes badly, so do not judge the system by it.

---

## What you are looking at

Type a command a driver might say. Every turn shows all four steps of the workflow:

```
[C_llm · shipped · S] > 把主驾温度调到25度

  recognised   set_temperature{temperature: 25.0, position: driver}    band=MEDIUM  → resolved by LLM
  executed     climate.driver/temperature   24.0 → 25.0
  reply        已将主驾温度设置为25°C。
```

| line | |
|---|---|
| `recognised` | which function it matched, with what parameters, and how confident it was |
| `executed` | **the actual row that moved in the car's database** |
| `reply` | what the driver would hear |

The prompt always states which system you are talking to, so you never have to remember: the
router's arm, the confidence gate, and which half of the [scene engine](#the-fifth-thing-let-the-car-start-the-conversation)
is attached.

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
| `/scene <key>=<value> [conf=] [ttl=]` | one perception event, as if the cabin camera saw it — see below |
| `/context` | every live observation: value, confidence, source, age, time to expiry |
| `/clock +30 \| -5` | move the session clock, to elapse a cooldown or expire an observation |
| `/scene-llm on\|off` | attach or detach the scene fallback — a second model |
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
[C · shipped · S] > 把主驾温度调到99度
  rejected     validation · 目标温度只能设置在16到32度之间 · never reached the car
  reply        目标温度只能设置在16到32度之间。
```

Now the same words with the LLM on:

```
[C_llm · shipped · S] > 把主驾温度调到99度
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
[C_llm · shipped · S] > /scene rear_occupant=child conf=0.9
  scene        rear_child_window_lock
  rule         rear_child_window_lock  match       all conditions met
  reply        后排有小孩，要打开儿童锁吗？

[C_llm · shipped · S] > 好

  scene        consent
  executed     window.all/window_child_lock   False → True
  reply        已为您打开车窗儿童锁。

[C_llm · shipped · S] > 开车窗

  recognised   open_window{is_open: True}    band=MEDIUM  → resolved by LLM
  refused      vehicle · 车窗儿童锁已开启 · nothing changed
  reply        车窗儿童锁已开启。
```

The `rule` line in the first beat is the engine showing its work. It matters most when the reply is
empty, which is most of the time — see [why nothing happened](#why-nothing-happened), below.

**Read the third beat again.** You asked to open a window and the car refused — because of something
*it* did, two turns earlier, after asking your permission. A proactive action changed what a later
driver command is allowed to do, and the refusal came back with its reason rather than a shrug. Both
entry points, one car:

```
[C_llm · shipped · S] > /car
  window.all/window_child_lock = True
[C_llm · shipped · S] > /log
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
  [C_llm · shipped · S] > 好像有点热

    recognised   set_temperature    band=MEDIUM  → resolved by LLM
    reply        抱歉，我不太确定您的意思，可以换个说法吗？

  [C_llm · shipped · S] > /car
    (the car is as it was seeded)
  ```

  A command must never be mistakable for consent, and `好` is inside `好像` — a substring test would
  have opened the lock here.
- **`conf=` is the perception confidence**, and this rule fires at 0.80. Below that but at or above
  0.50 the event is a *near-miss*: the rules will not act on it, and it is the only thing the scene
  fallback is ever offered. Below 0.50 the rule stops considering it at all.
- **Silence is the normal answer, and it now says why.** A weak detection, an observation no rule
  anticipated, a lock that is already on, a cooldown still running — all four are correct silences,
  and each one prints the reason it happened. That is the next section.

Only one scene exists so far — a child in the rear with the window child lock off — so
`rear_occupant=child` is the single event with anything to say. That is the whole shipped rule set,
not a sample of it, which is why every transcript below has exactly one `rule` line.

### Why nothing happened

This subsystem's commonest correct outcome is to say nothing, and a bare `scene —` cannot tell you
*which* nothing you got. The camera confidence was too low; the child lock is already on; a cooldown
from the last time it asked is still running; a question is already open and unanswered. Four
different decisions, one blank line — and no way to tell any of them from a bug.

So every scene turn prints one `rule` line per rule: the verdict, the reason behind it, and, when a
rule matched and *still* stayed quiet, what stopped it. Below that, a `fallback` line says what the
constrained model did or why it was not asked.

Type this into a fresh session — one perception event, repeated at two confidences and across one
cooldown:

```
[C_llm · shipped · S] > /scene rear_occupant=child conf=0.6
  scene        —
  rule         rear_child_window_lock  near_miss   inside.rear_occupant conf 0.60 in [0.50, 0.80)
  fallback     no model attached
  reply        —  (nothing spoken)

[C_llm · shipped · S] > /context
  inside.rear_occupant     child        conf 0.60  cabin_cam    age 0s  expires in 300s
[C_llm · shipped · S] > /scene rear_occupant=child conf=0.9
  scene        rear_child_window_lock
  rule         rear_child_window_lock  match       all conditions met
  reply        后排有小孩，要打开儿童锁吗？

[C_llm · shipped · S] > 不用

  scene        consent
  reply        好的。

[C_llm · shipped · S] > /scene rear_occupant=child conf=0.9
  scene        —
  rule         rear_child_window_lock  match       all conditions met  · suppressed: cooldown, 120s left
  fallback     no model attached
  reply        —  (nothing spoken)

[C_llm · shipped · S] > /clock +121
  → clock offset +121s
[C_llm · shipped · S] > /scene rear_occupant=child conf=0.9
  scene        rear_child_window_lock
  rule         rear_child_window_lock  match       all conditions met
  reply        后排有小孩，要打开儿童锁吗？
```

The `/scene` after `不用` is the one to look at. It is a `match` — every condition of the rule held,
exactly as they did when it asked — and the car said nothing, because the rule has already spoken
once and will not speak again for 120 seconds. Without `suppressed:` that turn and the very first
one print the same blank reply for opposite reasons.

Declining with `不用` rather than `好` is deliberate. Consent would have turned the lock on, and every
later event would then be a `reject` — you would never see the cooldown.

Four verdicts:

| verdict | what it means |
|---|---|
| `match` | every condition held. The rule spoke — unless `· suppressed:` names what stopped it: a cooldown with the seconds left on it, a question already asked and awaiting an answer, or a question the router is holding |
| `near_miss` | the right observation, not strong enough — between the rule's floor (0.50) and its threshold (0.80) — or not held for long enough. The rules will not act on it; it is what the scene fallback exists for |
| `reject` | a vehicle signal has already settled it. Nothing to ask about, and no model is consulted either |
| `not_applicable` | the rule has nothing to say about what was observed: no live observation for the key it watches, a different value, or a confidence below the floor |

`reject` is the one people meet by accident, one turn after saying `好`:

```
[C_llm · shipped · S] > /scene rear_occupant=child conf=0.9
  scene        —
  rule         rear_child_window_lock  reject      window.all/window_child_lock is already True
  fallback     no model attached
  reply        —  (nothing spoken)
```

The reason names the signal and its current value, so the silence is traceable to a row `/car` will
show you. And an observation the rule set never mentions is `not_applicable` — the rule is not
declining, it is not involved:

```
[C_llm · shipped · S] > /scene driver_state=drowsy conf=0.9
  scene        —
  rule         rear_child_window_lock  not_applicable  no live observation for inside.rear_occupant
  fallback     no model attached
  reply        —  (nothing spoken)
```

These strings are diagnostics, not speech. They name signals, keys and thresholds on purpose, and no
driver ever hears them.

### What perception believes: `/context`

`/car` shows the vehicle; `/context` shows the other half. Every observation the engine is currently
holding, with the confidence it arrived with, where it came from, how old it is, and how long it has
left:

```
[C_llm · shipped · S] > /scene driver_state=drowsy conf=0.9
  scene        —
  rule         rear_child_window_lock  not_applicable  no live observation for inside.rear_occupant
  fallback     no model attached
  reply        —  (nothing spoken)

[C_llm · shipped · S] > /scene outside.weather=rain conf=0.9
  scene        —
  rule         rear_child_window_lock  not_applicable  no live observation for inside.rear_occupant
  fallback     no model attached
  reply        —  (nothing spoken)

[C_llm · shipped · S] > /context
  inside.driver_state      drowsy       conf 0.90  cabin_cam    age 0s  expires in 300s
  outside.weather          rain         conf 0.90  cabin_cam    age 0s  expires in 300s
```

Keys are namespaced — `inside.`, `outside.`, `vehicle.`. A bare key is given `inside.`, which is why
`rear_occupant=child` works; write the namespace yourself to reach the others.

Only *live* observations are listed. An expired one is no longer part of what the engine reasons
over, and showing it would explain a decision by a belief that was not held. An empty context prints
`(no live observations)` rather than nothing, for the same reason the silences do.

### Moving the clock: `/clock +121`

Every decision the engine makes is a function of the time: an observation lives 300 seconds, the
rule will not repeat itself for 120, a question stays answerable for 30, the fallback has a
30-second budget. All of those are minutes long, so by hand they are unwatchable — you would sit at
the prompt waiting.

`/clock +121` moves the session's clock forward 121 seconds; a negative number goes back. Nothing
sleeps and nothing is re-evaluated — the offset is read the next time anything asks what time it is,
exactly as a real elapsed interval would be. In the transcript above it is what turns
`suppressed: cooldown, 120s left` back into a question. It is also the only way to watch an
observation expire:

```
[C_llm · shipped · S] > /scene rear_occupant=child conf=0.9
  scene        rear_child_window_lock
  rule         rear_child_window_lock  match       all conditions met
  reply        后排有小孩，要打开儿童锁吗？

[C_llm · shipped · S] > /context
  inside.rear_occupant     child        conf 0.90  cabin_cam    age 0s  expires in 300s
[C_llm · shipped · S] > /clock +301
  → clock offset +301s
[C_llm · shipped · S] > /context
  (no live observations)
```

The offset moves *now* forward for everything that asks afterwards; it does not backdate what is
already recorded. An observation you type after `/clock +121` is stamped at the shifted clock and
shows `age 0s` — it is a new observation, not a two-minute-old one.

### The second model: `/scene-llm on`

The rules are half of the scene subsystem. The other half is a constrained model, offered exactly
what the rules would not act on — a near-miss, or an observation no rule mentions — which decides
whether it is worth a sentence. `/scene-llm on` attaches it and `--scene-llm` starts with it
attached. It is a second copy of Qwen3-0.6B and takes about a minute to load. The prompt's last
segment tells you which half you have: `S` for rules only, `S_llm` with the fallback attached.

**Expect it to change nothing.** That is the measurement, not modesty about the demo: across the 13
rows of the scene gold set the fallback is reached on 2 — the other 11 are settled by a rule or by
the absence of one — and on both of those it declines to speak. `scene_recall` and
`scene_false_speech_rate` are identical in the two arms; the only figure that moves is
`avg_llm_calls_per_event`, from 0.0000 to 0.1538. Attaching it buys you two decodes and two more
silences.

Here it is, being consulted on the same 0.6 near-miss this section opened with:

```
[C_llm · shipped · S] > /scene-llm on
  loading the scene fallback (about a minute) ...
  → C_llm · shipped · S_llm
[C_llm · shipped · S_llm] > /scene rear_occupant=child conf=0.6
  scene        —
  rule         rear_child_window_lock  near_miss   inside.rear_occupant conf 0.60 in [0.50, 0.80)
  fallback     no_action · The scene does not match the provided information.
  reply        —  (nothing spoken)
```

Same silence as before, one decode more expensive — and the `fallback` line is the only place that
difference is visible. `no_action` is the model's decision and the rest is its own stated reason,
printed verbatim — it is a diagnostic, so nothing normalises it, including the language it arrives
in.

Most events never reach it at all, and the line says which ones. Started with `--scene-llm`, three
events in a row:

```
[C_llm · shipped · S_llm] > /scene rear_occupant=adult conf=0.9
  scene        —
  rule         rear_child_window_lock  not_applicable  inside.rear_occupant is 'adult', not 'child'
  fallback     no near-miss or unconsumed observation
  reply        —  (nothing spoken)

[C_llm · shipped · S_llm] > /scene driver_state=drowsy conf=0.9
  scene        —
  rule         rear_child_window_lock  not_applicable  inside.rear_occupant is 'adult', not 'child'
  fallback     no_action · The scene information is not matched with the available data.
  reply        —  (nothing spoken)

[C_llm · shipped · S_llm] > /scene rear_occupant=child conf=0.6
  scene        —
  rule         rear_child_window_lock  near_miss   inside.rear_occupant conf 0.60 in [0.50, 0.80)
  fallback     budget: 27s remaining
  reply        —  (nothing spoken)
```

An adult in the rear is a key the rules already account for, so there is nothing to hand over. A
drowsy driver is a key no rule mentions, so it goes to the model — which spends a decode and
declines. The near-miss immediately after would have gone too, but the fallback runs at most once
every 30 seconds and 27 of them were left. Three silences, three different causes, one of them
costing real time. With no model attached the line reads `no model attached` throughout, which is
what every transcript above this section shows.

Under `--fake` the command refuses rather than attaching a scripted stand-in:

```
[C_llm · shipped · S · FAKE] > /scene-llm on
  the scene fallback needs a real model — restart without --fake
```

A fake here would fabricate the one thing someone attaching the model wants to observe: the line on
screen would be a decision this tool wrote, not one a model made.

---

## The same session, in a browser

```bash
python3 -m ui
```

Then open `http://127.0.0.1:8770`. Same flags as the terminal — `--fake`, `--no-llm`, `--gate`,
`--db`, plus `--port`.

This is not a second system. It builds a session exactly the way `python3 -m cli` does and drives it
through the same methods, so anything you can do here you can do there. It exists because **a text
table cannot show perception decaying.** `/context` tells you an observation has 240 seconds left;
the page shows the bar draining, and you watch the belief age out.

Four panes:

| | |
|---|---|
| **Perception** | one card per observation, with a TTL bar that drains continuously and a confidence bar marked with the rule's floor and threshold — a near-miss reads as a *position*, not a number to compare in your head |
| **The car** | signals that differ from the seeded vehicle, flashing when they move |
| **Rules** | every rule with its verdict, its reason, and what suppressed it — the same thing the terminal prints, but standing still instead of scrolling past |
| **Conversation** | the transcript, plus the pending question with a live countdown and **Yes** / **No** buttons |

Those two buttons submit the utterances `好` and `不用`. They are not a shortcut past the consent
lexicon — there is exactly one route to the car and the page uses it, the same one you type into at
the terminal. A button wired straight to the vehicle would be a second route with different rules,
which is the thing this design spends most of its effort preventing.

The header carries the clock control and the scene-fallback toggle, so the whole of `/clock` and
`/scene-llm` is there too.

**Two things worth knowing.** The server is single-threaded on purpose — the simulated car's SQLite
connection belongs to one thread, and a threaded server would quietly serve a page showing a car
that never moved rather than failing loudly. And an observation is validated before it is recorded:
an empty key, an empty value, or a confidence outside 0–1 is refused, because every rule's bands
live in 0–1 and an observation at 7.0 would clear any of them trivially.

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
