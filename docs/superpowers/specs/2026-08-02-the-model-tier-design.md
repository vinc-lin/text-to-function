# The model tier — end-to-end without the stand-in embedder

**Date:** 2026-08-02
**Goal:** run the end-to-end suite against the **real** embedder, so that "the utterance reached
the right function and the car really moved" is a claim about the shipped system rather than
about a hashed-n-gram stand-in.

> Every number below was measured on this tree, not estimated. Machine: the development box, CUDA
> available, `Qwen/Qwen3-Embedding-0.6B` at `mrl_dim=512`. The design was measured before it was
> written; the passages marked **as built** / **discharged** were added afterwards, from the
> implementation, where it moved past what this document predicted.

---

## 1. What is wrong today

`tests/e2e/` has 120 tests and none of them has ever seen the embedding model. They use
`FakeEmbedder` — md5-hashed character 3-grams into 256 buckets, L2-normalised, **no semantics**.

The tests know this and say so. `test_s6_success_matrix.py` is candid in its own docstring:

> `FakeEmbedder` is a hashed-n-gram stand-in with NO semantics, so over 92 cards many plausible
> utterances misroute badly (「关闭屏幕」 reaches `next_track`, 「座椅往后移」 reaches `play_radio`).
> **Every utterance below was therefore probed against the full catalog first and kept only
> because it reached the function it names.**

That last sentence is the problem, and it is worth being precise about *why*. It is not that the
assertions were weakened — they were not, and the file is careful to say so. It is that the
**inputs were selected until they passed**. A suite whose stimuli are chosen by the system under
test measures the selection, not the system. It cannot fail for the reason it exists to catch.

So the shipped question — *does the real Qwen3 embedder, over the real 92-card catalog, route
these utterances to these functions?* — has no test. It has a metric, in an offline harness, that
CI does not run.

### What is NOT wrong today

One accommodation looks like a test hack and turns out not to be. Every one of these files sets

```python
Thresholds(high_top1=0.2, high_margin=0.0, low_top1=0.05)
```

and `tests/e2e/conftest.py` calls them *"loosened so the FakeEmbedder reaches the HIGH band"*.
They are byte-identical to `cli/session.py:30`:

```python
PERMISSIVE = Thresholds(high_top1=0.2, high_margin=0.0, low_top1=0.05)
```

— the gate behind `python3 -m cli --gate permissive` and the `/gate permissive` command. **The
e2e suite runs a supported product mode, not an invented one.** The docstring describing it as a
test accommodation is simply wrong, and is corrected by this work.

The triple is currently re-typed in **eleven files, twelve times** — nine test files, the eval
runner, and `cli/session.py`, the only one that gives it a name. Nothing keeps them equal; changing
the CLI's `PERMISSIVE` would leave eleven silent copies of the old gate behind.

## 2. The measurement that shaped this design

Three files already use the real 92-card catalog: `test_s5_simulator.py` (7),
`test_s6_success_matrix.py` (24), `test_s7_failure_matrix.py` (26) — **57 tests**. They are also
the three that assert a row in the vehicle database really moves. Swapping *only* the embedder,
leaving the permissive gate exactly as it is:

```
57 tests, real Qwen3-Embedding-0.6B, real catalog, permissive gate
    56 passed, 1 failed        5.6 s after model load
```

**Fifty-six of fifty-seven pass unchanged.** The tier is nearly free, and what it buys is that
those 56 assertions stop depending on a stand-in.

### The one failure is a real finding

`test_a_numeric_parameter_given_a_word_is_told_it_needs_a_number` forces the MEDIUM band by
demanding an unreachable score:

```python
_pipeline(Thresholds(high_top1=0.6, high_margin=0.0, low_top1=0.05), llm_client=llm)
```

| embedder | top1 for 「空调温度调一下」 | band under a 0.6 bar |
|---|---|---|
| `FakeEmbedder` | 0.4979 | MEDIUM — the LLM path the test wants |
| real | **0.6305** | HIGH — the deterministic path, so the test's premise is gone |

Both rows are under `Config.default()`, which is what every e2e file used when this was measured.
Under the shipped weights the `real` profile finally adopted, the same words score **0.8087** —
further over the bar, not less, so the conclusion is unchanged.

The bar was set just above what the fake embedder happens to score. It is not a wrong test; it is
a test whose *mechanism* silently depends on the embedder. The fix is to make the intent
explicit — force MEDIUM with a bar no embedder can clear — rather than to re-tune a number. As
built it is `high_top1=2.0`: unreachable by arithmetic while the weights sum to 1.0 and every
component signal is ≤ 1.0, so no embedder clears it, present or future.

### What the second witness actually caught

One case in `test_s7_failure_matrix.py` behaves differently under the real profile, and it is
the kind of thing this tier was built to find. 「空调模式调到3」 — *set the AC **mode** to 3* —
is a bad-enum case: the right answer is to name the five valid modes and dispatch nothing.

| configuration | winner | runner-up | reply |
|---|---|---|---|
| fake + default weights | `set_ac_mode` 0.3979 | | names the modes — correct |
| real + default weights | `set_ac_mode` 0.4610 | `set_temperature` 0.4225 | correct |
| **real + shipped weights** | **`set_temperature` 0.6760** | `set_ac_mode` **0.6709** | 您想设置到多少度？ |

**The margin is 0.0051.** That number is what makes this reportable rather than alarming: it is
a coin-flip on a confusable pair, not a confident misclassification. Under `config.yaml`'s own
`high_margin: 0.12` the clause lands in MEDIUM and goes to the LLM tier — precisely the case the
shipped gate's margin requirement exists to catch, as `config.yaml` says at the point of
decision. It is visible here only because the harness runs `PERMISSIVE`, which zeroes that
requirement.

So the finding is a **ranking** weakness under embedding-dominant weights, on a pair the lexical
signals used to separate — recorded as `xfail(strict=True)` on the real profile alone, so that
if ranking improves the suite says so instead of going quietly green.

The safety half of that case — nothing dispatched, the car unchanged — still holds and is still
asserted, in its own always-green test. An assertion inside an `xfail` body is unguarded:
`xfail(strict)` reports the same result whichever line trips, so a safety check placed after a
failing one is silently absorbed. `tests/e2e/test_s4b_failure_cause.py:73-79`
(`test_s4b_06_bad_enum_dispatches_nothing`) had already found this and written it down; the tier
follows that pattern rather than inventing a second one. It is now in `CLAUDE.md`'s Gotchas, having
come up twice.

### What the shipped gate costs, measured separately

Running the same utterances through the genuine product assembly — `build_pipeline` +
`config.yaml` (its weights, its domain keywords, and the shipped `0.35/0.12/0.15` gate; **not** its
`classifier:` block, which `build_pipeline` does not read) + real embedder:

```
s6:  correct function 22/22        reaches HIGH and executes 10/22
s5:  correct function  2/2         reaches HIGH and executes  0/2
```

**Recognition is not the problem. The margin requirement is** — and more literally than that
sentence first suggested. The lowest top1 across all 22 is **0.7545**, against a `high_top1` of
0.35, so the score floor rejects *nothing* on this matrix. The entire twelve-case cost is
`high_margin` alone, exactly as `config.yaml` says of itself.

The boundary is thin on both sides: the tightest clearance is `misc-washer` at margin 0.1298
(0.0098 over the bar) and the closest miss is `display-hud` at 0.1187 — **0.0013 under**. An
embedder change worth a hundredth of a point moves this set, which is why the test asserts its
membership rather than its size.

This is not a defect, and the tier must not assert it away. `config.yaml` says so at the point of
decision:

> Conservative HIGH band keeps incorrect execution low at the cost of coverage; the medium band is
> what the Spec-2 LLM will resolve. The binding constraint is `high_margin`.

A test demanding that the shipped gate execute these would be a test against the design. What is
missing is not an assertion that it *should* execute — it is any record, in CI, of **how much**
coverage that choice costs. Today that lives in `docs/TEST_REPORT.md` §8 finding 2 and moves
without anything noticing.

## 3. The design

### Two profiles over the same test bodies

`tests/e2e/conftest.py` gains a parametrized `profile` fixture:

| profile | embedder | gate | when it runs |
|---|---|---|---|
| `fake` | `FakeEmbedder(256)` | permissive | always — the fast suite, unchanged |
| `real` | `TransformersEmbedder` | permissive | `-m model` only |

The `real` param carries `marks=pytest.mark.model`, so `addopts = "-m 'not model'"` deselects it
by default and `python3 -m pytest -m model` selects it. **The test bodies do not change.** One
suite, two witnesses — and any assertion that only holds for one of them is now visible.

The real embedder loads **once per session** and memoises `encode`, because 57 pipeline
constructions would otherwise re-embed 92 cards 57 times. Measured: 5.6 s for all 57 with the
memo, against a ~1 min model load that happens once. As built, `pytest -m model tests/e2e/` is 69
tests in **13 s** — the same three files, whose rows grew; see §5 for where the extra twelve came
from.

Two things had to be added to keep the fixture from costing the default run, neither of which this
section anticipated. `profile` resolves its embedder through `request.getfixturevalue` rather than
declaring `real_embedder` as a parameter — a fixture argument is resolved before the body runs, so
naming it would load the model for the `fake` param too, on every default run. And a
`pytest_collection_modifyitems` hook fails collection if any test declares `real_embedder` without
the `model` marker, because that mistake reads as "the suite got slow", not as a mistake.

### The profile is an embedder *and* a weight set

Found while building the fixture, by mutation rather than by reading: zero the embedder and
**10 of the 29 cases still pass**. They never needed it. `Config.default()` — which every e2e
file uses — weights the coarse lexical signals heavily enough to clear the permissive floor
without any embedding signal at all:

| | `keyword_alias` | `param_compat` | `domain_prior` | sum | clears `high_top1=0.2`? |
|---|---|---|---|---|---|
| `Config.default()` | 0.15 | 0.25 | 0.05 | **0.45** | yes |
| `config.yaml` (shipped) | 0.04 | 0.05 | 0.03 | **0.12** | no |

**The shipped row omits a fifth non-embedding weight, and the omission is load-bearing.**
`config.yaml:9` also weights `classifier_prob` at 0.15, which would take the row to 0.27 and clear
the floor. It does not, because `t2f/build.py` wires no classifier source: `t2f/score.py:29` reads
`cp` as 0.0 for every card, so the term is dead everywhere reachable from the product factory —
`cli --no-llm`, the e2e profiles, and the gate-cost test. Only `eval/arms.py::build_arm_d` supplies
one, and it builds its `Pipeline` directly rather than through the factory.
`Config.default()` needs no such caveat; its `classifier_prob` is 0.0 outright. Anyone wiring a
classifier into `build_pipeline` invalidates the conclusion below, and nothing will fail to say so.

The shipped weights are embedding-dominant on purpose — `config.yaml` records that a dev-set
sweep found the lexical signals do not improve ranking over the embedder alone. So a tier that
kept `Config.default()` would hand a third of its cases to features the product barely uses.

Measured, all three:

```
real embedder + shipped weights              29 passed
zeroed embedder + Config.default() weights   10 passed, 19 failed
zeroed embedder + shipped weights             1 passed, 28 failed
```

**The `real` profile therefore carries the shipped weights, and `fake` keeps `Config.default()`.**
Not an inconsistency: `FakeEmbedder` under embedding-dominant weights misroutes badly (measured
— `打开主驾车门` breaks), so the fast profile genuinely needs the lexical help, while the real
one must not have it. Each profile runs the configuration under which it means something.

**Weights only, not `config.yaml` wholesale.** Its `thresholds` are the shipped gate, under
which twelve of s6's cases land in MEDIUM — that is the subject of the gate-cost test below,
not something to import silently into every case.

### Scope: the three real-catalog files, and not the other five

`s2`, `s3`, `s4a`, `s4b`, `s8` (63 tests) run on a **3-card fixture catalog**. They test reply
composition, refusal causes and the contract sweep — mechanism that is catalog-independent and
embedder-independent. Giving them the real embedder would mean rewriting 63 utterances against
92 cards to buy nothing: with three cards there is no confusion for a better embedder to resolve.

They stay as they are. Naming this is part of the design, not an omission.

### The gate's cost becomes an assertion

One new file, in the model tier, over s6's own 22 utterances through the **shipped** assembly:

- every utterance reaches its intended function — `22/22`, asserted exactly;
- the cases that clear the shipped gate are asserted **by identity** — the set of ten case ids,
  not the count. A bare `len(...) == 10` stays green through a change that promotes one case and
  demotes another, which is the "sweep silently shrank to fit" shape this repo has hit twice.

Both directions matter. If recognition regresses the first fails. If the gate is retuned, or the
embedder improves, or the margin problem is fixed, the second fails, names which cases moved and
in which direction, and asks for a fresh look at `TEST_REPORT` §8. It converts a figure that
drifts silently into one that cannot change without someone deciding it should.

As built it is four tests rather than one, and the extra two are not padding. `HIGH` and `the car
moved` are asserted equal in a test of their own, because an assertion living inside the set test
would never run on the day the set moves — `pytest.fail` takes the report first, which is exactly
when a case that reached HIGH and then asked a question instead of acting would matter. And "the
margin is what stops them" is separated from the set, because the two thresholds are edited
together and cost very different things: a config where `high_top1` does the rejecting is a
different product, and this file's finding would need rewriting rather than renumbering.

### One home for the permissive gate

`PERMISSIVE` moves to `t2f/gate.py`, beside the `Thresholds` it is an instance of, and the eleven
files import it. It is a product mode, so it belongs with the product rather than in a dev tool
that tests reach into. `cli/session.py` keeps the name it exports today.

## 4. What is not changing

- `t2f/` behaviour — the move of `PERMISSIVE` is a constant changing modules, nothing else.
- The default `python3 -m pytest` run: same runtime, and the model tier deselected exactly like
  the five `@model` tests already are. **Not** the same count, as built: it went 1125 → 1135,
  because ten rows of S7's bad-value table had their safety half split into an always-green test
  once one of them started xfailing under the real profile.
- The fixture catalog, `FakeEmbedder`, and the five files that use them. The fast tier keeps
  being the fast tier; it stops being the *only* tier for the three files where that mattered.
- The shipped gate. Its cost is recorded, not litigated.

## 5. Proof obligations

```bash
python3 -m eval.run_eval --arm C --dataset data/eval/gold.jsonl --fake --permissive
python3 -m eval.run_scene_eval --arm S
```

**Byte-identical.** Both reports are gitignored, so diff them explicitly; arm C's two latency
lines are wall-clock jitter. Task 1 touches `t2f/gate.py` and is the one that could move them.

Plus: `python3 -m pytest -q`, **1125 passed, 1 skipped, 5 deselected** before the work.

**Discharged.** Against the commit this branch was cut from: `run_scene_eval --arm S`
byte-identical, `run_eval --arm C --fake --permissive` identical but for `p50_latency_ms` and
`p95_latency_ms`. The default run ended at **1135 passed, 1 skipped, 74 deselected** — the ten
extra passes are accounted for above.

Deselected was predicted here as **62** — the 5 existing `@model` tests plus the 57 of §2, all
three files included. It came out at **74**, and the twelve are three separate things, none of
them a change of scope:

| | | |
|---|---:|---|
| s6 has two tests that do not take the fixture | **−2** | `test_every_case_touches_a_distinct_function` and `test_the_matrix_spans_every_domain_it_can` assert over `CASES` and never build a pipeline, so 22 of its 24 are second-witness cases |
| s7's ten bad-value rows split their safety half out | **+10** | forced by the xfail in §2, not by the profile |
| S9 shipped four tests where §3 predicted one | **+4** | the reason is under *The gate's cost becomes an assertion* |

`5 + 7 + 22 + 36 + 4 = 74`, against the predicted `5 + 7 + 24 + 26 = 62`.

## 6. Risks

1. **The model tier will not run in most environments.** It needs the model and a GPU to be
   tolerable. That is already true of the five existing `@model` tests; this makes the
   deselected fraction large enough to be forgotten — as built, 74 of 1210 tests. Mitigated only
   by saying so in `CLAUDE.md` next to the command that runs it, which is done, and in
   `TEST_REPORT` §15's account of what the tier does not prove.
2. **A memoised embedder is not quite the shipped one.** It caches by `(text, is_query)` — never
   text alone, because `TransformersEmbedder.encode` prepends the query instruction when
   `is_query` is set, so one string has two correct and different vectors and a text-keyed cache
   would hand back whichever was asked for first. Safe for a deterministic encoder, and what makes
   69 tests take 13 s instead of minutes. If the encoder ever becomes context-dependent the cache
   silently lies. Stated here; the tier builds it in one place so there is one thing to delete.
3. **A profile that silently degraded would look like success.** A fixture falling back to
   `FakeEmbedder` on a load error, or a cache handing back the wrong vector, would produce a
   green run indistinguishable from a real one — the exact failure mode this repo has hit
   twice with contract sweeps that shrank to fit. The tier is therefore verified by mutation:
   zero the embedder and the real profile must collapse. A green run is not evidence.
4. **The ten that clear the shipped gate will need updating.** That is the point of asserting
   them, but a reviewer who sees the test fail and simply edits the set has defeated it — and the
   boundary is thin enough (0.0098 over on one side, 0.0013 under on the other) that an embedder
   change worth a hundredth of a point moves it. The test says so in its own message: which way a
   case moved is the finding, and the set changes because the decision changed, not because the
   test was red.
