"""S9 — what the SHIPPED gate costs, on the cases S6 already proves the router can do.

Every other file in tests/e2e/ runs at `PERMISSIVE` (t2f/gate.py) — a real shipped mode, the
one behind `--gate permissive`, but not the default one. That is a deliberate choice there:
those files are about execution, confirmations and refusals, and a gate that stops half the
cases before the car moves would test nothing about any of it.

This file runs the SAME 22 utterances through the gate a person actually gets, and asserts
what it costs. The assembly below is the product, not a reconstruction of it: `Config.load`
on config.yaml wholesale — its weights, its thresholds, its domain keywords — the real
embedder, `t2f.build.build_pipeline`, and a seeded `SqliteExecutor`. That is precisely what
cli/session.py:130-142 builds for `python3 -m cli --no-llm`.

The measurement, which is the whole content of this file:

    correct function reached        22 / 22
    clears the gate and executes    10 / 22

Recognition is not the problem. All 22 reach the right function, and all 22 clear `high_top1`
(0.35) with room to spare — the lowest top1 on 2026-08-02 was 0.7545 (打开玻璃水喷水), and
`test_the_margin_is_what_stops_them` below asserts the floor rejects nothing, so that claim
does not rest on this paragraph staying true.

The twelve that stop stop on `high_margin` alone: config.yaml wants a 0.12 gap between top1
and top2 before anything is dispatched with no LLM to check it, and in a 92-card catalog the
runner-up is a near-twin. Measured 2026-08-02, the three that show the shape:

    屏幕亮度调到80%   set_screen_brightness 0.8025  vs set_dashboard_brightness 0.7742  → 0.0283
    关闭车窗          open_window           0.7971  vs set_window_child_lock     0.7567  → 0.0404
    打开大灯          set_headlight         0.8714  vs set_high_beam             0.7871  → 0.0843

Note what the runner-up is NOT. `open_window` and `set_window_position` are one actuator in
the simulator (test_s6_success_matrix.py:112-113), which makes them sound like the confusable
pair, and they are not: on 关闭车窗 `set_window_position` ranks #5 at 0.6431, and on
主驾车窗开到百分之三十 `open_window` ranks #3 at 0.7358 behind `set_sunroof_position`. The
embedder confuses each with a *sibling control* — child lock, high beam, dashboard brightness
— not with its own other spelling. Do not transpose the simulator's actuator map onto the
ranking; the two disagree, and this sentence was wrong in exactly that way once.

**That is a decision, and this file does not argue with it.** config.yaml says so at the
threshold line: "a deliberate precision-over-coverage choice for zero-LLM safety". A test
demanding the shipped gate execute these twelve would be a test against the design. What was
missing was any record IN CI of the price — it lived in docs/TEST_REPORT.md §8 finding 2,
where it could go stale without a single test noticing.

So: if you are here because this file is red, the constant below is a MEASUREMENT of a
shipped decision, not a number to be kept green. Editing `_CLEARS_THE_SHIPPED_GATE` to match
a new run, with no accompanying change to config.yaml's thresholds and no note about why the
gate now admits or refuses something different, deletes the only thing this file was built to
say. The failure message tells you which cases moved and in which direction; that direction is
the finding. Write it down somewhere a person will read it.

Marked `model` in full: with `FakeEmbedder` these numbers are the hashed n-grams' opinion of
Chinese, not the router's, and a coverage figure measured on a stand-in is worth nothing.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from t2f.build import build_pipeline
from t2f.cards import load_catalog
from t2f.config import Config
from t2f.types import Band

from sim.executor import SqliteExecutor
from sim.seed import seed_from_catalog
from sim.vehicle import SqliteVehicle

# The same table S6 asserts against, imported and never copied: two lists of 22 utterances
# would be two lists the day someone adds a case to one of them, and this file's claim is
# specifically about the cases S6 proves are routable.
from .test_s6_success_matrix import CASES

pytestmark = pytest.mark.model

CARDS = load_catalog("data/catalog")
BY = {c.name: c for c in CARDS}


# The ten of S6's twenty-two that reach HIGH under config.yaml's thresholds and execute.
# Measured, not chosen. The identities and not the count: a bare `len(...) == 10` stays green
# through a change that promotes one case and demotes another, which is exactly the shape of
# the "sweep silently shrank to fit" failure this repo has hit twice (CLAUDE.md, Gotchas).
#
# The trailing margins and utterances are a convenience copy of what each case was on
# 2026-08-02, so the set reads as sentences rather than as slugs. Nothing checks them: if S6
# rewords a case they go quietly wrong, and only the ids are enforced.
#
# Read them for how thin the boundary is. The smallest margin that clears is misc-washer at
# 0.1298 — 0.0098 over the bar — and the largest that does not is display-hud at 0.1187,
# 0.0013 under it. So this set is decided at the third decimal place on both sides, and an
# embedder or weight change worth ~0.01 is enough to move it. That is a reason to read a
# failure here carefully, not a reason to distrust it: which way a case moves is still the
# finding, and the gate really does behave differently on either side of 0.12.
_CLEARS_THE_SHIPPED_GATE = {
    "climate-temperature-passenger",     # 副驾温度调到26度        margin 0.146
    "climate-fan-speed",                 # 风速调到三档            margin 0.188
    "seat-heating-driver",               # 主驾座椅加热开到三档      margin 0.155
    "media-volume",                      # 音量调到三档            margin 0.188
    "media-mute",                        # 打开静音               margin 0.154
    "media-shuffle",                     # 开启随机播放模式         margin 0.138
    "door-trunk",                        # 打开后备箱             margin 0.196
    "navigation-traffic-display",        # 打开路况显示            margin 0.213
    "misc-wiper-speed",                  # 雨刷调到三档            margin 0.139
    "misc-washer",                       # 打开玻璃水喷水           margin 0.130  ← thinnest pass
}


@dataclass(frozen=True)
class Outcome:
    """One utterance's fate at the shipped gate."""
    case_id: str
    utterance: str
    intended: str               # the function S6 says this utterance means
    chosen: str | None          # the function the router picked; None only at LOW
    band: Band
    top1: float
    margin: float
    executed: bool              # the vehicle's own log says the operation happened

    def __str__(self) -> str:
        return (f"{self.case_id:<28} {self.band.value:<6} top1 {self.top1:.3f}  "
                f"margin {self.margin:.3f}  -> {self.chosen}")


@pytest.fixture(scope="session")
def outcomes(real_embedder) -> dict[str, Outcome]:
    """Route all 22 through the shipped product once, and hand back what happened.

    Session-scoped because the four tests below are four readings of ONE measurement; running
    the matrix once per test would let them disagree about the same run.

    Every case gets its own freshly seeded car and its own pipeline, as S6 does — a shared car
    would let case order decide outcomes, and the memoised embedder makes rebuilding cheap.
    """
    config = Config.load("config.yaml")
    results = {}
    for case_id, utterance, intended, *_ in CASES:
        car = SqliteVehicle(":memory:")
        car.init_schema()
        seed_from_catalog(car, CARDS)
        executor = SqliteExecutor(car, BY)
        # No llm_client, no ood_texts, no `thresholds=` override: `--no-llm` passes none of
        # them either, so the gate that decides here is config.yaml's own.
        pipe = build_pipeline(CARDS, real_embedder, config, executor=executor)

        result = pipe.route(utterance)
        assert len(result.clauses) == 1, (
            f"{case_id}: expected one clause, got {len(result.clauses)} — this file measures "
            "the gate, and a re-segmented utterance is no longer the case S6 asserts")
        decision = result.clauses[0].decision
        log = car.recent_operations()
        results[case_id] = Outcome(
            case_id=case_id, utterance=utterance, intended=intended,
            chosen=decision.chosen, band=decision.band,
            top1=decision.features.get("top1", 0.0),
            margin=decision.features.get("margin", 0.0),
            executed=[(r["function"], r["outcome"]) for r in log] == [(intended, "executed")])
    return results


def test_every_case_reaches_its_intended_function(outcomes):
    """Recognition, measured apart from the gate.

    This is the load-bearing half of the finding: the twelve that do not execute are not
    twelve failures of the router. If this test ever fails, the gate discussion below is moot
    — something upstream of it broke, and the coverage number stops meaning what it says.
    """
    wrong = [o for o in outcomes.values() if o.chosen != o.intended]
    assert not wrong, (
        "the shipped assembly no longer routes these to the function S6 says they mean:\n"
        + "\n".join(f"    {o.case_id:<28} {o.utterance}\n"
                    f"        wanted {o.intended}, got {o.chosen} at {o.band.value} "
                    f"(top1 {o.top1:.3f})" for o in wrong)
        + "\n\n  This is a recognition regression, not a gate finding. The embedder, the "
          "fusion\n  weights in config.yaml, or the catalog's prototypes changed. Do not "
          "reach for\n  the threshold constants — nothing in this file is about them yet.")


def test_exactly_these_cases_clear_the_shipped_gate(outcomes):
    """The price of the shipped gate, pinned by identity.

    Ten of twenty-two. Which ten matters as much as how many: see the constant's comment.
    """
    reached_high = {o.case_id for o in outcomes.values() if o.band == Band.HIGH}

    # `.get`, not `[...]`: an id in the constant that S6 has since renamed or removed is a case
    # a person very much needs the message for, and a bare KeyError would replace it.
    def describe(case_id):
        o = outcomes.get(case_id)
        return f"      {o}" if o else f"      {case_id:<28} — no longer a case in S6 at all"

    gained = sorted(reached_high - _CLEARS_THE_SHIPPED_GATE)
    lost = sorted(_CLEARS_THE_SHIPPED_GATE - reached_high)
    if gained or lost:
        lines = ["the set of S6 cases that clear the SHIPPED gate has moved: "
                 f"{len(_CLEARS_THE_SHIPPED_GATE)} -> {len(reached_high)} of {len(CASES)}.", ""]
        if gained:
            lines.append("  now HIGH, and so now executed without an LLM:")
            lines += [describe(c) for c in gained]
        if lost:
            lines.append("  no longer HIGH — these used to execute and now stop at MEDIUM:")
            lines += [describe(c) for c in lost]
        lines += [
            "",
            "  config.yaml ships high_top1 0.35 / high_margin 0.12, and the margin is what",
            "  binds: see test_the_margin_is_what_stops_them below. If you changed the",
            "  thresholds, the fusion weights, the catalog or the embedder, you changed what",
            "  the product executes with no LLM present — which is a safety decision, and the",
            "  reason this set is written down rather than counted.",
            "",
            "  Updating the constant so the suite is green again, and nothing else, throws",
            "  away the only record of that. Record what moved and why it was worth it; then",
            "  update the constant because the decision changed, not because the test was red.",
        ]
        pytest.fail("\n".join(lines))


def test_clearing_the_gate_is_the_same_as_the_car_moving(outcomes):
    """HIGH and executed are one set — asserted where nothing can absorb it.

    Its own test rather than a second assertion in the one above, because an assertion that
    only runs when the set is unchanged is not verified: the moment the set moves, the earlier
    `pytest.fail` takes the report and this divergence never gets mentioned. Which is precisely
    when it would matter — a case newly at HIGH that then clarifies instead of acting, or one
    the vehicle refuses, is a very different event from a case that changed band.

    Nothing else in the file would catch it. `test_every_case_reaches_its_intended_function`
    reads the decision, and the set test reads the band; both are upstream of the car. This one
    reads the vehicle's own operation log, which is the only witness that an operation happened
    (S6's whole argument, and S5's).
    """
    reached_high = {o.case_id for o in outcomes.values() if o.band == Band.HIGH}
    executed = {o.case_id for o in outcomes.values() if o.executed}
    assert executed == reached_high, (
        "reaching HIGH and the car moving have come apart:\n"
        + "".join(f"    HIGH, but the car did not move: {outcomes[c]}\n"
                  for c in sorted(reached_high - executed))
        + "".join(f"    the car moved without HIGH:     {outcomes[c]}\n"
                  for c in sorted(executed - reached_high))
        + "  The coverage number this file reports counts operations the vehicle really\n"
          "  performed, not decisions the gate liked, and those have just stopped agreeing.\n"
          "  A HIGH case that stops short is asking a question (a required parameter went\n"
          "  missing) or being refused (a precondition, a physical limit); the operation log\n"
          "  and the clause's exec_error say which. Settle this before reading the set test —\n"
          "  until it holds, '10 of 22 execute' is not a claim this file can make.")


def test_the_margin_is_what_stops_them(outcomes):
    """Why the twelve stop — the claim config.yaml makes about itself, checked.

    "The binding constraint is high_margin: on confusable clusters top1/top2 are close, so
    only clear-margin cases go HIGH." The measurement is the FIRST assertion: every one of the
    22 clears `high_top1`, so the floor rejects nothing here at all. That is a fact about this
    matrix and could stop being true tomorrow.

    Be clear about the second assertion, which is not a second measurement. Given today's
    t2f/gate.py:40-44 the gate has exactly two ways to refuse — the floor and the margin — so
    once the floor is shown to reject nothing, "everything that stopped, stopped on the margin"
    follows by arithmetic. It is written down as a canary, not as evidence: it is what fails
    the day `ConfidenceGate.decide` grows a third rejection condition (an OOD score, a
    param_compat requirement, a per-domain rule), which is exactly when this file's account of
    the product would quietly become incomplete while every other test stayed green.

    Kept separate from the set test because the two thresholds are edited together and cost
    very different things. A config where `high_top1` does the rejecting is a different
    product — the utterance is no longer recognised confidently, rather than recognised
    confidently among near-twins — and the finding above would need rewriting, not renumbering.
    """
    config = Config.load("config.yaml")
    t = config.thresholds

    below_floor = sorted(o.case_id for o in outcomes.values() if o.top1 < t.high_top1)
    assert not below_floor, (
        f"high_top1 ({t.high_top1}) now rejects cases it used to admit: {below_floor}\n"
        "  This file's finding — 'recognition is fine, the margin is the cost' — no longer\n"
        "  describes the product. Re-read it before touching anything else here.")

    stopped = [o for o in outcomes.values() if o.band != Band.HIGH]
    not_on_margin = sorted(o.case_id for o in stopped if o.margin >= t.high_margin)
    assert not not_on_margin, (
        f"these stopped short of HIGH for some reason other than the {t.high_margin} margin: "
        f"{not_on_margin}\n"
        "  Both shipped thresholds now bind, so 'the margin is what it costs' has become an\n"
        "  incomplete account of the gate.")
